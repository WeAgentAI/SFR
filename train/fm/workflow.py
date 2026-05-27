# Copyright 2025 the LlamaFactory team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Workflow for AR + Flow Matching joint training.

Architecture:
  Phase 0 (pre-training): Encode all response texts using a frozen sentence
      encoder in a **subprocess** (fully isolated from DeepSpeed/DDP).
      Embeddings are written back to the dataset as a new column.

  Phase 1 (training): Standard SFT data pipeline + CustomFMTrainer that reads
      pre-computed z_1 from each batch and computes the FM auxiliary loss.
      No target encoder exists in this phase.
"""

from typing import TYPE_CHECKING, Optional

import torch

from ...data import SFTDataCollatorWith4DAttentionMask, get_dataset, get_template_and_fix_tokenizer
from ...extras.constants import IGNORE_INDEX
from ...extras.logging import get_logger
from ...extras.misc import calculate_tps
from ...extras.ploting import plot_loss
from ...model import load_model, load_tokenizer
from ...model.model_utils.fm_head import (
    FlowMatchingHead,
    OnlineTargetEncoder,
    encode_texts_with_target_encoder,
    get_target_encoder_hidden_size,
)
from ..trainer_utils import create_modelcard_and_push
from .trainer import CustomFMTrainer


if TYPE_CHECKING:
    from datasets import Dataset
    from transformers import Seq2SeqTrainingArguments, TrainerCallback

    from ...hparams import DataArguments, FinetuningArguments, GeneratingArguments, ModelArguments


logger = get_logger(__name__)


def _search_existing_cache(cache_filename: str, parent_dir: str) -> Optional[str]:
    """Search for an existing embedding cache file in .fm_embedding_cache_* dirs under parent_dir.

    Scans all directories matching the pattern ``.fm_embedding_cache_*`` under
    ``parent_dir``, looking for ``cache_filename`` (e.g. "train_embeddings.pt").
    Returns the first match found, or None.

    This allows reusing embeddings from a previous run even if the cache_id
    (hash) differs, as long as the same filename exists.
    """
    import glob
    import os

    pattern = os.path.join(parent_dir, ".fm_embedding_cache_*")
    for cache_dir_candidate in sorted(glob.glob(pattern)):
        if not os.path.isdir(cache_dir_candidate):
            continue
        candidate = os.path.join(cache_dir_candidate, cache_filename)
        if os.path.isfile(candidate):
            return candidate
    return None


def _precompute_target_embeddings(
    dataset: "Dataset",
    tokenizer,
    encoder_name_or_path: str,
    cache_path: str,
    is_main_process: bool,
    batch_size: int = 512,
    max_length: int = 512,
    parent_dir: Optional[str] = None,
) -> "Dataset":
    """Pre-compute global z_1 = E_target(y_full) for each sample.

    One (d,) vector per sample, stored as list[float] in the dataset.
    Rank 0 computes and saves to cache; other ranks load from cache.

    Before computing, scans ``parent_dir`` for existing ``.fm_embedding_cache_*``
    directories that contain a file with the same name.  If found, copies it to
    ``cache_path`` and skips the expensive encoding step.
    """
    import os
    import shutil

    cache_filename = os.path.basename(cache_path)

    if is_main_process:
        if os.path.exists(cache_path):
            logger.info(f"[FM] Found cached embeddings at {cache_path}, skipping encoding.")
        else:
            # Scan parent_dir for .fm_embedding_cache_* dirs that have the same file
            found_path = None
            if parent_dir:
                found_path = _search_existing_cache(cache_filename, parent_dir)

            if found_path is not None:
                logger.info(f"[FM] Found existing cache at {found_path}, copying to {cache_path}")
                os.makedirs(os.path.dirname(cache_path), exist_ok=True)
                shutil.copy2(found_path, cache_path)
            else:
                import time
                logger.info(f"[FM] Rank 0: Pre-computing global embeddings for {len(dataset)} samples...")

                t0 = time.time()
                all_labels = dataset["labels"]
                all_texts = []
                for label_ids in all_labels:
                    valid_ids = [tid for tid in label_ids if tid != IGNORE_INDEX]
                    if len(valid_ids) == 0:
                        all_texts.append("")
                    else:
                        all_texts.append(tokenizer.decode(valid_ids, skip_special_tokens=True))
                logger.info(f"[FM] Decoded {len(all_texts)} responses in {time.time()-t0:.1f}s")

                logger.info(f"[FM] Encoding {len(all_texts)} texts (batch_size={batch_size})...")
                embeddings_tensor = encode_texts_with_target_encoder(
                    model_name_or_path=encoder_name_or_path,
                    texts=all_texts,
                    batch_size=batch_size,
                    max_length=max_length,
                )
                all_embeddings = embeddings_tensor.tolist()
                logger.info(f"[FM] Encoding done. embed_dim={embeddings_tensor.size(1)}")

                os.makedirs(os.path.dirname(cache_path), exist_ok=True)
                torch.save(all_embeddings, cache_path)
                logger.info(f"[FM] Saved embeddings cache to {cache_path}")
    else:
        logger.info(f"[FM] Rank != 0: Waiting for embeddings cache at {cache_path}...")

    all_embeddings = torch.load(cache_path, map_location="cpu", weights_only=True)
    logger.info(f"[FM] Loaded {len(all_embeddings)} embeddings from cache.")

    dataset = dataset.add_column("fm_target_embedding", all_embeddings)
    return dataset


def run_fm(
    model_args: "ModelArguments",
    data_args: "DataArguments",
    training_args: "Seq2SeqTrainingArguments",
    finetuning_args: "FinetuningArguments",
    generating_args: "GeneratingArguments",
    callbacks: Optional[list["TrainerCallback"]] = None,
):
    # Step 1: Load tokenizer
    tokenizer_module = load_tokenizer(model_args)
    tokenizer = tokenizer_module["tokenizer"]
    template = get_template_and_fix_tokenizer(tokenizer, data_args)

    # Step 2: Load dataset (reuse SFT data pipeline)
    dataset_module = get_dataset(template, model_args, data_args, training_args, stage="sft", **tokenizer_module)

    if data_args.streaming:
        raise ValueError("FM stage does not support streaming datasets. Please set `streaming: false`.")

    # Step 3: Get target encoder hidden size (reads config.json only, no nn.Module)
    encoder_path = finetuning_args.target_encoder_name_or_path
    target_dim = get_target_encoder_hidden_size(encoder_path)
    logger.info(f"Target encoder: {encoder_path}, hidden size: {target_dim}")

    # Step 4: Prepare FM target embeddings
    # Two modes:
    #   (a) fm_suffix_tokens = 0 (default): pre-compute global embeddings in subprocess
    #   (b) fm_suffix_tokens > 0: load encoder into GPU for online per-token suffix encoding
    import hashlib
    import os

    use_online_encoding = finetuning_args.fm_suffix_tokens > 0
    online_encoder = None

    if use_online_encoding:
        # Online mode: encoder will be loaded after LLM (Step 5) to share GPU.
        # No pre-computed embeddings needed.
        logger.info(
            f"[FM] Online suffix encoding enabled: fm_suffix_tokens={finetuning_args.fm_suffix_tokens}. "
            f"Target encoder will be loaded on GPU during training. Skipping pre-computation."
        )
    else:
        # Legacy mode: pre-compute global embeddings
        dataset_fingerprint = ""
        if "train_dataset" in dataset_module and dataset_module["train_dataset"] is not None:
            dataset_fingerprint = getattr(dataset_module["train_dataset"], "_fingerprint", "default")
        cache_id = hashlib.md5(f"{encoder_path}_{dataset_fingerprint}".encode()).hexdigest()[:12]
        cache_dir = os.path.join(os.path.dirname(training_args.output_dir), f".fm_embedding_cache_{cache_id}")
        is_main = training_args.process_index == 0
        logger.info(f"[FM] Embedding cache dir: {cache_dir}, is_main_process: {is_main}")

        encoder_max_len = finetuning_args.fm_target_encoder_max_length
        cache_parent_dir = os.path.dirname(training_args.output_dir)
        logger.info(f"[FM] Will scan {cache_parent_dir}/.fm_embedding_cache_* for existing caches before encoding.")

        with training_args.main_process_first(desc="pre-compute FM target embeddings"):
            if "train_dataset" in dataset_module and dataset_module["train_dataset"] is not None:
                dataset_module["train_dataset"] = _precompute_target_embeddings(
                    dataset_module["train_dataset"], tokenizer, encoder_path,
                    cache_path=os.path.join(cache_dir, "train_embeddings.pt"),
                    is_main_process=is_main,
                    max_length=encoder_max_len,
                    parent_dir=cache_parent_dir,
                )

            if "eval_dataset" in dataset_module and dataset_module["eval_dataset"] is not None:
                eval_ds = dataset_module["eval_dataset"]
                if isinstance(eval_ds, dict):
                    for key in eval_ds:
                        eval_ds[key] = _precompute_target_embeddings(
                            eval_ds[key], tokenizer, encoder_path,
                            cache_path=os.path.join(cache_dir, f"eval_{key}_embeddings.pt"),
                            is_main_process=is_main,
                            max_length=encoder_max_len,
                            parent_dir=cache_parent_dir,
                        )
                else:
                    dataset_module["eval_dataset"] = _precompute_target_embeddings(
                        eval_ds, tokenizer, encoder_path,
                        cache_path=os.path.join(cache_dir, "eval_embeddings.pt"),
                        is_main_process=is_main,
                        max_length=encoder_max_len,
                        parent_dir=cache_parent_dir,
                    )

        logger.info("FM target embeddings are now part of the dataset. No encoder in memory.")

    # Step 4.5: Load online target encoder BEFORE LLM (if suffix mode enabled).
    # IMPORTANT: Must be loaded before load_model() / DeepSpeed initialization.
    # DeepSpeed ZeRO-3 hooks into torch.nn.Module.__init__ globally after init,
    # which would corrupt the encoder's embedding weights (making them 1-D partitions).
    # Loading here ensures the encoder is created in a clean PyTorch environment.
    if use_online_encoding:
        import os as _os
        local_rank = int(_os.environ.get("LOCAL_RANK", "0"))
        encoder_device = f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu"
        logger.info(
            f"[FM] Loading online target encoder '{encoder_path}' onto {encoder_device} (fp16) "
            f"[BEFORE DeepSpeed init to avoid ZeRO-3 interference]..."
        )
        online_encoder = OnlineTargetEncoder(
            model_name_or_path=encoder_path,
            device=encoder_device,
            dtype=torch.float16,
        )
        logger.info(
            f"[FM] Online target encoder loaded. hidden_size={online_encoder.hidden_size}, "
            f"suffix_tokens={finetuning_args.fm_suffix_tokens}, "
            f"batch_size={finetuning_args.fm_suffix_batch_size}"
        )

    # Step 5: Load the LLM backbone
    model = load_model(tokenizer, model_args, finetuning_args, training_args.do_train)

    if getattr(model, "is_quantized", False) and not training_args.do_train:
        setattr(model, "_hf_peft_config_loaded", True)

    # Step 6: Create FM head and register as model submodule
    if hasattr(model.config, "hidden_size"):
        backbone_hidden_size = model.config.hidden_size
    elif hasattr(model.config, "text_config") and hasattr(model.config.text_config, "hidden_size"):
        backbone_hidden_size = model.config.text_config.hidden_size
    else:
        raise ValueError("Cannot determine backbone hidden size from model config.")

    fm_head = FlowMatchingHead(
        condition_dim=backbone_hidden_size,
        target_dim=target_dim,
        hidden_dim=finetuning_args.fm_head_hidden_dim,
        num_layers=finetuning_args.fm_head_num_layers,
    )
    logger.info(
        f"Created FM head: condition_dim={backbone_hidden_size}, target_dim={target_dim}, "
        f"hidden_dim={finetuning_args.fm_head_hidden_dim}, num_layers={finetuning_args.fm_head_num_layers}"
    )
    model.fm_head = fm_head

    # Step 7: Data collator
    data_collator = SFTDataCollatorWith4DAttentionMask(
        template=template,
        model=model if not training_args.predict_with_generate else None,
        pad_to_multiple_of=8 if training_args.do_train else None,
        label_pad_token_id=IGNORE_INDEX if data_args.ignore_pad_token_for_loss else tokenizer.pad_token_id,
        block_diag_attn=model_args.block_diag_attn,
        attn_implementation=getattr(model.config, "_attn_implementation", None),
        compute_dtype=model_args.compute_dtype,
        **tokenizer_module,
    )

    gen_kwargs = generating_args.to_dict(obey_generation_config=True)
    gen_kwargs["eos_token_id"] = [tokenizer.eos_token_id] + tokenizer.additional_special_tokens_ids
    gen_kwargs["pad_token_id"] = tokenizer.pad_token_id

    # Step 8: Initialize CustomFMTrainer
    trainer = CustomFMTrainer(
        model=model,
        args=training_args,
        finetuning_args=finetuning_args,
        data_collator=data_collator,
        callbacks=callbacks,
        gen_kwargs=gen_kwargs,
        online_encoder=online_encoder,
        **dataset_module,
        **tokenizer_module,
    )

    # Step 9: Training
    if training_args.do_train:
        train_result = trainer.train(resume_from_checkpoint=training_args.resume_from_checkpoint)
        trainer.save_model()
        if finetuning_args.include_effective_tokens_per_second:
            train_result.metrics["effective_tokens_per_sec"] = calculate_tps(
                dataset_module["train_dataset"], train_result.metrics, stage="sft"
            )

        trainer.log_metrics("train", train_result.metrics)
        trainer.save_metrics("train", train_result.metrics)
        trainer.save_state()
        if trainer.is_world_process_zero() and finetuning_args.plot_loss:
            keys = ["loss", "ar_loss", "fm_loss"]
            if isinstance(dataset_module.get("eval_dataset"), dict):
                keys += sum(
                    [[f"eval_{key}_loss"] for key in dataset_module["eval_dataset"].keys()], []
                )
            else:
                keys += ["eval_loss"]
            plot_loss(training_args.output_dir, keys=keys)

    # Step 10: Evaluation
    if training_args.do_eval:
        metrics = trainer.evaluate(metric_key_prefix="eval")
        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)

    # Step 10.5: Prediction
    if training_args.do_predict:
        predict_dataset = dataset_module.get("eval_dataset")
        if predict_dataset is not None:
            if isinstance(predict_dataset, dict):
                for key, ds in predict_dataset.items():
                    predict_results = trainer.predict(ds, metric_key_prefix=f"predict_{key}")
                    trainer.log_metrics(f"predict_{key}", predict_results.metrics)
                    trainer.save_metrics(f"predict_{key}", predict_results.metrics)
            else:
                predict_results = trainer.predict(predict_dataset, metric_key_prefix="predict")
                trainer.log_metrics("predict", predict_results.metrics)
                trainer.save_metrics("predict", predict_results.metrics)

    # Step 11: Create model card
    create_modelcard_and_push(trainer, model_args, data_args, training_args, finetuning_args)
