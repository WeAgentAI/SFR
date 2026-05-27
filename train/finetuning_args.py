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

from dataclasses import asdict, dataclass, field
from typing import Any, Literal, Optional


@dataclass
class FreezeArguments:
    r"""Arguments pertaining to the freeze (partial-parameter) training."""

    freeze_trainable_layers: int = field(
        default=2,
        metadata={
            "help": (
                "The number of trainable layers for freeze (partial-parameter) fine-tuning. "
                "Positive numbers mean the last n layers are set as trainable, "
                "negative numbers mean the first n layers are set as trainable."
            )
        },
    )
    freeze_trainable_modules: str = field(
        default="all",
        metadata={
            "help": (
                "Name(s) of trainable modules for freeze (partial-parameter) fine-tuning. "
                "Use commas to separate multiple modules. "
                "Use `all` to specify all the available modules."
            )
        },
    )
    freeze_extra_modules: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "Name(s) of modules apart from hidden layers to be set as trainable "
                "for freeze (partial-parameter) fine-tuning. "
                "Use commas to separate multiple modules."
            )
        },
    )


@dataclass
class LoraArguments:
    r"""Arguments pertaining to the LoRA training."""

    additional_target: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "Name(s) of modules apart from LoRA layers to be set as trainable "
                "and saved in the final checkpoint. "
                "Use commas to separate multiple modules."
            )
        },
    )
    lora_alpha: Optional[int] = field(
        default=None,
        metadata={"help": "The scale factor for LoRA fine-tuning (default: lora_rank * 2)."},
    )
    lora_dropout: float = field(
        default=0.0,
        metadata={"help": "Dropout rate for the LoRA fine-tuning."},
    )
    lora_rank: int = field(
        default=8,
        metadata={"help": "The intrinsic dimension for LoRA fine-tuning."},
    )
    lora_target: str = field(
        default="all",
        metadata={
            "help": (
                "Name(s) of target modules to apply LoRA. "
                "Use commas to separate multiple modules. "
                "Use `all` to specify all the linear modules."
            )
        },
    )
    loraplus_lr_ratio: Optional[float] = field(
        default=None,
        metadata={"help": "LoRA plus learning rate ratio (lr_B / lr_A)."},
    )
    loraplus_lr_embedding: float = field(
        default=1e-6,
        metadata={"help": "LoRA plus learning rate for lora embedding layers."},
    )
    use_rslora: bool = field(
        default=False,
        metadata={"help": "Whether or not to use the rank stabilization scaling factor for LoRA layer."},
    )
    use_dora: bool = field(
        default=False,
        metadata={"help": "Whether or not to use the weight-decomposed lora method (DoRA)."},
    )
    pissa_init: bool = field(
        default=False,
        metadata={"help": "Whether or not to initialize a PiSSA adapter."},
    )
    pissa_iter: int = field(
        default=16,
        metadata={"help": "The number of iteration steps performed by FSVD in PiSSA. Use -1 to disable it."},
    )
    pissa_convert: bool = field(
        default=False,
        metadata={"help": "Whether or not to convert the PiSSA adapter to a normal LoRA adapter."},
    )
    create_new_adapter: bool = field(
        default=False,
        metadata={"help": "Whether or not to create a new adapter with randomly initialized weight."},
    )


@dataclass
class RLHFArguments:
    r"""Arguments pertaining to the PPO, DPO and KTO training."""

    pref_beta: float = field(
        default=0.1,
        metadata={"help": "The beta parameter in the preference loss."},
    )
    pref_ftx: float = field(
        default=0.0,
        metadata={"help": "The supervised fine-tuning loss coefficient in DPO training."},
    )
    pref_loss: Literal["sigmoid", "hinge", "ipo", "kto_pair", "orpo", "simpo"] = field(
        default="sigmoid",
        metadata={"help": "The type of DPO loss to use."},
    )
    dpo_label_smoothing: float = field(
        default=0.0,
        metadata={"help": "The robust DPO label smoothing parameter in cDPO that should be between 0 and 0.5."},
    )
    kto_chosen_weight: float = field(
        default=1.0,
        metadata={"help": "The weight factor of the desirable losses in KTO training."},
    )
    kto_rejected_weight: float = field(
        default=1.0,
        metadata={"help": "The weight factor of the undesirable losses in KTO training."},
    )
    simpo_gamma: float = field(
        default=0.5,
        metadata={"help": "The target reward margin term in SimPO loss."},
    )
    ppo_buffer_size: int = field(
        default=1,
        metadata={"help": "The number of mini-batches to make experience buffer in a PPO optimization step."},
    )
    ppo_epochs: int = field(
        default=4,
        metadata={"help": "The number of epochs to perform in a PPO optimization step."},
    )
    ppo_score_norm: bool = field(
        default=False,
        metadata={"help": "Use score normalization in PPO training."},
    )
    ppo_target: float = field(
        default=6.0,
        metadata={"help": "Target KL value for adaptive KL control in PPO training."},
    )
    ppo_whiten_rewards: bool = field(
        default=False,
        metadata={"help": "Whiten the rewards before compute advantages in PPO training."},
    )
    ref_model: Optional[str] = field(
        default=None,
        metadata={"help": "Path to the reference model used for the PPO or DPO training."},
    )
    ref_model_adapters: Optional[str] = field(
        default=None,
        metadata={"help": "Path to the adapters of the reference model."},
    )
    ref_model_quantization_bit: Optional[int] = field(
        default=None,
        metadata={"help": "The number of bits to quantize the reference model."},
    )
    reward_model: Optional[str] = field(
        default=None,
        metadata={"help": "Path to the reward model used for the PPO training."},
    )
    reward_model_adapters: Optional[str] = field(
        default=None,
        metadata={"help": "Path to the adapters of the reward model."},
    )
    reward_model_quantization_bit: Optional[int] = field(
        default=None,
        metadata={"help": "The number of bits to quantize the reward model."},
    )
    reward_model_type: Literal["lora", "full", "api"] = field(
        default="lora",
        metadata={"help": "The type of the reward model in PPO training. Lora model only supports lora training."},
    )


@dataclass
class GaloreArguments:
    r"""Arguments pertaining to the GaLore algorithm."""

    use_galore: bool = field(
        default=False,
        metadata={"help": "Whether or not to use the gradient low-Rank projection (GaLore)."},
    )
    galore_target: str = field(
        default="all",
        metadata={
            "help": (
                "Name(s) of modules to apply GaLore. Use commas to separate multiple modules. "
                "Use `all` to specify all the linear modules."
            )
        },
    )
    galore_rank: int = field(
        default=16,
        metadata={"help": "The rank of GaLore gradients."},
    )
    galore_update_interval: int = field(
        default=200,
        metadata={"help": "Number of steps to update the GaLore projection."},
    )
    galore_scale: float = field(
        default=2.0,
        metadata={"help": "GaLore scaling coefficient."},
    )
    galore_proj_type: Literal["std", "reverse_std", "right", "left", "full"] = field(
        default="std",
        metadata={"help": "Type of GaLore projection."},
    )
    galore_layerwise: bool = field(
        default=False,
        metadata={"help": "Whether or not to enable layer-wise update to further save memory."},
    )


@dataclass
class ApolloArguments:
    r"""Arguments pertaining to the APOLLO algorithm."""

    use_apollo: bool = field(
        default=False,
        metadata={"help": "Whether or not to use the APOLLO optimizer."},
    )
    apollo_target: str = field(
        default="all",
        metadata={
            "help": (
                "Name(s) of modules to apply APOLLO. Use commas to separate multiple modules. "
                "Use `all` to specify all the linear modules."
            )
        },
    )
    apollo_rank: int = field(
        default=16,
        metadata={"help": "The rank of APOLLO gradients."},
    )
    apollo_update_interval: int = field(
        default=200,
        metadata={"help": "Number of steps to update the APOLLO projection."},
    )
    apollo_scale: float = field(
        default=32.0,
        metadata={"help": "APOLLO scaling coefficient."},
    )
    apollo_proj: Literal["svd", "random"] = field(
        default="random",
        metadata={"help": "Type of APOLLO low-rank projection algorithm (svd or random)."},
    )
    apollo_proj_type: Literal["std", "right", "left"] = field(
        default="std",
        metadata={"help": "Type of APOLLO projection."},
    )
    apollo_scale_type: Literal["channel", "tensor"] = field(
        default="channel",
        metadata={"help": "Type of APOLLO scaling (channel or tensor)."},
    )
    apollo_layerwise: bool = field(
        default=False,
        metadata={"help": "Whether or not to enable layer-wise update to further save memory."},
    )
    apollo_scale_front: bool = field(
        default=False,
        metadata={"help": "Whether or not to use the norm-growth limiter in front of gradient scaling."},
    )


@dataclass
class BAdamArgument:
    r"""Arguments pertaining to the BAdam optimizer."""

    use_badam: bool = field(
        default=False,
        metadata={"help": "Whether or not to use the BAdam optimizer."},
    )
    badam_mode: Literal["layer", "ratio"] = field(
        default="layer",
        metadata={"help": "Whether to use layer-wise or ratio-wise BAdam optimizer."},
    )
    badam_start_block: Optional[int] = field(
        default=None,
        metadata={"help": "The starting block index for layer-wise BAdam."},
    )
    badam_switch_mode: Optional[Literal["ascending", "descending", "random", "fixed"]] = field(
        default="ascending",
        metadata={"help": "the strategy of picking block to update for layer-wise BAdam."},
    )
    badam_switch_interval: Optional[int] = field(
        default=50,
        metadata={
            "help": "Number of steps to update the block for layer-wise BAdam. Use -1 to disable the block update."
        },
    )
    badam_update_ratio: float = field(
        default=0.05,
        metadata={"help": "The ratio of the update for ratio-wise BAdam."},
    )
    badam_mask_mode: Literal["adjacent", "scatter"] = field(
        default="adjacent",
        metadata={
            "help": (
                "The mode of the mask for BAdam optimizer. "
                "`adjacent` means that the trainable parameters are adjacent to each other, "
                "`scatter` means that trainable parameters are randomly choosed from the weight."
            )
        },
    )
    badam_verbose: int = field(
        default=0,
        metadata={
            "help": (
                "The verbosity level of BAdam optimizer. "
                "0 for no print, 1 for print the block prefix, 2 for print trainable parameters."
            )
        },
    )


@dataclass
class SwanLabArguments:
    use_swanlab: bool = field(
        default=False,
        metadata={"help": "Whether or not to use the SwanLab (an experiment tracking and visualization tool)."},
    )
    swanlab_project: Optional[str] = field(
        default="llamafactory",
        metadata={"help": "The project name in SwanLab."},
    )
    swanlab_workspace: Optional[str] = field(
        default=None,
        metadata={"help": "The workspace name in SwanLab."},
    )
    swanlab_run_name: Optional[str] = field(
        default=None,
        metadata={"help": "The experiment name in SwanLab."},
    )
    swanlab_mode: Literal["cloud", "local"] = field(
        default="cloud",
        metadata={"help": "The mode of SwanLab."},
    )
    swanlab_api_key: Optional[str] = field(
        default=None,
        metadata={"help": "The API key for SwanLab."},
    )
    swanlab_logdir: Optional[str] = field(
        default=None,
        metadata={"help": "The log directory for SwanLab."},
    )
    swanlab_lark_webhook_url: Optional[str] = field(
        default=None,
        metadata={"help": "The Lark(飞书) webhook URL for SwanLab."},
    )
    swanlab_lark_secret: Optional[str] = field(
        default=None,
        metadata={"help": "The Lark(飞书) secret for SwanLab."},
    )


@dataclass
class FlowMatchingArguments:
    r"""Arguments pertaining to the Flow Matching (FM) auxiliary training."""

    fm_loss_weight: float = field(
        default=1.0,
        metadata={"help": "The weight coefficient for the Flow Matching loss (lambda)."},
    )
    fm_head_num_layers: int = field(
        default=3,
        metadata={"help": "Number of layers in the FM vector field MLP head."},
    )
    fm_head_hidden_dim: int = field(
        default=1024,
        metadata={"help": "Hidden dimension of each layer in the FM vector field MLP head."},
    )
    fm_backbone_layer_index: int = field(
        default=-1,
        metadata={"help": "Which backbone hidden layer to use as condition for FM head. -1 means the last layer."},
    )
    target_encoder_name_or_path: str = field(
        default="BAAI/bge-large-zh",
        metadata={"help": "Path or name of the frozen target encoder (e.g. BGE-large) for FM training."},
    )
    fm_loss_weight_schedule: Literal["constant", "linear_decay", "cosine_decay"] = field(
        default="constant",
        metadata={"help": "Schedule for the FM loss weight over training. 'constant' keeps it fixed."},
    )
    fm_target_encoder_max_length: int = field(
        default=512,
        metadata={"help": "Maximum token length for the target encoder input during FM embedding pre-computation."},
    )
    fm_suffix_tokens: int = field(
        default=0,
        metadata={
            "help": (
                "Number of subsequent tokens to encode as the FM target at each response position. "
                "When > 0, enables ONLINE token-level suffix encoding: for each response token at "
                "position t, the target z_1[t] = Encoder(y_{t+1}, ..., y_{t+k}) where k = fm_suffix_tokens. "
                "This replaces the pre-computed global embedding with a fine-grained per-position target. "
                "The target encoder is loaded into GPU memory during training (frozen, fp16). "
                "0 (default) means use pre-computed global embeddings (original behavior)."
            )
        },
    )
    fm_suffix_batch_size: int = field(
        default=64,
        metadata={
            "help": (
                "Batch size for the online target encoder when fm_suffix_tokens > 0. "
                "Controls how many suffix sequences are encoded per encoder forward pass. "
                "Larger = faster but uses more GPU memory. Only used when fm_suffix_tokens > 0."
            )
        },
    )
    fm_suffix_stride: int = field(
        default=1,
        metadata={
            "help": (
                "Stride for sparse suffix encoding. Only every `stride`-th response position is "
                "actually encoded; intermediate positions get the embedding of their nearest "
                "encoded anchor (nearest-neighbor fill). "
                "stride=1 (default): encode every position (most accurate, slowest). "
                "stride=32: encode every 32nd position (32x faster, slight approximation). "
                "This makes online encoding practical for long sequences. "
                "Example: response_len=2048, stride=32 → only 64 encoder calls instead of 2048."
            )
        },
    )
    fm_head_warmup_steps: int = field(
        default=0,
        metadata={
            "help": (
                "Number of initial training steps during which the LLM backbone is frozen "
                "and only the FM head is trained. This lets the FM head adapt to the backbone's "
                "hidden state distribution before joint optimization begins. 0 means no warmup."
            )
        },
    )
    fm_loss_warmup_steps: int = field(
        default=0,
        metadata={
            "help": (
                "Number of steps after FM head warmup during which the FM loss weight linearly "
                "ramps from 0 to fm_loss_weight. This prevents large FM gradients from disrupting "
                "the backbone when joint training starts. 0 means no ramp (full weight immediately)."
            )
        },
    )
    fm_loss_type: Literal["flow_matching", "cosine", "mse", "cka"] = field(
        default="flow_matching",
        metadata={
            "help": (
                "Which auxiliary loss to use on top of the AR loss. "
                "'flow_matching' (default): original Flow Matching MSE on the predicted "
                "velocity field. "
                "'cosine' ablation: compute 1 - cos(h_proj, z_1) where "
                "h_proj = fm_head(0, 0, h) reuses the FM head as a non-linear projection. "
                "'mse' ablation: compute ||h_proj - z_1||^2 (mean over d) with the same "
                "h_proj = fm_head(0, 0, h). "
                "'cka' ablation: compute 1 - LinearCKA(pooled_h, z_1) using the unbiased "
                "HSIC estimator; supports heterogeneous dims so no projection is needed "
                "(h is used directly, fm_head is NOT called)."
            )
        },
    )
    fm_grad_gate_enabled: bool = field(
        default=False,
        metadata={
            "help": (
                "Enable adaptive gradient gating for FM loss. When enabled, if the FM loss "
                "drops significantly (indicating FM head is 'solving' the task itself), the "
                "backbone is detached from the FM loss computation graph for that step. "
                "This prevents the FM head from short-circuiting the backbone learning."
            )
        },
    )
    fm_grad_gate_threshold: float = field(
        default=1.5,
        metadata={
            "help": (
                "Threshold for FM gradient gating. Interpretation depends on fm_grad_gate_mode: "
                "- 'static': absolute fm_loss value below which gating triggers (e.g. 0.1). "
                "- 'dynamic': divisor on EMA of fm_loss. Gate when current_loss < EMA / threshold. "
                "  E.g. 1.5 means gate when loss drops below 67% of its running average. "
                "  2.0 means gate when loss drops below 50% of average. "
                "Only used when fm_grad_gate_enabled=True."
            )
        },
    )
    fm_grad_gate_mode: Literal["static", "dynamic"] = field(
        default="dynamic",
        metadata={
            "help": (
                "Mode for FM gradient gating threshold: "
                "'static': gate when fm_loss drops below fm_grad_gate_threshold (absolute value). "
                "'dynamic': gate when fm_loss drops below EMA(fm_loss) / fm_grad_gate_threshold "
                "(adapts to training dynamics automatically, recommended)."
            )
        },
    )
    fm_grad_gate_ema_decay: float = field(
        default=0.99,
        metadata={
            "help": (
                "EMA decay factor for tracking fm_loss running average in gradient gating. "
                "Higher values = smoother baseline, less sensitive to short-term fluctuations. "
                "Only used when fm_grad_gate_enabled=True."
            )
        },
    )
    fm_ema_teacher_enabled: bool = field(
        default=False,
        metadata={
            "help": (
                "Enable EMA parameter smoothing for FM head. When enabled, after each "
                "optimizer step, the FM head's parameters are replaced with an EMA-smoothed "
                "version (slow-moving average). This prevents the FM head from updating too "
                "fast and 'short-circuiting' the backbone — ensuring it provides stable "
                "auxiliary supervision throughout training. "
                "Effective learning rate of FM head is reduced by factor (1-decay)."
            )
        },
    )
    fm_ema_teacher_decay: float = field(
        default=0.999,
        metadata={
            "help": (
                "EMA decay factor for FM head parameter smoothing. Higher values mean slower "
                "FM head updates: 0.999 → effective step size is 0.001x the optimizer's. "
                "0.99 → 0.01x. 0.9 → 0.1x. "
                "Recommended: 0.999 for aggressive smoothing, 0.99 for moderate. "
                "Only used when fm_ema_teacher_enabled=True."
            )
        },
    )
    fm_ablation_cosine: bool = field(
        default=False,
        metadata={
            "help": (
                "[Deprecated] Kept for backward compatibility. If True, forces "
                "fm_loss_type='cosine'. Prefer setting fm_loss_type directly."
            )
        },
    )


@dataclass
class FinetuningArguments(
    FlowMatchingArguments, SwanLabArguments, BAdamArgument, ApolloArguments, GaloreArguments, RLHFArguments, LoraArguments, FreezeArguments
):
    r"""Arguments pertaining to which techniques we are going to fine-tuning with."""

    pure_bf16: bool = field(
        default=False,
        metadata={"help": "Whether or not to train model in purely bf16 precision (without AMP)."},
    )
    stage: Literal["pt", "sft", "rm", "ppo", "dpo", "kto", "fm"] = field(
        default="sft",
        metadata={"help": "Which stage will be performed in training."},
    )
    finetuning_type: Literal["lora", "freeze", "full"] = field(
        default="lora",
        metadata={"help": "Which fine-tuning method to use."},
    )
    use_llama_pro: bool = field(
        default=False,
        metadata={"help": "Whether or not to make only the parameters in the expanded blocks trainable."},
    )
    use_adam_mini: bool = field(
        default=False,
        metadata={"help": "Whether or not to use the Adam-mini optimizer."},
    )
    use_muon: bool = field(
        default=False,
        metadata={"help": "Whether or not to use the Muon optimizer."},
    )
    freeze_vision_tower: bool = field(
        default=True,
        metadata={"help": "Whether ot not to freeze the vision tower in MLLM training."},
    )
    freeze_multi_modal_projector: bool = field(
        default=True,
        metadata={"help": "Whether or not to freeze the multi modal projector in MLLM training."},
    )
    freeze_language_model: bool = field(
        default=False,
        metadata={"help": "Whether or not to freeze the language model in MLLM training."},
    )
    compute_accuracy: bool = field(
        default=False,
        metadata={"help": "Whether or not to compute the token-level accuracy at evaluation."},
    )
    disable_shuffling: bool = field(
        default=False,
        metadata={"help": "Whether or not to disable the shuffling of the training set."},
    )
    early_stopping_steps: Optional[int] = field(
        default=None,
        metadata={"help": "Number of steps to stop training if the `metric_for_best_model` does not improve."},
    )
    plot_loss: bool = field(
        default=False,
        metadata={"help": "Whether or not to save the training loss curves."},
    )
    include_effective_tokens_per_second: bool = field(
        default=False,
        metadata={"help": "Whether or not to compute effective tokens per second."},
    )

    def __post_init__(self):
        def split_arg(arg):
            if isinstance(arg, str):
                return [item.strip() for item in arg.split(",")]
            return arg

        self.freeze_trainable_modules: list[str] = split_arg(self.freeze_trainable_modules)
        self.freeze_extra_modules: Optional[list[str]] = split_arg(self.freeze_extra_modules)
        self.lora_alpha: int = self.lora_alpha or self.lora_rank * 2
        self.lora_target: list[str] = split_arg(self.lora_target)
        self.additional_target: Optional[list[str]] = split_arg(self.additional_target)
        self.galore_target: list[str] = split_arg(self.galore_target)
        self.apollo_target: list[str] = split_arg(self.apollo_target)
        self.use_ref_model = self.stage == "dpo" and self.pref_loss not in ["orpo", "simpo"]

        assert self.finetuning_type in ["lora", "freeze", "full"], "Invalid fine-tuning method."
        assert self.ref_model_quantization_bit in [None, 8, 4], "We only accept 4-bit or 8-bit quantization."
        assert self.reward_model_quantization_bit in [None, 8, 4], "We only accept 4-bit or 8-bit quantization."

        if self.stage == "ppo" and self.reward_model is None:
            raise ValueError("`reward_model` is necessary for PPO training.")

        if self.stage == "ppo" and self.reward_model_type == "lora" and self.finetuning_type != "lora":
            raise ValueError("`reward_model_type` cannot be lora for Freeze/Full PPO training.")

        if self.stage == "dpo" and self.pref_loss != "sigmoid" and self.dpo_label_smoothing > 1e-6:
            raise ValueError("`dpo_label_smoothing` is only valid for sigmoid loss function.")

        if self.use_llama_pro and self.finetuning_type == "full":
            raise ValueError("`use_llama_pro` is only valid for Freeze or LoRA training.")

        if self.finetuning_type == "lora" and (self.use_galore or self.use_apollo or self.use_badam):
            raise ValueError("Cannot use LoRA with GaLore, APOLLO or BAdam together.")

        if int(self.use_galore) + int(self.use_apollo) + (self.use_badam) > 1:
            raise ValueError("Cannot use GaLore, APOLLO or BAdam together.")

        if self.pissa_init and (self.stage in ["ppo", "kto"] or self.use_ref_model):
            raise ValueError("Cannot use PiSSA for current training stage.")

        if self.finetuning_type != "lora":
            if self.loraplus_lr_ratio is not None:
                raise ValueError("`loraplus_lr_ratio` is only valid for LoRA training.")

            if self.use_rslora:
                raise ValueError("`use_rslora` is only valid for LoRA training.")

            if self.use_dora:
                raise ValueError("`use_dora` is only valid for LoRA training.")

            if self.pissa_init:
                raise ValueError("`pissa_init` is only valid for LoRA training.")

        # FM-specific validation
        if self.stage == "fm":
            if self.fm_loss_weight < 0:
                raise ValueError("`fm_loss_weight` must be non-negative.")

            if self.fm_head_num_layers < 1:
                raise ValueError("`fm_head_num_layers` must be at least 1.")

            if self.fm_head_hidden_dim < 1:
                raise ValueError("`fm_head_hidden_dim` must be at least 1.")

            if not self.target_encoder_name_or_path:
                raise ValueError("`target_encoder_name_or_path` must be specified for FM training.")

            if self.fm_target_encoder_max_length < 1:
                raise ValueError("`fm_target_encoder_max_length` must be at least 1.")

            # Back-compat: fm_ablation_cosine=True is equivalent to fm_loss_type="cosine".
            if self.fm_ablation_cosine:
                if self.fm_loss_type == "flow_matching":
                    self.fm_loss_type = "cosine"
                elif self.fm_loss_type != "cosine":
                    raise ValueError(
                        "`fm_ablation_cosine=True` conflicts with `fm_loss_type="
                        f"'{self.fm_loss_type}'`. Remove `fm_ablation_cosine` and "
                        "use `fm_loss_type` only."
                    )

    def to_dict(self) -> dict[str, Any]:
        args = asdict(self)
        args = {k: f"<{k.upper()}>" if k.endswith("api_key") else v for k, v in args.items()}
        return args
