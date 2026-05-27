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

"""Custom trainer for AR + Flow Matching (FM) joint training.

FM head is registered as model.fm_head (a submodule of the LLM).
This lets DeepSpeed/DDP manage its parameters together with the backbone.

Token-level suffix FM: each response position t has its own target
z_1[t] = E_target(y_{>=t}), enabling "future content prediction" at every step.
"""

import os
from types import MethodType
from typing import TYPE_CHECKING, Any, Optional

import torch
import torch.nn as nn
import torch.distributed as dist
from transformers import Seq2SeqTrainer
from typing_extensions import override

from ...extras import logging
from ...extras.constants import IGNORE_INDEX
from ...extras.packages import is_transformers_version_greater_than
from ...model.model_utils.fm_head import linear_cka_loss
from ..callbacks import SaveProcessorCallback
from ..trainer_utils import create_custom_optimizer, create_custom_scheduler


if TYPE_CHECKING:
    from transformers import PreTrainedTokenizer, ProcessorMixin

    from ...hparams import FinetuningArguments


logger = logging.get_logger(__name__)


def _unwrap_model(model):
    """Unwrap model from DeepSpeed/DDP/FSDP/PeftModel wrappers."""
    if hasattr(model, "module"):
        model = model.module
    if hasattr(model, "base_model"):
        model = model.base_model
    if hasattr(model, "model") and not isinstance(model, nn.Sequential):
        model = model.model
    return model


class CustomFMTrainer(Seq2SeqTrainer):
    r"""Trainer for AR + Flow Matching joint optimization.

    FM head is a submodule of self.model (model.fm_head), so its parameters
    are automatically included in the main optimizer and managed by DeepSpeed.
    """

    def __init__(
        self,
        finetuning_args: "FinetuningArguments",
        gen_kwargs: Optional[dict[str, Any]] = None,
        online_encoder: Optional[Any] = None,
        **kwargs,
    ) -> None:
        processor: Optional["ProcessorMixin"] = kwargs.pop("processor", None)

        if is_transformers_version_greater_than("4.46"):
            kwargs["processing_class"] = kwargs.pop("tokenizer")
        else:
            self.processing_class: PreTrainedTokenizer = kwargs.get("tokenizer")

        super().__init__(**kwargs)
        if processor is not None:
            self.model_accepts_loss_kwargs = False

        self.finetuning_args = finetuning_args

        if gen_kwargs is not None:
            self._gen_kwargs = gen_kwargs

        if processor is not None:
            self.add_callback(SaveProcessorCallback(processor))

        if finetuning_args.use_badam:
            from badam import BAdamCallback, clip_grad_norm_old_version  # type: ignore
            self.accelerator.clip_grad_norm_ = MethodType(clip_grad_norm_old_version, self.accelerator)
            self.add_callback(BAdamCallback)

        self._stored_fm_metrics: dict[str, float] = {}
        self._backbone_frozen: bool = False  # tracks whether backbone is currently frozen for FM warmup
        self._logged_warmup_phase: bool = False  # avoid spamming logs
        self._logged_joint_phase: bool = False

        # ── Gradient gating state ──
        self._fm_grad_norm_ema: float = 0.0  # EMA of FM head gradient norm
        self._last_fm_grad_norm: float = 0.0  # raw grad norm from the last step
        self._fm_grad_gate_active: bool = False  # whether gate was triggered this step
        self._fm_grad_ema_initialized: bool = False  # first-step initialization flag

        # ── EMA teacher state ──
        # When enabled, we maintain a shadow copy of fm_head (EMA teacher).
        # The EMA teacher does the forward pass for computing FM loss → gradients
        # flow to backbone but NOT through the (fast-changing) student fm_head.
        # The student fm_head is updated by optimizer as usual; then EMA is applied.
        self._ema_fm_head: Optional[nn.Module] = None  # initialized lazily on first step
        self._ema_teacher_initialized: bool = False

        # ── Online target encoder (for token-level suffix encoding) ──
        # When fm_suffix_tokens > 0, this encoder is loaded on GPU and used every step
        # to encode suffix tokens into per-position targets z_1[t].
        # NOT registered as model submodule — excluded from optimizer/checkpoint.
        self._online_encoder = online_encoder

    def _get_fm_head(self, model):
        """Get fm_head from a possibly-wrapped model."""
        if hasattr(model, "module") and hasattr(model.module, "fm_head"):
            return model.module.fm_head
        if hasattr(model, "fm_head"):
            return model.fm_head
        raise AttributeError("Cannot find fm_head on model.")

    def _get_fm_loss_weight(self) -> float:
        """Get the FM loss weight considering warmup and schedule.

        Three-phase weight logic:
          Phase A (step < fm_head_warmup_steps): backbone frozen, FM weight = fm_loss_weight
              (only FM head trains, full FM signal)
          Phase B (step < fm_head_warmup_steps + fm_loss_warmup_steps): backbone unfrozen,
              FM weight linearly ramps from 0 → fm_loss_weight
          Phase C (remaining): normal schedule (constant / linear_decay / cosine_decay)
        """
        import math as _math

        base_weight = self.finetuning_args.fm_loss_weight
        head_warmup = self.finetuning_args.fm_head_warmup_steps
        loss_warmup = self.finetuning_args.fm_loss_warmup_steps
        step = self.state.global_step

        # Phase A: backbone frozen warmup — full FM weight to train FM head effectively
        if head_warmup > 0 and step < head_warmup:
            return base_weight

        # Phase B: FM loss ramp-up after backbone is unfrozen
        joint_step = step - head_warmup  # steps since joint training started
        if loss_warmup > 0 and joint_step < loss_warmup:
            ramp = joint_step / loss_warmup
            return base_weight * ramp

        # Phase C: normal schedule (applied to remaining steps after both warmups)
        schedule = self.finetuning_args.fm_loss_weight_schedule
        if schedule == "constant" or self.state.max_steps <= 0:
            return base_weight

        total_after_warmup = self.state.max_steps - head_warmup - loss_warmup
        if total_after_warmup <= 0:
            return base_weight

        progress = (step - head_warmup - loss_warmup) / total_after_warmup
        progress = min(max(progress, 0.0), 1.0)
        if schedule == "linear_decay":
            return base_weight * (1.0 - progress)
        elif schedule == "cosine_decay":
            return base_weight * 0.5 * (1.0 + _math.cos(_math.pi * progress))
        return base_weight

    def _compute_fm_head_grad_norm(self, model) -> float:
        """Compute the L2 gradient norm of FM head parameters.

        Tries multiple unwrapping strategies to find fm_head with gradients.
        Works with DDP, DeepSpeed, PEFT/LoRA wrappers.
        """
        # Strategy 1: try self.model (Trainer's reference)
        fm_head = None
        for candidate in [model, self.model]:
            try:
                if hasattr(candidate, "module") and hasattr(candidate.module, "fm_head"):
                    fm_head = candidate.module.fm_head
                    break
                elif hasattr(candidate, "fm_head"):
                    fm_head = candidate.fm_head
                    break
            except Exception:
                continue

        if fm_head is None:
            # Last resort: unwrap fully
            try:
                unwrapped = self.accelerator.unwrap_model(self.model)
                fm_head = getattr(unwrapped, "fm_head", None)
            except Exception:
                pass

        if fm_head is None:
            return 0.0

        total_norm_sq = 0.0
        for p in fm_head.parameters():
            if p.grad is not None:
                total_norm_sq += p.grad.data.float().norm().item() ** 2
        return total_norm_sq ** 0.5

    def _should_gate_fm_gradient(self) -> bool:
        """Decide whether to detach backbone hidden states from FM loss graph.

        Uses fm_loss value tracking (EMA) to detect when FM head is learning too fast.
        When current fm_loss drops significantly below the EMA (head "solving" the task),
        we gate the backbone to prevent the head from short-circuiting.

        Modes:
          - 'static': gate if current fm_loss < threshold (absolute, e.g. 0.1)
          - 'dynamic': gate if current fm_loss < EMA / threshold
                       (e.g. threshold=1.5: gate when loss is below 67% of average)
        """
        if not self.finetuning_args.fm_grad_gate_enabled:
            return False
        if self._backbone_frozen:
            return False
        if not self._fm_grad_ema_initialized:
            return False

        threshold = self.finetuning_args.fm_grad_gate_threshold
        mode = self.finetuning_args.fm_grad_gate_mode

        if mode == "static":
            # Gate when fm_loss drops below absolute threshold
            return self._last_fm_grad_norm < threshold
        else:  # dynamic
            # Gate when current fm_loss is significantly below the running average
            # threshold=1.5 means: gate when current_loss < ema / 1.5 (= ema * 0.67)
            if self._fm_grad_norm_ema < 1e-10:
                return False
            return self._last_fm_grad_norm < self._fm_grad_norm_ema / threshold

    def _dummy_fm_head_forward(self, model, h: torch.Tensor, target_dim: int) -> None:
        """Execute a dummy forward pass through FM head to maintain consistent submodule order.

        DeepSpeed ZeRO-3 tracks the order of submodule execution across ranks. If any rank
        skips the FM head forward (e.g., due to empty response masks), the submodule order
        diverges, causing NCCL collective mismatch errors.

        This method performs a minimal forward pass (single token, no grad) that touches
        all FM head submodules in the correct order, ensuring ZeRO-3 consistency.
        """
        fm_head = self._get_fm_head_for_forward(model)
        device = h.device
        dtype = h.dtype
        # Minimal tensors: (1, 1, dim) to touch all submodules
        dummy_z = torch.zeros(1, 1, target_dim, device=device, dtype=dtype)
        dummy_t = torch.zeros(1, device=device, dtype=dtype)
        dummy_h = torch.zeros(1, 1, h.shape[-1], device=device, dtype=dtype)
        with torch.no_grad():
            _ = fm_head(dummy_z, dummy_t, dummy_h)

    def _sync_grad_gate_decision(self) -> bool:
        """Synchronize the gradient gating decision across all ranks.

        The grad gate decision depends on per-rank EMA state which can diverge because
        each rank sees different data. To prevent divergent computation graphs (which can
        cause issues with gradient synchronization), we use an all-reduce to ensure all
        ranks make the same gating decision: gate if ANY rank wants to gate.

        Returns the synchronized decision (True = gate active on all ranks).
        """
        local_decision = self._should_gate_fm_gradient()
        if not dist.is_initialized():
            return local_decision

        # Use all-reduce MAX: if any rank wants to gate, all ranks gate.
        decision_tensor = torch.tensor(
            [1.0 if local_decision else 0.0],
            device=torch.cuda.current_device() if torch.cuda.is_available() else "cpu",
        )
        try:
            dist.all_reduce(decision_tensor, op=dist.ReduceOp.MAX)
        except RuntimeError:
            # If all_reduce fails (e.g., in non-distributed testing), fall back to local
            return local_decision
        return decision_tensor.item() > 0.5

    def _init_ema_teacher(self, fm_head: nn.Module) -> None:
        """Lazily initialize the EMA teacher as a deep copy of fm_head.

        The EMA teacher stores the slow-moving shadow weights. After each optimizer
        step, the student fm_head's parameters are blended back toward the EMA:
          student_p = ema_p  (hard replacement)
          ema_p = decay * ema_p + (1-decay) * updated_student_p

        This effectively limits how fast fm_head can change per step, preventing
        it from learning a "shortcut" mapping that bypasses backbone improvement.
        The backbone still receives gradients through the student head normally.
        """
        import copy
        self._ema_fm_head = copy.deepcopy(fm_head)
        for p in self._ema_fm_head.parameters():
            p.requires_grad_(False)
        self._ema_teacher_initialized = True
        logger.info_rank0(
            f"[FM EMA Teacher] Initialized with decay={self.finetuning_args.fm_ema_teacher_decay}. "
            f"FM head parameters will be EMA-smoothed after each step."
        )

    @torch.no_grad()
    def _update_ema_teacher(self, fm_head: nn.Module) -> None:
        """Update EMA teacher from student, then replace student with EMA weights.

        Two-step process after optimizer.step():
          1. ema_p = decay * ema_p + (1-decay) * student_p  (student just got updated by optimizer)
          2. student_p = ema_p  (force student to use smoothed weights for next forward)

        Effect: fm_head's effective learning rate is reduced by factor (1-decay).
        With decay=0.999, a step that would move params by Δ only moves them by 0.001*Δ.
        This keeps the FM head as a "slow learner" that provides stable auxiliary signal.
        """
        if self._ema_fm_head is None:
            return
        decay = self.finetuning_args.fm_ema_teacher_decay
        for ema_p, student_p in zip(self._ema_fm_head.parameters(), fm_head.parameters()):
            # Step 1: update EMA with the freshly-optimized student weights
            ema_p.data.mul_(decay).add_(student_p.data, alpha=1.0 - decay)
            # Step 2: replace student with EMA (smoothed) weights
            student_p.data.copy_(ema_p.data)

    def _get_fm_head_for_forward(self, model) -> nn.Module:
        """Get the FM head for forward pass, initializing EMA teacher if needed.

        When EMA teacher is enabled, this triggers lazy initialization on first call.
        The student fm_head is always returned (it's in the computation graph),
        but its weights are EMA-smoothed after each optimizer step.
        """
        fm_head = self._get_fm_head(model)

        if self.finetuning_args.fm_ema_teacher_enabled and not self._ema_teacher_initialized:
            self._init_ema_teacher(fm_head)

        return fm_head

    def _manage_backbone_freeze(self, model) -> None:
        """Track backbone freeze phase based on current training step.

        During FM head warmup (step < fm_head_warmup_steps), backbone gradients
        are blocked via detach in compute_loss. After warmup, gradients flow normally.

        IMPORTANT: We do NOT modify requires_grad at runtime. Doing so under DeepSpeed
        ZeRO causes NCCL communication deadlocks (optimizer param_groups become
        inconsistent with the actual gradient state, leading to silent hangs at the
        phase transition step). Instead, we track the phase here and use h.detach()
        in compute_loss to prevent backbone gradients during Phase A.
        """
        head_warmup = self.finetuning_args.fm_head_warmup_steps
        if head_warmup <= 0:
            return  # no warmup configured, nothing to do

        step = self.state.global_step
        should_freeze = (step < head_warmup)

        if should_freeze and not self._backbone_frozen:
            self._backbone_frozen = True
            if not self._logged_warmup_phase:
                logger.info_rank0(
                    f"[FM Warmup] Phase A: Blocking backbone gradients (via detach), only training FM head "
                    f"for {head_warmup} steps (current step: {step})."
                )
                self._logged_warmup_phase = True

        elif not should_freeze and self._backbone_frozen:
            self._backbone_frozen = False
            if not self._logged_joint_phase:
                logger.info_rank0(
                    f"[FM Warmup] Phase B: Enabling backbone gradients, starting joint training "
                    f"(step: {step}). FM loss will ramp over {self.finetuning_args.fm_loss_warmup_steps} steps."
                )
                self._logged_joint_phase = True

    @override
    def create_optimizer(self) -> "torch.optim.Optimizer":
        if self.optimizer is None:
            self.optimizer = create_custom_optimizer(self.model, self.args, self.finetuning_args)
        return super().create_optimizer()

    @override
    def create_scheduler(
        self, num_training_steps: int, optimizer: Optional["torch.optim.Optimizer"] = None
    ) -> "torch.optim.lr_scheduler.LRScheduler":
        create_custom_scheduler(self.args, num_training_steps, optimizer)
        return super().create_scheduler(num_training_steps, optimizer)

    @override
    def _get_train_sampler(self) -> Optional["torch.utils.data.Sampler"]:
        if self.finetuning_args.disable_shuffling:
            return torch.utils.data.SequentialSampler(self.train_dataset)
        return super()._get_train_sampler()

    @override
    def training_step(self, model, inputs, *args, **kwargs):
        """Override training_step to update EMA teacher after optimizer step."""
        loss = super().training_step(model, inputs, *args, **kwargs)

        # Update EMA teacher: smooth fm_head parameters after optimizer step.
        # IMPORTANT: Skip during Phase A (backbone frozen / FM head warmup).
        # During Phase A, FM head needs to learn freely to adapt to backbone's
        # hidden state distribution. EMA would kill its learning rate.
        # EMA only activates once joint training begins (Phase B/C).
        if (
            self.finetuning_args.fm_ema_teacher_enabled
            and self._ema_teacher_initialized
            and not self._backbone_frozen  # ← key: skip during Phase A
        ):
            try:
                fm_head = self._get_fm_head(model)
                self._update_ema_teacher(fm_head)
            except (AttributeError, RuntimeError):
                pass

        # Re-initialize EMA teacher when transitioning from Phase A → Phase B.
        # This resets the EMA to the current (well-trained) FM head state.
        if (
            self.finetuning_args.fm_ema_teacher_enabled
            and self._ema_teacher_initialized
            and not self._backbone_frozen
            and not hasattr(self, "_ema_reinitialized_for_joint")
        ):
            try:
                import copy
                fm_head = self._get_fm_head(model)
                self._ema_fm_head = copy.deepcopy(fm_head)
                for p in self._ema_fm_head.parameters():
                    p.requires_grad_(False)
                self._ema_reinitialized_for_joint = True
                logger.info_rank0(
                    "[FM EMA Teacher] Re-initialized EMA from trained FM head at Phase B start."
                )
            except (AttributeError, RuntimeError):
                pass

        return loss

    @override
    def compute_loss(self, model, inputs, *args, **kwargs):
        """Compute joint AR + FM loss.

        Supports two target modes:
          - Global (fm_suffix_tokens=0): pre-computed z_1 (B, d) broadcast to all positions.
          - Online suffix (fm_suffix_tokens>0): per-position z_1[t] = Encoder(y_{t+1:t+k}),
            computed on-the-fly using the online target encoder.

        Supports two-phase warmup:
          Phase A (step < fm_head_warmup_steps): backbone frozen, only FM head trains.
          Phase B (next fm_loss_warmup_steps): backbone unfrozen, FM weight ramps 0 → target.
          Phase C (remaining): normal FM weight schedule.
        """
        # Manage backbone freeze/unfreeze based on current step
        self._manage_backbone_freeze(model)

        # Determine encoding mode
        suffix_k = self.finetuning_args.fm_suffix_tokens
        use_online_encoding = (suffix_k > 0 and self._online_encoder is not None)

        # Pop pre-computed embedding (may be None in online mode)
        z_1_global = inputs.pop("fm_target_embedding", None)

        # If neither online encoder nor pre-computed embedding is available, fall back to AR-only.
        # IMPORTANT: Still execute a dummy FM head forward to keep DeepSpeed ZeRO-3 submodule
        # order consistent across all ranks (prevents NCCL collective mismatch).
        if not use_online_encoding and z_1_global is None:
            ar_loss = super().compute_loss(model, inputs, *args, **kwargs)
            # We need hidden states to determine dimensions for dummy forward
            inner_model = _unwrap_model(model)
            fm_head = self._get_fm_head_for_forward(model)
            target_dim = fm_head.target_dim
            # Create a minimal dummy hidden state to run fm_head forward
            device = next(inner_model.parameters()).device
            dtype = next(inner_model.parameters()).dtype
            dummy_h = torch.zeros(1, 1, fm_head.condition_dim, device=device, dtype=dtype)
            self._dummy_fm_head_forward(model, dummy_h, target_dim)
            return ar_loss

        fm_head = self._get_fm_head_for_forward(model)
        labels = inputs.get("labels")

        # ── Step 1: forward with hidden states ──
        layer_idx = self.finetuning_args.fm_backbone_layer_index
        need_all_hidden = (layer_idx != -1)

        inner_model = _unwrap_model(model)
        old_output_hs = getattr(inner_model.config, "output_hidden_states", False)
        if need_all_hidden:
            inner_model.config.output_hidden_states = True

        outputs = model(**inputs, output_hidden_states=True)

        if need_all_hidden:
            inner_model.config.output_hidden_states = old_output_hs
        ar_loss = outputs.loss

        if outputs.hidden_states is None:
            logger.warning_rank0("hidden_states is None, falling back to AR-only loss.")
            fm_head = self._get_fm_head_for_forward(model)
            device = ar_loss.device
            dtype = ar_loss.dtype
            dummy_h = torch.zeros(1, 1, fm_head.condition_dim, device=device, dtype=dtype)
            self._dummy_fm_head_forward(model, dummy_h, fm_head.target_dim)
            return ar_loss

        # ── Step 2: hidden states (B, L, hidden_dim) ──
        h = outputs.hidden_states[layer_idx]
        B, L, H = h.shape

        # ── Step 2.5: Gradient gating ──
        # During Phase A (backbone frozen), detach h so FM loss gradients only
        # update fm_head, not backbone. This replaces the unsafe requires_grad_
        # toggling which caused DeepSpeed NCCL deadlocks.
        # NOTE: Use synchronized decision to ensure all ranks make the same choice,
        # preventing divergent computation graphs across ranks.
        self._fm_grad_gate_active = self._sync_grad_gate_decision()
        if self._backbone_frozen or self._fm_grad_gate_active:
            h_for_fm = h.detach()
        else:
            h_for_fm = h

        # ── Step 3: build response mask ──
        if labels is None:
            self._dummy_fm_head_forward(model, h, fm_head.target_dim)
            return ar_loss
        response_mask = (labels != IGNORE_INDEX).float()  # (B, L), 1 at response positions
        total_valid = response_mask.sum()
        if total_valid < 1.0:
            self._dummy_fm_head_forward(model, h, fm_head.target_dim)
            return ar_loss

        # ── Step 4: compute z_1 targets ──
        if use_online_encoding:
            # Online token-level suffix encoding: z_1 has shape (B, L, d)
            # Each response position t gets z_1[t] = Encoder(y_{t+1}, ..., y_{t+k})
            llm_tokenizer = self.processing_class if hasattr(self, "processing_class") else self.tokenizer
            z_1 = self._online_encoder.encode_suffix_tokens(
                labels=labels,
                response_mask=response_mask,
                suffix_tokens=suffix_k,
                llm_tokenizer=llm_tokenizer,
                batch_size=self.finetuning_args.fm_suffix_batch_size,
                max_length=self.finetuning_args.fm_target_encoder_max_length,
                stride=self.finetuning_args.fm_suffix_stride,
            )
            z_1 = z_1.to(device=h.device, dtype=h.dtype)  # (B, L, d)
            d = z_1.size(-1)
            z_1_global_for_cka = z_1.mean(dim=1)  # fallback for CKA mode: (B, d)
        else:
            # Legacy: pre-computed global embedding broadcast to all positions
            z_1_global = z_1_global.to(device=h.device, dtype=h.dtype)  # (B, d)
            d = z_1_global.size(-1)
            z_1 = z_1_global.unsqueeze(1).expand(B, L, d)  # (B, L, d)
            z_1_global_for_cka = z_1_global

        # ── Step 5: auxiliary loss (four modes) ──
        # The FM head architecture is shared across all modes. The modes differ
        # in how the target is compared and whether fm_head is invoked:
        #
        #   (a) flow_matching (default): sample z_0, t; v_hat = fm_head(z_t, t, h);
        #       per-token loss = || v_hat - v_target ||^2 (mean over d).
        #
        #   (b) cosine:  h_proj = fm_head(0, 0, h) (head degenerates to projection);
        #       per-token loss = 1 - cos(h_proj, z_1).
        #
        #   (c) mse:     h_proj = fm_head(0, 0, h);
        #       per-token loss = || h_proj - z_1 ||^2 (mean over d).
        #
        #   (d) cka:     sample-level. X = mean_pool(h over response mask) (B, d_llm),
        #       Y = z_1_global (B, d_enc); linear CKA with unbiased HSIC.
        #       NOT token-level, NOT position-weighted. fm_head is NOT used.
        #       Requires batch size >= 4 at the current step.
        loss_type = self.finetuning_args.fm_loss_type
        per_token_loss: Optional[torch.Tensor] = None  # set for a/b/c; None for d

        if loss_type == "flow_matching":
            z_0 = torch.randn_like(z_1)                           # (B, L, d)
            t = torch.rand(B, device=h.device, dtype=h.dtype)     # (B,) one t per sample
            t_exp = t.view(B, 1, 1)                                # (B, 1, 1)
            z_t = t_exp * z_1 + (1.0 - t_exp) * z_0              # (B, L, d)
            v_target = z_1 - z_0                                   # (B, L, d)
            v_hat = fm_head(z_t, t, h_for_fm)                      # (B, L, d) — uses gated h
            per_token_loss = (v_hat - v_target).pow(2).mean(dim=-1)  # (B, L)

        elif loss_type in ("cosine", "mse"):
            z_t_zero = torch.zeros(B, L, d, device=h.device, dtype=h.dtype)  # (B, L, d)
            t_zero = torch.zeros(B, device=h.device, dtype=h.dtype)          # (B,)
            h_proj = fm_head(z_t_zero, t_zero, h_for_fm)                      # (B, L, d) — uses gated h
            if loss_type == "cosine":
                cos_sim = torch.nn.functional.cosine_similarity(h_proj, z_1, dim=-1, eps=1e-8)
                per_token_loss = 1.0 - cos_sim                                # (B, L)
            else:  # "mse"
                per_token_loss = (h_proj - z_1).pow(2).mean(dim=-1)           # (B, L)

        elif loss_type == "cka":
            # Sample-level CKA — pool hidden states over response positions to (B, d_llm).
            # z_1_global (B, d_enc) is already pooled by the target encoder.
            if B < 4:
                logger.warning_rank0(
                    f"[FM/CKA] batch size B={B} < 4, unbiased HSIC is ill-defined. "
                    "Falling back to AR-only loss for this step. Increase "
                    "per_device_train_batch_size (or use grad-accumulation caching) "
                    "to B >= 4 (recommended >= 32) for stable CKA."
                )
                # Still run FM head forward to keep submodule order consistent
                self._dummy_fm_head_forward(model, h, d)
                return ar_loss
            # Mean-pool h_for_fm over response tokens (ignores prompt/pad). Uses gated h.
            resp_mask_3d = response_mask.unsqueeze(-1)                        # (B, L, 1)
            sum_h = (h_for_fm * resp_mask_3d).sum(dim=1)                       # (B, H)
            denom = resp_mask_3d.sum(dim=1).clamp(min=1.0)                     # (B, 1)
            X = sum_h / denom                                                   # (B, d_llm)
            Y = z_1_global_for_cka                                                  # (B, d_enc)
            # CKA is numerically more stable in fp32, especially under bf16 training.
            fm_loss = linear_cka_loss(X.float(), Y.float()).to(ar_loss.dtype)
        else:
            raise ValueError(f"Unknown fm_loss_type: {loss_type}")

        # ── Step 6: position decay weighting (token-level losses only) ──
        # CKA is already a scalar; it skips this step.
        if per_token_loss is not None:
            # remaining_ratio[t] = (resp_len - pos + 1) / resp_len
            # First response token → ~1.0 (most future), last → ~1/N (least future), prompt → 0
            position_in_resp = response_mask.cumsum(dim=1) * response_mask  # 1-indexed at resp positions, 0 at prompt
            resp_len = response_mask.sum(dim=1, keepdim=True).clamp(min=1)  # (B, 1)
            remaining_ratio = ((resp_len - position_in_resp + 1.0) / resp_len * response_mask).clamp(min=0.0, max=1.0)

            weighted_mask = response_mask * remaining_ratio  # (B, L)
            weighted_total = weighted_mask.sum().clamp(min=1.0)

            fm_loss = (per_token_loss * weighted_mask).sum() / weighted_total

        # ── Step 7: total loss ──
        fm_weight = self._get_fm_loss_weight()
        # During Phase A (backbone frozen), only FM loss drives optimization.
        # AR loss is still logged but detached from the computation graph to
        # avoid wasting compute on gradients that won't update frozen params.
        if self._backbone_frozen:
            total_loss = fm_weight * fm_loss
        else:
            total_loss = ar_loss + fm_weight * fm_loss

        # ── DEBUG: diagnostic print (first 5 steps only) ──
        if self.state.global_step < 5:
            _diag_parts = []
            _diag_parts.append(f"step={self.state.global_step}")
            _diag_parts.append(f"fm_loss={fm_loss.item():.6f}")
            _diag_parts.append(f"fm_loss.requires_grad={fm_loss.requires_grad}")
            _diag_parts.append(f"total_loss={total_loss.item():.6f}")
            _diag_parts.append(f"total_loss.requires_grad={total_loss.requires_grad}")
            _diag_parts.append(f"fm_weight={fm_weight:.4f}")
            _diag_parts.append(f"backbone_frozen={self._backbone_frozen}")
            # Check FM head params
            _fm_h = self._get_fm_head(model)
            _n_params = sum(p.numel() for p in _fm_h.parameters())
            _n_grad = sum(p.numel() for p in _fm_h.parameters() if p.requires_grad)
            _diag_parts.append(f"fm_head: {_n_grad}/{_n_params} params require grad")
            # Check output_proj weight (should be near zero initially)
            _op = _fm_h.output_proj.weight
            _diag_parts.append(f"output_proj.weight: mean={_op.data.mean().item():.6e}, std={_op.data.std().item():.6e}")
            # Check if v_hat is all zeros
            if loss_type == "flow_matching" and per_token_loss is not None:
                _diag_parts.append(f"per_token_loss: mean={per_token_loss.mean().item():.6f}, max={per_token_loss.max().item():.6f}")
            # Check z_1
            _diag_parts.append(f"z_1: shape={z_1.shape}, mean={z_1.mean().item():.4f}, norm={z_1.norm().item():.4f}")
            # Check h_for_fm
            _diag_parts.append(f"h_for_fm: detached={not h_for_fm.requires_grad}")
            logger.info_rank0(f"[FM DEBUG] {' | '.join(_diag_parts)}")
        # ── END DEBUG ──

        self._stored_fm_metrics = {
            "ar_loss": ar_loss.detach().item(),
            "fm_loss": fm_loss.detach().item(),
            "fm_weight": fm_weight,
            "fm_phase": 0.0 if self._backbone_frozen else 1.0,
        }
        # Add gradient gating metrics when enabled
        if self.finetuning_args.fm_grad_gate_enabled:
            # Use fm_loss value as proxy for FM head activity.
            # Track fm_loss EMA — when current fm_loss drops significantly below EMA,
            # it means FM head is learning fast (potential short-circuiting).
            # NOTE: Synchronize fm_loss across ranks to prevent EMA divergence,
            # which would cause grad gate decisions to diverge over time.
            current_fm_loss = fm_loss.detach().item()
            if dist.is_initialized():
                fm_loss_tensor = torch.tensor([current_fm_loss], device=ar_loss.device)
                try:
                    dist.all_reduce(fm_loss_tensor, op=dist.ReduceOp.SUM)
                    fm_loss_tensor /= dist.get_world_size()
                    current_fm_loss = fm_loss_tensor.item()
                except RuntimeError:
                    pass  # fallback to local value in non-distributed testing
            decay = self.finetuning_args.fm_grad_gate_ema_decay
            if not self._fm_grad_ema_initialized:
                self._fm_grad_norm_ema = current_fm_loss
                self._last_fm_grad_norm = current_fm_loss
                self._fm_grad_ema_initialized = True
            else:
                self._last_fm_grad_norm = current_fm_loss
                self._fm_grad_norm_ema = decay * self._fm_grad_norm_ema + (1 - decay) * current_fm_loss

            self._stored_fm_metrics["fm_grad_norm"] = self._last_fm_grad_norm
            self._stored_fm_metrics["fm_grad_norm_ema"] = self._fm_grad_norm_ema
            self._stored_fm_metrics["fm_grad_gated"] = 1.0 if self._fm_grad_gate_active else 0.0

        return total_loss

    @override
    def log(self, logs: dict[str, float], *args, **kwargs) -> None:
        if self._stored_fm_metrics:
            logs.update(self._stored_fm_metrics)
            self._stored_fm_metrics = {}
        return super().log(logs, *args, **kwargs)

    def save_fm_head(self, output_dir: str) -> None:
        """Save fm_head weights separately.

        Handles DeepSpeed ZeRO-3 by gathering partitioned parameters before
        saving. All ranks must call this method (for the collective gather),
        but only rank 0 actually writes to disk.
        """
        unwrapped = self.accelerator.unwrap_model(self.model)
        fm_head = getattr(unwrapped, "fm_head", None)
        if fm_head is None:
            return

        fm_dir = os.path.join(output_dir, "fm_head")
        fm_path = os.path.join(fm_dir, "fm_head.pt")

        # Detect ZeRO-3: check if any fm_head param has a DeepSpeed partition ID
        is_zero3 = any(hasattr(p, "ds_id") for p in fm_head.parameters())

        if is_zero3:
            import deepspeed

            with deepspeed.zero.GatheredParameters(
                list(fm_head.parameters()), modifier_rank=0
            ):
                if self.is_world_process_zero():
                    os.makedirs(fm_dir, exist_ok=True)
                    torch.save(fm_head.state_dict(), fm_path)
                    logger.info_rank0(f"FM head saved (ZeRO-3 gathered) at: {fm_path}")
        else:
            if self.is_world_process_zero():
                os.makedirs(fm_dir, exist_ok=True)
                torch.save(fm_head.state_dict(), fm_path)
                logger.info_rank0(f"FM head saved at: {fm_path}")

    @override
    def save_model(self, output_dir: Optional[str] = None, _internal_call: bool = False):
        if output_dir is None:
            output_dir = self.args.output_dir

        # Temporarily detach fm_head so LLM checkpoint stays clean.
        # All ranks must detach/re-attach to keep model state consistent across DDP.
        # Using _modules dict for safer manipulation of nn.Module internals.
        unwrapped = self.accelerator.unwrap_model(self.model)
        fm_head_ref = unwrapped._modules.pop("fm_head", None)

        try:
            super().save_model(output_dir, _internal_call)
        finally:
            # Re-attach fm_head regardless of save success/failure
            if fm_head_ref is not None:
                unwrapped._modules["fm_head"] = fm_head_ref

        # Save fm_head separately — ALL ranks must call this (ZeRO-3 gather is collective)
        self.save_fm_head(output_dir)
