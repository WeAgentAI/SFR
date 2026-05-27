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

"""Flow Matching Head and Target Encoder modules.

This module implements:
1. FlowMatchingHead: A conditional MLP that takes noisy features z_t, time step t,
   and a conditioning hidden state h from the LLM backbone, and predicts the velocity
   field v_hat for the flow matching objective.
2. encode_texts_with_target_encoder: A subprocess-based function that encodes texts
   using a frozen sentence encoder, completely isolated from DeepSpeed/DDP.
"""

import math
from typing import Optional

import torch
import torch.nn as nn


class TimestepEmbedding(nn.Module):
    """Sinusoidal timestep embedding, maps scalar t to a vector."""

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.dim = dim
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.SiLU(),
            nn.Linear(dim * 4, dim),
        )

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        half_dim = self.dim // 2
        emb = math.log(10000.0) / max(half_dim - 1, 1)
        emb = torch.exp(torch.arange(half_dim, device=t.device, dtype=t.dtype) * -emb)
        emb = t.unsqueeze(-1) * emb
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)
        if self.dim % 2 == 1:
            emb = nn.functional.pad(emb, (0, 1))
        return self.mlp(emb)


class _ResidualBlock(nn.Module):
    """A single residual MLP block: Linear → LayerNorm → SiLU, with skip connection."""

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.linear = nn.Linear(dim, dim)
        self.norm = nn.LayerNorm(dim)
        self.act = nn.SiLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.act(self.norm(self.linear(x)))


class FlowMatchingHead(nn.Module):
    """Conditional MLP for predicting the velocity field in Flow Matching.

    Architecture: input projection → N-1 residual blocks → output projection.
    Residual connections improve gradient flow for deeper heads.
    """

    def __init__(
        self,
        condition_dim: int,
        target_dim: int,
        hidden_dim: int = 1024,
        num_layers: int = 3,
    ) -> None:
        super().__init__()
        self.condition_dim = condition_dim
        self.target_dim = target_dim
        self.time_embed = TimestepEmbedding(target_dim)

        input_dim = target_dim + target_dim + condition_dim

        # Input projection
        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
        )

        # Residual blocks (num_layers - 2 intermediate blocks, minimum 0)
        num_res_blocks = max(num_layers - 2, 0)
        self.res_blocks = nn.Sequential(*[_ResidualBlock(hidden_dim) for _ in range(num_res_blocks)])

        # Output projection — zero-initialized so that initial v_hat ≈ 0,
        # producing a mild initial FM loss (≈ E[||v_target||²]) and preventing
        # large gradients from disrupting a pre-trained backbone.
        self.output_proj = nn.Linear(hidden_dim, target_dim)
        nn.init.zeros_(self.output_proj.weight)
        nn.init.zeros_(self.output_proj.bias)

    def forward(self, z_t: torch.Tensor, t: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
        t_emb = self.time_embed(t)
        if z_t.dim() == 3 and t_emb.dim() == 2:
            t_emb = t_emb.unsqueeze(1).expand_as(z_t)
        x = torch.cat([z_t, t_emb, h], dim=-1)
        x = self.input_proj(x)
        x = self.res_blocks(x)
        return self.output_proj(x)


# ============================================================================
# Subprocess-based target encoding — uses subprocess.run with a temp script
# to achieve 100% process isolation (no DeepSpeed, no DDP, no NCCL, nothing)
# ============================================================================

_ENCODE_SCRIPT_TEMPLATE = r'''
import json, os, sys, time

print(f"[TargetEncoder] PID={os.getpid()}", flush=True)

import torch
from transformers import AutoModel, AutoTokenizer

model_path  = sys.argv[1]
texts_file  = sys.argv[2]
output_file = sys.argv[3]
batch_size  = int(sys.argv[4])
gpu_id      = sys.argv[5] if len(sys.argv) > 5 else ""

# ── Device selection ──
if gpu_id and torch.cuda.is_available():
    device = torch.device(f"cuda:{gpu_id}")
    use_fp16 = True
    print(f"[TargetEncoder] Using GPU cuda:{gpu_id}, fp16=True", flush=True)
else:
    device = torch.device("cpu")
    use_fp16 = False
    print("[TargetEncoder] Using CPU, fp32", flush=True)

# ── Load model ──
t0 = time.time()
tokenizer = AutoTokenizer.from_pretrained(model_path)
encoder = AutoModel.from_pretrained(
    model_path,
    torch_dtype=torch.float16 if use_fp16 else torch.float32,
)
encoder.eval()
for p in encoder.parameters():
    p.requires_grad_(False)
encoder.to(device)
embed_dim = encoder.config.hidden_size
print(f"[TargetEncoder] Model loaded in {time.time()-t0:.1f}s, embed_dim={embed_dim}", flush=True)

# ── Load texts ──
with open(texts_file, "r", encoding="utf-8") as f:
    texts = json.load(f)
N = len(texts)
print(f"[TargetEncoder] {N} texts loaded", flush=True)

# ── Encode ──
all_embeddings = torch.zeros(N, embed_dim, dtype=torch.float32)
t_start = time.time()
for i in range(0, N, batch_size):
    batch = texts[i : i + batch_size]
    ne_idx = [j for j, t in enumerate(batch) if t.strip()]
    ne_txt = [batch[j] for j in ne_idx]
    if ne_txt:
        inputs = tokenizer(ne_txt, padding=True, truncation=True, max_length=512, return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}
        with torch.no_grad(), torch.amp.autocast("cuda", enabled=use_fp16):
            out = encoder(**inputs)
            mask = inputs["attention_mask"].unsqueeze(-1).float()
            se = (out.last_hidden_state.float() * mask).sum(1) / mask.sum(1).clamp(min=1e-9)
            se = torch.nn.functional.normalize(se, p=2, dim=-1)
        for k, b in enumerate(ne_idx):
            all_embeddings[i + b] = se[k].cpu()
    done = min(i + batch_size, N)
    if (done // batch_size) % 20 == 0 or done == N:
        el = time.time() - t_start
        print(f"[TargetEncoder] {done}/{N} ({100*done/N:.0f}%) {done/el:.0f} texts/s", flush=True)

torch.save(all_embeddings, output_file)
print(f"[TargetEncoder] Saved {all_embeddings.shape}, total {time.time()-t_start:.1f}s", flush=True)
'''


def encode_texts_with_target_encoder(
    model_name_or_path: str,
    texts: list[str],
    batch_size: int = 256,
    max_length: int = 512,
) -> "torch.Tensor":
    """Encode texts in a fully isolated subprocess. Uses GPU if available.

    Returns a (N, embed_dim) float32 tensor (on CPU).

    The subprocess has ALL distributed env vars stripped so DeepSpeed/NCCL/DDP
    cannot interfere. It sees exactly one GPU (the first visible one) or falls
    back to CPU.

    WARNING: This function should only be called from the main process (rank 0).
    Calling from multiple ranks simultaneously may cause GPU memory contention.
    """
    import json
    import os
    import subprocess
    import sys
    import tempfile

    from ...extras.logging import get_logger
    _logger = get_logger(__name__)

    # Warn if called from non-rank-0 process
    local_rank = os.environ.get("LOCAL_RANK", "0")
    if local_rank != "0":
        _logger.warning(
            f"[FM encode] Called from LOCAL_RANK={local_rank}. "
            "This function is intended to be called only from the main process."
        )

    with tempfile.TemporaryDirectory() as tmp_dir:
        texts_file  = os.path.join(tmp_dir, "texts.json")
        output_file = os.path.join(tmp_dir, "embeddings.pt")
        script_file = os.path.join(tmp_dir, "encode.py")

        _logger.info(f"[FM encode] Writing {len(texts)} texts to temp file")
        with open(texts_file, "w", encoding="utf-8") as f:
            json.dump(texts, f, ensure_ascii=False)
        with open(script_file, "w", encoding="utf-8") as f:
            f.write(_ENCODE_SCRIPT_TEMPLATE)

        # ── Clean environment: strip ALL distributed / DeepSpeed vars ──
        dist_vars = {
            "RANK", "LOCAL_RANK", "WORLD_SIZE", "LOCAL_WORLD_SIZE",
            "MASTER_ADDR", "MASTER_PORT", "GROUP_RANK", "ROLE_RANK",
            "TORCHELASTIC_RUN_ID", "TORCHELASTIC_RESTART_COUNT",
            "TORCHELASTIC_MAX_RESTARTS", "TORCHELASTIC_ERROR_FILE",
            "OMP_NUM_THREADS",
            "NCCL_ASYNC_ERROR_HANDLING", "NCCL_DEBUG",
            "DEEPSPEED_ZERO_STAGE", "ACCELERATE_USE_DEEPSPEED",
            "ACCELERATE_DEEPSPEED_ZERO3_INIT", "ACCELERATE_DEEPSPEED_CONFIG_FILE",
        }
        clean_env = {k: v for k, v in os.environ.items() if k not in dist_vars}
        clean_env["DS_ACCELERATOR"] = "cpu"  # prevent DeepSpeed auto-init

        # ── Pick one GPU for the subprocess ──
        # Use LOCAL_RANK-aware selection to avoid multi-rank GPU contention
        local_rank = os.environ.get("LOCAL_RANK", "0")
        original_cuda = os.environ.get("CUDA_VISIBLE_DEVICES", "")
        if original_cuda:
            gpu_list = original_cuda.split(",")
            gpu_idx = int(local_rank) if int(local_rank) < len(gpu_list) else 0
            first_gpu = gpu_list[gpu_idx]
        else:
            first_gpu = local_rank
        clean_env["CUDA_VISIBLE_DEVICES"] = first_gpu
        gpu_arg = "0"

        _logger.info(
            f"[FM encode] Subprocess: GPU={first_gpu}, batch_size={batch_size}, "
            f"texts={len(texts)}, model={model_name_or_path}"
        )

        cmd = [
            sys.executable, script_file,
            model_name_or_path, texts_file, output_file, str(batch_size), gpu_arg,
        ]

        result = subprocess.run(cmd, env=clean_env, capture_output=True, text=True)

        # Log output
        for line in (result.stdout or "").strip().split("\n"):
            if line:
                _logger.info(f"[FM encode] {line}")
        if result.returncode != 0:
            for line in (result.stderr or "").strip().split("\n"):
                if line:
                    _logger.error(f"[FM encode ERR] {line}")
            raise RuntimeError(
                f"Target encoder subprocess failed (rc={result.returncode}).\n"
                f"stderr: {result.stderr[-2000:]}"
            )

        _logger.info(f"[FM encode] Loading embeddings from {output_file}")
        embeddings = torch.load(output_file, map_location="cpu", weights_only=True)
        _logger.info(f"[FM encode] shape={embeddings.shape}")

    return embeddings


def get_target_encoder_hidden_size(model_name_or_path: str) -> int:
    """Get the hidden size of the target encoder without loading the full model."""
    from transformers import AutoConfig
    config = AutoConfig.from_pretrained(model_name_or_path)
    return config.hidden_size


class OnlineTargetEncoder:
    """Frozen target encoder for online suffix encoding during training.

    Loaded onto GPU in fp16, all parameters frozen. Encodes suffix token sequences
    into L2-normalized embeddings via mean pooling of last_hidden_state.

    NOT an nn.Module subclass — this is intentional to prevent DeepSpeed ZeRO-3
    from intercepting its __init__ and sharding the encoder weights (which causes
    'weight must be 2-D' errors in the embedding layer).

    Usage:
        encoder = OnlineTargetEncoder("BAAI/bge-large-zh", device="cuda:0")
        embeddings = encoder.encode_texts(["hello world"])
    """

    def __init__(self, model_name_or_path: str, device: str = "cuda", dtype=None):
        from transformers import AutoModel, AutoTokenizer

        if dtype is None:
            dtype = torch.float16

        self._tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)

        # Disable DeepSpeed ZeRO-3 to prevent weight sharding of this encoder.
        # Problem: `from_pretrained` internally checks `is_deepspeed_zero3_enabled()`
        # and if True, wraps model init in `deepspeed.zero.Init()` which partitions
        # embedding weights into 1-D shards → causes "weight must be 2-D" error.
        # Solution: temporarily clear the global HF DeepSpeed config weakref so that
        # `is_deepspeed_zero3_enabled()` returns False during our from_pretrained call.
        _ds_config_ref_backup = None
        try:
            import transformers.integrations.deepspeed as _ds_module
            _ds_config_ref_backup = _ds_module._hf_deepspeed_config_weak_ref
            _ds_module._hf_deepspeed_config_weak_ref = None
        except (ImportError, AttributeError):
            pass

        try:
            self._encoder = AutoModel.from_pretrained(model_name_or_path, torch_dtype=dtype)
        finally:
            # Restore the global DeepSpeed config reference
            if _ds_config_ref_backup is not None:
                try:
                    import transformers.integrations.deepspeed as _ds_module
                    _ds_module._hf_deepspeed_config_weak_ref = _ds_config_ref_backup
                except (ImportError, AttributeError):
                    pass

        self._encoder.eval()
        for p in self._encoder.parameters():
            p.requires_grad_(False)
        self._encoder.to(device)
        self._device = device
        self._dtype = dtype
        self.hidden_size = self._encoder.config.hidden_size

    @property
    def tokenizer(self):
        return self._tokenizer

    @torch.no_grad()
    def encode_texts(self, texts: list[str], max_length: int = 512) -> torch.Tensor:
        """Encode a list of texts into L2-normalized embeddings.

        Args:
            texts: list of text strings to encode
            max_length: maximum token length for truncation

        Returns:
            (N, hidden_dim) tensor of L2-normalized embeddings on self._device
        """
        if not texts:
            return torch.zeros(0, self.hidden_size, device=self._device, dtype=torch.float32)

        inputs = self._tokenizer(
            texts, padding=True, truncation=True, max_length=max_length, return_tensors="pt"
        )
        inputs = {k: v.to(self._device) for k, v in inputs.items()}

        with torch.amp.autocast("cuda", enabled=(self._dtype == torch.float16)):
            outputs = self._encoder(**inputs)

        # Mean pooling over non-padding tokens
        mask = inputs["attention_mask"].unsqueeze(-1).float()  # (N, L, 1)
        hidden = outputs.last_hidden_state.float()  # (N, L, H)
        pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-9)  # (N, H)
        # L2 normalize
        pooled = torch.nn.functional.normalize(pooled, p=2, dim=-1)
        return pooled

    @torch.no_grad()
    def encode_suffix_tokens(
        self,
        labels: torch.Tensor,
        response_mask: torch.Tensor,
        suffix_tokens: int,
        llm_tokenizer,
        batch_size: int = 64,
        max_length: int = 512,
        stride: int = 1,
        min_suffix_len: int = 16,
    ) -> torch.Tensor:
        """Encode suffix token sequences for sampled response positions.

        Uses sparse sampling with stride: only every `stride`-th response position is
        encoded. Intermediate positions are filled with the nearest anchor's embedding
        (nearest-neighbor interpolation). This reduces encoder calls from O(resp_len)
        to O(resp_len / stride).

        When stride=1, every position is encoded (original behavior, no approximation).
        When stride=32, a 2048-token response only needs ~64 encoder calls.

        Positions where the suffix is shorter than `min_suffix_len` tokens are skipped
        (their z_1 remains zero and will be masked out by response_mask weighting).

        Args:
            labels: (B, L) tensor with IGNORE_INDEX at prompt/pad, token IDs at response
            response_mask: (B, L) float tensor, 1.0 at response positions
            suffix_tokens: number of subsequent tokens to encode (k)
            llm_tokenizer: the LLM's tokenizer (for decoding token IDs to text)
            batch_size: encoder batch size
            max_length: max length for encoder tokenizer
            stride: encode every stride-th position; fill in between with nearest anchor
            min_suffix_len: skip positions with fewer than this many suffix tokens

        Returns:
            (B, L, d) tensor of suffix embeddings. Non-response/skipped positions are zero.
        """
        B, L = labels.shape
        d = self.hidden_size
        device = labels.device

        # Output tensor
        z_1 = torch.zeros(B, L, d, device=device, dtype=torch.float32)

        # Collect anchor positions and their suffix texts
        # anchor_info[b] = list of (position_index, text) for sample b
        all_texts: list[str] = []
        all_positions: list[tuple[int, int]] = []  # (batch_idx, seq_position)
        # Track which positions are anchors per sample for nearest-neighbor fill
        anchor_positions_per_sample: list[list[int]] = [[] for _ in range(B)]

        for b in range(B):
            resp_positions = (response_mask[b] > 0.5).nonzero(as_tuple=True)[0]
            if len(resp_positions) == 0:
                continue

            label_row = labels[b]
            num_resp = len(resp_positions)

            # Select anchor indices: every `stride`-th within the response positions
            anchor_indices = list(range(0, num_resp, stride))
            # Always include the first position
            if 0 not in anchor_indices:
                anchor_indices = [0] + anchor_indices

            for ai in anchor_indices:
                t_idx = resp_positions[ai].item()

                # Extract suffix token IDs
                suffix_start = t_idx + 1
                suffix_end = min(t_idx + 1 + suffix_tokens, L)

                suffix_ids = []
                for s in range(suffix_start, suffix_end):
                    tok_id = label_row[s].item()
                    if tok_id != -100:  # IGNORE_INDEX
                        suffix_ids.append(tok_id)

                # Skip if suffix too short
                if len(suffix_ids) < min_suffix_len:
                    continue

                suffix_text = llm_tokenizer.decode(suffix_ids, skip_special_tokens=True)
                if suffix_text.strip():
                    all_texts.append(suffix_text)
                    all_positions.append((b, t_idx))
                    anchor_positions_per_sample[b].append(t_idx)

        if not all_texts:
            return z_1.to(device)

        # Batch encode all anchor suffix texts
        all_embeddings = []
        for i in range(0, len(all_texts), batch_size):
            batch_texts = all_texts[i:i + batch_size]
            embs = self.encode_texts(batch_texts, max_length=max_length)  # (bs, d)
            all_embeddings.append(embs)
        all_embeddings = torch.cat(all_embeddings, dim=0)  # (N_anchors, d)

        # Place anchor embeddings into z_1
        for idx, (b, t) in enumerate(all_positions):
            z_1[b, t] = all_embeddings[idx]

        # Nearest-neighbor fill: for each non-anchor response position, copy from nearest anchor
        if stride > 1:
            for b in range(B):
                anchors = anchor_positions_per_sample[b]
                if not anchors:
                    continue

                resp_positions = (response_mask[b] > 0.5).nonzero(as_tuple=True)[0]
                anchors_tensor = torch.tensor(anchors, device=device)

                for t in resp_positions:
                    t_idx = t.item()
                    if t_idx in anchors:
                        continue  # already has embedding

                    # Find nearest anchor
                    dists = (anchors_tensor - t_idx).abs()
                    nearest_anchor = anchors[dists.argmin().item()]

                    # Copy embedding from nearest anchor
                    z_1[b, t_idx] = z_1[b, nearest_anchor]

        return z_1.to(device)


# ============================================================================
# CKA (Centered Kernel Alignment) — linear CKA with unbiased HSIC estimator.
# Reference: Kornblith et al. 2019 (ICML); Song et al. 2012 (unbiased HSIC);
# Dasgupta & Cohn 2025 (ICLR) for LLM hidden-state alignment.
# Used as an ablation loss for FM training; supports heterogeneous dims
# (X.shape[-1] != Y.shape[-1]), so no projection of h is required.
# ============================================================================


def _unbiased_hsic(K: torch.Tensor, L: torch.Tensor) -> torch.Tensor:
    """Song et al. 2012 unbiased HSIC estimator.

    K, L: (n, n) symmetric kernel matrices (e.g. linear kernel K = X @ X.T).
    Requires n >= 4.
    """
    n = K.shape[0]
    # Zero out the diagonals
    K = K - torch.diag(torch.diag(K))
    L = L - torch.diag(torch.diag(L))

    trace = (K * L).sum()                            # = tr(K L) since both symmetric
    sum_K = K.sum()
    sum_L = L.sum()
    ones_K_L_ones = (K.sum(dim=0) * L.sum(dim=0)).sum()  # = 1^T K L 1

    hsic = (
        trace
        + sum_K * sum_L / ((n - 1) * (n - 2))
        - 2.0 * ones_K_L_ones / (n - 2)
    ) / (n * (n - 3))
    return hsic


def linear_cka_loss(X: torch.Tensor, Y: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Linear CKA loss = 1 - CKA(X, Y).

    Supports heterogeneous dims (X.shape[1] may differ from Y.shape[1]).
    Inputs should be column-centered internally here.

    X: (n, d1) — e.g. pooled LLM hidden states for n samples
    Y: (n, d2) — e.g. pooled target-encoder embeddings for n samples
    """
    # Column-centering (critical for CKA)
    X = X - X.mean(dim=0, keepdim=True)
    Y = Y - Y.mean(dim=0, keepdim=True)

    # Linear kernel (Gram matrices in sample space)
    K = X @ X.T   # (n, n)
    L = Y @ Y.T   # (n, n)

    hsic_xy = _unbiased_hsic(K, L)
    hsic_xx = _unbiased_hsic(K, K)
    hsic_yy = _unbiased_hsic(L, L)

    denom = torch.sqrt(torch.clamp(hsic_xx * hsic_yy, min=eps))
    cka = hsic_xy / (denom + eps)
    cka = torch.clamp(cka, min=0.0, max=1.0)

    return 1.0 - cka

