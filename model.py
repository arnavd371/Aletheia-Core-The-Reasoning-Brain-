"""
model.py — Aletheia-Core Decoder-only Transformer
===================================================
Architecture
------------
* 12 transformer layers, 12 attention heads, 768 hidden dimensions.
* Rotary Positional Embeddings (RoPE) for length-generalisation on math sequences.
* Uses FlashAttention (flash_attn library) when available; falls back to a
  memory-efficient scaled dot-product attention via PyTorch 2.x otherwise.
* A dedicated "Reasoning Head" that, at each position, predicts the next
  algebraic action from a fixed action vocabulary:
      {Expand, Factor, Simplify, Substitute, Transpose, Combine, Evaluate, Done}
"""

from __future__ import annotations

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

# ---------------------------------------------------------------------------
# Optional FlashAttention import
# ---------------------------------------------------------------------------
try:
    from flash_attn import flash_attn_qkvpacked_func, flash_attn_func  # type: ignore

    _FLASH_AVAILABLE = True
except ImportError:
    _FLASH_AVAILABLE = False

# ---------------------------------------------------------------------------
# Action vocabulary for the Reasoning Head
# ---------------------------------------------------------------------------
ACTION_VOCAB = [
    "Expand",
    "Factor",
    "Simplify",
    "Substitute",
    "Transpose",
    "Combine",
    "Evaluate",
    "Done",
]
NUM_ACTIONS = len(ACTION_VOCAB)


# ---------------------------------------------------------------------------
# Rotary Positional Embeddings (RoPE)
# ---------------------------------------------------------------------------
def _rope_sincos(seq_len: int, head_dim: int, device: torch.device, dtype: torch.dtype):
    """Pre-compute cos/sin tables for RoPE.

    Returns tensors of shape ``(seq_len, head_dim)``.
    """
    assert head_dim % 2 == 0, "head_dim must be even for RoPE"
    theta = 1.0 / (
        10000.0 ** (torch.arange(0, head_dim, 2, device=device, dtype=dtype) / head_dim)
    )
    positions = torch.arange(seq_len, device=device, dtype=dtype)
    angles = torch.outer(positions, theta)  # (seq_len, head_dim/2)
    cos = torch.cat([angles.cos(), angles.cos()], dim=-1)  # (seq_len, head_dim)
    sin = torch.cat([angles.sin(), angles.sin()], dim=-1)
    return cos, sin


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    """Rotate the second half of the last dimension by 180° (used for RoPE)."""
    half = x.shape[-1] // 2
    x1, x2 = x[..., :half], x[..., half:]
    return torch.cat([-x2, x1], dim=-1)


def apply_rope(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Apply RoPE to query and key tensors.

    Args:
        q: ``(batch, heads, seq_len, head_dim)``
        k: ``(batch, heads, seq_len, head_dim)``
        cos: ``(seq_len, head_dim)``
        sin: ``(seq_len, head_dim)``
    """
    cos = cos.unsqueeze(0).unsqueeze(0)  # (1, 1, seq_len, head_dim)
    sin = sin.unsqueeze(0).unsqueeze(0)
    q_rot = q * cos + _rotate_half(q) * sin
    k_rot = k * cos + _rotate_half(k) * sin
    return q_rot, k_rot


# ---------------------------------------------------------------------------
# Multi-Head Self-Attention with RoPE
# ---------------------------------------------------------------------------
class MultiHeadSelfAttention(nn.Module):
    """Causal multi-head self-attention.

    Automatically uses FlashAttention when the library is installed and the
    device is CUDA; otherwise falls back to PyTorch's built-in
    ``scaled_dot_product_attention`` (which already includes an efficient
    flash-attention kernel on compatible hardware via ``torch.backends``).
    """

    def __init__(self, hidden_dim: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        assert hidden_dim % num_heads == 0
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads

        self.qkv_proj = nn.Linear(hidden_dim, 3 * hidden_dim, bias=False)
        self.out_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.attn_drop = dropout

    def forward(
        self,
        x: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        B, T, C = x.shape
        qkv = self.qkv_proj(x)  # (B, T, 3*C)
        q, k, v = qkv.split(self.hidden_dim, dim=-1)

        # Reshape to (B, heads, T, head_dim)
        def _reshape(t: torch.Tensor) -> torch.Tensor:
            return t.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)

        q, k, v = _reshape(q), _reshape(k), _reshape(v)

        # Apply RoPE
        q, k = apply_rope(q, k, cos, sin)

        if _FLASH_AVAILABLE and x.is_cuda:
            # flash_attn expects (B, T, heads, head_dim) and fp16/bf16.
            # Prefer bfloat16 when the device supports it, else fall back to fp16.
            orig_dtype = q.dtype
            flash_dtype = (
                torch.bfloat16
                if torch.cuda.is_bf16_supported()
                else torch.float16
            )
            q = q.transpose(1, 2).to(flash_dtype)
            k = k.transpose(1, 2).to(flash_dtype)
            v = v.transpose(1, 2).to(flash_dtype)
            dropout_p = self.attn_drop if self.training else 0.0
            out = flash_attn_func(q, k, v, dropout_p=dropout_p, causal=True)
            out = out.to(orig_dtype)  # (B, T, heads, head_dim)
            out = out.reshape(B, T, C)
        else:
            # PyTorch 2.x scaled_dot_product_attention (uses flash kernel when possible)
            dropout_p = self.attn_drop if self.training else 0.0
            out = F.scaled_dot_product_attention(
                q, k, v,
                attn_mask=mask,
                dropout_p=dropout_p,
                is_causal=(mask is None),
            )  # (B, heads, T, head_dim)
            out = out.transpose(1, 2).reshape(B, T, C)

        return self.out_proj(out)


# ---------------------------------------------------------------------------
# Feed-Forward Network (SwiGLU variant for better performance)
# ---------------------------------------------------------------------------
class FeedForward(nn.Module):
    """SwiGLU feed-forward block used in modern transformer variants."""

    def __init__(self, hidden_dim: int, ffn_dim: Optional[int] = None, dropout: float = 0.1):
        super().__init__()
        ffn_dim = ffn_dim or 4 * hidden_dim
        # Gate + up projections fused; down projection
        self.gate_proj = nn.Linear(hidden_dim, ffn_dim, bias=False)
        self.up_proj = nn.Linear(hidden_dim, ffn_dim, bias=False)
        self.down_proj = nn.Linear(ffn_dim, hidden_dim, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x)))


# ---------------------------------------------------------------------------
# Single Transformer Decoder Layer
# ---------------------------------------------------------------------------
class DecoderLayer(nn.Module):
    def __init__(self, hidden_dim: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        self.attn_norm = nn.RMSNorm(hidden_dim)
        self.ffn_norm = nn.RMSNorm(hidden_dim)
        self.attn = MultiHeadSelfAttention(hidden_dim, num_heads, dropout)
        self.ffn = FeedForward(hidden_dim, dropout=dropout)

    def forward(
        self,
        x: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        x = x + self.attn(self.attn_norm(x), cos, sin, mask)
        x = x + self.ffn(self.ffn_norm(x))
        return x


# ---------------------------------------------------------------------------
# Aletheia-Core Model
# ---------------------------------------------------------------------------
class AletheiaCore(nn.Module):
    """Decoder-only Transformer for step-by-step algebraic reasoning.

    Parameters
    ----------
    vocab_size:
        Size of the token vocabulary (set to match your tokeniser).
    hidden_dim:
        Model width.  Default: 768.
    num_layers:
        Number of decoder layers.  Default: 12.
    num_heads:
        Number of attention heads.  Default: 12.
    max_seq_len:
        Maximum supported sequence length.  Default: 2048.
    dropout:
        Dropout probability used throughout.  Default: 0.1.
    """

    def __init__(
        self,
        vocab_size: int,
        hidden_dim: int = 768,
        num_layers: int = 12,
        num_heads: int = 12,
        max_seq_len: int = 2048,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        self.max_seq_len = max_seq_len

        # Token embedding
        self.token_embedding = nn.Embedding(vocab_size, hidden_dim)
        self.embed_drop = nn.Dropout(dropout)

        # Transformer layers
        self.layers = nn.ModuleList(
            [DecoderLayer(hidden_dim, num_heads, dropout) for _ in range(num_layers)]
        )
        self.norm = nn.RMSNorm(hidden_dim)

        # Language modelling head (tied weights)
        self.lm_head = nn.Linear(hidden_dim, vocab_size, bias=False)
        self.lm_head.weight = self.token_embedding.weight  # weight tying

        # Reasoning Head — predicts the next algebraic action at every position
        self.reasoning_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, NUM_ACTIONS),
        )

        # Pre-compute RoPE tables once; re-compute lazily if seq_len grows
        self._register_rope_buffers(max_seq_len)

        self._init_weights()

    # ------------------------------------------------------------------
    # Initialisation
    # ------------------------------------------------------------------
    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, std=0.02)

    def _register_rope_buffers(self, seq_len: int, device: Optional[torch.device] = None):
        cos, sin = _rope_sincos(
            seq_len,
            self.head_dim,
            device=device or torch.device("cpu"),
            dtype=torch.float32,
        )
        self.register_buffer("rope_cos", cos, persistent=False)
        self.register_buffer("rope_sin", sin, persistent=False)

    # ------------------------------------------------------------------
    # Forward pass
    # ------------------------------------------------------------------
    def forward(
        self,
        input_ids: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> dict[str, torch.Tensor]:
        """Run a forward pass.

        Args:
            input_ids: ``(batch, seq_len)`` integer token ids.
            mask: Optional attention mask ``(batch, 1, seq_len, seq_len)``.

        Returns:
            A dict with keys:
            * ``"logits"``  — LM logits ``(batch, seq_len, vocab_size)``.
            * ``"action_logits"`` — Reasoning-head logits
              ``(batch, seq_len, NUM_ACTIONS)``.
        """
        B, T = input_ids.shape

        # Lazily extend RoPE buffers if needed
        if T > self.rope_cos.shape[0]:
            self._register_rope_buffers(T, device=input_ids.device)

        cos = self.rope_cos[:T].to(input_ids.device)
        sin = self.rope_sin[:T].to(input_ids.device)

        x = self.embed_drop(self.token_embedding(input_ids))

        for layer in self.layers:
            x = layer(x, cos, sin, mask)

        x = self.norm(x)

        logits = self.lm_head(x)
        action_logits = self.reasoning_head(x)

        return {"logits": logits, "action_logits": action_logits}

    # ------------------------------------------------------------------
    # Convenience: parameter count
    # ------------------------------------------------------------------
    def num_parameters(self, trainable_only: bool = True) -> int:
        """Return the total number of (trainable) parameters."""
        params = self.parameters() if not trainable_only else filter(
            lambda p: p.requires_grad, self.parameters()
        )
        return sum(p.numel() for p in params)

    # ------------------------------------------------------------------
    # Greedy / sampling generation helper
    # ------------------------------------------------------------------
    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int = 256,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
        eos_token_id: Optional[int] = None,
    ) -> torch.Tensor:
        """Auto-regressively generate tokens.

        Args:
            input_ids: Prompt token ids ``(batch, seq_len)``.
            max_new_tokens: Maximum tokens to generate.
            temperature: Sampling temperature (1.0 = unscaled).
            top_k: If set, restrict sampling to the top-k logits.
            eos_token_id: Stop generation when this token is produced.

        Returns:
            ``(batch, seq_len + generated_len)`` token ids.
        """
        for _ in range(max_new_tokens):
            # Truncate context to max_seq_len
            context = input_ids[:, -self.max_seq_len:]
            outputs = self(context)
            next_logits = outputs["logits"][:, -1, :]  # (B, vocab_size)

            if temperature != 1.0:
                next_logits = next_logits / max(temperature, 1e-8)

            if top_k is not None:
                # next_logits is 2D (B, vocab_size); take the k-th value per row
                threshold = torch.topk(next_logits, top_k).values[:, -1].unsqueeze(-1)
                next_logits = next_logits.masked_fill(next_logits < threshold, float("-inf"))

            probs = F.softmax(next_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)  # (B, 1)
            input_ids = torch.cat([input_ids, next_token], dim=1)

            if eos_token_id is not None and (next_token == eos_token_id).all():
                break

        return input_ids


# ---------------------------------------------------------------------------
# Factory helpers
# ---------------------------------------------------------------------------
def build_aletheia_core(vocab_size: int, **kwargs) -> AletheiaCore:
    """Construct the default Aletheia-Core configuration.

    Keyword args are forwarded to :class:`AletheiaCore` and can be used to
    override the default hyperparameters (e.g. for ablations or unit tests).
    """
    defaults = dict(
        hidden_dim=768,
        num_layers=12,
        num_heads=12,
        max_seq_len=2048,
        dropout=0.1,
    )
    defaults.update(kwargs)
    return AletheiaCore(vocab_size=vocab_size, **defaults)
