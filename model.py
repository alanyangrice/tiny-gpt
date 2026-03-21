"""
Modern GPT with Block Attention Residuals (Moonshot AI).

Architecture: LLaMA-style decoder-only transformer with RMSNorm, RoPE,
Grouped-Query Attention, SwiGLU FFN, and Block AttnRes replacing fixed
residual connections.

Reference pseudocode: Figure 2 of the Attention Residuals paper.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from config import GPTConfig


RMSNorm = nn.RMSNorm


# ---------------------------------------------------------------------------
# Rotary Position Embeddings (RoPE)
# ---------------------------------------------------------------------------
class RotaryEmbedding(nn.Module):
    def __init__(self, dim: int, max_seq_len: int = 8192, theta: float = 10000.0):
        super().__init__()
        inv_freq = 1.0 / (theta ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self._build_cache(max_seq_len)

    def _build_cache(self, seq_len: int) -> None:
        t = torch.arange(seq_len, dtype=torch.float32)
        freqs = torch.outer(t, self.inv_freq)
        emb = torch.cat([freqs, freqs], dim=-1)
        self.register_buffer("cos_cached", emb.cos(), persistent=False)
        self.register_buffer("sin_cached", emb.sin(), persistent=False)

    def forward(self, seq_len: int) -> tuple[Tensor, Tensor]:
        if seq_len > self.cos_cached.size(0):
            self._build_cache(seq_len)
        return self.cos_cached[:seq_len], self.sin_cached[:seq_len]


def _rotate_half(x: Tensor) -> Tensor:
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_emb(x: Tensor, cos: Tensor, sin: Tensor) -> Tensor:
    """Apply RoPE to tensor of shape [B, n_heads, T, head_dim]."""
    cos = cos[None, None, :, :]
    sin = sin[None, None, :, :]
    return x * cos + _rotate_half(x) * sin


# ---------------------------------------------------------------------------
# Grouped-Query Causal Self-Attention with Flash Attention
# ---------------------------------------------------------------------------
class GQACausalSelfAttention(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.n_head = config.n_head
        self.n_kv_head = config.n_kv_head
        self.head_dim = config.head_dim
        self.n_rep = self.n_head // self.n_kv_head

        self.q_proj = nn.Linear(config.d_model, config.n_head * self.head_dim, bias=config.bias)
        self.k_proj = nn.Linear(config.d_model, config.n_kv_head * self.head_dim, bias=config.bias)
        self.v_proj = nn.Linear(config.d_model, config.n_kv_head * self.head_dim, bias=config.bias)
        self.o_proj = nn.Linear(config.n_head * self.head_dim, config.d_model, bias=config.bias)
        self.attn_dropout = config.dropout
        self.resid_dropout = nn.Dropout(config.dropout)

    def forward(self, x: Tensor, rope_cos: Tensor, rope_sin: Tensor) -> Tensor:
        B, T, _ = x.shape

        q = self.q_proj(x).view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, T, self.n_kv_head, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, T, self.n_kv_head, self.head_dim).transpose(1, 2)

        q = apply_rotary_emb(q, rope_cos, rope_sin)
        k = apply_rotary_emb(k, rope_cos, rope_sin)

        y = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=None,
            dropout_p=self.attn_dropout if self.training else 0.0,
            is_causal=True,
            enable_gqa=self.n_rep > 1,
        )

        y = y.transpose(1, 2).contiguous().view(B, T, -1)
        return self.resid_dropout(self.o_proj(y))


# ---------------------------------------------------------------------------
# SwiGLU Feed-Forward Network
# ---------------------------------------------------------------------------
class SwiGLUFFN(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        d_ff = config.d_ff
        self.w_gate = nn.Linear(config.d_model, d_ff, bias=config.bias)
        self.w_up = nn.Linear(config.d_model, d_ff, bias=config.bias)
        self.w_down = nn.Linear(d_ff, config.d_model, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x: Tensor) -> Tensor:
        return self.dropout(self.w_down(F.silu(self.w_gate(x)) * self.w_up(x)))


# ---------------------------------------------------------------------------
# Block Attention Residuals Operator
# ---------------------------------------------------------------------------
class BlockAttnResOp(nn.Module):
    """
    Depth-wise softmax attention over block representations.
    Each instance owns one zero-initialized pseudo-query w_l and one RMSNorm
    for key normalization.  (Moonshot, Attention Residuals, Figure 2)
    """

    def __init__(self, d_model: int):
        super().__init__()
        self.pseudo_query = nn.Parameter(torch.zeros(d_model))
        self.key_norm = RMSNorm(d_model)

    def forward(self, blocks: list[Tensor], partial_block: Tensor | None) -> Tensor:
        if partial_block is not None:
            sources = torch.stack(blocks + [partial_block], dim=0)
        else:
            sources = torch.stack(blocks, dim=0)
        K = self.key_norm(sources)
        logits = torch.einsum("d, n b t d -> n b t", self.pseudo_query, K)
        weights = logits.softmax(dim=0)
        return torch.einsum("n b t, n b t d -> b t d", weights, sources)


# ---------------------------------------------------------------------------
# Single Transformer Layer with Block AttnRes wiring
#   Follows the paper's Figure 2 pseudocode exactly.
# ---------------------------------------------------------------------------
class TransformerLayer(nn.Module):
    def __init__(self, config: GPTConfig, layer_idx: int):
        super().__init__()
        self.layer_idx = layer_idx
        self.layers_per_block = config.layers_per_attn_res_block
        self.is_block_boundary = (layer_idx % self.layers_per_block == 0)

        self.attn_res_attn = BlockAttnResOp(config.d_model)
        self.attn_norm = RMSNorm(config.d_model)
        self.attn = GQACausalSelfAttention(config)

        self.attn_res_mlp = BlockAttnResOp(config.d_model)
        self.mlp_norm = RMSNorm(config.d_model)
        self.mlp = SwiGLUFFN(config)

    def forward(
        self,
        blocks: list[Tensor],
        partial_block: Tensor | None,
        rope_cos: Tensor,
        rope_sin: Tensor,
    ) -> tuple[list[Tensor], Tensor | None]:
        # ---- pre-attention depth aggregation ----
        h = self.attn_res_attn(blocks, partial_block)

        # ---- block boundary (between AttnRes and attention, per paper Fig 2) ----
        if self.is_block_boundary:
            if partial_block is not None:
                blocks = blocks + [partial_block]
            partial_block = None

        # ---- self-attention ----
        attn_out = self.attn(self.attn_norm(h), rope_cos, rope_sin)
        if partial_block is not None:
            partial_block = partial_block + attn_out
        else:
            partial_block = attn_out

        # ---- pre-MLP depth aggregation ----
        h = self.attn_res_mlp(blocks, partial_block)

        # ---- FFN ----
        mlp_out = self.mlp(self.mlp_norm(h))
        partial_block = partial_block + mlp_out

        return blocks, partial_block


# ---------------------------------------------------------------------------
# GPT Model
# ---------------------------------------------------------------------------
class GPT(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.config = config

        self.tok_emb = nn.Embedding(config.vocab_size, config.d_model)
        self.drop = nn.Dropout(config.dropout)

        self.rope = RotaryEmbedding(config.head_dim, config.block_size, config.rope_theta)

        self.layers = nn.ModuleList(
            TransformerLayer(config, i) for i in range(config.n_layer)
        )

        self.final_attn_res = BlockAttnResOp(config.d_model)
        self.final_norm = RMSNorm(config.d_model)

        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)
        self.tok_emb.weight = self.lm_head.weight

        self.apply(self._init_weights)
        for m in self.modules():
            if isinstance(m, BlockAttnResOp):
                nn.init.zeros_(m.pseudo_query)

        if config.use_gradient_checkpointing:
            self._enable_gradient_checkpointing()

        n_params = sum(p.numel() for p in self.parameters())
        print(f"Model parameters: {n_params / 1e6:.1f}M")

    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def _enable_gradient_checkpointing(self) -> None:
        for layer in self.layers:
            layer._orig_forward = layer.forward

            def _make_ckpt(mod: TransformerLayer):
                def _ckpt_fwd(blocks, partial_block, rope_cos, rope_sin):
                    return torch.utils.checkpoint.checkpoint(
                        mod._orig_forward,
                        blocks, partial_block, rope_cos, rope_sin,
                        use_reentrant=False,
                    )
                return _ckpt_fwd

            layer.forward = _make_ckpt(layer)

    def forward(
        self,
        idx: Tensor,
        targets: Optional[Tensor] = None,
    ) -> tuple[Tensor, Optional[Tensor]]:
        B, T = idx.shape
        assert T <= self.config.block_size, (
            f"Sequence length {T} exceeds block_size {self.config.block_size}"
        )

        x = self.drop(self.tok_emb(idx))
        rope_cos, rope_sin = self.rope(T)
        rope_cos = rope_cos.to(x.device, dtype=x.dtype)
        rope_sin = rope_sin.to(x.device, dtype=x.dtype)

        blocks: list[Tensor] = [x]          # b0 = token embedding
        partial_block: Tensor | None = None  # no sub-layer outputs yet

        for layer in self.layers:
            blocks, partial_block = layer(blocks, partial_block, rope_cos, rope_sin)

        h = self.final_attn_res(blocks, partial_block)
        h = self.final_norm(h)
        logits = self.lm_head(h)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))

        return logits, loss

    @torch.no_grad()
    def generate(
        self,
        idx: Tensor,
        max_new_tokens: int,
        temperature: float = 1.0,
        top_k: int | None = None,
        top_p: float | None = None,
    ) -> Tensor:
        for _ in range(max_new_tokens):
            idx_cond = idx if idx.size(1) <= self.config.block_size else idx[:, -self.config.block_size :]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] / temperature

            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = float("-inf")

            if top_p is not None:
                sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                cumulative_probs = sorted_logits.softmax(dim=-1).cumsum(dim=-1)
                mask = cumulative_probs - sorted_logits.softmax(dim=-1) >= top_p
                sorted_logits[mask] = float("-inf")
                logits = sorted_logits.scatter(1, sorted_indices, sorted_logits)

            probs = logits.softmax(dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            idx = torch.cat([idx, next_token], dim=1)
        return idx
