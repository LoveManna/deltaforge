"""Hypothesis 005 — fused partial mRoPE.

**Mechanism.** `reference.apply_partial_rope` slices the head dimension into a rotary part
and a passthrough part, builds `rotate_half` with a `torch.cat` of two negated slices,
multiplies, and concatenates the result back together. On this model that is 64 rotary
dims of 256, so **three quarters of every element it touches is copied for no reason
other than to rebuild a contiguous tensor.** One kernel reads the row once, rotates the
first 64 lanes in registers, passes the other 192 through, and writes once.

**Ceiling: 0.004% of per-token bytes** — the smallest in the backlog, because the Q/K/V
intermediates are 0.33 MB/token across all 8 full-attention layers while the projections'
weights are the real cost. Predicted `inconclusive`, and it is in this batch only because
a slot is now cheap enough that a measured null beats an argued one.

**The rotation, spelled out.** With ``half = rotary_dim // 2``, the reference computes
``rot * cos + rotate_half(rot) * sin`` where ``rotate_half(x) = cat(-x[half:], x[:half])``:

    i <  half:  out[i] = rot[i] * cos[i] - rot[i + half] * sin[i]
    i >= half:  out[i] = rot[i] * cos[i] + rot[i - half] * sin[i]

so each lane needs its partner lane — one extra gather of the 64 rotary elements, and
nothing else. This is **mRoPE-interleaved and partial**; the interleaving lives in
`RotaryEmbedding._interleave`, which builds `cos`/`sin`, and is untouched here.

Strides are passed explicitly rather than requiring contiguity: the reference receives
`query`/`key` straight from a `.transpose(1, 2)` and is perfectly happy with the
non-contiguous result, so a kernel that demanded contiguity would charge the candidate a
copy the baseline never pays and lose on its own accounting.
"""

from __future__ import annotations

import torch
from torch import Tensor

from ._triton import HAS_TRITON, next_power_of_two, require_cuda, tl, triton

__all__ = ["apply_partial_rope", "correctness_checks", "install", "is_installed"]


if HAS_TRITON:

    @triton.jit
    def _partial_rope_kernel(
        X,
        COS,
        SIN,
        OUT,
        stride_xb,
        stride_xh,
        stride_xs,
        stride_ob,
        stride_oh,
        stride_os,
        stride_cb,
        stride_cs,
        num_heads,
        seq_len,
        HEAD_DIM: tl.constexpr,
        ROTARY_DIM: tl.constexpr,
        BLOCK: tl.constexpr,
    ):
        row = tl.program_id(0)
        # x is (batch, heads, seq, head_dim); cos/sin are (batch, seq, rotary_dim), so the
        # head index has to be divided back out to find the right cos row.
        s = row % seq_len
        h = (row // seq_len) % num_heads
        b = row // (seq_len * num_heads)

        x_base = b * stride_xb + h * stride_xh + s * stride_xs
        o_base = b * stride_ob + h * stride_oh + s * stride_os
        c_base = b * stride_cb + s * stride_cs

        cols = tl.arange(0, BLOCK)
        in_row = cols < HEAD_DIM
        in_rot = cols < ROTARY_DIM
        half = ROTARY_DIM // 2
        is_low = cols < half

        x = tl.load(X + x_base + cols, mask=in_row, other=0.0).to(tl.float32)
        # The partner lane: i + half for the low half, i - half for the high half.
        partner = tl.where(is_low, cols + half, cols - half)
        xp = tl.load(X + x_base + partner, mask=in_rot, other=0.0).to(tl.float32)

        c = tl.load(COS + c_base + cols, mask=in_rot, other=0.0).to(tl.float32)
        s_val = tl.load(SIN + c_base + cols, mask=in_rot, other=0.0).to(tl.float32)

        # -sin on the low half, +sin on the high half: that is all rotate_half's negation
        # of the upper slice amounts to once the cat is gone.
        sign = tl.where(is_low, -1.0, 1.0)
        rotated = x * c + sign * xp * s_val

        out = tl.where(in_rot, rotated, x)
        tl.store(OUT + o_base + cols, out.to(OUT.dtype.element_ty), mask=in_row)


@torch.library.custom_op("deltaforge::partial_rope", mutates_args=())
def apply_partial_rope(x: Tensor, cos: Tensor, sin: Tensor) -> Tensor:
    """`reference.apply_partial_rope`, in one pass over the head dimension.

    ``x`` is ``(batch, heads, seq, head_dim)``; ``cos``/``sin`` are
    ``(batch, seq, rotary_dim)`` — the reference unsqueezes a head axis into them, which
    this kernel does by index arithmetic instead.
    """
    if x.dim() != 4:
        raise ValueError(f"expected (batch, heads, seq, head_dim), got {tuple(x.shape)}")
    if cos.shape != sin.shape:
        raise ValueError(f"cos shape {tuple(cos.shape)} != sin shape {tuple(sin.shape)}")
    if cos.dim() != 3:
        raise ValueError(f"expected cos as (batch, seq, rotary_dim), got {tuple(cos.shape)}")
    batch, heads, seq_len, head_dim = x.shape
    rotary_dim = cos.shape[-1]
    if cos.shape[0] != batch or cos.shape[1] != seq_len:
        raise ValueError(f"cos {tuple(cos.shape)} does not match x {tuple(x.shape)} on batch/seq")
    if rotary_dim > head_dim:
        raise ValueError(f"rotary_dim {rotary_dim} exceeds head_dim {head_dim}")
    if rotary_dim % 2:
        raise ValueError(f"rotary_dim must be even to rotate in halves, got {rotary_dim}")
    if x.stride(-1) != 1:
        raise ValueError("the head dimension must be the innermost contiguous axis")
    require_cuda(x, "partial_rope")

    cos = cos.contiguous()
    sin = sin.contiguous()
    out = torch.empty_like(x)
    block = max(next_power_of_two(head_dim), 32)
    rows = batch * heads * seq_len
    if rows == 0:  # pragma: no cover - defensive
        return out
    _partial_rope_kernel[(rows,)](
        x,
        cos,
        sin,
        out,
        x.stride(0),
        x.stride(1),
        x.stride(2),
        out.stride(0),
        out.stride(1),
        out.stride(2),
        cos.stride(0),
        cos.stride(1),
        heads,
        seq_len,
        HEAD_DIM=head_dim,
        ROTARY_DIM=rotary_dim,
        BLOCK=block,
        num_warps=4,
    )
    return out


@apply_partial_rope.register_fake
def _partial_rope_fake(x: Tensor, cos: Tensor, sin: Tensor) -> Tensor:
    return torch.empty_like(x)


_PATCHED: dict[str, type] = {}


def _roped_attention_class() -> type:
    """The patched attention class, built once.

    The forward is written out rather than delegating to the reference's with the
    module-level `apply_partial_rope` name temporarily rebound. That trick is tempting —
    it would keep one copy of the attention body — but rebinding a module global is
    exactly what Dynamo cannot trace: `torch.compile` would graph-break at the rebind, and
    a graph break in the candidate column makes the candidate lose for a reason that has
    nothing to do with the kernel. `reference_parity_test.py` pins this copy against the
    reference so the duplication cannot drift silently.
    """
    if "attn" not in _PATCHED:
        import torch.nn.functional as F

        from ..reference import GatedAttention, apply_output_gate

        class RopedGatedAttention(GatedAttention):
            def forward(self, hidden_states, cos, sin, cache=None, cache_offset=0):
                config = self.config
                batch, seq_len, _ = hidden_states.shape

                projected = self.q_proj(hidden_states)
                if config.attn_output_gate:
                    projected = projected.view(batch, seq_len, config.num_attention_heads, 2 * self.head_dim)
                    query, gate = torch.chunk(projected, 2, dim=-1)
                    gate = gate.reshape(batch, seq_len, -1)
                else:
                    query = projected.view(batch, seq_len, config.num_attention_heads, self.head_dim)
                    gate = None

                query = self.q_norm(query).transpose(1, 2)
                key = self.k_norm(
                    self.k_proj(hidden_states).view(batch, seq_len, config.num_key_value_heads, self.head_dim)
                ).transpose(1, 2)
                value = (
                    self.v_proj(hidden_states)
                    .view(batch, seq_len, config.num_key_value_heads, self.head_dim)
                    .transpose(1, 2)
                )

                # The two lines this hypothesis exists to change.
                query = apply_partial_rope(query, cos, sin)
                key = apply_partial_rope(key, cos, sin)

                if cache is not None:
                    end = cache_offset + seq_len
                    cache.keys[:, :, cache_offset:end] = key.to(cache.keys.dtype)
                    cache.values[:, :, cache_offset:end] = value.to(cache.values.dtype)
                    key = cache.keys[:, :, :end]
                    value = cache.values[:, :, :end]

                groups = config.num_key_value_groups
                if groups > 1:
                    key = key.repeat_interleave(groups, dim=1)
                    value = value.repeat_interleave(groups, dim=1)

                scores = torch.matmul(query, key.transpose(2, 3)) * self.scaling
                key_len = key.shape[2]
                if seq_len > 1:
                    query_pos = torch.arange(seq_len, device=scores.device) + cache_offset
                    key_pos = torch.arange(key_len, device=scores.device)
                    blocked = key_pos.unsqueeze(0) > query_pos.unsqueeze(1)
                    scores = scores.masked_fill(blocked, torch.finfo(scores.dtype).min)
                weights = F.softmax(scores, dim=-1, dtype=torch.float32).to(query.dtype)
                attn = torch.matmul(weights, value)

                attn = attn.transpose(1, 2).reshape(batch, seq_len, -1)
                if gate is not None:
                    attn = attn * apply_output_gate(gate, self.config.output_gate_type)
                return self.o_proj(attn)

        _PATCHED["attn"] = RopedGatedAttention
    return _PATCHED["attn"]


def install(model, entry=None) -> None:
    if getattr(model, "_deltaforge_fused_rope", False):
        return
    patched = _roped_attention_class()
    installed = 0
    for layer in model.layers:
        attn = getattr(layer, "self_attn", None)
        if attn is not None:
            attn.__class__ = patched
            installed += 1
    if installed == 0:
        raise RuntimeError("no full-attention layers found; this hypothesis installed nothing")
    model._deltaforge_fused_rope = True


def is_installed(model) -> bool:
    return bool(getattr(model, "_deltaforge_fused_rope", False))


def correctness_checks(model, *, device="cuda", dtype=None, seed: int = 0):
    from ..harness.correctness import check_kernel
    from ..reference import apply_partial_rope as reference_rope

    config = model.config
    target_dtype = dtype if dtype is not None else next(model.parameters()).dtype
    generator = torch.Generator(device=device).manual_seed(seed)

    checks = []
    for batch, seq, heads, label in (
        (1, 1, config.num_attention_heads, "headline decode step: batch 1, one token, query heads"),
        (1, 1, config.num_key_value_heads, "headline decode step: key heads"),
        (32, 1, config.num_attention_heads, "secondary decode step: batch 32"),
        (1, 2048, config.num_attention_heads, "prefill: batch 1, 2048 tokens"),
    ):
        x = torch.randn(
            (batch, heads, seq, config.head_dim), device=device, dtype=target_dtype, generator=generator
        )
        cos = torch.randn(
            (batch, seq, config.rotary_dim), device=device, dtype=target_dtype, generator=generator
        )
        sin = torch.randn(
            (batch, seq, config.rotary_dim), device=device, dtype=target_dtype, generator=generator
        )
        checks.append(
            check_kernel(
                "fused_rope.partial_rope",
                reference_rope,
                apply_partial_rope,
                args=(x, cos, sin),
                replaces="qkv_projection_rope",
                note=label,
            )
        )
    return tuple(checks)
