"""Hypothesis 008 — split-KV flash decode: manufacturing parallelism at batch 1.

**Mechanism, category C.** Hypothesis 006 gives every ``(batch, query_head)`` pair one
program and walks the whole KV cache inside it. At batch 1 that is 16 programs on a card
with 170 SMs: **90% of the GPU is idle** while 16 programs each stream 2048 keys. Splitting
the KV length across programs and combining their partial softmax results turns one long
serial scan into many short parallel ones.

This is a mathematical reassociation — the online-softmax rescaling identity that lets
partial maxima and denominators be merged after the fact — not a fusion or tiling decision.
A scheduler will not derive it, which is what makes it category C rather than category A.

**Prediction: `inconclusive` at the headline workload, and this is a prediction about the
experiment rather than about the kernel.** The KV cache read is 0.78% of per-token bytes
at context 2048, and the model spends 91.85% of its traffic streaming weights. Parallelism
that the rest of the model does not have cannot show up in an end-to-end decode ratio when
attention is not the bottleneck. `docs/HYPOTHESES.md` is explicit that this hypothesis
needs a 32k or 128k workload to express itself, and batch 001 does not have one.

It is in this batch anyway, deliberately, for two reasons. It is the **control for 006**:
006 and 008 differ only in how the KV scan is parallelised, so 008 measuring the same as
006 is evidence that the win 006 may show comes from not materialising the expansion
rather than from anything about the scan. And it puts the kernel on disk, gated and
measured at short context, so the session that adds a long-context workload starts from a
working kernel instead of from a blank file.

**Two stages.** The first computes a partial ``(max, denominator, weighted sum)`` per
split; the second merges them with the standard rescaling. The merge is a separate kernel
rather than an atomic, so the result does not depend on the order splits happen to finish
in — a benchmark whose output changes run to run is not a benchmark.
"""

from __future__ import annotations

import torch
from torch import Tensor

from ._triton import HAS_TRITON, require_cuda, tl, triton

__all__ = ["correctness_checks", "install", "is_installed", "split_kv_decode_attention"]


#: How many pieces the KV scan is cut into. 8 puts 128 programs on the card at batch 1
#: with 16 query heads, which is the point of the hypothesis; past that the merge starts
#: costing more than the parallelism buys at this context length.
DEFAULT_SPLITS = 8


if HAS_TRITON:

    @triton.jit
    def _split_kv_partial_kernel(
        Q,
        K,
        V,
        PARTIAL_ACC,  # (batch, heads_q, splits, head_dim) fp32
        PARTIAL_M,  # (batch, heads_q, splits) fp32
        PARTIAL_L,  # (batch, heads_q, splits) fp32
        stride_qb,
        stride_qh,
        stride_kb,
        stride_kh,
        stride_kn,
        stride_vb,
        stride_vh,
        stride_vn,
        stride_ab,
        stride_ah,
        stride_as,
        stride_mb,
        stride_mh,
        key_len,
        scale,
        HEADS_Q: tl.constexpr,
        GROUPS: tl.constexpr,
        HEAD_DIM: tl.constexpr,
        BLOCK_N: tl.constexpr,
        SPLITS: tl.constexpr,
    ):
        pid = tl.program_id(0)
        split = tl.program_id(1)
        hq = pid % HEADS_Q
        b = pid // HEADS_Q
        hkv = hq // GROUPS

        # Ceiling division so the last split absorbs the remainder rather than the work
        # being silently truncated.
        per_split = (key_len + SPLITS - 1) // SPLITS
        start_n = split * per_split
        end_n = tl.minimum(start_n + per_split, key_len)

        d = tl.arange(0, HEAD_DIM)
        q = tl.load(Q + b * stride_qb + hq * stride_qh + d).to(tl.float32) * scale

        k_base = K + b * stride_kb + hkv * stride_kh
        v_base = V + b * stride_vb + hkv * stride_vh

        m_i = float("-inf")
        l_i = 0.0
        acc = tl.zeros((HEAD_DIM,), dtype=tl.float32)

        for start in range(start_n, end_n, BLOCK_N):
            offs_n = start + tl.arange(0, BLOCK_N)
            mask_n = offs_n < end_n

            k = tl.load(
                k_base + offs_n[:, None] * stride_kn + d[None, :],
                mask=mask_n[:, None],
                other=0.0,
            ).to(tl.float32)
            scores = tl.sum(k * q[None, :], axis=1)
            scores = tl.where(mask_n, scores, float("-inf"))

            m_new = tl.maximum(m_i, tl.max(scores, axis=0))
            alpha = tl.exp(m_i - m_new)
            p = tl.exp(scores - m_new)

            v = tl.load(
                v_base + offs_n[:, None] * stride_vn + d[None, :],
                mask=mask_n[:, None],
                other=0.0,
            ).to(tl.float32)

            acc = acc * alpha + tl.sum(p[:, None] * v, axis=0)
            l_i = l_i * alpha + tl.sum(p, axis=0)
            m_i = m_new

        # An empty split — key_len smaller than SPLITS — writes an identity element the
        # merge will ignore, rather than a NaN it would propagate.
        offs = b * stride_ab + hq * stride_ah + split * stride_as + d
        tl.store(PARTIAL_ACC + offs, acc)
        tl.store(PARTIAL_M + b * stride_mb + hq * stride_mh + split, m_i)
        tl.store(PARTIAL_L + b * stride_mb + hq * stride_mh + split, l_i)

    @triton.jit
    def _split_kv_merge_kernel(
        PARTIAL_ACC,
        PARTIAL_M,
        PARTIAL_L,
        OUT,
        stride_ab,
        stride_ah,
        stride_as,
        stride_mb,
        stride_mh,
        stride_ob,
        stride_oh,
        HEADS_Q: tl.constexpr,
        HEAD_DIM: tl.constexpr,
        SPLITS: tl.constexpr,
    ):
        pid = tl.program_id(0)
        hq = pid % HEADS_Q
        b = pid // HEADS_Q

        d = tl.arange(0, HEAD_DIM)
        splits = tl.arange(0, SPLITS)

        m = tl.load(PARTIAL_M + b * stride_mb + hq * stride_mh + splits)
        l = tl.load(PARTIAL_L + b * stride_mb + hq * stride_mh + splits)

        # Rescale every split onto the global maximum, then sum. This is the identity that
        # makes the split legitimate: exp(s - m_g) = exp(s - m_i) * exp(m_i - m_g).
        m_global = tl.max(m, axis=0)
        rescale = tl.exp(m - m_global)
        # A split that saw nothing has m_i = -inf, giving rescale = 0 and contributing
        # nothing — which is exactly right, and is why -inf is the identity here.
        rescale = tl.where(l > 0.0, rescale, 0.0)

        acc = tl.load(PARTIAL_ACC + b * stride_ab + hq * stride_ah + splits[:, None] * stride_as + d[None, :])
        numerator = tl.sum(acc * rescale[:, None], axis=0)
        denominator = tl.sum(l * rescale, axis=0)

        tl.store(OUT + b * stride_ob + hq * stride_oh + d, (numerator / denominator).to(OUT.dtype.element_ty))


@torch.library.custom_op("deltaforge::split_kv_decode_attention", mutates_args=())
def split_kv_decode_attention(query: Tensor, key: Tensor, value: Tensor, scale: float, splits: int) -> Tensor:
    """Single-token GQA attention with the KV scan split across ``splits`` programs.

    Same signature and same result as `gqa_decode.gqa_decode_attention`; the two differ
    only in how the scan is parallelised, which is what makes 008 a usable control for 006.
    """
    if query.dim() != 3:
        raise ValueError(f"expected query (batch, heads_q, head_dim), got {tuple(query.shape)}")
    if key.shape != value.shape:
        raise ValueError(f"key {tuple(key.shape)} and value {tuple(value.shape)} must match")
    batch, heads_q, head_dim = query.shape
    kv_batch, heads_kv, key_len, kv_dim = key.shape
    if kv_batch != batch or kv_dim != head_dim:
        raise ValueError(f"key {tuple(key.shape)} does not match query {tuple(query.shape)}")
    if heads_kv == 0 or heads_q % heads_kv:
        raise ValueError(f"{heads_q} query heads do not divide into {heads_kv} KV heads")
    if key_len < 1:
        raise ValueError("cannot attend over an empty KV cache")
    if splits < 1 or splits & (splits - 1):
        raise ValueError(f"splits must be a power of two so the merge can index it, got {splits}")
    require_cuda(query, "split_kv_decode_attention")

    query = query.contiguous()
    key = key.contiguous()
    value = value.contiguous()

    partial_acc = torch.empty((batch, heads_q, splits, head_dim), device=query.device, dtype=torch.float32)
    partial_m = torch.empty((batch, heads_q, splits), device=query.device, dtype=torch.float32)
    partial_l = torch.empty((batch, heads_q, splits), device=query.device, dtype=torch.float32)
    out = torch.empty_like(query)

    _split_kv_partial_kernel[(batch * heads_q, splits)](
        query,
        key,
        value,
        partial_acc,
        partial_m,
        partial_l,
        query.stride(0),
        query.stride(1),
        key.stride(0),
        key.stride(1),
        key.stride(2),
        value.stride(0),
        value.stride(1),
        value.stride(2),
        partial_acc.stride(0),
        partial_acc.stride(1),
        partial_acc.stride(2),
        partial_m.stride(0),
        partial_m.stride(1),
        key_len,
        scale,
        HEADS_Q=heads_q,
        GROUPS=heads_q // heads_kv,
        HEAD_DIM=head_dim,
        BLOCK_N=32,
        SPLITS=splits,
        num_warps=8,
        num_stages=2,
    )
    _split_kv_merge_kernel[(batch * heads_q,)](
        partial_acc,
        partial_m,
        partial_l,
        out,
        partial_acc.stride(0),
        partial_acc.stride(1),
        partial_acc.stride(2),
        partial_m.stride(0),
        partial_m.stride(1),
        out.stride(0),
        out.stride(1),
        HEADS_Q=heads_q,
        HEAD_DIM=head_dim,
        SPLITS=splits,
        num_warps=4,
    )
    return out


@split_kv_decode_attention.register_fake
def _split_kv_fake(query: Tensor, key: Tensor, value: Tensor, scale: float, splits: int) -> Tensor:
    return torch.empty_like(query)


_PATCHED: dict[str, type] = {}


def _split_kv_attention_class() -> type:
    if "attn" not in _PATCHED:
        from .gqa_decode import _unexpanded_attention_class

        base = _unexpanded_attention_class()

        class SplitKVGatedAttention(base):  # type: ignore[misc, valid-type]
            """006's layer with only the decode call swapped.

            Subclassing 006 rather than copying it is deliberate: the two hypotheses must
            differ *only* in the scan, or 008 is not a control for 006.
            """

            def forward(self, hidden_states, cos, sin, cache=None, cache_offset=0):
                if hidden_states.shape[1] != 1:
                    return super().forward(hidden_states, cos, sin, cache, cache_offset)
                return _split_kv_forward(self, hidden_states, cos, sin, cache, cache_offset)

        _PATCHED["attn"] = SplitKVGatedAttention
    return _PATCHED["attn"]


def _split_kv_forward(self, hidden_states, cos, sin, cache, cache_offset):
    from ..reference import apply_output_gate, apply_partial_rope

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

    query = apply_partial_rope(query, cos, sin)
    key = apply_partial_rope(key, cos, sin)

    if cache is not None:
        end = cache_offset + seq_len
        cache.keys[:, :, cache_offset:end] = key.to(cache.keys.dtype)
        cache.values[:, :, cache_offset:end] = value.to(cache.values.dtype)
        key = cache.keys[:, :, :end]
        value = cache.values[:, :, :end]

    attn = split_kv_decode_attention(
        query[:, :, 0].contiguous(), key.contiguous(), value.contiguous(), self.scaling, DEFAULT_SPLITS
    ).unsqueeze(2)

    attn = attn.transpose(1, 2).reshape(batch, seq_len, -1)
    if gate is not None:
        attn = attn * apply_output_gate(gate, self.config.output_gate_type)
    return self.o_proj(attn)


def install(model, entry=None) -> None:
    if getattr(model, "_deltaforge_split_kv", False):
        return
    patched = _split_kv_attention_class()
    installed = 0
    for layer in model.layers:
        attn = getattr(layer, "self_attn", None)
        if attn is not None:
            attn.__class__ = patched
            installed += 1
    if installed == 0:
        raise RuntimeError("no full-attention layers found; this hypothesis installed nothing")
    model._deltaforge_split_kv = True
    # 006's marker is set too: this class *is* 006's, with the scan replaced.
    model._deltaforge_gqa_no_expand = True


def is_installed(model) -> bool:
    return bool(getattr(model, "_deltaforge_split_kv", False))


def correctness_checks(model, *, device="cuda", dtype=None, seed: int = 0):
    from ..harness.correctness import check_kernel
    from .gqa_decode import _reference_decode_attention

    config = model.config
    target_dtype = dtype if dtype is not None else next(model.parameters()).dtype
    generator = torch.Generator(device=device).manual_seed(seed)
    scale = config.head_dim**-0.5

    checks = []
    for batch, key_len, label in (
        (1, 2048, "headline decode step: batch 1, context 2048"),
        (32, 2048, "secondary decode step: batch 32, context 2048"),
        # Fewer keys than splits: most splits see nothing at all, and the merge has to
        # treat an empty split as an identity rather than propagating its -inf.
        (1, 3, "context shorter than the split count"),
        (1, 33, "context that is not a multiple of BLOCK_N or of the split count"),
    ):
        query = torch.randn(
            (batch, config.num_attention_heads, config.head_dim),
            device=device,
            dtype=target_dtype,
            generator=generator,
        )
        key = torch.randn(
            (batch, config.num_key_value_heads, key_len, config.head_dim),
            device=device,
            dtype=target_dtype,
            generator=generator,
        )
        value = torch.randn(
            (batch, config.num_key_value_heads, key_len, config.head_dim),
            device=device,
            dtype=target_dtype,
            generator=generator,
        )
        checks.append(
            check_kernel(
                "flash_decode_splitkv.split_kv_decode_attention",
                lambda q, k, v: _reference_decode_attention(q, k, v, scale),
                lambda q, k, v: split_kv_decode_attention(q, k, v, scale, DEFAULT_SPLITS),
                args=(query, key, value),
                replaces="gqa_attention",
                note=label,
            )
        )
    return tuple(checks)
