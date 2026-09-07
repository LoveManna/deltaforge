"""Hypothesis 006 — attention decode that never materialises the GQA head expansion.

**This is the only slot in batch 001 predicted to win, and the prediction is conditional.**

**Mechanism.** `reference.py:579` expands 4 KV heads to 16 query heads with a real
`repeat_interleave`, which allocates and writes a 4x copy of the KV cache and then reads
it back through the attention matmul. At batch 1, context 2048 that is **570 MB/token —
6.23% of all per-token traffic, and 300x more than every elementwise fusion in the
backlog combined.** A kernel that indexes the unexpanded cache directly never pays it:
each program maps its query head to `hq // groups` and reads the one KV head that
actually exists.

**Category B**: it moves fewer bytes. That is a choice about representation, not about
scheduling, which is why a scheduler is not obliged to find it.

**Why the prediction is conditional.** Inductor may already fold the `repeat_interleave`'s
index arithmetic into its consumer instead of materialising it. If it does, `compiled`
already has this win and there is nothing here to take — and that is a genuinely
interesting fact about inductor, to be recorded as the finding rather than buried as a
null. `docs/HYPOTHESES.md` flags this as the thing to check first, and the honest way to
check it is `TORCH_LOGS=output_code`, which costs nothing. Until someone reads that dump,
6.23% is an upper bound on a win, not a prediction of one.

**Also honest about the baseline.** This 6.23% is partly a property of *our* reference
choosing to copy rather than alias. The claim is "against `torch.compile(max-autotune)`",
never "against eager", and never "attention libraries do this wrong".

**The share grows with context.** It scales with the KV cache: at 32k it is already 32.5%
of per-token bytes on the 27B config. `docs/roofline.py --context N` before deciding this
is a 6% hypothesis for the context you actually mean to measure.

## The two paths, and why both are the candidate

The Triton kernel handles **single-token decode**, which is the only thing the benchmark
times. Prefill runs a broadcast-view formulation — `query.view(B, HKV, groups, S, D)`
against `key.unsqueeze(2)` — which also never materialises the expansion, in pure PyTorch.

Neither path is the reference. That matters: this package has no silent fallbacks, because
a candidate that quietly ran the baseline would be recorded as the kernel under test. The
prefill path is the *candidate's* prefill, it is stated here, and it is outside the timed
region regardless.

**Numerics.** The online softmax accumulates in fp32 throughout, where the reference casts
its fp32 softmax weights back to bf16 before the value matmul. The kernel is therefore
slightly *more* accurate than what it replaces, not less. The layer-1 gate compares
against the reference at rtol/atol 1e-2, which bf16 attention sits well inside.
"""

from __future__ import annotations

import torch
from torch import Tensor

from ._triton import HAS_TRITON, require_cuda, tl, triton

__all__ = [
    "correctness_checks",
    "gqa_decode_attention",
    "install",
    "is_installed",
    "reference_style_expanded_attention",
]


if HAS_TRITON:

    @triton.jit
    def _gqa_decode_kernel(
        Q,  # (batch, heads_q, head_dim)
        K,  # (batch, heads_kv, key_len, head_dim)
        V,  # (batch, heads_kv, key_len, head_dim)
        OUT,  # (batch, heads_q, head_dim)
        stride_qb,
        stride_qh,
        stride_kb,
        stride_kh,
        stride_kn,
        stride_vb,
        stride_vh,
        stride_vn,
        stride_ob,
        stride_oh,
        key_len,
        scale,
        HEADS_Q: tl.constexpr,
        GROUPS: tl.constexpr,
        HEAD_DIM: tl.constexpr,
        BLOCK_N: tl.constexpr,
    ):
        pid = tl.program_id(0)
        hq = pid % HEADS_Q
        b = pid // HEADS_Q
        # The whole hypothesis, in one line: the query head reads the KV head it shares
        # rather than a private copy of it.
        hkv = hq // GROUPS

        d = tl.arange(0, HEAD_DIM)
        q = tl.load(Q + b * stride_qb + hq * stride_qh + d).to(tl.float32) * scale

        k_base = K + b * stride_kb + hkv * stride_kh
        v_base = V + b * stride_vb + hkv * stride_vh

        # Online softmax: running max, running denominator, running weighted sum.
        m_i = float("-inf")
        l_i = 0.0
        acc = tl.zeros((HEAD_DIM,), dtype=tl.float32)

        for start in range(0, key_len, BLOCK_N):
            offs_n = start + tl.arange(0, BLOCK_N)
            mask_n = offs_n < key_len

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

        out = acc / l_i
        tl.store(OUT + b * stride_ob + hq * stride_oh + d, out.to(OUT.dtype.element_ty))


@torch.library.custom_op("deltaforge::gqa_decode_attention", mutates_args=())
def gqa_decode_attention(query: Tensor, key: Tensor, value: Tensor, scale: float) -> Tensor:
    """Single-token GQA attention against an **unexpanded** KV cache.

    ``query`` is ``(batch, heads_q, head_dim)``, ``key``/``value`` are
    ``(batch, heads_kv, key_len, head_dim)`` with ``heads_q % heads_kv == 0``. Returns
    ``(batch, heads_q, head_dim)``.

    An opaque custom op so inductor may CUDA-graph around it but may not decompose it back
    into the matmul-softmax-matmul this hypothesis exists to replace.
    """
    if query.dim() != 3:
        raise ValueError(f"expected query (batch, heads_q, head_dim), got {tuple(query.shape)}")
    if key.dim() != 4 or value.dim() != 4:
        raise ValueError(
            f"expected key/value (batch, heads_kv, key_len, head_dim), got "
            f"{tuple(key.shape)} and {tuple(value.shape)}"
        )
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
    for name, t in (("query", query), ("key", key), ("value", value)):
        if t.stride(-1) != 1:
            raise ValueError(f"{name} must have the head dimension innermost and contiguous")
    require_cuda(query, "gqa_decode_attention")

    out = torch.empty_like(query)
    # 32 keys x 256 head dims of fp32 is the largest tile that reliably stays in registers
    # on an SM with 8 warps; wider spills and spilling costs more than the extra loop trips.
    block_n = 32
    _gqa_decode_kernel[(batch * heads_q,)](
        query,
        key,
        value,
        out,
        query.stride(0),
        query.stride(1),
        key.stride(0),
        key.stride(1),
        key.stride(2),
        value.stride(0),
        value.stride(1),
        value.stride(2),
        out.stride(0),
        out.stride(1),
        key_len,
        scale,
        HEADS_Q=heads_q,
        GROUPS=heads_q // heads_kv,
        HEAD_DIM=head_dim,
        BLOCK_N=block_n,
        num_warps=8,
        num_stages=2,
    )
    return out


@gqa_decode_attention.register_fake
def _gqa_decode_fake(query: Tensor, key: Tensor, value: Tensor, scale: float) -> Tensor:
    return torch.empty_like(query)


def reference_style_expanded_attention(
    query: Tensor, key: Tensor, value: Tensor, scale: float, cache_offset: int = 0
) -> Tensor:
    """The candidate's **prefill** path: broadcasting, not expansion.

    ``query`` is ``(batch, heads_q, seq, head_dim)``, ``key``/``value`` are
    ``(batch, heads_kv, key_len, head_dim)``.

    Views the query as ``(batch, heads_kv, groups, seq, head_dim)`` and lets the matmul
    broadcast against an unsqueezed key, which reaches the same result as
    `repeat_interleave` without allocating the 4x copy. Every operation here is a view or
    a matmul; nothing materialises an expanded cache.

    Not used in the timed region — the benchmark times decode only — but it is the
    candidate's own code rather than a fallback to the reference, which is the rule this
    package does not bend.
    """
    import torch.nn.functional as F

    batch, heads_q, seq_len, head_dim = query.shape
    heads_kv, key_len = key.shape[1], key.shape[2]
    groups = heads_q // heads_kv

    q = query.view(batch, heads_kv, groups, seq_len, head_dim)
    scores = torch.matmul(q, key.unsqueeze(2).transpose(-1, -2)) * scale
    if seq_len > 1:
        query_pos = torch.arange(seq_len, device=scores.device) + cache_offset
        key_pos = torch.arange(key_len, device=scores.device)
        blocked = key_pos.unsqueeze(0) > query_pos.unsqueeze(1)
        scores = scores.masked_fill(blocked, torch.finfo(scores.dtype).min)
    weights = F.softmax(scores, dim=-1, dtype=torch.float32).to(query.dtype)
    attn = torch.matmul(weights, value.unsqueeze(2))
    return attn.view(batch, heads_q, seq_len, head_dim)


_PATCHED: dict[str, type] = {}


def _unexpanded_attention_class() -> type:
    if "attn" not in _PATCHED:
        from ..reference import GatedAttention, apply_output_gate, apply_partial_rope

        class UnexpandedGatedAttention(GatedAttention):
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

                query = apply_partial_rope(query, cos, sin)
                key = apply_partial_rope(key, cos, sin)

                if cache is not None:
                    end = cache_offset + seq_len
                    cache.keys[:, :, cache_offset:end] = key.to(cache.keys.dtype)
                    cache.values[:, :, cache_offset:end] = value.to(cache.values.dtype)
                    key = cache.keys[:, :, :end]
                    value = cache.values[:, :, :end]

                # No repeat_interleave anywhere below. This is the hypothesis.
                if seq_len == 1:
                    attn = gqa_decode_attention(
                        query[:, :, 0].contiguous(),
                        key.contiguous(),
                        value.contiguous(),
                        self.scaling,
                    ).unsqueeze(2)
                else:
                    attn = reference_style_expanded_attention(query, key, value, self.scaling, cache_offset)

                attn = attn.transpose(1, 2).reshape(batch, seq_len, -1)
                if gate is not None:
                    attn = attn * apply_output_gate(gate, self.config.output_gate_type)
                return self.o_proj(attn)

        _PATCHED["attn"] = UnexpandedGatedAttention
    return _PATCHED["attn"]


def install(model, entry=None) -> None:
    if getattr(model, "_deltaforge_gqa_no_expand", False):
        return
    patched = _unexpanded_attention_class()
    installed = 0
    for layer in model.layers:
        attn = getattr(layer, "self_attn", None)
        if attn is not None:
            attn.__class__ = patched
            installed += 1
    if installed == 0:
        raise RuntimeError("no full-attention layers found; this hypothesis installed nothing")
    model._deltaforge_gqa_no_expand = True


def is_installed(model) -> bool:
    return bool(getattr(model, "_deltaforge_gqa_no_expand", False))


def _reference_decode_attention(query, key, value, scale):
    """Exactly what `reference.GatedAttention.forward` does at seq_len 1, expansion and all.

    Written out here rather than called through the module so the check compares against
    the *operation*, on tensors the gate controls, rather than against a whole layer.
    """
    import torch.nn.functional as F

    groups = query.shape[1] // key.shape[1]
    k = key.repeat_interleave(groups, dim=1)
    v = value.repeat_interleave(groups, dim=1)
    scores = torch.matmul(query.unsqueeze(2), k.transpose(2, 3)) * scale
    weights = F.softmax(scores, dim=-1, dtype=torch.float32).to(query.dtype)
    return torch.matmul(weights, v).squeeze(2)


def correctness_checks(model, *, device="cuda", dtype=None, seed: int = 0):
    from ..harness.correctness import check_kernel

    config = model.config
    target_dtype = dtype if dtype is not None else next(model.parameters()).dtype
    generator = torch.Generator(device=device).manual_seed(seed)
    scale = config.head_dim**-0.5

    checks = []
    for batch, key_len, label in (
        (1, 2048, "headline decode step: batch 1, context 2048"),
        (32, 2048, "secondary decode step: batch 32, context 2048"),
        (1, 1, "first decoded token: a single key"),
        (1, 33, "context that is not a multiple of BLOCK_N"),
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
                "gqa_decode.gqa_decode_attention",
                lambda q, k, v: _reference_decode_attention(q, k, v, scale),
                lambda q, k, v: gqa_decode_attention(q, k, v, scale),
                args=(query, key, value),
                replaces="gqa_attention",
                note=label,
            )
        )
    return tuple(checks)
