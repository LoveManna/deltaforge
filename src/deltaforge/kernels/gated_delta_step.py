"""Hypothesis 007 — the gated delta rule's recurrent step, fused.

**Mechanism, category A.** `reference.recurrent_gated_delta_rule` runs five separate
PyTorch operations over the ``(128, 128)`` fp32 state, per head, per token:

    state = state * exp(g)                       # read + write the whole state
    recalled = (state * k[:, None]).sum(-2)      # read it again
    delta = (v - recalled) * beta
    state = state + k[:, None] * delta[None, :]  # read and write again
    out = (state * q[:, None]).sum(-2)           # read again

That is **five passes over the state where one suffices.** A single program can load its
head's state tile once, keep it in registers through the decay, the recall, the rank-1
correction and the read-out, and write it back once. 24 of 32 layers are linear-attention
layers, so this touches three quarters of the model.

**Share of per-token bytes: 1.10%**, and the ceiling is not that number but the fraction
of it the extra passes waste — the state has to be read and written once no matter what.
Predicted `inconclusive`: 1.10% is already close to the noise band before the ceiling is
discounted, and inductor's home turf is exactly this kind of memory-bound elementwise
chain, so it may well already fuse most of the passes. Reading its output code would
settle that for free and is the first thing the next session should do with this entry.

**This is not hypothesis 3 from the backlog.** The chunked parallel scan is a different
and much larger claim — a reassociation of the recurrence — and `docs/HYPOTHESES.md` is
explicit that it cannot express itself at batch-1 single-token decode, where there is no
sequence to chunk. This kernel changes *how the same recurrence is executed*, not what
recurrence is computed, and is honest about being the smaller claim.

**The state stays fp32** (`mamba_ssm_dtype`) whatever the surrounding weights are. That is
a config-level contract, not a choice this kernel gets to make.

**Prefill loops the same kernel** one token at a time, which is what the reference already
does in Python. Slower is fine and expected: prefill is outside the timed region. What
matters is that it is the candidate's own code — this package has no fallbacks.
"""

from __future__ import annotations

import torch
from torch import Tensor

from ._triton import HAS_TRITON, require_cuda, tl, triton

__all__ = ["correctness_checks", "delta_rule_step", "install", "is_installed"]


if HAS_TRITON:

    @triton.jit
    def _delta_step_kernel(
        STATE_IN,  # (batch, heads, key_dim, value_dim) fp32
        STATE_OUT,
        Q,  # (batch, heads, key_dim) fp32, already l2-normed and scaled
        K,  # (batch, heads, key_dim) fp32, already l2-normed
        V,  # (batch, heads, value_dim) fp32
        G,  # (batch, heads) fp32, log-space decay (<= 0)
        BETA,  # (batch, heads) fp32
        OUT,  # (batch, heads, value_dim) fp32
        stride_sb,
        stride_sh,
        stride_sk,
        stride_qb,
        stride_qh,
        stride_vb,
        stride_vh,
        stride_gb,
        stride_gh,
        HEADS: tl.constexpr,
        KEY_DIM: tl.constexpr,
        BLOCK_V: tl.constexpr,
        VALUE_DIM: tl.constexpr,
    ):
        pid = tl.program_id(0)
        v_block = tl.program_id(1)
        h = pid % HEADS
        b = pid // HEADS

        offs_k = tl.arange(0, KEY_DIM)
        offs_v = v_block * BLOCK_V + tl.arange(0, BLOCK_V)
        mask_v = offs_v < VALUE_DIM

        q = tl.load(Q + b * stride_qb + h * stride_qh + offs_k)
        k = tl.load(K + b * stride_qb + h * stride_qh + offs_k)
        v = tl.load(V + b * stride_vb + h * stride_vh + offs_v, mask=mask_v, other=0.0)
        g = tl.load(G + b * stride_gb + h * stride_gh)
        beta = tl.load(BETA + b * stride_gb + h * stride_gh)

        # The state tile for this (head, value-block): loaded once, written once.
        state_ptr = STATE_IN + b * stride_sb + h * stride_sh + offs_k[:, None] * stride_sk + offs_v[None, :]
        state = tl.load(state_ptr, mask=mask_v[None, :], other=0.0)

        # 1. Decay. `g` is a log-space decay, so this is a multiply by exp(g) <= 1.
        state = state * tl.exp(g)
        # 2. Recall what the state currently associates with this key.
        recalled = tl.sum(state * k[:, None], axis=0)
        # 3. Correct towards the new value by beta: how much this token overwrites.
        delta = (v - recalled) * beta
        # 4. Rank-1 update.
        state = state + k[:, None] * delta[None, :]
        # 5. Read out with the query.
        out = tl.sum(state * q[:, None], axis=0)

        out_ptr = STATE_OUT + b * stride_sb + h * stride_sh + offs_k[:, None] * stride_sk + offs_v[None, :]
        tl.store(out_ptr, state, mask=mask_v[None, :])
        tl.store(OUT + b * stride_vb + h * stride_vh + offs_v, out, mask=mask_v)


@torch.library.custom_op("deltaforge::delta_rule_step", mutates_args=())
def delta_rule_step(
    state: Tensor, query: Tensor, key: Tensor, value: Tensor, g: Tensor, beta: Tensor
) -> tuple[Tensor, Tensor]:
    """One delta-rule step. Returns ``(output, new_state)``.

    ``state`` is ``(batch, heads, key_dim, value_dim)`` fp32; ``query``/``key`` are
    ``(batch, heads, key_dim)`` **already l2-normed, with the query already scaled by
    ``1/sqrt(key_dim)``**; ``value`` is ``(batch, heads, value_dim)``; ``g`` and ``beta``
    are ``(batch, heads)``.

    The normalisation is left to the caller on purpose: it is three cheap elementwise ops
    on a 128-wide vector, and doing it here would mean this kernel had two jobs and a
    correctness gate that could not tell which one was wrong.
    """
    if state.dim() != 4:
        raise ValueError(f"expected state (batch, heads, key_dim, value_dim), got {tuple(state.shape)}")
    batch, heads, key_dim, value_dim = state.shape
    for name, t, shape in (
        ("query", query, (batch, heads, key_dim)),
        ("key", key, (batch, heads, key_dim)),
        ("value", value, (batch, heads, value_dim)),
        ("g", g, (batch, heads)),
        ("beta", beta, (batch, heads)),
    ):
        if tuple(t.shape) != shape:
            raise ValueError(f"{name} has shape {tuple(t.shape)}, expected {shape}")
    if state.dtype != torch.float32:
        raise ValueError(
            f"the recurrent state is fp32 by config (mamba_ssm_dtype), got {state.dtype}. "
            "Running it in bf16 would change the model, not just the kernel."
        )
    require_cuda(state, "delta_rule_step")

    state = state.contiguous()
    query, key, value = query.contiguous(), key.contiguous(), value.contiguous()
    g, beta = g.contiguous(), beta.contiguous()

    state_out = torch.empty_like(state)
    out = torch.empty((batch, heads, value_dim), device=state.device, dtype=torch.float32)

    # 128 key dims x 32 value dims of fp32 per program keeps the tile register-resident.
    block_v = 32
    grid = (batch * heads, (value_dim + block_v - 1) // block_v)
    _delta_step_kernel[grid](
        state,
        state_out,
        query,
        key,
        value,
        g,
        beta,
        out,
        state.stride(0),
        state.stride(1),
        state.stride(2),
        query.stride(0),
        query.stride(1),
        value.stride(0),
        value.stride(1),
        g.stride(0),
        g.stride(1),
        HEADS=heads,
        KEY_DIM=key_dim,
        BLOCK_V=block_v,
        VALUE_DIM=value_dim,
        num_warps=4,
    )
    return out, state_out


@delta_rule_step.register_fake
def _delta_step_fake(state, query, key, value, g, beta):
    batch, heads, _key_dim, value_dim = state.shape
    return (
        torch.empty((batch, heads, value_dim), device=state.device, dtype=torch.float32),
        torch.empty_like(state),
    )


def fused_gated_delta_rule(query, key, value, g, beta, initial_state=None):
    """`reference.recurrent_gated_delta_rule`'s signature, driven by the Triton step.

    The pre-normalisation is kept identical to the reference — same order, same fp32
    casts, same ``1/sqrt(key_dim)`` scaling — because those choices are part of what the
    correctness gate compares against.
    """
    import math

    from ..config import STATE_DTYPE
    from ..reference import l2norm

    initial_dtype = query.dtype
    query, key, value, beta, g = (
        x.transpose(1, 2).to(STATE_DTYPE).contiguous() for x in (query, key, value, beta, g)
    )
    query = l2norm(query)
    key = l2norm(key)
    query = query / math.sqrt(query.shape[-1])

    batch, heads, seq_len, key_head_dim = key.shape
    value_head_dim = value.shape[-1]

    if initial_state is None:
        state = torch.zeros(
            (batch, heads, key_head_dim, value_head_dim), dtype=STATE_DTYPE, device=value.device
        )
    else:
        state = initial_state.to(STATE_DTYPE)

    out = torch.empty_like(value)
    for t in range(seq_len):
        step_out, state = delta_rule_step(
            state, query[:, :, t], key[:, :, t], value[:, :, t], g[:, :, t], beta[:, :, t]
        )
        out[:, :, t] = step_out

    return out.transpose(1, 2).contiguous().to(initial_dtype), state


_PATCHED: dict[str, type] = {}


def _fused_delta_class() -> type:
    if "linear_attn" not in _PATCHED:
        import torch.nn.functional as F

        from ..reference import GatedDeltaNet

        class FusedGatedDeltaNet(GatedDeltaNet):
            def forward(self, hidden_states, cache=None):
                config = self.config
                batch, seq_len, _ = hidden_states.shape

                qkv = self.in_proj_qkv(hidden_states).transpose(1, 2)
                qkv = self._causal_conv(qkv, cache).transpose(1, 2)
                query, key, value = torch.split(
                    qkv,
                    [config.linear_key_dim, config.linear_key_dim, config.linear_value_dim],
                    dim=-1,
                )
                query = query.reshape(batch, seq_len, config.linear_num_key_heads, config.linear_key_head_dim)
                key = key.reshape(batch, seq_len, config.linear_num_key_heads, config.linear_key_head_dim)
                value = value.reshape(
                    batch, seq_len, config.linear_num_value_heads, config.linear_value_head_dim
                )

                z = self.in_proj_z(hidden_states).reshape(
                    batch, seq_len, config.linear_num_value_heads, config.linear_value_head_dim
                )
                beta = self.in_proj_b(hidden_states).sigmoid()
                g = -self.A_log.float().exp() * F.softplus(
                    self.in_proj_a(hidden_states).float() + self.dt_bias.float()
                )

                groups = config.linear_value_groups
                if groups > 1:
                    query = query.repeat_interleave(groups, dim=2)
                    key = key.repeat_interleave(groups, dim=2)

                initial_state = cache.recurrent if cache is not None else None
                core_out, final_state = fused_gated_delta_rule(
                    query, key, value, g=g, beta=beta, initial_state=initial_state
                )
                if cache is not None:
                    cache.recurrent.copy_(final_state)

                core_out = self.norm(
                    core_out.reshape(-1, config.linear_value_head_dim),
                    z.reshape(-1, config.linear_value_head_dim),
                ).reshape(batch, seq_len, config.linear_value_dim)
                return self.out_proj(core_out)

        _PATCHED["linear_attn"] = FusedGatedDeltaNet
    return _PATCHED["linear_attn"]


def install(model, entry=None) -> None:
    if getattr(model, "_deltaforge_fused_delta", False):
        return
    patched = _fused_delta_class()
    installed = 0
    for layer in model.layers:
        linear = getattr(layer, "linear_attn", None)
        if linear is not None:
            linear.__class__ = patched
            installed += 1
    if installed == 0:
        raise RuntimeError("no linear-attention layers found; this hypothesis installed nothing")
    model._deltaforge_fused_delta = True


def is_installed(model) -> bool:
    return bool(getattr(model, "_deltaforge_fused_delta", False))


def correctness_checks(model, *, device="cuda", dtype=None, seed: int = 0):
    from ..harness.correctness import check_kernel
    from ..reference import recurrent_gated_delta_rule

    config = model.config
    target_dtype = dtype if dtype is not None else next(model.parameters()).dtype
    generator = torch.Generator(device=device).manual_seed(seed)

    heads = config.linear_num_value_heads
    key_dim = config.linear_key_head_dim
    value_dim = config.linear_value_head_dim

    checks = []
    for batch, seq_len, label in (
        (1, 1, "headline decode step: batch 1, one token"),
        (32, 1, "secondary decode step: batch 32, one token"),
        # Several tokens with a carried state: this is where an off-by-one in the
        # recurrence shows up, and a single-token check cannot see it.
        (1, 8, "short prefill: the state must carry across steps"),
    ):
        query = torch.randn(
            (batch, seq_len, heads, key_dim), device=device, dtype=target_dtype, generator=generator
        )
        key = torch.randn(
            (batch, seq_len, heads, key_dim), device=device, dtype=target_dtype, generator=generator
        )
        value = torch.randn(
            (batch, seq_len, heads, value_dim), device=device, dtype=target_dtype, generator=generator
        )
        beta = torch.rand((batch, seq_len, heads), device=device, dtype=target_dtype, generator=generator)
        # g is a log-space decay: strictly non-positive, or the state grows without bound.
        g = -torch.rand((batch, seq_len, heads), device=device, dtype=torch.float32, generator=generator)
        initial = torch.randn(
            (batch, heads, key_dim, value_dim),
            device=device,
            dtype=torch.float32,
            generator=generator,
        )
        checks.append(
            check_kernel(
                "gated_delta_step.delta_rule_step",
                lambda q, k, v, gg, bb, s: recurrent_gated_delta_rule(q, k, v, gg, bb, s),
                lambda q, k, v, gg, bb, s: fused_gated_delta_rule(q, k, v, gg, bb, s),
                args=(query, key, value, g, beta, initial),
                replaces="gated_delta_rule",
                note=label,
            )
        )
    return tuple(checks)
