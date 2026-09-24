"""The baseline: Qwen3.5-4B text decode in plain PyTorch.

This file *is* the definition of the baseline. Changes to it invalidate every stored
result, so it changes rarely and deliberately.

**Invariant: no custom kernels, ever.** No Triton, no FlashAttention, no
`flash-linear-attention`, and no HuggingFace modeling code in the forward path.
``reference_no_custom_kernels_test.py`` enforces this by inspecting the module.

Where the line is drawn
-----------------------
"Plain PyTorch" needs a precise meaning, because ``F.linear`` also lands in a
hand-written cuBLAS kernel. The rule used here:

* **Allowed** — primitives that are the compiler's own building blocks and that
  ``torch.compile`` can see through or dispatch normally: ``F.linear``, ``F.conv1d``,
  ``matmul``, ``softmax``, elementwise math. Beating cuBLAS on dense GEMM is an explicit
  non-goal, so using it in the baseline costs us nothing we were going to claim.
* **Excluded** — any op that is *itself the fused algorithm we intend to hand-write*.
  That means `F.scaled_dot_product_attention` (it dispatches to FlashAttention — using
  it would mean benchmarking hand-written Triton against hand-written CUDA while claiming
  to beat a compiler) and anything from `flash-linear-attention` (its chunked delta-rule
  scan is hypothesis 4). Attention here is therefore written out as matmul + softmax.

The delta-rule recurrence below is written in a plain, obviously-correct **sequential**
form. Restructuring it into a chunked/parallel form is hypothesis 4 — it is the single
largest expected win in the project. Putting that restructuring in the baseline would
destroy the very thing the project exists to measure. Do not "optimise" it here.

Architecture facts this file honours, all verified against the published checkpoint
(see ``docs/ARCHITECTURE.md``):

* 32 layers as 8 repetitions of ``3 x linear_attention`` then ``1 x full_attention``.
* Full attention: 16 query heads, 4 KV heads, ``head_dim`` 256 — which is *not*
  ``hidden_size / num_heads``, so the projections are not square.
* The attention output is gated (``attn_output_gate``); ``q_proj`` emits
  ``num_heads * head_dim * 2`` and is split into query and gate.
* Partial RoPE: only the first 64 of each 256-wide head is rotated.
* mRoPE with interleaved sections; for text-only decode all three sections carry the
  same position index, which reduces to standard RoPE (proved in ``reference_test.py``).
* RMSNorm applies ``(1 + weight)``, not ``weight`` — the checkpoint stores zero-centred
  norm weights. The gated RMSNorm inside Gated DeltaNet uses plain ``weight`` instead,
  and normalises *before* gating.
* The Gated DeltaNet recurrent state is fp32 even though the weights are bf16.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from .config import FULL_ATTENTION, LINEAR_ATTENTION, SIGMOID_GATE, SWISH_GATE, ModelConfig

__all__ = [
    "CacheSnapshot",
    "DecodeCache",
    "GatedAttention",
    "GatedDeltaNet",
    "ReferenceModel",
    "RMSNorm",
    "SwiGLUMLP",
    "apply_output_gate",
    "recurrent_gated_delta_rule",
]

# The recurrent state dtype is fixed by the checkpoint's ``mamba_ssm_dtype``.
STATE_DTYPE = torch.float32


# --------------------------------------------------------------------------------------
# Normalisation
# --------------------------------------------------------------------------------------


class RMSNorm(nn.Module):
    """RMSNorm with a zero-centred weight: the effective scale is ``1 + weight``.

    Qwen3.5 stores norm weights centred on zero, so a freshly initialised module with
    ``weight = 0`` is the identity scale. Reading this as plain ``weight`` produces an
    all-zero activation and is the most common way to get this model silently wrong.
    """

    def __init__(self, dim: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.zeros(dim))

    def forward(self, x: Tensor) -> Tensor:
        out = x.float()
        out = out * torch.rsqrt(out.pow(2).mean(-1, keepdim=True) + self.eps)
        out = out * (1.0 + self.weight.float())
        return out.type_as(x)

    def extra_repr(self) -> str:
        return f"{tuple(self.weight.shape)}, eps={self.eps}"


class GatedRMSNorm(nn.Module):
    """RMSNorm followed by a SiLU gate, as used inside Gated DeltaNet.

    Two things differ from :class:`RMSNorm` and both matter:
    the scale is plain ``weight`` (not ``1 + weight``), and the normalisation happens
    *before* the gate is applied rather than after.
    """

    def __init__(self, dim: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim, dtype=STATE_DTYPE))

    def forward(self, x: Tensor, gate: Tensor) -> Tensor:
        input_dtype = x.dtype
        out = x.float()
        out = out * torch.rsqrt(out.pow(2).mean(-1, keepdim=True) + self.eps)
        out = self.weight * out.to(input_dtype)
        out = out * F.silu(gate.float())
        return out.to(input_dtype)

    def extra_repr(self) -> str:
        return f"{tuple(self.weight.shape)}, eps={self.eps}"


# --------------------------------------------------------------------------------------
# Rotary embeddings (partial, mRoPE)
# --------------------------------------------------------------------------------------


def rotate_half(x: Tensor) -> Tensor:
    half = x.shape[-1] // 2
    return torch.cat((-x[..., half:], x[..., :half]), dim=-1)


def apply_partial_rope(x: Tensor, cos: Tensor, sin: Tensor) -> Tensor:
    """Rotate the leading ``cos.shape[-1]`` dims of ``x`` and pass the rest through.

    ``x`` is ``(batch, heads, seq, head_dim)``; ``cos``/``sin`` are ``(batch, seq, rotary_dim)``.
    With ``partial_rotary_factor`` 0.25 on this model, ``rotary_dim`` is 64 of 256 and the
    trailing 192 dims are untouched.
    """
    cos = cos.unsqueeze(1)
    sin = sin.unsqueeze(1)
    rotary_dim = cos.shape[-1]
    rot, passthrough = x[..., :rotary_dim], x[..., rotary_dim:]
    rotated = (rot * cos) + (rotate_half(rot) * sin)
    return torch.cat((rotated, passthrough), dim=-1)


class RotaryEmbedding(nn.Module):
    """Interleaved mRoPE.

    The model is multimodal, so positions are 3-dimensional ``(t, h, w)`` and the
    frequency bands are interleaved between the three sections rather than laid out in
    contiguous chunks.

    For **text-only decode all three rows carry the same position index**, so whichever
    row each band is drawn from, it sees the same value — the whole construction collapses
    to standard RoPE at the token position. That reduction is asserted as an exact
    identity in ``reference_test.py`` rather than assumed.
    """

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.rotary_dim = config.rotary_dim
        self.mrope_section = tuple(config.mrope_section)
        self.mrope_interleaved = config.mrope_interleaved
        inv_freq = 1.0 / (
            config.rope_theta ** (torch.arange(0, self.rotary_dim, 2, dtype=torch.float32) / self.rotary_dim)
        )
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def _interleave(self, freqs: Tensor) -> Tensor:
        """``(3, batch, seq, rotary_dim//2)`` -> ``(batch, seq, rotary_dim//2)``.

        Band ``i`` is taken from section ``i % 3`` for the first ``3 * section[k]`` bands
        of each section, which is what "interleaved" means here: the layout is
        ``T H W T H W ... T T`` rather than ``TTT... HHH... WWW...``.
        """
        out = freqs[0].clone()
        for section, offset in enumerate((1, 2), start=1):
            length = self.mrope_section[section] * 3
            idx = slice(offset, length, 3)
            out[..., idx] = freqs[section][..., idx]
        return out

    def forward(self, position_ids: Tensor, dtype: torch.dtype) -> tuple[Tensor, Tensor]:
        """``position_ids`` is ``(batch, seq)`` or ``(3, batch, seq)``."""
        if position_ids.ndim == 2:
            position_ids = position_ids.unsqueeze(0).expand(3, -1, -1)
        if position_ids.shape[0] != 3:
            raise ValueError(f"expected 3 mRoPE sections, got {position_ids.shape[0]}")

        # (3, batch, seq, rotary_dim // 2), always computed in fp32.
        freqs = position_ids.float().unsqueeze(-1) * self.inv_freq.to(position_ids.device)
        freqs = self._interleave(freqs) if self.mrope_interleaved else freqs[0]
        emb = torch.cat((freqs, freqs), dim=-1)
        return emb.cos().to(dtype), emb.sin().to(dtype)


# --------------------------------------------------------------------------------------
# Cache
# --------------------------------------------------------------------------------------


@dataclass
class _AttentionLayerCache:
    keys: Tensor  # (batch, num_kv_heads, max_seq_len, head_dim)
    values: Tensor


@dataclass
class _LinearLayerCache:
    conv: Tensor  # (batch, conv_dim, kernel_size - 1), the causal conv's history
    recurrent: Tensor  # (batch, num_v_heads, key_head_dim, value_head_dim), always fp32


@dataclass(frozen=True)
class CacheSnapshot:
    """A copy of a cache's whole state, and the position it was taken at.

    The benchmark is what wants this. Every scoring round has to start from the same
    post-prefill state, and the only way it had to get there was to re-run the prefill:
    `cache.reset()` and a 2048-token forward pass, once per column, once per round. That
    work is excluded from the timed region — so it never moved a ratio — and rental 46
    still paid ~18 s a round for it against a timed region of ~2 s, which is ~180 s per
    slot at 15 rounds. A snapshot restores the same state with a device-to-device copy of
    ~120 MB.

    Held as a flat tuple in the cache's own traversal order rather than as a structure:
    the only operations are "copy all of it out" and "copy all of it back in", and a flat
    tuple makes the shape check that guards the second one trivial.
    """

    seq_len: int
    tensors: tuple[Tensor, ...]


class DecodeCache:
    """Per-layer decode state: KV for attention layers, conv + recurrent for linear ones.

    The KV cache is **preallocated** to ``max_seq_len`` and filled by slice assignment
    rather than grown with ``torch.cat``. That is not an optimisation smuggled into the
    baseline — repeatedly reallocating and copying a growing cache is quadratic, and it
    would make the baseline artificially slow, which would inflate every ratio we report.
    An honest baseline is the whole point of this file.

    ``seq_len`` counts tokens already committed to the cache, so it is both the write
    offset and the position index of the next token.
    """

    def __init__(
        self,
        config: ModelConfig,
        batch_size: int,
        max_seq_len: int,
        device: torch.device | str = "cpu",
        dtype: torch.dtype = torch.float32,
    ) -> None:
        self.config = config
        self.batch_size = batch_size
        self.max_seq_len = max_seq_len
        self.device = torch.device(device)
        self.dtype = dtype
        self.seq_len = 0

        self.layers: list[_AttentionLayerCache | _LinearLayerCache] = []
        for layer_type in config.layer_types:
            if layer_type == FULL_ATTENTION:
                shape = (batch_size, config.num_key_value_heads, max_seq_len, config.head_dim)
                self.layers.append(
                    _AttentionLayerCache(
                        keys=torch.zeros(shape, device=device, dtype=dtype),
                        values=torch.zeros(shape, device=device, dtype=dtype),
                    )
                )
            else:
                conv_history = max(config.linear_conv_kernel_dim - 1, 0)
                self.layers.append(
                    _LinearLayerCache(
                        conv=torch.zeros(
                            (batch_size, config.linear_conv_dim, conv_history),
                            device=device,
                            dtype=dtype,
                        ),
                        recurrent=torch.zeros(
                            (
                                batch_size,
                                config.linear_num_value_heads,
                                config.linear_key_head_dim,
                                config.linear_value_head_dim,
                            ),
                            device=device,
                            dtype=STATE_DTYPE,
                        ),
                    )
                )

    def reset(self) -> None:
        """Zero every layer's state and rewind to position 0, keeping the allocation."""
        self.seq_len = 0
        for layer in self.layers:
            if isinstance(layer, _AttentionLayerCache):
                layer.keys.zero_()
                layer.values.zero_()
            else:
                layer.conv.zero_()
                layer.recurrent.zero_()

    def state_tensors(self) -> tuple[Tensor, ...]:
        """Every tensor holding decode state, in a fixed order. The snapshot's contract."""
        tensors: list[Tensor] = []
        for layer in self.layers:
            if isinstance(layer, _AttentionLayerCache):
                tensors.extend((layer.keys, layer.values))
            else:
                tensors.extend((layer.conv, layer.recurrent))
        return tuple(tensors)

    def snapshot(self) -> CacheSnapshot:
        """Copy the whole state out, so a round can be restarted without a prefill."""
        return CacheSnapshot(
            seq_len=self.seq_len,
            tensors=tuple(tensor.clone() for tensor in self.state_tensors()),
        )

    def restore(self, snapshot: CacheSnapshot) -> None:
        """Copy a snapshot back **in place**, keeping every tensor's storage.

        In place is the load-bearing word. `034-static-cache-cudagraphs` won 2.0% on the
        promise that these tensors never move — `mark_static_address` puts them in
        inductor's `static_input_idxs`, which is what skips the per-call alignment check —
        and rebinding them to fresh storage between rounds would void that silently.
        """
        live = self.state_tensors()
        if len(live) != len(snapshot.tensors) or any(
            a.shape != b.shape or a.dtype != b.dtype for a, b in zip(live, snapshot.tensors)
        ):
            raise ValueError(
                "snapshot came from a different cache: it holds "
                f"{[tuple(t.shape) for t in snapshot.tensors][:3]}... against this cache's "
                f"{[tuple(t.shape) for t in live][:3]}..."
            )
        for tensor, saved in zip(live, snapshot.tensors):
            tensor.copy_(saved)
        self.seq_len = snapshot.seq_len

    def attention(self, layer_idx: int) -> _AttentionLayerCache:
        layer = self.layers[layer_idx]
        if not isinstance(layer, _AttentionLayerCache):
            raise TypeError(f"layer {layer_idx} is not a full-attention layer")
        return layer

    def linear(self, layer_idx: int) -> _LinearLayerCache:
        layer = self.layers[layer_idx]
        if not isinstance(layer, _LinearLayerCache):
            raise TypeError(f"layer {layer_idx} is not a linear-attention layer")
        return layer

    def ensure_capacity(self, num_tokens: int) -> None:
        """Check that ``num_tokens`` more positions fit, *before* anything is written.

        Called at the top of the forward pass rather than at the end: overflowing part
        way through a layer stack would surface as a confusing tensor-shape error from
        deep inside attention, with the cache already half-written.
        """
        if self.seq_len + num_tokens > self.max_seq_len:
            raise ValueError(
                f"cache overflow: {self.seq_len} + {num_tokens} > max_seq_len={self.max_seq_len}"
            )

    def advance(self, num_tokens: int) -> None:
        """Commit ``num_tokens`` positions. Called once per forward, by the model."""
        self.ensure_capacity(num_tokens)
        self.seq_len += num_tokens


# --------------------------------------------------------------------------------------
# Gated DeltaNet (linear attention)
# --------------------------------------------------------------------------------------


def l2norm(x: Tensor, eps: float = 1e-6) -> Tensor:
    return x * torch.rsqrt((x * x).sum(dim=-1, keepdim=True) + eps)


def recurrent_gated_delta_rule(
    query: Tensor,
    key: Tensor,
    value: Tensor,
    g: Tensor,
    beta: Tensor,
    initial_state: Tensor | None = None,
) -> tuple[Tensor, Tensor]:
    """The delta rule, one token at a time.

    Shapes: ``query``/``key`` are ``(batch, seq, num_v_heads, key_head_dim)`` — already
    expanded from key heads to value heads by the caller — ``value`` is
    ``(batch, seq, num_v_heads, value_head_dim)``, and ``g``/``beta`` are
    ``(batch, seq, num_v_heads)``. ``g`` is a log-space decay, so entries are <= 0.

    Returns ``(output, final_state)`` where the state is
    ``(batch, num_v_heads, key_head_dim, value_head_dim)`` in fp32.

    This is deliberately the naive sequential form. It is the reference the chunked
    parallel version (hypothesis 4) must match and beat; a chunked implementation here
    would erase the win the project exists to measure. See the module docstring.
    """
    initial_dtype = query.dtype

    # (batch, heads, seq, dim), fp32 throughout: the recurrent state is fp32 by config.
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

    out = torch.zeros_like(value)
    for t in range(seq_len):
        q_t, k_t, v_t = query[:, :, t], key[:, :, t], value[:, :, t]
        # Decay the state, then correct it towards the new value by beta ("how much of
        # this token overwrites what is already stored").
        state = state * g[:, :, t].exp()[..., None, None]
        recalled = (state * k_t.unsqueeze(-1)).sum(dim=-2)
        delta = (v_t - recalled) * beta[:, :, t].unsqueeze(-1)
        state = state + k_t.unsqueeze(-1) * delta.unsqueeze(-2)
        out[:, :, t] = (state * q_t.unsqueeze(-1)).sum(dim=-2)

    out = out.transpose(1, 2).contiguous().to(initial_dtype)
    return out, state


class GatedDeltaNet(nn.Module):
    """Linear-attention layer: causal depthwise conv, then the gated delta rule.

    Note the projection layout, which differs from Qwen3-Next: this checkpoint keeps
    ``in_proj_qkv`` fused but ``z``, ``b`` and ``a`` as three separate projections, so
    there is no per-key-head interleaving to undo.
    """

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.config = config
        self.conv_kernel_size = config.linear_conv_kernel_dim

        self.in_proj_qkv = nn.Linear(config.hidden_size, config.linear_conv_dim, bias=False)
        self.in_proj_z = nn.Linear(config.hidden_size, config.linear_value_dim, bias=False)
        self.in_proj_b = nn.Linear(config.hidden_size, config.linear_num_value_heads, bias=False)
        self.in_proj_a = nn.Linear(config.hidden_size, config.linear_num_value_heads, bias=False)

        self.conv1d = nn.Conv1d(
            in_channels=config.linear_conv_dim,
            out_channels=config.linear_conv_dim,
            kernel_size=self.conv_kernel_size,
            groups=config.linear_conv_dim,
            bias=False,
        )

        self.dt_bias = nn.Parameter(torch.zeros(config.linear_num_value_heads))
        self.A_log = nn.Parameter(torch.zeros(config.linear_num_value_heads, dtype=STATE_DTYPE))

        self.norm = GatedRMSNorm(config.linear_value_head_dim, eps=config.rms_norm_eps)
        self.out_proj = nn.Linear(config.linear_value_dim, config.hidden_size, bias=False)

    def _causal_conv(self, x: Tensor, cache: _LinearLayerCache | None) -> Tensor:
        """Depthwise causal conv over ``(batch, conv_dim, seq)``.

        History from the cache is prepended, so prefill and single-token decode take the
        same path: zeros for the first call are exactly the left padding a causal conv
        would apply anyway.
        """
        history = self.conv1d.kernel_size[0] - 1
        if history:
            prefix = cache.conv if cache is not None else x.new_zeros((x.shape[0], x.shape[1], history))
            x = torch.cat((prefix.to(x.dtype), x), dim=-1)
        out = F.conv1d(x, self.conv1d.weight, groups=self.conv1d.groups)
        if cache is not None and history:
            cache.conv.copy_(x[..., -history:].to(cache.conv.dtype))
        return F.silu(out)

    def forward(self, hidden_states: Tensor, cache: _LinearLayerCache | None = None) -> Tensor:
        config = self.config
        batch, seq_len, _ = hidden_states.shape

        qkv = self.in_proj_qkv(hidden_states).transpose(1, 2)
        qkv = self._causal_conv(qkv, cache).transpose(1, 2)
        query, key, value = torch.split(
            qkv, [config.linear_key_dim, config.linear_key_dim, config.linear_value_dim], dim=-1
        )
        query = query.reshape(batch, seq_len, config.linear_num_key_heads, config.linear_key_head_dim)
        key = key.reshape(batch, seq_len, config.linear_num_key_heads, config.linear_key_head_dim)
        value = value.reshape(batch, seq_len, config.linear_num_value_heads, config.linear_value_head_dim)

        z = self.in_proj_z(hidden_states).reshape(
            batch, seq_len, config.linear_num_value_heads, config.linear_value_head_dim
        )
        beta = self.in_proj_b(hidden_states).sigmoid()
        # ``.float()`` on A_log matters: in fp16 the exponential can reach -inf.
        g = -self.A_log.float().exp() * F.softplus(
            self.in_proj_a(hidden_states).float() + self.dt_bias.float()
        )

        groups = config.linear_value_groups
        if groups > 1:
            query = query.repeat_interleave(groups, dim=2)
            key = key.repeat_interleave(groups, dim=2)

        initial_state = cache.recurrent if cache is not None else None
        core_out, final_state = recurrent_gated_delta_rule(
            query, key, value, g=g, beta=beta, initial_state=initial_state
        )
        if cache is not None:
            cache.recurrent.copy_(final_state)

        core_out = self.norm(
            core_out.reshape(-1, config.linear_value_head_dim),
            z.reshape(-1, config.linear_value_head_dim),
        ).reshape(batch, seq_len, config.linear_value_dim)
        return self.out_proj(core_out)


# --------------------------------------------------------------------------------------
# Gated attention (full attention)
# --------------------------------------------------------------------------------------


def apply_output_gate(gate: Tensor, gate_type: str) -> Tensor:
    """The attention output gate's nonlinearity, selected by config.

    ``sigmoid`` for Qwen3.5, ``swish`` (SiLU) for Qwen3.8. Computed in fp32 and cast
    back, matching how the checkpoints were trained.
    """
    if gate_type == SIGMOID_GATE:
        return torch.sigmoid(gate.float()).to(gate.dtype)
    if gate_type == SWISH_GATE:
        return F.silu(gate.float()).to(gate.dtype)
    raise ValueError(f"unknown output_gate_type {gate_type!r}")


class GatedAttention(nn.Module):
    """GQA softmax attention with a gated output and partial RoPE.

    ``head_dim`` is 256 while ``hidden_size / num_heads`` is 160 (Qwen3.5-4B) or 213
    (Qwen3.8-27B), so none of these projections are square. ``q_proj`` emits twice the
    query width: the second half is the output gate.

    The gate's nonlinearity is **config-driven**, not fixed. Qwen3.5 omits
    ``output_gate_type`` and means sigmoid; Qwen3.8 declares ``swish``. Hardcoding
    either one gives a model that runs and emits plausible logits on the other
    checkpoint, which is the failure mode this project is least able to detect.
    """

    def __init__(self, config: ModelConfig, layer_idx: int) -> None:
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.head_dim = config.head_dim
        self.scaling = config.head_dim**-0.5

        query_multiplier = 2 if config.attn_output_gate else 1
        self.q_proj = nn.Linear(
            config.hidden_size,
            config.num_attention_heads * config.head_dim * query_multiplier,
            bias=config.attention_bias,
        )
        self.k_proj = nn.Linear(
            config.hidden_size, config.num_key_value_heads * config.head_dim, bias=config.attention_bias
        )
        self.v_proj = nn.Linear(
            config.hidden_size, config.num_key_value_heads * config.head_dim, bias=config.attention_bias
        )
        self.o_proj = nn.Linear(
            config.num_attention_heads * config.head_dim, config.hidden_size, bias=config.attention_bias
        )
        # Both norms are over the head dimension only, applied before RoPE.
        self.q_norm = RMSNorm(config.head_dim, eps=config.rms_norm_eps)
        self.k_norm = RMSNorm(config.head_dim, eps=config.rms_norm_eps)

    def forward(
        self,
        hidden_states: Tensor,
        cos: Tensor,
        sin: Tensor,
        cache: _AttentionLayerCache | None = None,
        cache_offset: int = 0,
    ) -> Tensor:
        config = self.config
        batch, seq_len, _ = hidden_states.shape

        projected = self.q_proj(hidden_states)
        if config.attn_output_gate:
            # (batch, seq, heads, 2 * head_dim) -> query, gate. The split is per head,
            # not a split of the whole flat projection.
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

        # Expand KV heads to query heads. `expand` would alias into matmul; a real
        # broadcast copy keeps this straightforwardly correct, and KV-cache layout is
        # itself hypothesis 6.
        groups = config.num_key_value_groups
        if groups > 1:
            key = key.repeat_interleave(groups, dim=1)
            value = value.repeat_interleave(groups, dim=1)

        # Written out rather than calling F.scaled_dot_product_attention: SDPA *is* the
        # fused attention kernel under test (hypothesis 5). See the module docstring.
        scores = torch.matmul(query, key.transpose(2, 3)) * self.scaling
        key_len = key.shape[2]
        if seq_len > 1:
            # Causal mask. Query position i (absolute: cache_offset + i) may attend to
            # every key position <= it.
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


# --------------------------------------------------------------------------------------
# FFN
# --------------------------------------------------------------------------------------


class SwiGLUMLP(nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.gate_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.up_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.down_proj = nn.Linear(config.intermediate_size, config.hidden_size, bias=False)

    def forward(self, x: Tensor) -> Tensor:
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


# --------------------------------------------------------------------------------------
# Decoder layer and model
# --------------------------------------------------------------------------------------


class DecoderLayer(nn.Module):
    def __init__(self, config: ModelConfig, layer_idx: int) -> None:
        super().__init__()
        self.layer_idx = layer_idx
        self.layer_type = config.layer_types[layer_idx]
        if self.layer_type == LINEAR_ATTENTION:
            self.linear_attn = GatedDeltaNet(config)
        else:
            self.self_attn = GatedAttention(config, layer_idx)
        self.mlp = SwiGLUMLP(config)
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        hidden_states: Tensor,
        cos: Tensor,
        sin: Tensor,
        cache: DecodeCache | None = None,
        cache_offset: int = 0,
    ) -> Tensor:
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        if self.layer_type == LINEAR_ATTENTION:
            layer_cache = cache.linear(self.layer_idx) if cache is not None else None
            hidden_states = self.linear_attn(hidden_states, layer_cache)
        else:
            layer_cache = cache.attention(self.layer_idx) if cache is not None else None
            hidden_states = self.self_attn(hidden_states, cos, sin, layer_cache, cache_offset)
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        return residual + hidden_states


class ReferenceModel(nn.Module):
    """``ReferenceModel(config).forward(input_ids, cache) -> (logits, cache)``.

    The LM head is tied to the embedding matrix, so there is no separate weight.
    """

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.config = config
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList(DecoderLayer(config, i) for i in range(config.num_hidden_layers))
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.rotary_emb = RotaryEmbedding(config)
        if not config.tie_word_embeddings:
            self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

    @property
    def lm_head_weight(self) -> Tensor:
        if self.config.tie_word_embeddings:
            return self.embed_tokens.weight
        return self.lm_head.weight

    def new_cache(self, batch_size: int, max_seq_len: int) -> DecodeCache:
        param = next(self.parameters())
        return DecodeCache(self.config, batch_size, max_seq_len, device=param.device, dtype=param.dtype)

    def project_logits(self, hidden_states: Tensor) -> Tensor:
        """The LM head projection, as a callable boundary a kernel can replace.

        The expression is the one `forward` used inline, moved verbatim and nothing else.
        It is here because the head is **15.1% of per-token weight bytes** and, with
        ``tie_word_embeddings``, has no `nn.Linear` for a kernel to swap — every other
        replaceable operation in this model is a module, and this one was a statement.

        Extracting it is not the accommodation `AGENT.md` forbids: no kernel enters this
        file, no arithmetic changes, and `reference_purity_test.py` still holds. What would
        invalidate stored results is a change to what the baseline *computes*, and this
        changes only where the same computation is written down.
        """
        return F.linear(hidden_states, self.lm_head_weight.to(hidden_states.dtype))

    def forward(
        self,
        input_ids: Tensor,
        cache: DecodeCache | None = None,
        *,
        position_ids: Tensor | None = None,
        num_logits_to_keep: int = 0,
    ) -> tuple[Tensor, DecodeCache | None]:
        """Run the decode forward pass.

        ``num_logits_to_keep`` of 0 returns logits for every position; greedy decode
        passes 1, because materialising a 248320-wide vocabulary over a 2048-token
        prefill costs about a gigabyte and none of it is used.
        """
        batch, seq_len = input_ids.shape
        cache_offset = 0
        if cache is not None:
            cache.ensure_capacity(seq_len)
            cache_offset = cache.seq_len

        if position_ids is None:
            position_ids = (
                (torch.arange(seq_len, device=input_ids.device) + cache_offset).unsqueeze(0).expand(batch, -1)
            )

        hidden_states = self.embed_tokens(input_ids)
        cos, sin = self.rotary_emb(position_ids, hidden_states.dtype)

        for layer in self.layers:
            hidden_states = layer(hidden_states, cos, sin, cache, cache_offset)

        hidden_states = self.norm(hidden_states)
        if num_logits_to_keep:
            hidden_states = hidden_states[:, -num_logits_to_keep:]
        logits = self.project_logits(hidden_states)

        if cache is not None:
            cache.advance(seq_len)
        return logits, cache
