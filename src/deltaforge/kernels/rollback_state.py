"""Per-step recurrent state, so a rejected draft costs a copy instead of a weight stream.

24 of 32 layers carry a `(batch, 32, 128, 128)` fp32 state updated in place, once per
token. A KV cache rewinds by moving `seq_len`; this does not rewind at all — nothing in it
is indexed by position. So a verify pass over `k+1` tokens that keeps only `j` of them has
to put the state back to where it was `k+1-j` tokens ago.

Spec §3 costs the three ways of doing that. Re-running the accepted prefix through the
model is a second weight stream, ~7 ms, and ends the hypothesis. This module takes the
cheap one: record the state after every step of the verify, then keep the one the loop
asks for. `(k+1) x 50.33 MB` of writes at the model's real size, ~0.2 ms at k=4 — the
2.9% already inside the spec's `gamma`.

The conv window is free: `_causal_conv` has already concatenated history and input into
one tensor, so the window as of step `j` is a slice of a tensor the layer is holding.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor

from ..reference import STATE_DTYPE, GatedDeltaNet, recurrent_gated_delta_rule

#: Longest forward this layer will record per-step versions for.
#:
#: `install_speculative_loop` lowers this to the block size it was built with. The default
#: covers any block this project can register (`SPECULATIVE_CACHE_HEADROOM` is 16) so that
#: the kernel is safe on its own, and sits far below any prefill.
DEFAULT_MAX_RECORDED_STEPS = 17


class RollbackState:
    """The per-step versions one patched layer recorded on its last forward."""

    def __init__(self, max_recorded_steps: int = DEFAULT_MAX_RECORDED_STEPS) -> None:
        self.cache = None
        self.states: list[Tensor] = []
        self.conv_windows: list[Tensor] = []
        self.max_recorded_steps = max_recorded_steps

    def records(self, seq_len: int) -> bool:
        """Whether a forward of `seq_len` tokens is one the loop may later roll back.

        Only a verify is, and a verify is `block_size + 1` tokens. Everything longer is a
        prefill, which the loop never rewinds into: it commits the prefill's token before
        the first cycle and rewinds only to positions inside a verify.

        Bounding this by *length* rather than by a flag the loop sets is deliberate. The
        candidate column is compiled, and a Python flag that switches code paths is a new
        axis for dynamo to specialise on -- the kind of thing that voided six of nine slots
        in batch 008 through `recompile_limit`. Shape is an axis it already specialises on,
        so this branch costs no cache entry that the two call shapes did not already.
        """
        return seq_len <= self.max_recorded_steps

    def record(self, cache, states: list[Tensor], conv_windows: list[Tensor]) -> None:
        self.cache = cache
        self.states = states
        self.conv_windows = conv_windows

    def release(self) -> None:
        """Drop the recorded versions, keeping `cache`.

        Called by the loop once a cycle is committed, and by a non-recording forward.
        Deliberately *not* called by `keep`: `keep` copying a version and the record's
        lifetime ending are two different events, and `keep` is what the property tests
        inspect the record through.

        Without it the last verify's `(k+1)` states and windows stay referenced for the
        rest of the candidate's life. Rental 53 showed 20.77 GiB still allocated *after* a
        failed slot released, which pushed every later slot's ceiling down.
        """
        self.states = []
        self.conv_windows = []

    def keep(self, step: int) -> None:
        """Commit the state as of ``step`` tokens of the last forward being accepted.

        ``step`` counts tokens of that forward, so 0 means "none of it happened" and
        ``len(states) - 1`` (the number of tokens in that forward) means "all of it did".
        Copied in place, because the address of a cache tensor is a promise the compiled
        graph was given.
        """
        if self.cache is None:
            raise RuntimeError("keep() before any forward recorded a state")
        if not 0 <= step < len(self.states):
            raise ValueError(f"step {step} outside the valid range 0..{len(self.states) - 1}")
        _state = self.states[step]
        _window = self.conv_windows[step]
        self.cache.recurrent.copy_(_state)
        # `forward` only writes `cache.conv` when `history` is nonzero; this copy is
        # unconditional. Harmless: at `history == 0`, `cache.conv` and every recorded
        # window are both zero-width (sized from `max(kernel_size - 1, 0)`), so the copy
        # is a no-op there rather than a divergence from `forward`'s guard.
        self.cache.conv.copy_(_window)


def _patched_delta_net_class():
    class RollbackGatedDeltaNet(GatedDeltaNet):
        """`GatedDeltaNet` that keeps one state and one conv window per input token."""

        def forward(self, hidden_states: Tensor, cache=None) -> Tensor:
            if cache is None:
                return super().forward(hidden_states, cache)
            config = self.config
            batch, seq_len, _ = hidden_states.shape

            # A forward too long to be a verify gets the parent's implementation, and the
            # reason is not only memory. The loop below is one `recurrent_gated_delta_rule`
            # call per token: at a verify's five tokens that is the point of this class, and
            # on the benchmark's 2048-token prefill it is 2048 sequential launches per layer
            # plus 2049 cloned states, which is how rental 53 lost three slots to an OOM at
            # 23.03 GiB -- the same figure at k=4 and at k=2, because none of it scales with
            # k. `002-compile-cost` closed an unrolled prefill scan once already (6980 s
            # unfinished, then 268 s); this is that defect wearing a different kernel's
            # name, and the bound is what stops it coming back a third time.
            if not self._deltaforge_rollback.records(seq_len):
                self._deltaforge_rollback.release()
                return super().forward(hidden_states, cache)

            history = self.conv1d.kernel_size[0] - 1
            qkv = self.in_proj_qkv(hidden_states).transpose(1, 2)
            padded = torch.cat((cache.conv.to(qkv.dtype), qkv), dim=-1) if history else qkv
            conv_out = F.silu(F.conv1d(padded, self.conv1d.weight, groups=self.conv1d.groups))
            # The window after t of these tokens is a slice of what we already built.
            conv_windows = [
                padded[..., t : t + history].to(cache.conv.dtype).clone() for t in range(seq_len + 1)
            ]
            qkv = conv_out.transpose(1, 2)

            query, key, value = torch.split(
                qkv,
                [config.linear_key_dim, config.linear_key_dim, config.linear_value_dim],
                dim=-1,
            )
            query = query.reshape(batch, seq_len, config.linear_num_key_heads, config.linear_key_head_dim)
            key = key.reshape(batch, seq_len, config.linear_num_key_heads, config.linear_key_head_dim)
            value = value.reshape(batch, seq_len, config.linear_num_value_heads, config.linear_value_head_dim)
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

            # One scan call per token, so every intermediate state is a value we hold.
            # `recurrent_gated_delta_rule` returns only its final state, which is the whole
            # reason this class exists.
            state = cache.recurrent
            states = [state.clone()]
            outputs = []
            for t in range(seq_len):
                step_out, state = recurrent_gated_delta_rule(
                    query[:, t : t + 1],
                    key[:, t : t + 1],
                    value[:, t : t + 1],
                    g=g[:, t : t + 1],
                    beta=beta[:, t : t + 1],
                    initial_state=state,
                )
                outputs.append(step_out)
                states.append(state.clone())
            core_out = torch.cat(outputs, dim=1)

            cache.recurrent.copy_(state.to(STATE_DTYPE))
            if history:
                cache.conv.copy_(conv_windows[seq_len])
            self._deltaforge_rollback.record(cache, states, conv_windows)

            core_out = self.norm(
                core_out.reshape(-1, config.linear_value_head_dim),
                z.reshape(-1, config.linear_value_head_dim),
            ).reshape(batch, seq_len, config.linear_value_dim)
            return self.out_proj(core_out)

    return RollbackGatedDeltaNet


def _delta_nets(model):
    return [module for module in model.modules() if isinstance(module, GatedDeltaNet)]


def install_rollback_state(model, entry=None) -> None:
    if getattr(model, "_deltaforge_rollback_state", False):
        return
    model._deltaforge_rollback_state = True
    patched = _patched_delta_net_class()
    for module in _delta_nets(model):
        module.__class__ = patched
        module._deltaforge_rollback = RollbackState()


def rollback_states(model) -> tuple[RollbackState, ...]:
    return tuple(module._deltaforge_rollback for module in _delta_nets(model))
