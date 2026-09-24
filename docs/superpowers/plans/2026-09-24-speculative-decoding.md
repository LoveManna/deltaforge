# Speculative decoding implementation plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Emit more than one token per weight-stream, by drafting `k` tokens cheaply and
verifying them in one forward pass, with the recurrent state rolled back correctly when a
draft is rejected.

**Architecture:** A decode *loop* replaces the decode *step*. `greedy_decode` dispatches to
`model.decode_loop` when a kernel installed one; the loop drafts, verifies with a single
`model(draft_tokens, cache, num_logits_to_keep=k+1)` call, accepts the longest prefix the
verifier's argmax agrees with, and rewinds the cache. The 24 gated delta-rule layers are the
hard part: their state is updated in place and does not rewind, so a module swap records one
state version per verified step and the loop selects the one it kept.

**Tech Stack:** PyTorch 2.11 + inductor (`max-autotune`), no Triton anywhere in this plan.
Everything here is testable on a CPU laptop with `tiny_config()`; the GPU buys measurement,
not development.

## Global Constraints

- **`src/deltaforge/reference.py` never gains a kernel, a branch for a candidate, or an
  accommodation.** `reference_purity_test.py` enforces it. The only reference changes in this
  plan are `DecodeCache` methods, which are cache mechanics, not arithmetic.
- **Tests must pass with no CUDA present**: `uv run pytest`, `testpaths = src, remote`,
  colocated as `<module>_test.py`, named as sentences (`test_a_rejected_draft_restores_...`).
- **Lint:** `uv run ruff check . && uv run ruff format --check .`, line length 110.
- **Nothing in `batch.py` may import torch at module scope.**
- **A prediction registered in a manifest is never edited after a rental.** Batch 010 is
  registered by Task 9 and committed before any GPU runs.
- **`k` is the block size; `j` is the number of draft tokens accepted in a cycle, `0 ≤ j ≤ k`.**
  A cycle always emits `j + 1` tokens: the accepted drafts plus the verifier's own token.
- Spec: `docs/superpowers/specs/2026-09-24-speculative-decoding-design.md`. Read §3, §4 and
  §9 before Task 3.

---

## File structure

| File | Responsibility |
|---|---|
| `src/deltaforge/reference.py` | **Modify.** `DecodeCache.rewind`; nothing else. |
| `src/deltaforge/model.py` | **Modify.** `greedy_decode` dispatches to an installed loop. |
| `src/deltaforge/speculative.py` | **Create.** The loop, the drafter protocol, the two cheap drafters, and the per-cycle acceptance record. No torch-free constraint, but no Triton and no custom ops. |
| `src/deltaforge/kernels/rollback_state.py` | **Create.** The `GatedDeltaNet` swap that records one recurrent state and conv window per verified step, and restores a chosen one. |
| `src/deltaforge/harness/correctness.py` | **Modify.** `check_sequence`: free-run both models, report the first divergence and the reference's top-2 gap there. |
| `src/deltaforge/batch.py` | **Modify.** `"sequence"` policy, its threshold field, its validation. |
| `src/deltaforge/batch_run.py` | **Modify.** Run the new gate; let a loop-only candidate past the no-op guard; record acceptance in `SlotResult`. |
| `src/deltaforge/cli.py` | **Modify.** `headline_text` workload. |
| `src/deltaforge/batches.py` | **Modify.** Batch 010. |

Tasks 1–5 are the mechanism and are independent of any drafter. Task 6 is the gate. Tasks
7–9 are harness and manifest. Task 10 is the int4 self-draft and is **blocked on `061`**.

---

### Task 1: `DecodeCache.rewind`

**Files:**
- Modify: `src/deltaforge/reference.py` (the `DecodeCache` class, after `advance`)
- Test: `src/deltaforge/reference_test.py`

**Interfaces:**
- Consumes: `DecodeCache.seq_len`, `DecodeCache.advance`, `DecodeCache.snapshot`
- Produces: `DecodeCache.rewind(to_seq_len: int) -> None`

- [ ] **Step 1: Write the failing tests**

```python
def test_rewinding_the_cache_puts_the_next_write_back_where_it_was(model, config):
    """A rejected draft has to un-commit positions the verifier already wrote.

    KV positions past `seq_len` are never read — attention masks to the committed length —
    so rewinding the counter is the whole operation for the attention layers. The recurrent
    layers are not rewound by this and Task 3 is why.
    """
    ids = torch.randint(0, config.vocab_size, (1, 4))
    cache = model.new_cache(batch_size=1, max_seq_len=16)
    model(ids, cache)
    committed = cache.seq_len

    cache.advance(3)
    cache.rewind(committed)

    assert cache.seq_len == committed


def test_rewinding_forward_is_refused(model):
    cache = model.new_cache(batch_size=1, max_seq_len=16)
    cache.advance(2)

    with pytest.raises(ValueError, match="only rewind backwards"):
        cache.rewind(5)


def test_rewinding_below_zero_is_refused(model):
    cache = model.new_cache(batch_size=1, max_seq_len=16)

    with pytest.raises(ValueError, match="cannot rewind past 0"):
        cache.rewind(-1)
```

- [ ] **Step 2: Run them and watch them fail**

Run: `uv run pytest src/deltaforge/reference_test.py -k rewind -q`
Expected: FAIL, `AttributeError: 'DecodeCache' object has no attribute 'rewind'`

- [ ] **Step 3: Implement**

```python
    def rewind(self, to_seq_len: int) -> None:
        """Un-commit positions back to ``to_seq_len``. The counter, and only the counter.

        A verify pass writes ``k+1`` positions and may keep only ``j+1`` of them. For the
        full-attention layers that is all that is needed: attention reads ``[0, seq_len)``,
        so a position past the counter is never read and the next write overwrites it.

        It is deliberately *not* enough for the linear-attention layers, whose recurrent
        state was updated in place and carries no position index at all. Those are restored
        by the candidate that owns them — see `kernels/rollback_state.py`. A cache method
        that silently did half the job would be worse than one that does a named half.
        """
        if to_seq_len < 0:
            raise ValueError(f"cannot rewind past 0, got {to_seq_len}")
        if to_seq_len > self.seq_len:
            raise ValueError(
                f"can only rewind backwards: asked for {to_seq_len} from {self.seq_len}"
            )
        self.seq_len = to_seq_len
```

- [ ] **Step 4: Run them and watch them pass**

Run: `uv run pytest src/deltaforge/reference_test.py -q`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/deltaforge/reference.py src/deltaforge/reference_test.py
git commit -m "A cache that can un-commit the positions a rejected draft wrote"
```

---

### Task 2: `greedy_decode` dispatches to an installed decode loop

**Files:**
- Modify: `src/deltaforge/model.py` (`greedy_decode`)
- Test: `src/deltaforge/model_test.py`

**Interfaces:**
- Consumes: `greedy_decode(model, input_ids, max_new_tokens, cache=None)`
- Produces: the attribute contract `model.decode_loop(runnable, input_ids, max_new_tokens, cache) -> Tensor`,
  returning generated ids of shape `(batch, max_new_tokens)`

- [ ] **Step 1: Write the failing tests**

```python
def test_greedy_decode_uses_an_installed_decode_loop(model, config):
    """The bench times `greedy_decode`, so a candidate that replaces the loop must be
    reachable from there — a speculative decoder is not a module swap, and
    `apply_champions` has no other way to put one in front of the benchmark."""
    seen = {}

    def loop(runnable, input_ids, max_new_tokens, cache):
        seen["args"] = (runnable, input_ids.shape, max_new_tokens, cache)
        return torch.zeros((input_ids.shape[0], max_new_tokens), dtype=torch.long)

    model.decode_loop = loop
    ids = torch.randint(0, config.vocab_size, (1, 3))
    cache = model.new_cache(1, 16)

    out = greedy_decode(model, ids, 4, cache=cache)

    assert out.shape == (1, 4)
    assert seen["args"][0] is model
    assert seen["args"][2] == 4
    assert seen["args"][3] is cache


def test_greedy_decode_without_a_loop_is_unchanged(model, config):
    ids = torch.randint(0, config.vocab_size, (1, 3))

    out = greedy_decode(model, ids, 4)

    assert out.shape == (1, 4)


def test_a_loop_returning_the_wrong_number_of_tokens_is_refused(model, config):
    """The ratio divides by a fixed token count. A loop that emitted 130 tokens where the
    reference emitted 128 would look 1.6% faster for having done more work."""
    model.decode_loop = lambda runnable, input_ids, max_new_tokens, cache: torch.zeros(
        (1, max_new_tokens - 1), dtype=torch.long
    )
    ids = torch.randint(0, config.vocab_size, (1, 3))

    with pytest.raises(RuntimeError, match="returned 3 tokens, expected 4"):
        greedy_decode(model, ids, 4)
```

- [ ] **Step 2: Run them and watch them fail**

Run: `uv run pytest src/deltaforge/model_test.py -k decode_loop -q`
Expected: FAIL — the loop is never called, `seen` is empty.

- [ ] **Step 3: Implement**

Insert at the top of `greedy_decode`, after the `max_new_tokens` check and the cache
default:

```python
    # A candidate may replace the decode *loop* rather than a module: speculative decoding
    # changes how many tokens come out of one forward pass, which no module swap can
    # express. `torch.compile` wraps the model, and `OptimizedModule.__getattr__`
    # forwards to the original, so this reaches an installed loop through either.
    loop = getattr(model, "decode_loop", None)
    if loop is not None:
        generated = loop(model, input_ids, max_new_tokens, cache)
        if generated.shape[-1] != max_new_tokens:
            raise RuntimeError(
                f"{type(loop).__name__} returned {generated.shape[-1]} tokens, expected "
                f"{max_new_tokens}. The benchmark divides a fixed token count into the "
                "measured time, so a loop that emits a different number is not comparable."
            )
        return generated
```

- [ ] **Step 4: Run the file**

Run: `uv run pytest src/deltaforge/model_test.py -q`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/deltaforge/model.py src/deltaforge/model_test.py
git commit -m "Let a candidate replace the decode loop, not only a module"
```

---

### Task 3: per-step recurrent state, so a rejected draft can be undone

**Files:**
- Create: `src/deltaforge/kernels/rollback_state.py`
- Create: `src/deltaforge/kernels/rollback_state_test.py`
- Modify: `src/deltaforge/kernels/__init__.py` (register the kernel)
- Modify: `src/deltaforge/model.py` (register the installer)

**Interfaces:**
- Consumes: `reference.GatedDeltaNet`, `reference.recurrent_gated_delta_rule`,
  `reference._LinearLayerCache`
- Produces:
  - `install_rollback_state(model, entry=None) -> None`
  - `rollback_states(model) -> tuple[RollbackState, ...]` — one per patched layer
  - `RollbackState.keep(step: int) -> None` — commit the state as of `step` accepted tokens
  - kernel name `"rollback_state"`, replacing `"gated_delta_rule"`

**Read first:** spec §3. The table there says why re-running the accepted prefix through the
model is not an option (a second weight stream) and why per-step versions are.

- [ ] **Step 1: Write the failing test — the property that matters**

```python
"""The one correctness property: rolling back is indistinguishable from never having gone."""

from __future__ import annotations

import torch

from ..config import tiny_config
from ..kernels.rollback_state import install_rollback_state, rollback_states
from ..reference import ReferenceModel


def test_rolling_back_to_j_matches_never_having_run_past_j():
    """Run k+1 tokens, keep j, continue — against a model that only ever saw j.

    This is the whole hypothesis in one assertion. If it fails, every ratio the
    speculative slots produce is measuring a model with a corrupted recurrent state, and
    the tokens would be wrong in a way no timing would reveal.
    """
    config = tiny_config()
    torch.manual_seed(0)
    ids = torch.randint(0, config.vocab_size, (1, 9))
    accepted = 3  # j: tokens 5, 6, 7 of the draft block are kept, 8 is rejected

    speculative = ReferenceModel(config).eval()
    install_rollback_state(speculative)
    honest = ReferenceModel(config).eval()
    honest.load_state_dict(speculative.state_dict())

    with torch.no_grad():
        spec_cache = speculative.new_cache(1, 32)
        speculative(ids[:, :5], spec_cache)          # committed prefix
        speculative(ids[:, 5:9], spec_cache)         # the verify: 4 tokens
        for state in rollback_states(speculative):
            state.keep(accepted)
        spec_cache.rewind(5 + accepted)
        spec_out, _ = speculative(ids[:, 8:9], spec_cache, num_logits_to_keep=1)

        honest_cache = honest.new_cache(1, 32)
        honest(ids[:, : 5 + accepted], honest_cache)
        honest_out, _ = honest(ids[:, 8:9], honest_cache, num_logits_to_keep=1)

    torch.testing.assert_close(spec_out, honest_out, rtol=0, atol=0)


def test_keeping_every_step_is_the_same_as_not_rolling_back_at_all():
    config = tiny_config()
    torch.manual_seed(0)
    ids = torch.randint(0, config.vocab_size, (1, 9))

    model = ReferenceModel(config).eval()
    install_rollback_state(model)
    with torch.no_grad():
        cache = model.new_cache(1, 32)
        model(ids[:, :5], cache)
        model(ids[:, 5:9], cache)
        before = [layer.recurrent.clone() for layer in cache.layers if hasattr(layer, "recurrent")]
        for state in rollback_states(model):
            state.keep(4)
        after = [layer.recurrent for layer in cache.layers if hasattr(layer, "recurrent")]

    for old, new in zip(before, after):
        torch.testing.assert_close(old, new, rtol=0, atol=0)


def test_installing_it_changes_the_linear_attention_classes_and_nothing_else():
    """`batches_test` asserts every install changes something; this says exactly what."""
    config = tiny_config()
    model = ReferenceModel(config)
    before = {name: type(m).__name__ for name, m in model.named_modules()}

    install_rollback_state(model)

    after = {name: type(m).__name__ for name, m in model.named_modules()}
    changed = {name for name in after if after[name] != before[name]}
    assert changed
    assert all("linear_attn" in name for name in changed)
```

- [ ] **Step 2: Run them and watch them fail**

Run: `uv run pytest src/deltaforge/kernels/rollback_state_test.py -q`
Expected: FAIL, `ModuleNotFoundError: deltaforge.kernels.rollback_state`

- [ ] **Step 3: Implement the module**

```python
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


class RollbackState:
    """The per-step versions one patched layer recorded on its last forward."""

    def __init__(self) -> None:
        self.cache = None
        self.states: list[Tensor] = []
        self.conv_windows: list[Tensor] = []

    def record(self, cache, states: list[Tensor], conv_windows: list[Tensor]) -> None:
        self.cache = cache
        self.states = states
        self.conv_windows = conv_windows

    def keep(self, step: int) -> None:
        """Commit the state as of ``step`` tokens of the last forward being accepted.

        ``step`` counts tokens of that forward, so 0 means "none of it happened" and
        ``len(states)`` means "all of it did". Copied in place, because the address of a
        cache tensor is a promise the compiled graph was given.
        """
        if self.cache is None:
            raise RuntimeError("keep() before any forward recorded a state")
        if not 0 <= step <= len(self.states):
            raise ValueError(f"step {step} outside the {len(self.states)} recorded steps")
        self.cache.recurrent.copy_(self.states[step])
        self.cache.conv.copy_(self.conv_windows[step])


def _patched_delta_net_class():
    class RollbackGatedDeltaNet(GatedDeltaNet):
        """`GatedDeltaNet` that keeps one state and one conv window per input token."""

        def forward(self, hidden_states: Tensor, cache=None) -> Tensor:
            if cache is None:
                return super().forward(hidden_states, cache)
            config = self.config
            batch, seq_len, _ = hidden_states.shape

            history = self.conv1d.kernel_size[0] - 1
            qkv = self.in_proj_qkv(hidden_states).transpose(1, 2)
            padded = torch.cat((cache.conv.to(qkv.dtype), qkv), dim=-1) if history else qkv
            conv_out = F.silu(F.conv1d(padded, self.conv1d.weight, groups=self.conv1d.groups))
            # The window after t of these tokens is a slice of what we already built.
            conv_windows = [
                padded[..., t : t + history].to(cache.conv.dtype).clone()
                for t in range(seq_len + 1)
            ]
            qkv = conv_out.transpose(1, 2)

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
```

- [ ] **Step 4: Register the kernel and its installer**

In `src/deltaforge/kernels/__init__.py`, beside the other registrations:

```python
from . import rollback_state as _rollback_state

REGISTRY.register(
    "rollback_state",
    impl=_rollback_state.install_rollback_state,
    replaces="gated_delta_rule",
    status=KernelStatus.CANDIDATE,
    hypothesis="062-verify-inflation",
    notes=(
        "Records one recurrent state and one conv window per input token, so a rejected "
        "draft is undone by a copy rather than by a second weight stream. Not a kernel: "
        "no Triton, no custom op, and the arithmetic is the reference's, one token at a "
        "time. It exists because `recurrent_gated_delta_rule` returns only its final "
        "state. See docs/superpowers/specs/2026-09-24-speculative-decoding-design.md §3."
    ),
)
```

In `src/deltaforge/model.py`, beside the other installers:

```python
def _install_rollback_state(model: ReferenceModel, entry: KernelEntry) -> None:
    from .kernels.rollback_state import install_rollback_state  # noqa: PLC0415

    install_rollback_state(model, entry)


register_installer("rollback_state", _install_rollback_state)
```

- [ ] **Step 5: Run the tests**

Run: `uv run pytest src/deltaforge/kernels/rollback_state_test.py src/deltaforge/batches_test.py -q`
Expected: PASS. If the first test fails by a small numeric margin rather than exactly, that
is a **real failure**, not tolerance: both sides run the same fp32 scan in the same order.

- [ ] **Step 6: Commit**

```bash
git add src/deltaforge/kernels/rollback_state.py src/deltaforge/kernels/rollback_state_test.py \
        src/deltaforge/kernels/__init__.py src/deltaforge/model.py
git commit -m "Undo a rejected draft with a copy, not with a second weight stream"
```

---

### Task 4: the drafter protocol and two drafters that cost nothing

**Files:**
- Create: `src/deltaforge/speculative.py` (drafters only; the loop is Task 5)
- Create: `src/deltaforge/speculative_test.py`

**Interfaces:**
- Produces:
  - `class Drafter(Protocol)`: `propose(committed: Tensor, k: int) -> Tensor` returning
    `(batch, k)` int64; `commit(tokens: Tensor) -> None`
  - `FixedTokenDrafter(token_id: int)` — always proposes the same id
  - `NgramDrafter(n: int = 3)` — proposes the continuation that followed the last `n`
    committed tokens the last time they occurred

- [ ] **Step 1: Write the failing tests**

```python
"""Drafters: the cheapest half of the hypothesis, and the only half testable without a GPU."""

from __future__ import annotations

import torch

from .speculative import FixedTokenDrafter, NgramDrafter


def test_a_fixed_drafter_proposes_the_same_token_every_time():
    """The instrument for measuring `gamma`: acceptance is ~0 by construction, so the slot
    measures what a k+1-token verify costs and nothing else. Spec §6, slot `a`."""
    drafter = FixedTokenDrafter(token_id=7)
    drafter.commit(torch.tensor([[1, 2, 3]]))

    proposal = drafter.propose(torch.tensor([[1, 2, 3]]), k=4)

    assert proposal.shape == (1, 4)
    assert proposal.unique().tolist() == [7]


def test_an_ngram_drafter_copies_the_continuation_of_the_last_match():
    drafter = NgramDrafter(n=2)
    drafter.commit(torch.tensor([[5, 6, 7, 8, 9, 5, 6]]))

    proposal = drafter.propose(torch.tensor([[5, 6, 7, 8, 9, 5, 6]]), k=3)

    assert proposal.tolist() == [[7, 8, 9]]


def test_an_ngram_drafter_with_no_match_still_proposes_k_tokens():
    """A drafter that returned fewer would make the verify shape vary, and a varying shape
    recompiles: batch 008 lost six of nine slots to recompilation."""
    drafter = NgramDrafter(n=2)
    drafter.commit(torch.tensor([[1, 2, 3]]))

    proposal = drafter.propose(torch.tensor([[1, 2, 3]]), k=4)

    assert proposal.shape == (1, 4)


def test_an_ngram_drafter_never_proposes_from_a_match_at_the_very_end():
    """The last n tokens always match themselves; drafting from that proposes nothing."""
    drafter = NgramDrafter(n=2)
    history = torch.tensor([[4, 1, 2]])
    drafter.commit(history)

    proposal = drafter.propose(history, k=2)

    assert proposal.shape == (1, 2)
```

- [ ] **Step 2: Run them and watch them fail**

Run: `uv run pytest src/deltaforge/speculative_test.py -q`
Expected: FAIL, `ModuleNotFoundError: deltaforge.speculative`

- [ ] **Step 3: Implement**

```python
"""Speculative decoding: more than one token per weight-stream.

The weights are read once per forward pass, not once per token, so a pass that verifies k
drafted tokens costs about what a pass producing one costs. Draft k tokens cheaply, verify
them in a single forward, keep every one the verifier's own argmax agrees with. The emitted
sequence is the one greedy decoding would have produced, because every token emitted is an
argmax of logits this model computed.

`docs/superpowers/specs/2026-09-24-speculative-decoding-design.md` has the cost model, the
kill criteria and the reason the recurrent state is the hard part.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol

import torch
from torch import Tensor

__all__ = [
    "AcceptanceRecord",
    "Drafter",
    "FixedTokenDrafter",
    "NgramDrafter",
]


class Drafter(Protocol):
    """Proposes `k` tokens and is told what was committed. Always proposes exactly `k`.

    Exactly `k` matters more than it looks: the verify tensor's shape is `k+1`, and a
    drafter that sometimes proposed fewer would hand dynamo a second shape to compile.
    """

    def propose(self, committed: Tensor, k: int) -> Tensor: ...

    def commit(self, tokens: Tensor) -> None: ...


class FixedTokenDrafter:
    """Proposes one id, forever. Acceptance ~0 by construction.

    This is an instrument, not a candidate: it measures `gamma(k)`, the cost of a
    `k+1`-token verify, with no drafter quality in the number at all. Its slot is
    registered as a predicted loss and the ratio it returns is `1/gamma`.
    """

    def __init__(self, token_id: int = 0) -> None:
        self.token_id = token_id

    def propose(self, committed: Tensor, k: int) -> Tensor:
        return torch.full(
            (committed.shape[0], k), self.token_id, dtype=torch.long, device=committed.device
        )

    def commit(self, tokens: Tensor) -> None:
        return None


class NgramDrafter:
    """Prompt-lookup drafting: the continuation that followed these `n` tokens last time.

    Costs no weights and no forward pass, so `d = 0` and the whole downside is `gamma - 1`:
    spec §1 puts break-even at about 10% acceptance. It is also the drafter whose
    acceptance depends most on the prompt, which is why the benchmark grows a text
    workload — on the random token ids the headline workload generates, this measures
    something that is not decoding.
    """

    def __init__(self, n: int = 3, fill_token: int = 0) -> None:
        self.n = n
        self.fill_token = fill_token

    def propose(self, committed: Tensor, k: int) -> Tensor:
        batch = committed.shape[0]
        out = torch.full((batch, k), self.fill_token, dtype=torch.long, device=committed.device)
        for row in range(batch):
            ids = committed[row].tolist()
            if len(ids) <= self.n:
                continue
            needle = ids[-self.n :]
            # Search backwards from the most recent occurrence that is not the tail itself.
            for start in range(len(ids) - self.n - 1, -1, -1):
                if ids[start : start + self.n] != needle:
                    continue
                follow = ids[start + self.n : start + self.n + k]
                if not follow:
                    continue
                out[row, : len(follow)] = torch.tensor(
                    follow, dtype=torch.long, device=committed.device
                )
                break
        return out

    def commit(self, tokens: Tensor) -> None:
        return None


@dataclass
class AcceptanceRecord:
    """How many drafts each cycle kept. The diagnostic without which a ratio is unreadable.

    `034-static-cache-cudagraphs` won 2.0% for a mechanism that never fired, and the slot
    could not say so because it recorded only a ratio. A speculative slot that came back at
    0.95 could be a bad drafter or an expensive verify, and those license opposite next
    steps: this separates them.
    """

    block_size: int
    accepted: list[int] = field(default_factory=list)

    def observe(self, accepted: int) -> None:
        self.accepted.append(accepted)

    @property
    def cycles(self) -> int:
        return len(self.accepted)

    @property
    def mean_accepted(self) -> float:
        return sum(self.accepted) / len(self.accepted) if self.accepted else 0.0

    @property
    def histogram(self) -> dict[int, int]:
        counts = {step: 0 for step in range(self.block_size + 1)}
        for value in self.accepted:
            counts[value] = counts.get(value, 0) + 1
        return counts

    def to_dict(self) -> dict[str, object]:
        return {
            "block_size": self.block_size,
            "cycles": self.cycles,
            "mean_accepted": self.mean_accepted,
            "histogram": {str(k): v for k, v in self.histogram.items()},
        }
```

- [ ] **Step 4: Run the tests**

Run: `uv run pytest src/deltaforge/speculative_test.py -q`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/deltaforge/speculative.py src/deltaforge/speculative_test.py
git commit -m "Two drafters that cost nothing, and the acceptance record a ratio needs"
```

---

### Task 5: the loop

**Files:**
- Modify: `src/deltaforge/speculative.py`
- Modify: `src/deltaforge/speculative_test.py`

**Interfaces:**
- Consumes: Task 1 `DecodeCache.rewind`, Task 2's loop contract, Task 3
  `rollback_states(model)`, Task 4 `Drafter`, `AcceptanceRecord`
- Produces: `SpeculativeLoop(drafter, block_size, acceptance=None)` callable as
  `loop(runnable, input_ids, max_new_tokens, cache) -> Tensor`, and
  `install_speculative_loop(model, drafter, block_size)`

- [ ] **Step 1: Write the failing tests**

```python
def test_a_block_size_of_zero_decodes_exactly_like_the_reference():
    """The diagnostic that separates a plumbing bug from a reduction-order flip. It must be
    bit-identical, and it runs here rather than on a rented card."""
    config = tiny_config()
    torch.manual_seed(0)
    ids = torch.randint(0, config.vocab_size, (1, 6))

    plain = ReferenceModel(config).eval()
    spec = ReferenceModel(config).eval()
    spec.load_state_dict(plain.state_dict())
    install_rollback_state(spec)
    install_speculative_loop(spec, FixedTokenDrafter(token_id=3), block_size=0)

    with torch.no_grad():
        expected = greedy_decode(plain, ids, 8, cache=plain.new_cache(1, 32))
        got = greedy_decode(spec, ids, 8, cache=spec.new_cache(1, 32))

    torch.testing.assert_close(got, expected, rtol=0, atol=0)


def test_a_drafter_that_is_always_wrong_still_emits_the_reference_sequence():
    """Acceptance 0 is the worst case and it must still be *correct*, not merely slow."""
    config = tiny_config()
    torch.manual_seed(0)
    ids = torch.randint(0, config.vocab_size, (1, 6))

    plain = ReferenceModel(config).eval()
    spec = ReferenceModel(config).eval()
    spec.load_state_dict(plain.state_dict())
    install_rollback_state(spec)
    install_speculative_loop(spec, FixedTokenDrafter(token_id=3), block_size=4)

    with torch.no_grad():
        expected = greedy_decode(plain, ids, 8, cache=plain.new_cache(1, 40))
        got = greedy_decode(spec, ids, 8, cache=spec.new_cache(1, 40))

    torch.testing.assert_close(got, expected, rtol=0, atol=0)


def test_a_perfect_drafter_is_accepted_every_time_and_emits_the_same_tokens():
    """An oracle drafter proves the accept path, which the always-wrong drafter never takes."""
    config = tiny_config()
    torch.manual_seed(0)
    ids = torch.randint(0, config.vocab_size, (1, 6))

    plain = ReferenceModel(config).eval()
    with torch.no_grad():
        expected = greedy_decode(plain, ids, 8, cache=plain.new_cache(1, 40))

    spec = ReferenceModel(config).eval()
    spec.load_state_dict(plain.state_dict())
    install_rollback_state(spec)
    record = AcceptanceRecord(block_size=4)
    install_speculative_loop(spec, _OracleDrafter(expected), block_size=4, acceptance=record)

    with torch.no_grad():
        got = greedy_decode(spec, ids, 8, cache=spec.new_cache(1, 40))

    torch.testing.assert_close(got, expected, rtol=0, atol=0)
    assert record.mean_accepted == 4.0
    assert record.cycles == 2


def test_the_loop_records_what_it_accepted():
    config = tiny_config()
    torch.manual_seed(0)
    ids = torch.randint(0, config.vocab_size, (1, 6))
    spec = ReferenceModel(config).eval()
    install_rollback_state(spec)
    record = AcceptanceRecord(block_size=2)
    install_speculative_loop(spec, FixedTokenDrafter(token_id=3), block_size=2, acceptance=record)

    with torch.no_grad():
        greedy_decode(spec, ids, 6, cache=spec.new_cache(1, 40))

    assert record.cycles == 6  # acceptance 0 emits one token per cycle
    assert record.histogram[0] == 6


class _OracleDrafter:
    """Drafts the reference's own continuation. Only a test fixture: it has the answer."""

    def __init__(self, truth):
        self.truth = truth
        self.emitted = 0

    def propose(self, committed, k):
        window = self.truth[:, self.emitted : self.emitted + k]
        if window.shape[1] < k:
            pad = torch.zeros((window.shape[0], k - window.shape[1]), dtype=torch.long)
            window = torch.cat((window, pad), dim=1)
        return window

    def commit(self, tokens):
        self.emitted += tokens.shape[1]
```

- [ ] **Step 2: Run them and watch them fail**

Run: `uv run pytest src/deltaforge/speculative_test.py -k loop -q`
Expected: FAIL, `ImportError: cannot import name 'install_speculative_loop'`

- [ ] **Step 3: Implement**

Append to `src/deltaforge/speculative.py`:

```python
class SpeculativeLoop:
    """Draft `k`, verify in one pass, keep the prefix the verifier agrees with.

    One cycle:

    1. the drafter proposes `k` tokens from what is committed;
    2. one forward over `[last_committed_token, draft_0 ... draft_{k-1}]` returns `k+1`
       logits — position `i` is the model's own answer for what follows the first `i`
       drafted tokens;
    3. accept the longest prefix where the drafts match those argmaxes, and emit the
       argmax at the first mismatch as well, which is always a correct token;
    4. rewind the cache by `k - j` and tell each layer which state version to keep.

    Step 3 is why the emitted sequence is greedy decoding's: every token emitted is an
    argmax of logits this model computed, and a rejected draft contributes nothing.
    """

    def __init__(self, drafter: Drafter, block_size: int, acceptance: AcceptanceRecord | None = None):
        if block_size < 0:
            raise ValueError(f"block_size must be non-negative, got {block_size}")
        self.drafter = drafter
        self.block_size = block_size
        self.acceptance = acceptance

    def __call__(self, runnable, input_ids: Tensor, max_new_tokens: int, cache) -> Tensor:
        from .kernels.rollback_state import rollback_states  # noqa: PLC0415

        states = rollback_states(runnable)
        logits, _ = runnable(input_ids, cache, num_logits_to_keep=1)
        token = logits[:, -1].argmax(dim=-1, keepdim=True)
        generated = [token]
        self.drafter.commit(torch.cat([input_ids, token], dim=1))

        while sum(piece.shape[1] for piece in generated) < max_new_tokens:
            committed = torch.cat([input_ids, *generated], dim=1)
            if self.block_size == 0:
                logits, _ = runnable(token, cache, num_logits_to_keep=1)
                token = logits[:, -1].argmax(dim=-1, keepdim=True)
                generated.append(token)
                self.drafter.commit(token)
                continue

            drafts = self.drafter.propose(committed, self.block_size)
            verified = torch.cat([token, drafts], dim=1)
            before = cache.seq_len
            logits, _ = runnable(verified, cache, num_logits_to_keep=self.block_size + 1)
            proposals = logits.argmax(dim=-1)

            # The first position where the model's own answer differs from the draft. Batch
            # 1 is what this project measures, so this reads row 0 and is honest about it.
            accepted = 0
            while accepted < self.block_size and bool(
                (proposals[:, accepted] == drafts[:, accepted]).all()
            ):
                accepted += 1

            kept = torch.cat(
                [drafts[:, :accepted], proposals[:, accepted : accepted + 1]], dim=1
            )
            # The verify wrote `block_size + 1` positions; `accepted + 1` of them survive.
            cache.rewind(before + accepted + 1)
            for state in states:
                state.keep(accepted + 1)
            if self.acceptance is not None:
                self.acceptance.observe(accepted)

            generated.append(kept)
            token = kept[:, -1:]
            self.drafter.commit(kept)

        return torch.cat(generated, dim=1)[:, :max_new_tokens]


def install_speculative_loop(
    model, drafter: Drafter, block_size: int, acceptance: AcceptanceRecord | None = None
) -> None:
    """Put the loop where `greedy_decode` will find it.

    The drafter is held **here**, on the loop, and never registered as a submodule.
    `cli._assert_parameters_are_shared` requires every candidate parameter to share storage
    with a reference parameter of the same name — the check that catches a candidate which
    silently loaded a second 8.4 GB copy of the weights — and draft weights have no
    counterpart. Keeping them off the module tree leaves that check exactly as strict.
    """
    model.decode_loop = SpeculativeLoop(drafter, block_size, acceptance)
```

- [ ] **Step 4: Run the tests**

Run: `uv run pytest src/deltaforge/speculative_test.py -q`
Expected: PASS. The always-wrong-drafter test is the one that fails if `rewind` and
`keep` disagree about what "accepted" counts.

- [ ] **Step 5: Run the whole suite, then lint**

Run: `uv run pytest -q && uv run ruff check . && uv run ruff format --check .`
Expected: PASS

- [ ] **Step 6: Commit**

```bash
git add src/deltaforge/speculative.py src/deltaforge/speculative_test.py
git commit -m "The loop: draft k, verify once, keep the prefix the model agrees with"
```

---

### Task 6: a gate that can fail

**Files:**
- Modify: `src/deltaforge/harness/correctness.py`
- Modify: `src/deltaforge/harness/correctness_test.py`
- Modify: `src/deltaforge/batch.py` (`CORRECTNESS_POLICIES`, `Hypothesis` fields + validation)
- Modify: `src/deltaforge/batch_test.py`

**Interfaces:**
- Produces:
  - `check_sequence(reference_model, candidate_model, prompt_token_ids, *, max_new_tokens,
    gap_ceiling, prompt_digest="", device=None) -> SequenceCheck`
  - `SequenceCheck.first_divergence: int | None`, `.reference_top2_gap: float | None`,
    `.passed: bool`, `.to_dict()`
  - `Hypothesis.correctness == "sequence"` with `divergence_gap_ceiling: float | None`

**Read first:** spec §4. The short version: `approximate` teacher-forces the *models*, and a
speculative candidate has the reference's own weights, so it returns 264/264 with a
corrupted state. It is not a weak gate here, it is one that cannot fail.

- [ ] **Step 1: Write the failing tests**

```python
def test_a_sequence_check_passes_when_the_tokens_match(tiny_models):
    reference, candidate = tiny_models

    check = check_sequence(reference, candidate, [[1, 2, 3]], max_new_tokens=8, gap_ceiling=0.02)

    assert check.first_divergence is None
    assert check.passed


def test_a_divergence_on_a_confident_position_fails(monkeypatch, tiny_models):
    """A rollback bug looks exactly like this: the model was sure, and we emitted something
    else. No distribution statistic would have caught it, because the weights are identical."""
    reference, candidate = tiny_models
    monkeypatch.setattr(candidate, "decode_loop", _loop_that_flips_token(index=3))

    check = check_sequence(reference, candidate, [[1, 2, 3]], max_new_tokens=8, gap_ceiling=0.02)

    assert check.first_divergence == 3
    assert check.reference_top2_gap > 0.02
    assert not check.passed


def test_a_divergence_where_the_reference_was_indifferent_passes(monkeypatch, tiny_models):
    """One bf16 ULP flips an argmax on this checkpoint — `009-gemv-bf16-control` matched 1
    prompt of 5 on exactly that — so a flip on a position with no gap is the reduction
    order, not a bug."""
    reference, candidate = tiny_models
    monkeypatch.setattr(candidate, "decode_loop", _loop_that_flips_a_tied_token())

    check = check_sequence(reference, candidate, [[1, 2, 3]], max_new_tokens=8, gap_ceiling=0.02)

    assert check.first_divergence is not None
    assert check.reference_top2_gap <= 0.02
    assert check.passed
```

And in `batch_test.py`:

```python
def test_a_sequence_gated_hypothesis_must_register_a_gap_ceiling():
    with pytest.raises(ValueError, match="divergence_gap_ceiling"):
        make_hypothesis(slug="062-x", correctness="sequence")


def test_a_sequence_gated_hypothesis_carries_no_distribution_bars():
    with pytest.raises(ValueError, match="teacher-forced"):
        make_hypothesis(
            slug="062-x",
            correctness="sequence",
            divergence_gap_ceiling=0.02,
            top1_threshold=0.99,
            kl_threshold=0.01,
            correctness_positions=264,
        )
```

- [ ] **Step 2: Run them and watch them fail**

Run: `uv run pytest src/deltaforge/harness/correctness_test.py src/deltaforge/batch_test.py -k sequence -q`
Expected: FAIL, `ImportError: cannot import name 'check_sequence'`

- [ ] **Step 3: Implement the gate**

In `harness/correctness.py`:

```python
@dataclass(frozen=True)
class SequenceCheck:
    """Free-running token agreement, with the first divergence attributed.

    The policy the speculative candidates need, and the reason neither existing gate fits.
    `check_distribution` teacher-forces both models, so on a candidate that carries the
    reference's own weights it returns perfect agreement whatever the decode loop did with
    the recurrent state. `check_end_to_end` does free-run the loop, but requires the token
    ids to match exactly — and a verify pass over `k+1` positions reduces in a different
    order from `k+1` separate passes, which on this checkpoint flips an argmax whenever the
    top-2 logits are within a ULP of each other.

    So: report **where** the sequences first differ, and how confident the reference was
    there. A flip on a position the reference had no opinion about is the arithmetic; a
    flip on one it was sure about is a bug.
    """

    num_prompts: int
    first_divergence: int | None
    reference_top2_gap: float | None
    gap_ceiling: float
    prompt_digest: str = ""

    @property
    def passed(self) -> bool:
        if self.first_divergence is None:
            return True
        return self.reference_top2_gap is not None and self.reference_top2_gap <= self.gap_ceiling

    def to_dict(self) -> dict[str, object]:
        return {
            "kind": "sequence",
            "num_prompts": self.num_prompts,
            "first_divergence": self.first_divergence,
            "reference_top2_gap": self.reference_top2_gap,
            "gap_ceiling": self.gap_ceiling,
            "prompt_digest": self.prompt_digest,
            "passed": self.passed,
        }


def check_sequence(
    reference_model,
    candidate_model,
    prompt_token_ids,
    *,
    max_new_tokens: int = 128,
    gap_ceiling: float,
    prompt_digest: str = "",
    device=None,
) -> SequenceCheck:
    """Free-run both models and attribute the first divergence, if there is one."""
    if not prompt_token_ids:
        raise ValueError("no prompts supplied to the sequence gate")
    device = device or next(reference_model.parameters()).device

    worst: tuple[int, float] | None = None
    for ids in prompt_token_ids:
        prompt = torch.tensor([list(ids)], dtype=torch.long, device=device)
        with torch.no_grad():
            expected = greedy_decode(
                reference_model, prompt, max_new_tokens,
                cache=reference_model.new_cache(1, len(ids) + max_new_tokens + 16),
            )
            got = greedy_decode(
                candidate_model, prompt, max_new_tokens,
                cache=candidate_model.new_cache(1, len(ids) + max_new_tokens + 16),
            )
        differing = (expected != got).nonzero()
        if differing.numel() == 0:
            continue
        index = int(differing[0, 1])
        gap = _reference_top2_gap(reference_model, prompt, expected, index, device)
        if worst is None or gap > worst[1]:
            worst = (index, gap)

    if worst is None:
        return SequenceCheck(len(prompt_token_ids), None, None, gap_ceiling, prompt_digest)
    return SequenceCheck(len(prompt_token_ids), worst[0], worst[1], gap_ceiling, prompt_digest)


def _reference_top2_gap(model, prompt, expected, index: int, device) -> float:
    """How far apart the reference's top two logits were at the position that diverged.

    The same statistic `oracle_test.py::test_report_the_first_greedy_divergence` computes
    against HuggingFace, for the same purpose: telling a disagreement the arithmetic
    explains from one it does not.
    """
    context = torch.cat([prompt, expected[:, :index]], dim=1)
    with torch.no_grad():
        logits, _ = model(context, model.new_cache(1, context.shape[1] + 1), num_logits_to_keep=1)
    top2 = torch.topk(logits[0, -1].float(), 2)
    return float(top2.values[0] - top2.values[1])
```

- [ ] **Step 4: Implement the policy**

In `batch.py`:

```python
CORRECTNESS_POLICIES = ("exact", "approximate", "sequence")
```

Add the field to `Hypothesis`:

```python
    #: Only for `sequence`. The largest top-2 logit gap at which a token divergence is still
    #: attributable to reduction order rather than to a bug. Registered before the rental
    #: for the same reason every other bar is.
    divergence_gap_ceiling: float | None = None
```

And in `__post_init__`, beside the `approximate` branch:

```python
        if self.correctness == "sequence":
            if self.divergence_gap_ceiling is None:
                raise ValueError(
                    f"{self.slug!r} is gated on its token sequence and must register a "
                    "divergence_gap_ceiling before the rental. Without one the gate passes "
                    "any divergence, including a corrupted recurrent state."
                )
            if self.top1_threshold is not None or self.kl_threshold is not None:
                raise ValueError(
                    f"{self.slug!r} is gated 'sequence' but carries teacher-forced bars. "
                    "check_distribution never calls the decode loop, and a candidate that "
                    "replaces the loop while sharing the reference's weights scores "
                    "perfectly on it whatever it did."
                )
        elif self.divergence_gap_ceiling is not None:
            raise ValueError(
                f"{self.slug!r} registers a divergence_gap_ceiling but is gated "
                f"{self.correctness!r}, which never reads it."
            )
```

- [ ] **Step 5: Run the tests**

Run: `uv run pytest src/deltaforge/harness/correctness_test.py src/deltaforge/batch_test.py -q`
Expected: PASS

- [ ] **Step 6: Commit**

```bash
git add src/deltaforge/harness/correctness.py src/deltaforge/harness/correctness_test.py \
        src/deltaforge/batch.py src/deltaforge/batch_test.py
git commit -m "A gate that can fail: attribute the first divergence instead of counting agreement"
```

---

### Task 7: harness plumbing

**Files:**
- Modify: `src/deltaforge/batch_run.py` (`_run_correctness`, `_build_candidate`, `SlotResult`)
- Modify: `src/deltaforge/batch_run_test.py`

**Interfaces:**
- Consumes: Task 5 `SpeculativeLoop`, Task 6 `check_sequence`
- Produces: `SlotResult.acceptance: dict[str, Any]`, in `to_slot_dict()` under `"acceptance"`

- [ ] **Step 1: Write the failing tests**

```python
def test_a_candidate_that_only_installs_a_decode_loop_is_not_a_no_op():
    """The guard exists because a candidate identical to the reference measures 1.00 and
    reads as a well-behaved null. A loop installer changes no module class and is still the
    largest behavioural change any candidate here has made."""
    runner = _runner_with(hypothesis_kernels=("speculative_ngram_k2",))

    candidate = runner._build_candidate(runner.batch.get("064-spec-ngram-k2"))

    assert candidate.decode_loop is not None


def test_a_sequence_gated_slot_runs_the_sequence_gate(monkeypatch):
    calls = []
    monkeypatch.setattr(batch_run, "check_sequence", lambda *a, **k: calls.append(k) or _passing())

    runner.run_slot(hypothesis_gated_sequence)

    assert calls and calls[0]["gap_ceiling"] == 0.02


def test_the_slot_record_carries_what_the_loop_accepted():
    result = runner.run_slot(hypothesis_gated_sequence)

    assert result.to_slot_dict()["acceptance"]["mean_accepted"] >= 0.0
```

- [ ] **Step 2: Run them and watch them fail**

Run: `uv run pytest src/deltaforge/batch_run_test.py -k 'loop or sequence or accepted' -q`
Expected: FAIL

- [ ] **Step 3: Implement**

In `_build_candidate`, replace the `elif after == before:` branch:

```python
        elif after == before and getattr(candidate, "decode_loop", None) is None:
            # The failure this check exists for: a candidate identical to the reference
            # measures 1.00 and is indistinguishable from a well-behaved null result. A
            # decode-loop installer is the one legitimate way to change nothing structural
            # and still be a different program, so it is named here rather than exempted by
            # a flag nobody can see.
            raise RuntimeError(
                f"{hypothesis.slug!r} installed {applied} but changed no module class and "
                "installed no decode loop. Refusing to benchmark the reference while "
                "labelling it the candidate."
            )
```

In `_run_correctness`, before the `approximate` branch:

```python
        if hypothesis.correctness == "sequence":
            sequence = check_sequence(
                self.reference,
                candidate,
                self.prompt_ids,
                max_new_tokens=self.max_new_tokens,
                gap_ceiling=hypothesis.divergence_gap_ceiling,
                prompt_digest=PROMPT_DIGEST,
            )
            self.log(
                f"[batch] {hypothesis.slug}: first divergence "
                f"{sequence.first_divergence} (reference top-2 gap "
                f"{sequence.reference_top2_gap}, ceiling {sequence.gap_ceiling})"
            )
            return CorrectnessReport(kernel_checks=kernel_checks, sequence=sequence).to_dict()
```

In `SlotResult`, beside `card`:

```python
    #: What the decode loop accepted, when the candidate installed one: cycles, mean
    #: accepted per cycle, and the histogram. A speculative ratio at 0.95 is a bad drafter
    #: or an expensive verify, and only this says which.
    acceptance: dict[str, Any] = field(default_factory=dict)
```

Populate it in `run_slot` from the loop's `AcceptanceRecord` and add
`"acceptance": dict(self.acceptance)` to `to_slot_dict`.

- [ ] **Step 4: Run the tests**

Run: `uv run pytest src/deltaforge/batch_run_test.py -q`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/deltaforge/batch_run.py src/deltaforge/batch_run_test.py
git commit -m "Run the sequence gate, and record what the loop accepted"
```

---

### Task 8: a workload whose prompt is text

**Files:**
- Modify: `src/deltaforge/cli.py` (`DEFAULT_WORKLOADS`, `_build_columns`)
- Modify: `src/deltaforge/cli_test.py`

**Interfaces:**
- Produces: `DEFAULT_WORKLOADS["headline_text"]` with the same shape as `headline` plus
  `"prompt": "text"`

- [ ] **Step 1: Write the failing test**

```python
def test_a_text_workload_exists_with_the_headline_shape():
    """Acceptance on random token ids is not acceptance on text: the model's continuation
    of noise is its own distribution, and an n-gram drafter can look far better or far
    worse there than it ever will in use. Every hypothesis before this one was indifferent
    to the prompt."""
    text = DEFAULT_WORKLOADS["headline_text"]
    headline = DEFAULT_WORKLOADS["headline"]

    assert text["batch_size"] == headline["batch_size"]
    assert text["context_length"] == headline["context_length"]
    assert text["decode_tokens"] == headline["decode_tokens"]
    assert text["prompt"] == "text"
    assert headline.get("prompt", "random") == "random"
```

- [ ] **Step 2: Run it and watch it fail**

Run: `uv run pytest src/deltaforge/cli_test.py -k text_workload -q`
Expected: FAIL, `KeyError: 'headline_text'`

- [ ] **Step 3: Implement**

```python
DEFAULT_WORKLOADS = {
    "headline": {"batch_size": 1, "context_length": 2048, "decode_tokens": 128, "prompt": "random"},
    "batch32": {"batch_size": 32, "context_length": 2048, "decode_tokens": 128, "prompt": "random"},
    # Same shape, real tokens. Speculative decoding is the first hypothesis whose number
    # depends on what the prompt says: acceptance is a property of the token distribution,
    # and `torch.randint` does not have one. Reported beside `headline`, never instead.
    "headline_text": {
        "batch_size": 1,
        "context_length": 2048,
        "decode_tokens": 128,
        "prompt": "text",
    },
}
```

In `_build_columns`, where the prompt is made:

```python
    if workload.get("prompt") == "text":
        prompt = _text_prompt(Path(args.weights), batch, context, device="cuda")
    else:
        prompt = torch.randint(
            0, reference.config.vocab_size, (batch, context), device="cuda", dtype=torch.long
        )
```

```python
def _text_prompt(weights: Path, batch: int, context: int, device: str) -> "torch.Tensor":
    """`context` tokens of real text, tiled from the correctness prompt set.

    The prompts are already in the repo, already hashed into `PROMPT_DIGEST`, and already
    the input the correctness gates use — so the benchmark and the gate see the same kind
    of text, and the digest says which text it was.
    """
    import torch  # noqa: PLC0415

    from .harness.prompts import CORRECTNESS_PROMPTS  # noqa: PLC0415

    ids = [token for row in _tokenize_prompts(weights, CORRECTNESS_PROMPTS) for token in row]
    if not ids:
        raise SystemExit("the tokenizer returned no ids for the correctness prompts")
    repeated = (ids * (context // len(ids) + 1))[:context]
    return torch.tensor([repeated] * batch, dtype=torch.long, device=device)
```

- [ ] **Step 4: Run the tests**

Run: `uv run pytest src/deltaforge/cli_test.py -q`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/deltaforge/cli.py src/deltaforge/cli_test.py
git commit -m "A workload whose prompt is text, because acceptance depends on it"
```

---

### Task 9: batch 010, registered before the rental

**Files:**
- Modify: `src/deltaforge/batches.py`
- Modify: `src/deltaforge/batches_test.py`
- Modify: `src/deltaforge/kernels/__init__.py` and `src/deltaforge/model.py` (register
  `speculative_fixed_k4`, `speculative_fixed_k2`, `speculative_ngram_k2`,
  `speculative_ngram_k4`, each installing the loop with its drafter and block size)

**Interfaces:**
- Consumes: everything above
- Produces: `BATCH_010` in `BATCHES`

- [ ] **Step 1: Write the manifest**

Slots, in order. Every one carries `contrast_with` or is an ingredient the batch has not
measured, because `batch.unpaired_slots` refuses a manifest from 010 on that does neither.

```python
BATCH_010 = Batch(
    batch_id="010-speculative-verify",
    description=(
        "What a k+1-token verify costs, measured with no drafter in it, and then two "
        "drafters that cost nothing. Spec: docs/superpowers/specs/"
        "2026-09-24-speculative-decoding-design.md."
    ),
    hypotheses=(
        Hypothesis(
            slug="000-identity",
            kernels=(),
            category="calibration",
            byte_share=0.0,
            mechanism="Installs nothing; calibrates the harness against the card of the hour.",
            prediction="identity",
            rationale=(
                "The identity champion must return 1.00 +- noise or every other number in "
                "the batch is void. It also reports the reference column's achieved "
                "bandwidth before any candidate runs, which is how rental 43's 800 GB/s "
                "card was recognised as a card rather than as a result."
            ),
        ),
        Hypothesis(
            slug="062-verify-inflation-k4",
            kernels=("rollback_state", "speculative_fixed_k4"),
            category="C",
            byte_share=0.0,
            mechanism=(
                "The speculative loop with a drafter that always proposes the same token, "
                "so acceptance is 0 by construction and the ratio is 1/gamma(4)."
            ),
            prediction="loss",
            rationale=(
                "This slot is an instrument, not a candidate. gamma is the whole downside "
                "of the hypothesis and nothing has ever measured it: +8.2% of bytes for the "
                "five-token KV and state traffic, +2.9% for the state versioning, and an "
                "unknown dispatch term because the linear-attention scan runs five steps "
                "per layer instead of one. Predicted 0.87-0.95. Above 1.25 in 1/ratio terms "
                "and blocks longer than 2 are dead, which is registered as a kill criterion "
                "in the spec rather than decided after the number."
            ),
            correctness="sequence",
            divergence_gap_ceiling=0.02,
        ),
        Hypothesis(
            slug="063-verify-inflation-k2",
            kernels=("rollback_state", "speculative_fixed_k2"),
            category="C",
            byte_share=0.0,
            contrast_with="062-verify-inflation-k4",
            mechanism="The same instrument at k=2: how gamma scales with the block.",
            prediction="loss",
            rationale=(
                "One variable from 062, the block size. Traffic says gamma(2) ~ 1.06 "
                "against gamma(4) ~ 1.11, so this should lose about half as much. If it "
                "loses as much or more, gamma is dispatch rather than traffic and the "
                "spec's arithmetic is wrong in a way that matters more than the slot does."
            ),
            correctness="sequence",
            divergence_gap_ceiling=0.02,
        ),
        Hypothesis(
            slug="064-spec-ngram-k2",
            kernels=("rollback_state", "speculative_ngram_k2"),
            category="C",
            byte_share=0.0,
            contrast_with="063-verify-inflation-k2",
            mechanism=(
                "The same loop at k=2 with a prompt-lookup drafter: d=0, so the whole "
                "downside is gamma-1 and any acceptance above ~10% is a win."
            ),
            prediction="inconclusive",
            rationale=(
                "One variable from 063: the drafter. On the headline workload the prompt "
                "is torch.randint token ids, and an n-gram drafter over noise has no "
                "structure to find -- so the honest prediction here is that it lands within "
                "noise of 063, and the result worth having is the acceptance histogram "
                "rather than the ratio. The text workload is where this is a real question."
            ),
            correctness="sequence",
            divergence_gap_ceiling=0.02,
        ),
        Hypothesis(
            slug="065-spec-ngram-k4",
            kernels=("rollback_state", "speculative_ngram_k4"),
            category="C",
            byte_share=0.0,
            contrast_with="064-spec-ngram-k2",
            mechanism="The prompt-lookup drafter at k=4.",
            prediction="loss",
            rationale=(
                "A longer block costs more gamma and an n-gram drafter's acceptance decays "
                "fastest with depth, so at k=4 on random ids this should sit below 064. "
                "Registered as a loss so that a win is informative: it would mean "
                "acceptance is holding deeper than the drafter deserves, which is worth "
                "knowing before the int4 self-draft is built."
            ),
            requires=Precondition(
                slug="064-spec-ngram-k2",
                floor=0.0,
                versus="063-verify-inflation-k2",
                reason=(
                    "The n-gram drafter must at least not lose to the same loop with a "
                    "drafter that is wrong on purpose. If it does, drafting is costing more "
                    "than it saves at any block size and k=4 will not rescue it. The floor "
                    "is on the comparison because that is the proposition -- rental 46's "
                    "061 declined on an absolute ratio that meant nothing here."
                ),
            ),
            correctness="sequence",
            divergence_gap_ceiling=0.02,
        ),
    ),
)
```

- [ ] **Step 2: Register the four loop kernels**

Each is a two-line installer in `model.py` and a `REGISTRY.register` in `kernels/__init__.py`,
following Task 3's shape. For example:

```python
def _install_speculative_ngram_k2(model: ReferenceModel, entry: KernelEntry) -> None:
    from .speculative import AcceptanceRecord, NgramDrafter, install_speculative_loop  # noqa: PLC0415

    install_speculative_loop(
        model, NgramDrafter(n=3), block_size=2, acceptance=AcceptanceRecord(block_size=2)
    )


register_installer("speculative_ngram_k2", _install_speculative_ngram_k2)
```

- [ ] **Step 3: Run the manifest tests**

Run: `uv run pytest src/deltaforge/batches_test.py -q`
Expected: PASS — including `test_every_batch_from_010_on_is_a_set_of_controlled_contrasts`,
which is what refuses a slot whose comparison is implicit.

- [ ] **Step 4: Run the pre-rental checks**

```bash
uv run python -m deltaforge.cli fusion --batch 010-speculative-verify
uv run pytest -q && uv run ruff check . && uv run ruff format --check .
remote/run_remote.sh --dry-run --session-id smoke --batch 010-speculative-verify
```

Expected: no unpartnered barrier (nothing here is a custom op), suite green, dry run clean.

- [ ] **Step 5: Commit**

```bash
git add src/deltaforge/batches.py src/deltaforge/batches_test.py \
        src/deltaforge/kernels/__init__.py src/deltaforge/model.py
git commit -m "Register batch 010: what a verify costs, before what a drafter saves"
```

---

### Task 10: the int4 self-draft — **blocked**

**Do not start this task until `061-int4-mlp-torch-dequant` has run and cleared 1.0.**
Spec §7: `d` is a full forward of the quantised model, so if int4 in torch does not beat
bf16 on the whole model then `d ≈ v₁` and this is dead by arithmetic, whatever the
acceptance rate turns out to be. The slots in Task 9 do not depend on it.

When unblocked, the work is a drafter that holds its own quantised model and its own cache:

- the drafter is held by the loop, never registered as a submodule (Task 5 says why);
- its first `propose` of each cycle consumes the newly committed tokens **in the same
  forward** that produces the first draft, because a separate catch-up pass makes the cost
  `(k+1)·d` and costs the headline cell 0.2x (spec §1);
- that pass has a data-dependent length, so its sequence dimension is marked dynamic or the
  slot is voided by recompilation;
- acceptance is predicted before the slot runs: it is the layer-2 top-1 agreement of the
  int4 model against the reference, which the correctness gate already computes.

---

## Self-review

**Spec coverage.** §1 arithmetic → Tasks 4, 5, 9 (the instrument slots measure `γ`); §2
unchanged model → relied on, nothing to build; §3 state rollback → Tasks 1, 3; §4 gate →
Task 6; §5 harness items 1–3 → Tasks 2, 7; item 4 (acceptance recorded) → Tasks 4, 7; item 5
(no byte model) → Task 7 (`weight_bits` left empty, and `_bytes_per_token` already logs "no
byte model" rather than inventing one); item 6 text workload → Task 8; items 7–8 (rewind,
drafter off the module tree) → Tasks 1, 5; §6 slots → Task 9; §7 kill criteria → recorded in
the rationales of `062` and `063` and in Task 10's block; §8 → nothing to build.

**Placeholders.** None: every step carries the code it asks for. Task 7's tests are sketched
against fixtures that exist in `batch_run_test.py` and the implementer should follow that
file's existing runner fixture rather than inventing one — that is the only place this plan
defers to the codebase instead of quoting it, and it does so because copying the fixture
would date faster than the reference to it.

**Type consistency.** `RollbackState.keep(step)` takes a count of *tokens of the last
forward*, and Task 5 calls it with `accepted + 1` because the verify pass is `k+1` tokens of
which `accepted + 1` survive; `DecodeCache.rewind(to_seq_len)` takes an *absolute* length,
and Task 5 calls it with `before + accepted + 1`. Those two conventions differ on purpose
and the always-wrong-drafter test in Task 5 is what catches getting them backwards.
`Drafter.propose(committed, k)` and `.commit(tokens)` are the same names in Tasks 4, 5, 9.
