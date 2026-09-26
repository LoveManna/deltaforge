"""The one correctness property: rolling back is indistinguishable from never having gone."""

from __future__ import annotations

import pytest
import torch

from ..config import tiny_config
from ..kernels.rollback_state import install_rollback_state, rollback_states
from ..reference import ReferenceModel


def test_keep_restores_the_cache_to_exactly_the_recorded_step():
    """`keep(step)` is a copy of an already-computed tensor, and a copy is exact or wrong —
    there is no numerical argument that earns a tolerance here.

    Checked directly against a real patched layer's own recorded `states`/`conv_windows`
    from a `tiny_config()` model, at a fully rolled-back step (0), a partially accepted step
    (2), and the fully accepted step (4, `len(states) - 1` for a 4-token verify). A bug in
    the indexing or in the copy itself — as opposed to the *numerics*, which
    `test_rolling_back_to_j_matches_never_having_run_past_j` cannot check bit-exactly, see
    its docstring — would fail here.
    """
    config = tiny_config()
    torch.manual_seed(0)
    ids = torch.randint(0, config.vocab_size, (1, 9))

    model = ReferenceModel(config).eval()
    install_rollback_state(model)
    with torch.no_grad():
        cache = model.new_cache(1, 32)
        model(ids[:, :5], cache)  # committed prefix
        model(ids[:, 5:9], cache)  # the verify: 4 tokens, so 5 recorded steps (0..4)

    for state in rollback_states(model):
        for step in (0, 2, 4):
            state.keep(step)
            torch.testing.assert_close(state.cache.recurrent, state.states[step], rtol=0, atol=0)
            torch.testing.assert_close(state.cache.conv, state.conv_windows[step], rtol=0, atol=0)


def test_keep_accepts_the_full_forward_but_rejects_one_step_past_it():
    """The boundary TDD missed. `states` holds `seq_len + 1` entries — one per token plus the
    initial clone — so the full-accept case is `step == seq_len` (`len(states) - 1`), and
    `step == len(states)`, one past that, must raise rather than index past the list.
    """
    config = tiny_config()
    torch.manual_seed(0)
    ids = torch.randint(0, config.vocab_size, (1, 9))

    model = ReferenceModel(config).eval()
    install_rollback_state(model)
    with torch.no_grad():
        cache = model.new_cache(1, 32)
        model(ids[:, :5], cache)  # committed prefix
        model(ids[:, 5:9], cache)  # 4-token verify: steps 0..4 are valid, 5 is not

    for state in rollback_states(model):
        state.keep(4)  # the full-accept case: still succeeds
        with pytest.raises(ValueError, match="outside the valid range"):
            state.keep(5)  # one past the last recorded step


def test_rolling_back_to_j_matches_never_having_run_past_j():
    """Run k+1 tokens, keep j, continue — against a model that only ever saw j.

    This is the whole hypothesis in one assertion. If it fails, every ratio the
    speculative slots produce is measuring a model with a corrupted recurrent state, and
    the tokens would be wrong in a way no timing would reveal.

    ``honest`` is patched too. The per-token scan decomposition inside
    ``RollbackGatedDeltaNet`` is not the property this test measures — comparing it against
    plain ``GatedDeltaNet``'s single batched scan would measure fp32 reassociation order
    rather than the rollback itself. With both sides running the identical decomposition,
    the only remaining difference is the rollback, which is exactly what this asserts.

    The tolerance below is not laziness. Bit-identity against a differently-shaped reference
    is unavailable here: `in_proj_qkv` and `F.conv1d` — both unmodified, plain layers
    inherited from `GatedDeltaNet` — return different values for the *same* rows depending
    on how many rows the call has (GEMM blocking, not thread count: it persists under
    `torch.set_num_threads(1)`), and the divergence is already present in `query`/`key`/
    `value` before the recurrent scan ever runs. Measured residual: `5.7220458984375e-06`
    absolute, `1.7570137060829438e-05` relative — small enough to clear
    `reference_test.py:324`'s `test_incremental_decode_matches_a_single_full_forward`
    (`rtol=1e-4, atol=1e-4`) with zero violating elements, but large enough that
    `reference_test.py:230`'s tighter `rtol=1e-5, atol=1e-6` (for the *isolated* scan
    function in a single call) leaves 1 of 512 elements over the line by `2.86e-06`. This
    test compares whole-model logits across two different forward decompositions — a
    5-then-4-token verify plus a rollback, against one straight-through call — which is
    line 324's situation, not line 230's: the noise here has compounded through the
    full-attention layer, the RMSNorm, the MLP and the LM head, not just the scan. Matching
    line 324's tolerance rather than line 230's is matching precedent, not relaxing one.
    """
    config = tiny_config()
    torch.manual_seed(0)
    ids = torch.randint(0, config.vocab_size, (1, 9))
    accepted = 3  # j: tokens 5, 6, 7 of the draft block are kept, 8 is rejected

    speculative = ReferenceModel(config).eval()
    install_rollback_state(speculative)
    honest = ReferenceModel(config).eval()
    honest.load_state_dict(speculative.state_dict())
    install_rollback_state(honest)

    with torch.no_grad():
        spec_cache = speculative.new_cache(1, 32)
        speculative(ids[:, :5], spec_cache)  # committed prefix
        speculative(ids[:, 5:9], spec_cache)  # the verify: 4 tokens
        for state in rollback_states(speculative):
            state.keep(accepted)
        spec_cache.rewind(5 + accepted)
        spec_out, _ = speculative(ids[:, 8:9], spec_cache, num_logits_to_keep=1)

        honest_cache = honest.new_cache(1, 32)
        honest(ids[:, : 5 + accepted], honest_cache)
        honest_out, _ = honest(ids[:, 8:9], honest_cache, num_logits_to_keep=1)

    torch.testing.assert_close(spec_out, honest_out, rtol=1e-4, atol=1e-4)


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
