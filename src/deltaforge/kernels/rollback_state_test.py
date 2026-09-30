"""The one correctness property: rolling back is indistinguishable from never having gone."""

from __future__ import annotations

import pytest
import torch

from ..config import tiny_config
from ..kernels.rollback_state import install_rollback_state, rollback_states
from ..reference import STATE_DTYPE, GatedDeltaNet, ReferenceModel, _LinearLayerCache


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


def _delta_net(model) -> GatedDeltaNet:
    nets = [m for m in model.modules() if isinstance(m, GatedDeltaNet)]
    assert nets, "the tiny config has no linear-attention layer"
    return nets[0]


def _fresh_cache(config) -> _LinearLayerCache:
    history = max(config.linear_conv_kernel_dim - 1, 0)
    return _LinearLayerCache(
        conv=torch.zeros(1, config.linear_conv_dim, history),
        recurrent=torch.zeros(
            1,
            config.linear_num_value_heads,
            config.linear_key_head_dim,
            config.linear_value_head_dim,
            dtype=STATE_DTYPE,
        ),
    )


def test_rollback_gated_delta_net_forward_matches_the_reference_arithmetic():
    """I3: binds `RollbackGatedDeltaNet.forward` to `GatedDeltaNet.forward`'s arithmetic.

    `RollbackGatedDeltaNet` reimplements `GatedDeltaNet.forward`'s data flow decomposed per
    token -- forced by the reference-immutability constraint, since the per-step states this
    module needs cannot be recorded any other way without touching `reference.py`. Nothing
    else in this test suite compares the two: the rollback property tests above patch
    `install_rollback_state` onto *both* sides on purpose, to isolate the rollback from the
    decomposition (a deliberate human ruling, see the ledger), and `speculative_test.py`'s
    `test_a_block_size_of_zero_decodes_exactly_like_the_reference` compares token ids, which
    would only *probably* notice an arithmetic change. So a future edit to
    `reference.GatedDeltaNet.forward` could silently diverge from the patched copy here and
    produce wrong tokens no timing would ever reveal. This touches `reference.py` not at
    all -- it builds one plain `GatedDeltaNet` and one patched copy from the same state dict
    and compares them directly -- so it does not violate reference-immutability; it is the
    mitigation for the cost that constraint imposes. If `reference.py`'s arithmetic moves,
    this fails at edit time instead of at token time.

    `reference_test.py:230`'s isolated-scan tolerance (`rtol=1e-5, atol=1e-6`) is the right
    precedent here, not `rollback_state_test.py`'s own whole-model `rtol=1e-4, atol=1e-4`:
    this compares one layer's output and cache in isolation, with no compounding through the
    rest of the model, which is exactly `reference_test.py:230`'s situation.
    """
    config = tiny_config()
    torch.manual_seed(0)

    reference_model = ReferenceModel(config).eval()
    reference_net = _delta_net(reference_model)

    patched_model = ReferenceModel(config).eval()
    patched_model.load_state_dict(reference_model.state_dict())
    install_rollback_state(patched_model)
    patched_net = _delta_net(patched_model)

    for seq_len in (1, 4):
        hidden_states = torch.randn(1, seq_len, config.hidden_size)
        reference_cache = _fresh_cache(config)
        patched_cache = _fresh_cache(config)

        with torch.no_grad():
            reference_out = reference_net(hidden_states, reference_cache)
            patched_out = patched_net(hidden_states, patched_cache)

        torch.testing.assert_close(reference_out, patched_out, rtol=1e-5, atol=1e-6)
        torch.testing.assert_close(reference_cache.recurrent, patched_cache.recurrent, rtol=1e-5, atol=1e-6)
        torch.testing.assert_close(reference_cache.conv, patched_cache.conv, rtol=1e-5, atol=1e-6)


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


def test_a_prefill_length_forward_records_nothing_and_still_computes_the_reference():
    """Rental 53 lost three slots to an OOM at 23.03 GiB, in the benchmark's prefill.

    The per-step recording is one `recurrent_gated_delta_rule` call and one cloned state per
    token. At a verify's `k+1` tokens that is the whole point of this class; at the
    benchmark's 2048-token prefill it is thousands of sequential launches per layer and
    thousands of live clones. The peak was *identical* at k=4 and k=2 -- 23.03 GiB both
    times -- which is the proof it never scaled with the block at all.

    A forward longer than a verify must therefore take the parent's path: record nothing,
    and still produce the reference's arithmetic.
    """
    config = tiny_config()
    torch.manual_seed(0)

    reference_model = ReferenceModel(config).eval()
    patched_model = ReferenceModel(config).eval()
    patched_model.load_state_dict(reference_model.state_dict())
    install_rollback_state(patched_model)

    reference_net = _delta_net(reference_model)
    patched_net = _delta_net(patched_model)
    rollback = patched_net._deltaforge_rollback

    long_seq = rollback.max_recorded_steps + 8
    hidden_states = torch.randn(1, long_seq, config.hidden_size)
    reference_cache = _fresh_cache(config)
    patched_cache = _fresh_cache(config)

    with torch.no_grad():
        reference_out = reference_net(hidden_states, reference_cache)
        patched_out = patched_net(hidden_states, patched_cache)

    assert rollback.states == [], f"recorded {len(rollback.states)} states for a prefill"
    assert rollback.conv_windows == [], f"recorded {len(rollback.conv_windows)} windows for a prefill"

    torch.testing.assert_close(reference_out, patched_out, rtol=1e-5, atol=1e-6)
    torch.testing.assert_close(reference_cache.recurrent, patched_cache.recurrent, rtol=1e-5, atol=1e-6)
    torch.testing.assert_close(reference_cache.conv, patched_cache.conv, rtol=1e-5, atol=1e-6)


def test_a_verify_length_forward_still_records_every_step():
    """The bound must not switch off the thing the kernel exists for.

    A test that only checked the prefill path would pass on a module that recorded nothing
    ever -- and then `keep()` would raise on the first rejected draft, on a rented card.
    """
    config = tiny_config()
    torch.manual_seed(0)
    model = ReferenceModel(config).eval()
    install_rollback_state(model)
    net = _delta_net(model)
    net._deltaforge_rollback.max_recorded_steps = 3

    with torch.no_grad():
        net(torch.randn(1, 3, config.hidden_size), _fresh_cache(config))

    assert len(net._deltaforge_rollback.states) == 4, "a 3-token verify has 4 states: before, and one per token"
    assert len(net._deltaforge_rollback.conv_windows) == 4


def test_the_loop_releases_each_cycle_s_versions_once_it_has_committed():
    """Rental 53 held 20.77 GiB *after* a failed slot released.

    `RollbackState` kept the last forward's states and windows referenced for the rest of
    the candidate's life, so every slot after a failure started against a lower ceiling.
    Asserted through the loop rather than through `keep`, because committing a cycle is the
    event that ends the record's life -- `keep` only copies out of it, and the property
    tests above read the record through `keep` on purpose.
    """
    from ..speculative import FixedTokenDrafter, install_speculative_loop

    config = tiny_config()
    torch.manual_seed(0)
    model = ReferenceModel(config).eval()
    install_rollback_state(model)
    install_speculative_loop(model, FixedTokenDrafter(0), block_size=2)

    prompt = torch.randint(0, config.vocab_size, (1, 4))
    with torch.no_grad():
        model.decode_loop(model, prompt, 6, model.new_cache(1, 32))

    for state in rollback_states(model):
        assert state.states == [], "a committed cycle must not keep its versions alive"
        assert state.conv_windows == []


def test_installing_the_loop_bounds_recording_to_the_block_it_was_built_with():
    """The bound has to arrive from the loop, or it defaults to something merely safe.

    `install_rollback_state` cannot know `k` -- the two are separate kernels, installed by
    name in the order the hypothesis lists them -- so the loop's installer is what lowers
    it. A default that is merely safe would record 17 steps for a k=2 verify and let
    `keep()` index versions no cycle asked for.
    """
    from ..speculative import FixedTokenDrafter, install_speculative_loop

    config = tiny_config()
    model = ReferenceModel(config).eval()
    install_rollback_state(model)
    install_speculative_loop(model, FixedTokenDrafter(0), block_size=2)

    for state in rollback_states(model):
        assert state.max_recorded_steps == 3, "a k=2 verify is 3 tokens"
        assert not state.records(4), "4 tokens is longer than any verify this loop runs"
        assert state.records(3)
