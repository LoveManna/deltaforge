"""Manifest tests, and the structural install check for every hypothesis in a batch.

The install check is the highest-value test in this repo right now. A hypothesis whose
installer silently patches nothing produces a candidate identical to the reference,
measures 1.00, and is recorded as a well-behaved null result — indistinguishable from a
real finding unless something asserts otherwise. Under the old workflow that cost a
rental to discover. Here it costs a laptop test.

Triton is not present in the CPU environment, so these exercise *installation* — which
class each module ends up with — not numerics. The numerics are the GPU gate's job and
run on the rented box.
"""

from __future__ import annotations

import pytest
import torch

from .batch import scoped_registry
from .batches import (
    BATCH_001,
    BATCH_002,
    BATCH_003,
    BATCH_004,
    BATCH_005,
    BATCH_006,
    BATCH_007,
    BATCHES,
    get_batch,
)
from .config import tiny_config
from .kernels import CHECK_BUILDERS, REGISTRY
from .model import apply_champions
from .reference import ReferenceModel


def module_classes(model) -> dict[str, str]:
    """Every submodule's class name, keyed by path. Two of these differ iff a patch landed."""
    return {name: type(module).__name__ for name, module in model.named_modules()}


@pytest.fixture
def model():
    return ReferenceModel(tiny_config())


def test_batch_001_has_between_seven_and_twelve_hypotheses():
    # The batch exists to amortise a ~15-minute fixed rental cost. Fewer than 7 does not
    # justify it; more than 12 will not fit the 120-minute session gate.
    assert 7 <= len(BATCH_001) <= 12


def test_batch_001_starts_with_the_calibration_slot():
    """A broken harness must be discovered in three minutes, not at the end of a rental."""
    assert BATCH_001.hypotheses[0].is_identity
    assert BATCH_001.calibration_slug == "000-identity"


def test_every_hypothesis_names_kernels_that_exist():
    for hyp in BATCH_001:
        for name in hyp.kernels:
            assert name in REGISTRY, f"{hyp.slug!r} names unregistered kernel {name!r}"


def test_every_hypothesis_has_an_installer():
    from .model import INSTALLERS

    for hyp in BATCH_001:
        for name in hyp.kernels:
            assert name in INSTALLERS, f"{hyp.slug!r} names {name!r}, which has no installer"


def test_every_hypothesis_builds_a_scoped_registry():
    for hyp in BATCH_001:
        scoped = scoped_registry(hyp, REGISTRY)
        assert len(scoped) == len(hyp.kernels)


@pytest.mark.parametrize("hypothesis", list(BATCH_001), ids=lambda h: h.slug)
def test_every_hypothesis_installs_and_actually_changes_the_model(hypothesis, model):
    """The check that stops a no-op candidate being recorded as a null result."""
    before = module_classes(model)

    applied = apply_champions(model, scoped_registry(hypothesis, REGISTRY))

    after = module_classes(model)
    if hypothesis.is_identity:
        assert applied == ()
        assert after == before, "the identity champion must leave the model untouched"
    else:
        assert set(applied) == set(hypothesis.kernels)
        assert after != before, (
            f"{hypothesis.slug!r} installed {applied} but changed no module class. A "
            "candidate identical to the reference measures 1.00 and would be recorded as "
            "a null result rather than as the bug it is."
        )


@pytest.mark.parametrize("hypothesis", list(BATCH_001), ids=lambda h: h.slug)
def test_installing_a_hypothesis_is_idempotent(hypothesis, model):
    registry = scoped_registry(hypothesis, REGISTRY)
    apply_champions(model, registry)
    once = module_classes(model)

    apply_champions(model, registry)

    assert module_classes(model) == once


def test_each_hypothesis_patches_a_distinct_set_of_modules():
    """Two hypotheses that patch identically are the same experiment run twice.

    002 and 003 both install the same Triton RMSNorm and differ only in *where*; if that
    distinction ever collapsed, the batch would silently spend two slots measuring one
    thing and report them as independent results.
    """
    patched: dict[str, frozenset[str]] = {}
    for hyp in BATCH_001:
        if hyp.is_identity:
            continue
        model = ReferenceModel(tiny_config())
        before = module_classes(model)
        apply_champions(model, scoped_registry(hyp, REGISTRY))
        after = module_classes(model)
        patched[hyp.slug] = frozenset(k for k in after if after[k] != before[k])

    assert patched["002-rmsnorm-only"] != patched["003-qk-norm-triton"]
    for slug, sites in patched.items():
        assert sites, f"{slug!r} patched nothing"

    seen: dict[frozenset[str], str] = {}
    for slug, sites in patched.items():
        # 006 and 008 both replace the whole attention module and so patch the same sites;
        # that is intended, and 008's docstring says why it is the control for 006.
        if slug in ("006-gqa-no-expand", "008-flash-decode-splitkv"):
            continue
        assert sites not in seen, f"{slug!r} and {seen[sites]!r} patch identical sites"
        seen[sites] = slug


def test_002_and_003_touch_different_norms():
    """002 is the hidden-size norms, 003 is the head-dim norms. Neither may leak."""
    hidden = ReferenceModel(tiny_config())
    before = module_classes(hidden)
    apply_champions(hidden, scoped_registry(BATCH_001.get("002-rmsnorm-only"), REGISTRY))
    hidden_sites = {k for k in module_classes(hidden) if module_classes(hidden)[k] != before[k]}

    qk = ReferenceModel(tiny_config())
    apply_champions(qk, scoped_registry(BATCH_001.get("003-qk-norm-triton"), REGISTRY))
    qk_sites = {k for k in module_classes(qk) if module_classes(qk)[k] != before[k]}

    assert not any("q_norm" in s or "k_norm" in s for s in hidden_sites)
    assert qk_sites and all("q_norm" in s or "k_norm" in s for s in qk_sites)


def test_007_only_touches_linear_attention_layers():
    model = ReferenceModel(tiny_config())
    before = module_classes(model)
    apply_champions(model, scoped_registry(BATCH_001.get("007-gated-delta-fused-step"), REGISTRY))
    after = module_classes(model)

    changed = {k for k in after if after[k] != before[k]}
    assert changed and all("linear_attn" in s for s in changed)


def test_predictions_are_registered_for_every_slot():
    """The prediction is the result. A slot without one cannot be scored."""
    for hyp in BATCH_001:
        assert hyp.prediction
        assert len(hyp.rationale) > 80, f"{hyp.slug!r} has a rationale too thin to be a claim"


def test_only_the_highest_byte_share_slot_predicts_a_win():
    """A batch that predicted wins everywhere would not be a prediction, it would be hope."""
    winners = [h for h in BATCH_001 if h.prediction == "win"]
    assert [h.slug for h in winners] == ["006-gqa-no-expand"]
    others = [h.byte_share for h in BATCH_001 if h.prediction != "win"]
    assert winners[0].byte_share > max(others)


def test_get_batch_rejects_an_unknown_id():
    with pytest.raises(SystemExit, match="unknown batch"):
        get_batch("nope")


def test_get_batch_returns_the_manifest():
    assert get_batch("001-calibration") is BATCH_001
    assert "001-calibration" in BATCHES


def test_a_batch_holds_seven_to_twelve_hypotheses_unless_it_is_calibrating():
    """The floor amortises a rental's fixed cost across measurements.

    A calibration batch's product IS that cost, measured, so the argument for the floor
    cannot apply to it — and rental 22 showed the cost is not what `docs/BATCHES.md`
    assumed.
    """
    assert 7 <= len(BATCH_001) <= 12
    assert BATCH_001.is_calibration is False

    assert len(BATCH_002) < 7
    assert BATCH_002.is_calibration is True


def test_batch_002_opens_with_the_identity_champion():
    assert BATCH_002.hypotheses[0].is_identity
    assert BATCH_002.calibration_slug == "000-identity"


def test_batch_002_can_score_a_kernel_hypothesis():
    """Calibration is necessary and is not a result. A rental that measures only the
    identity slot has proved the harness works and scored no hypothesis."""
    assert sum(1 for h in BATCH_002 if not h.is_identity) >= 1


def test_batch_002_is_registered_and_fetchable():
    assert get_batch("002-compile-cost") is BATCH_002


# -- batch 003: weight-only quantisation ---------------------------------------------


def test_batch_003_is_a_full_batch_and_opens_with_calibration():
    assert 7 <= len(BATCH_003) <= 12
    assert BATCH_003.hypotheses[0].is_identity
    assert BATCH_003.calibration_slug == "000-identity"
    assert get_batch("003-int8-weight-only") is BATCH_003


def test_batch_003_names_registered_kernels_with_installers():
    from .model import INSTALLERS

    for hyp in BATCH_003:
        for name in hyp.kernels:
            assert name in REGISTRY, f"{hyp.slug!r} names unregistered kernel {name!r}"
            assert name in INSTALLERS, f"{hyp.slug!r} names {name!r}, which has no installer"


@pytest.mark.parametrize("hypothesis", list(BATCH_003), ids=lambda h: h.slug)
def test_every_003_hypothesis_installs_and_actually_changes_the_model(hypothesis, model):
    before = module_classes(model)

    applied = apply_champions(model, scoped_registry(hypothesis, REGISTRY))

    after = module_classes(model)
    if hypothesis.is_identity:
        assert applied == ()
        assert after == before
    else:
        assert set(applied) == set(hypothesis.kernels)
        assert after != before, f"{hypothesis.slug!r} installed {applied} but changed no module class"


@pytest.mark.parametrize("hypothesis", list(BATCH_003), ids=lambda h: h.slug)
def test_installing_a_003_hypothesis_is_idempotent(hypothesis, model):
    registry = scoped_registry(hypothesis, REGISTRY)
    apply_champions(model, registry)
    once = module_classes(model)

    apply_champions(model, registry)

    assert module_classes(model) == once


def test_the_003_slots_form_a_dose_response_ladder():
    """011, 012 and 013 are the same kernel over increasing shares of the weight stream.

    That is what makes them a test of the *mechanism* rather than three separate results:
    if a larger share does not buy a larger win, the win is not coming from bytes. Two of
    them accidentally covering the same sites would erase the ladder and nothing else
    would notice, because each would still return a plausible ratio.
    """
    sites = {}
    for slug in ("011-int8-mlp", "012-int8-all-linear", "013-int8-full"):
        model = ReferenceModel(tiny_config())
        before = module_classes(model)
        apply_champions(model, scoped_registry(BATCH_003.get(slug), REGISTRY))
        after = module_classes(model)
        sites[slug] = frozenset(k for k in after if after.get(k) != before.get(k))

    assert sites["011-int8-mlp"] < sites["012-int8-all-linear"] < sites["013-int8-full"]

    ladder = ("011-int8-mlp", "012-int8-all-linear", "013-int8-full")
    shares = [BATCH_003.get(slug).byte_share for slug in ladder]
    assert shares == sorted(shares)


def test_the_003_controls_are_not_predicted_to_win():
    """A batch predicting a win everywhere is not a prediction, it is hope — and these two
    are in the batch precisely to be the things a win is measured against."""
    assert BATCH_003.get("009-gemv-bf16-control").prediction == "inconclusive"
    assert BATCH_003.get("010-int8-dequant-torch").prediction == "loss"


def test_the_bf16_control_attacks_no_bytes_at_all():
    """Its whole point is that it moves exactly the bytes the baseline moves."""
    assert BATCH_003.get("009-gemv-bf16-control").byte_share == 0.0


def test_every_quantised_003_slot_registers_its_approximate_thresholds():
    """A quantised candidate cannot match bf16 tokens, so it is gated on agreement and KL.

    Those bars belong in this file, committed before the rental, for the same reason the
    prediction does — and `Hypothesis.__post_init__` refuses an approximate slot without
    them, so this test is really asserting that the *right* slots are approximate.
    """
    for hyp in BATCH_003:
        quantised = any(name.startswith(("int8_", "int4_")) for name in hyp.kernels)
        if quantised:
            assert hyp.correctness == "approximate", f"{hyp.slug!r} quantises but is gated exactly"
            assert hyp.top1_threshold is not None and hyp.kl_threshold is not None
        else:
            assert hyp.correctness == "exact", f"{hyp.slug!r} computes the same function; gate it exactly"


def test_the_int4_slot_carries_the_loosest_bars_and_runs_last():
    """Riskiest last, and honest about why it is riskiest."""
    assert BATCH_003.hypotheses[-1].slug == "014-int4-full"
    int4 = BATCH_003.get("014-int4-full")
    int8 = BATCH_003.get("013-int8-full")
    assert int4.top1_threshold < int8.top1_threshold
    assert int4.kl_threshold > int8.kl_threshold


def test_003_predictions_are_registered_with_real_rationales():
    for hyp in BATCH_003:
        assert hyp.prediction
        assert len(hyp.rationale) > 80, f"{hyp.slug!r} has a rationale too thin to be a claim"


def test_only_batches_001_to_003_carry_the_pre_2026_09_17_exact_gate():
    """`exact` asserts bit-identity, which only the candidate installing nothing has.

    Batches 001-003 registered it for kernels that merely compute the same *function*, and
    it cost `009-gemv-bf16-control` its slot: one bf16 ULP of reordered accumulation flips
    an argmax on this model, and it matched 1 of 5 prompts. Those manifests keep the gate
    they actually ran under -- rewriting it would falsify the record as surely as
    rewriting a prediction would -- and nothing written after them may use it.
    """
    for batch_id, batch in BATCHES.items():
        if batch_id in ("001-calibration", "002-compile-cost", "003-int8-weight-only"):
            continue
        for hypothesis in batch:
            assert not hypothesis.historical_exact_gate, (
                f"{batch_id}/{hypothesis.slug} claims a gate this project has retired"
            )


# -- batch 004 ------------------------------------------------------------------------


def test_batch_004_is_a_full_batch_and_opens_with_calibration():
    assert 7 <= len(BATCH_004) <= 12
    assert not BATCH_004.is_calibration
    assert BATCH_004.hypotheses[0].is_identity
    assert BATCH_004.calibration_slug == "000-identity"
    assert get_batch("004-bandwidth-bound-gemv") is BATCH_004


def test_batch_004_names_registered_kernels_with_installers():
    from .model import INSTALLERS

    for hyp in BATCH_004:
        for name in hyp.kernels:
            assert name in REGISTRY, f"{hyp.slug!r} names unregistered kernel {name!r}"
            assert name in INSTALLERS, f"{hyp.slug!r} names {name!r}, which has no installer"


@pytest.mark.parametrize("hypothesis", list(BATCH_004), ids=lambda h: h.slug)
def test_every_004_hypothesis_installs_and_actually_changes_the_model(hypothesis, model):
    before = module_classes(model)

    applied = apply_champions(model, scoped_registry(hypothesis, REGISTRY))

    after = module_classes(model)
    if hypothesis.is_identity:
        assert applied == ()
        assert after == before
    else:
        assert set(applied) == set(hypothesis.kernels)
        assert after != before, f"{hypothesis.slug!r} installed {applied} but changed no module class"


@pytest.mark.parametrize("hypothesis", list(BATCH_004), ids=lambda h: h.slug)
def test_installing_a_004_hypothesis_is_idempotent(hypothesis, model):
    registry = scoped_registry(hypothesis, REGISTRY)
    apply_champions(model, registry)
    once = module_classes(model)

    apply_champions(model, registry)

    assert module_classes(model) == once


def test_batch_004_gates_every_quantised_slot_on_the_bf16_control():
    """Batch 003 spent five slots on variants its first slot had already settled."""
    for hyp in BATCH_004:
        if hyp.weight_bits:
            assert hyp.requires is not None, f"{hyp.slug!r} would run regardless of the control"
            assert hyp.requires.floor >= 0.56


def test_the_bf16_control_is_not_predicted_to_win():
    control = BATCH_004.get("015-tiled-gemv-bf16")

    assert control.prediction == "inconclusive"
    assert control.weight_bits == {}
    assert control.byte_share == 0.0, "it moves not one byte fewer than the baseline"


def test_every_004_slot_is_gated_approximately_except_the_identity():
    for hyp in BATCH_004:
        assert (hyp.correctness == "exact") == hyp.is_identity
        assert not hyp.historical_exact_gate


def _attacked_share(config, hyp, *, compiled: bool) -> float:
    """The share of per-token bytes a manifest's ``weight_bits`` actually names.

    Zero-width stand-in: ``bits=0`` removes the attacked regions entirely, so the
    difference from bf16 is exactly the bytes those regions contribute.
    """
    from .harness.bytes_model import decode_bytes_per_token

    full = decode_bytes_per_token(config, weight_bits={}, context_length=2048, compiled=compiled)
    without = decode_bytes_per_token(
        config, weight_bits=dict.fromkeys(hyp.weight_bits, 0), context_length=2048, compiled=compiled
    )
    return (full - without) / full


def test_004_byte_shares_agree_with_the_eager_byte_model_they_were_written_against():
    """A manifest's byte_share and its weight_bits must describe the same candidate.

    `byte_share` in this repo is the share of per-token bytes a hypothesis *attacks*, not
    the share it saves -- 012 carried 0.779 for the layer projections it quantised, not the
    0.390 it removed. Deriving it from the same arithmetic the bench uses stops the two
    drifting.

    **Batches 001-004 were written against the eager total, 9158.23 MB/token**, and are
    checked against it here rather than rewritten. Rental 38 then read the generated code
    and found inductor folding the GQA expansion away, so the compiled columns move 8587.80
    and `decode_bytes_per_token` defaults to that. Restating a share a rental already ran
    under would falsify the record in exactly the way restating a prediction would; batch
    005 is written against the corrected denominator and checked against it below.
    """
    from .config import qwen3_5_4b_config

    config = qwen3_5_4b_config()
    for hyp in BATCH_004:
        if not hyp.weight_bits:
            continue
        assert abs(hyp.byte_share - _attacked_share(config, hyp, compiled=False)) < 0.01, hyp.slug


def test_005_byte_shares_agree_with_the_compiled_byte_model():
    """Batch 005's shares are against what the *compiled* column moves, which is the score."""
    from .config import qwen3_5_4b_config

    config = qwen3_5_4b_config()
    for hyp in BATCH_005:
        if not hyp.weight_bits:
            continue
        assert abs(hyp.byte_share - _attacked_share(config, hyp, compiled=True)) < 0.01, hyp.slug


def test_the_004_fp8_slots_form_a_dose_response_ladder():
    """018, 016 and 017 are one kernel over increasing shares of the weight stream.

    A larger share buying a larger win is what distinguishes "the mechanism works" from
    "something else moved". Two of them accidentally covering the same sites would erase
    the ladder and nothing else would notice: each would still return a plausible ratio.
    """
    ladder = [BATCH_004.get(slug) for slug in ("018-fp8-mlp", "016-fp8-all-linear", "017-fp8-full")]
    shares = [hyp.byte_share for hyp in ladder]

    assert shares == sorted(shares), shares
    assert len(set(shares)) == 3


def test_the_int8_slot_attacks_exactly_the_sites_the_fp8_slot_does():
    """019 minus 016 is the int8 conversion tax and nothing else. Batch 003 measured that
    tax at 1.438x the time of bf16; if the two slots differed in sites it would measure
    something else entirely."""
    fp8 = BATCH_004.get("016-fp8-all-linear")
    int8 = BATCH_004.get("019-int8-all-linear")

    assert fp8.weight_bits == int8.weight_bits
    assert fp8.byte_share == int8.byte_share
    assert fp8.replaces == int8.replaces


def test_the_int4_slot_waits_for_fp8_to_show_the_pipeline_is_clear():
    """int4's extra unpack is only worth trying once 8 bits has actually won something."""
    int4 = BATCH_004.get("020-int4-full")

    assert int4.requires.slug == "016-fp8-all-linear"
    assert int4.requires.floor >= 1.0
    assert BATCH_004.hypotheses[-1] is int4, "the riskiest kernel runs last"


def test_every_004_bar_is_derived_from_a_measured_point():
    """Batch 003's bars came from priors about int8 being mild. It is mild; it still flips
    3% of argmaxes on this checkpoint. Every bar here is written as a count out of the 264
    positions batch 003 actually scored, so the number a reader checks is a number of
    tokens rather than a fraction that looks precise."""
    for hyp in BATCH_004:
        if hyp.correctness != "approximate":
            continue
        assert hyp.correctness_positions == 264
        assert hyp.kl_threshold is not None
        flips = round((1.0 - hyp.top1_threshold) * 264)
        assert 0 < flips < 264, hyp.slug


def test_004_predictions_are_registered_with_real_rationales():
    for hyp in BATCH_004:
        assert hyp.prediction in ("win", "loss", "inconclusive", "identity")
        assert len(hyp.rationale) > 80, f"{hyp.slug!r} has a label, not a rationale"


# ======================================================================================
# Batch 005 — the dispatch path, and the one site with parallelism to spare
# ======================================================================================


def test_batch_005_is_a_full_batch_and_opens_with_calibration():
    assert 7 <= len(BATCH_005) <= 12
    assert not BATCH_005.is_calibration
    assert BATCH_005.hypotheses[0].is_identity
    assert BATCH_005.calibration_slug == "000-identity"
    assert get_batch("005-launch-and-head") is BATCH_005


def test_batch_005_names_registered_kernels_with_installers():
    from .model import INSTALLERS

    for hyp in BATCH_005:
        for name in hyp.kernels:
            assert name in REGISTRY, f"{hyp.slug!r} names unregistered kernel {name!r}"
            assert name in INSTALLERS, f"{hyp.slug!r} names {name!r}, which has no installer"


@pytest.mark.parametrize("hypothesis", list(BATCH_005), ids=lambda h: h.slug)
def test_every_005_hypothesis_installs_and_actually_changes_the_model(hypothesis, model):
    before = module_classes(model)

    applied = apply_champions(model, scoped_registry(hypothesis, REGISTRY))

    after = module_classes(model)
    if hypothesis.is_identity:
        assert applied == ()
        assert after == before
    else:
        assert set(applied) == set(hypothesis.kernels)
        assert after != before, f"{hypothesis.slug!r} installed {applied} but changed no module class"


@pytest.mark.parametrize("hypothesis", list(BATCH_005), ids=lambda h: h.slug)
def test_installing_a_005_hypothesis_is_idempotent(hypothesis, model):
    registry = scoped_registry(hypothesis, REGISTRY)
    apply_champions(model, registry)
    once = module_classes(model)

    apply_champions(model, registry)

    assert module_classes(model) == once


def test_composing_two_root_class_patches_keeps_both(model):
    """The bug this batch had to fix before it could run, and the reason it has a test.

    `tiled_int4_head` and `static_decode_cache` both work by replacing the *root* model's
    class. Anchored at `ReferenceModel`, whichever installed second discarded the first —
    and nothing would have caught it: `_build_candidate` only asks whether any module class
    changed, which is still true, so slots 026 and 027 would have run, compiled, passed
    correctness and returned a plausible ratio for a candidate holding one kernel of the
    two it claimed. Each factory now subclasses whatever the model already is.
    """
    composed = BATCH_005.get("026-int4-head-static-cache")

    apply_champions(model, scoped_registry(composed, REGISTRY))

    assert hasattr(model, "tiled_lm_head"), "the head install was discarded"
    assert getattr(model, "_deltaforge_static_cache", False), "the static-cache install was discarded"
    assert type(model).project_logits is not ReferenceModel.project_logits
    assert type(model).new_cache is not ReferenceModel.new_cache


def test_the_static_cache_slot_marks_every_cache_tensor(model):
    """63 of 64 is worth exactly as much as 0: inductor's mutation check is all-or-nothing
    over the region, so one unmarked tensor skips cudagraphs for the whole decode step."""
    from .kernels.static_cache import install_static_decode_cache

    install_static_decode_cache(model)
    cache = model.new_cache(1, 8)

    tensors = [value for layer in cache.layers for value in vars(layer).values() if torch.is_tensor(value)]
    assert tensors, "the fixture has no decode cache to mark"
    assert all(getattr(t, "_dynamo_static_input_type", None) is not None for t in tensors)
    assert model.deltaforge_static_cache_tensors == len(tensors)


def test_the_head_slots_leave_every_layer_projection_alone(model):
    """022-024 are the first slots in this project to install on exactly one site.

    Two rentals measured a GEMV on all 248 layer projections at once and reported one
    aggregate byte rate, which cannot tell a kernel that is slow everywhere from one that
    is slow where there is no parallelism to have. If a head slot also patched the layer
    linears it would reproduce that confound and nothing else would notice.
    """
    from torch import nn

    from .kernels.tiled_gemv import TiledLMHead

    linears_before = {id(m) for m in model.layers.modules() if isinstance(m, nn.Linear)}

    apply_champions(model, scoped_registry(BATCH_005.get("022-int4-head"), REGISTRY))

    assert isinstance(model.tiled_lm_head, TiledLMHead)
    assert all(type(m) is nn.Linear for m in model.layers.modules() if id(m) in linears_before)


def test_the_fused_conv_slot_patches_every_linear_attention_layer_and_nothing_else(model):
    from .reference import GatedAttention, GatedDeltaNet

    apply_champions(model, scoped_registry(BATCH_005.get("025-fused-causal-conv"), REGISTRY))

    nets = [m for m in model.modules() if isinstance(m, GatedDeltaNet)]
    assert nets, "the fixture has no linear-attention layers"
    assert all(type(net) is not GatedDeltaNet for net in nets)
    assert all(type(m) is GatedAttention for m in model.modules() if isinstance(m, GatedAttention))


def test_both_005_compositions_are_gated_on_the_cudagraph_slot():
    """A composition whose novel ingredient did not fire is a slot re-measuring a number
    the batch already has. Batch 004's preconditions saved 16 billed minutes doing this."""
    for slug in ("026-int4-head-static-cache", "027-conv-head-static-cache"):
        hyp = BATCH_005.get(slug)
        assert hyp.requires is not None, slug
        assert hyp.requires.slug == "021-static-cache-cudagraphs"
        assert hyp.requires.floor >= 1.02, "a floor at or below 1.0 is not a win"


def test_every_005_slot_is_gated_approximately_except_the_identity():
    for hyp in BATCH_005:
        assert (hyp.correctness == "exact") == hyp.is_identity
        assert not hyp.historical_exact_gate


def test_the_005_bars_are_counts_out_of_the_positions_that_were_scored():
    """Every bar is a whole number of tokens out of the 264 positions batch 003 scored.

    The one slot allowed zero flips is the CUDA-graph candidate, which changes no
    arithmetic at all: it runs inductor's own kernels in inductor's own order, so a single
    flip there is a defect rather than a dtype.
    """
    for hyp in BATCH_005:
        if hyp.correctness != "approximate":
            continue
        assert hyp.correctness_positions == 264
        assert hyp.kl_threshold is not None
        flips = (1.0 - hyp.top1_threshold) * 264
        assert abs(flips - round(flips)) < 1e-9, hyp.slug
        if hyp.slug == "021-static-cache-cudagraphs":
            assert round(flips) == 0 and hyp.kl_threshold <= 1e-6
        else:
            assert 0 < round(flips) < 264, hyp.slug


def test_the_three_head_slots_attack_the_same_site_at_three_encodings():
    """022/023/024 differ in bit width and dtype and in nothing else, so their ordering is
    readable: int4 ahead means bandwidth-bound at this site, int8 ahead means still
    issue-bound, and 023 minus 024 is the int8 conversion tax on its own."""
    int4, int8, fp8 = (BATCH_005.get(s) for s in ("022-int4-head", "023-int8-head", "024-fp8-head"))

    assert int4.replaces == int8.replaces == fp8.replaces == ("decode_step",)
    assert int4.byte_share == int8.byte_share == fp8.byte_share
    assert int4.weight_bits == {"head": 4}
    assert int8.weight_bits == fp8.weight_bits == {"head": 8}


def test_005_predictions_are_registered_with_real_rationales():
    for hyp in BATCH_005:
        assert hyp.prediction in ("win", "loss", "inconclusive", "identity")
        assert len(hyp.rationale) > 80, f"{hyp.slug!r} has a label, not a rationale"


# -- batch 006: the tile, and the sites that were never grid-starved --------------------


def test_batch_006_is_a_full_batch_and_opens_with_calibration():
    assert 7 <= len(BATCH_006) <= 12
    assert BATCH_006.hypotheses[0].is_identity
    assert BATCH_006.calibration_slug == "000-identity"
    assert get_batch("006-tile-and-sites") is BATCH_006


def test_batch_006_names_registered_kernels_with_installers():
    from .model import INSTALLERS

    for hyp in BATCH_006:
        for name in hyp.kernels:
            assert REGISTRY.get(name) is not None, name
            assert name in INSTALLERS, name
            assert name in CHECK_BUILDERS, f"{name} has no layer-1 checks"


@pytest.mark.parametrize("hypothesis", list(BATCH_006), ids=lambda h: h.slug)
def test_every_006_hypothesis_installs_and_actually_changes_the_model(hypothesis, model):
    before = {name: type(module) for name, module in model.named_modules()}

    applied = apply_champions(model, scoped_registry(hypothesis, REGISTRY))
    after = {name: type(module) for name, module in model.named_modules()}

    if hypothesis.is_identity:
        assert not applied and after == before
    else:
        assert applied, hypothesis.slug
        assert after != before or type(model) is not before[""], hypothesis.slug


@pytest.mark.parametrize("hypothesis", list(BATCH_006), ids=lambda h: h.slug)
def test_installing_a_006_hypothesis_is_idempotent(hypothesis, model):
    apply_champions(model, scoped_registry(hypothesis, REGISTRY))
    once = {name: type(module) for name, module in model.named_modules()}

    apply_champions(model, scoped_registry(hypothesis, REGISTRY))

    assert {name: type(module) for name, module in model.named_modules()} == once


def test_every_006_slot_can_beat_the_incumbent_champion():
    """The session's brief: no slot here is a control whose ceiling is 1.0.

    Batch 004's bf16 GEMV was the most valuable slot in its batch *and* could not win by
    construction. After batch 005 that trade is no longer necessary — the cheapest
    diagnostic available is `028`, which is the champion's own site with a measured tile,
    and it has a 1.1249x ceiling. So every non-calibration slot attacks either bytes or
    dispatch, and `034` is the only one whose byte share is zero.
    """
    incumbent = 1.0791  # 022-int4-head, rental 40
    for hyp in BATCH_006:
        if hyp.is_identity:
            continue
        if hyp.byte_share == 0.0:
            assert hyp.slug == "034-static-cache-cudagraphs", hyp.slug
            continue
        ceiling = 1.0 / (1.0 - hyp.byte_share * (1.0 - 0.2578))
        assert ceiling > incumbent, f"{hyp.slug} cannot reach the champion"


def test_the_006_slots_form_a_site_ladder_over_one_mechanism():
    """028 -> 030 -> 031 -> 032 is group-128 int4 on a growing share of the same bytes.

    Read as a ladder it prices the per-site costs that do not scale with traffic — the
    fusion inductor forfeits, the second launch, split-K's reduction pass — which two
    rentals of one aggregate number could not separate.
    """
    ladder = [
        BATCH_006.get(slug)
        for slug in (
            "028-int4-head-tuned",
            "030-int4-mlp",
            "031-int4-mlp-and-head",
            "032-int4-wide-and-head",
        )
    ]
    shares = [hyp.byte_share for hyp in ladder]
    assert shares == sorted(shares), shares
    assert all(4 in hyp.weight_bits.values() for hyp in ladder)


def test_the_wide_slots_leave_the_32_channel_gates_in_bf16():
    """`in_proj_a` and `in_proj_b` are starved at every tile and worth 0.05% of bytes.

    Including them is what folded a site no tile can fix into batches 003 and 004's one
    aggregate rate. The manifest has to be able to say so, which is why
    `linear_attn_gates` is its own byte region.
    """
    from .harness.bytes_model import WEIGHT_REGIONS

    assert "linear_attn_gates" in WEIGHT_REGIONS
    for slug in ("032-int4-wide-and-head", "033-int4-wide-head-and-conv"):
        bits = BATCH_006.get(slug).weight_bits
        assert bits.get("linear_attn") == 4
        assert "linear_attn_gates" not in bits, f"{slug} credits itself the gates"
        assert "layers" not in bits, f"{slug} must not use the alias: it includes the gates"


def test_the_006_compositions_are_gated_on_a_slot_that_measured_their_mechanism():
    gates = {
        "031-int4-mlp-and-head": ("030-int4-mlp", 1.00),
        "032-int4-wide-and-head": ("030-int4-mlp", 1.00),
        "033-int4-wide-head-and-conv": ("032-int4-wide-and-head", 1.05),
    }
    for slug, (required, floor) in gates.items():
        hyp = BATCH_006.get(slug)
        assert hyp.requires is not None, slug
        assert hyp.requires.slug == required, slug
        assert hyp.requires.floor == floor, slug
        assert len(hyp.requires.reason) > 60, f"{slug}'s floor has a number and no reason"


def test_the_two_slots_that_bank_a_champion_early_are_not_gated():
    """028 and 029 run whatever else happens: 029 is two measured winners composed, and a
    batch that gated it behind an untested mechanism could end with no result at all."""
    for slug in ("028-int4-head-tuned", "029-head-and-conv"):
        assert BATCH_006.get(slug).requires is None, slug


def test_every_006_slot_is_gated_approximately_except_the_identity():
    for hyp in BATCH_006:
        assert (hyp.correctness == "exact") == hyp.is_identity
        assert not hyp.historical_exact_gate


def test_the_006_bars_are_whole_tokens_and_loosen_with_the_perturbation():
    """Bars come from two measured points — `014` at 0.09185 nats over every site and the
    head, `022` at 0.01674 over the head alone — never from priors about int4."""
    for hyp in BATCH_006:
        if hyp.correctness != "approximate":
            continue
        assert hyp.correctness_positions == 264
        flips = (1.0 - hyp.top1_threshold) * 264
        assert abs(flips - round(flips)) < 1e-9, hyp.slug
    quantised = [
        BATCH_006.get(slug)
        for slug in ("028-int4-head-tuned", "030-int4-mlp", "031-int4-mlp-and-head", "032-int4-wide-and-head")
    ]
    kls = [hyp.kl_threshold for hyp in quantised]
    assert kls == sorted(kls), kls
    assert BATCH_006.get("032-int4-wide-and-head").kl_threshold > 0.09185, (
        "the bar must exceed what batch 003 measured while quantising more sites"
    )


def test_006_predictions_are_registered_with_real_rationales():
    for hyp in BATCH_006:
        assert hyp.prediction in ("win", "loss", "inconclusive", "identity")
        assert len(hyp.rationale) > 80, f"{hyp.slug!r} has a label, not a rationale"


# ======================================================================================
# Batch 007 — the champion re-measured, its tile pinned, its wins composed
# ======================================================================================

#: The number every slot in batch 007 is trying to beat: `022-int4-head`, rental 40.
INCUMBENT = 1.0791


def test_batch_007_is_a_full_batch_and_opens_with_calibration():
    assert 7 <= len(BATCH_007) <= 12
    assert not BATCH_007.is_calibration
    assert BATCH_007.hypotheses[0].is_identity
    assert BATCH_007.calibration_slug == "000-identity"
    assert get_batch("007-compose-and-retile") is BATCH_007


def test_batch_007_names_registered_kernels_with_installers_and_checks():
    from .model import INSTALLERS

    for hyp in BATCH_007:
        for name in hyp.kernels:
            assert REGISTRY.get(name) is not None, name
            assert name in INSTALLERS, name
            assert name in CHECK_BUILDERS, f"{name} has no layer-1 checks"


@pytest.mark.parametrize("hypothesis", list(BATCH_007), ids=lambda h: h.slug)
def test_every_007_hypothesis_installs_and_actually_changes_the_model(hypothesis, model):
    before = {name: type(module) for name, module in model.named_modules()}

    applied = apply_champions(model, scoped_registry(hypothesis, REGISTRY))
    after = {name: type(module) for name, module in model.named_modules()}

    if hypothesis.is_identity:
        assert not applied and after == before
    else:
        assert set(applied) == set(hypothesis.kernels), hypothesis.slug
        assert after != before or type(model) is not before[""], hypothesis.slug


@pytest.mark.parametrize("hypothesis", list(BATCH_007), ids=lambda h: h.slug)
def test_installing_a_007_hypothesis_is_idempotent(hypothesis, model):
    apply_champions(model, scoped_registry(hypothesis, REGISTRY))
    once = {name: type(module) for name, module in model.named_modules()}

    apply_champions(model, scoped_registry(hypothesis, REGISTRY))

    assert {name: type(module) for name, module in model.named_modules()} == once


@pytest.mark.parametrize(
    "slug",
    ("040-int4-head-conv-cache", "042-wide-tile-conv-cache"),
)
def test_the_three_way_compositions_keep_all_three_installs(slug, model):
    """Two root-class swaps and a module patch, all of which have to survive each other.

    `026` and `027` were built for rental 40 and declined; `029` was the first composition
    this project ever executed and it lost 20%. The one failure mode that would not show
    up as a bad ratio is an install silently discarded — `_build_candidate` only asks
    whether *any* class changed — so the composed slots assert each ingredient by name.
    """
    from .kernels.fused_causal_conv import _delta_nets
    from .kernels.tiled_gemv import TiledLMHead

    apply_champions(model, scoped_registry(BATCH_007.get(slug), REGISTRY))

    assert isinstance(model.tiled_lm_head, TiledLMHead), "the head install was discarded"
    assert type(model).project_logits is not ReferenceModel.project_logits
    assert getattr(model, "_deltaforge_static_cache", False), "the static-cache install was discarded"
    assert type(model).new_cache is not ReferenceModel.new_cache
    nets = _delta_nets(model)
    assert nets, "the fixture has no linear-attention layer to patch"
    assert all(type(net).__name__ != "GatedDeltaNet" for net in nets), "the conv install was discarded"


def test_a_pinned_tile_does_not_leak_into_the_next_slot(model):
    """The hazard a batch of pinned tiles creates, and the reason `_install_head` clears.

    `_TUNED` is process-global and a batch runs every slot in one process, so a slot that
    pins BLOCK_N=128 would leave it in force for `035-int4-head` — which would then report
    a ratio for a tile its manifest does not name, exactly the way rental 35's eager
    candidates reported ratios for a compilation that never happened.
    """
    from .kernels.tiled_gemv import (
        HEAD_WIDE_SHAPE,
        _heuristic_shape,
        _launch_shape,
        head_shape_key,
    )

    apply_champions(model, scoped_registry(BATCH_007.get("037-int4-head-wide-tile"), REGISTRY))
    kind, n, k = head_shape_key(model.tiled_lm_head)
    assert _launch_shape(n, k, kind) == HEAD_WIDE_SHAPE

    fresh = ReferenceModel(tiny_config())
    apply_champions(fresh, scoped_registry(BATCH_007.get("035-int4-head"), REGISTRY))

    assert _launch_shape(n, k, kind) == _heuristic_shape(n, k), "035 inherited 037's tile"


def test_every_007_slot_can_beat_the_incumbent_champion():
    """The session's brief: no slot here is a control whose ceiling is 1.0.

    Every kernel slot carries the head at 4 bits, whose byte ceiling is 1.1249x, and the
    two extra mechanisms are measured savings on top of it. An 8-bit head would fail this
    test at 1.0799 — 0.0008 above the incumbent — which is why neither `023` nor `024` is
    re-run here however unfinished their business is.
    """
    for hyp in BATCH_007:
        if hyp.is_identity:
            continue
        assert hyp.weight_bits == {"head": 4}, hyp.slug
        ceiling = 1.0 / (1.0 - hyp.byte_share * (1.0 - 0.2578))
        assert ceiling > INCUMBENT, f"{hyp.slug} cannot reach the champion"


def test_the_007_slots_that_re_measure_something_are_not_gated():
    """035, 036, 037, 038 and 039 run whatever else happens.

    The champion's re-measurement is what every other slot is read against, and a batch
    that gated its tail behind one slot would have no result if that slot errored. 039 is
    ungated for the opposite reason: its 2-in-3 outcome is the one this batch most needs
    on disk either way.
    """
    for slug in (
        "035-int4-head",
        "036-int4-head-static-cache",
        "037-int4-head-wide-tile",
        "038-int4-head-deep-pipe",
        "039-int4-head-and-conv",
    ):
        assert BATCH_007.get(slug).requires is None, slug


def test_the_007_compositions_are_gated_on_a_slot_that_measured_their_ingredient():
    gates = {
        "040-int4-head-conv-cache": ("039-int4-head-and-conv", 1.05),
        "041-wide-tile-and-cache": ("037-int4-head-wide-tile", 1.08),
        "042-wide-tile-conv-cache": ("041-wide-tile-and-cache", 1.10),
    }
    for slug, (required, floor) in gates.items():
        hyp = BATCH_007.get(slug)
        assert hyp.requires is not None, slug
        assert hyp.requires.slug == required, slug
        assert hyp.requires.floor == floor, slug
        assert len(hyp.requires.reason) > 60, f"{slug}'s floor has a number and no reason"
    # Every floor is above the incumbent-adjacent band the batch is arguing about, except
    # 039's, which only has to show the conv composition is not the 20% regression again.
    assert BATCH_007.get("041-wide-tile-and-cache").requires.floor > INCUMBENT


def test_every_007_slot_is_gated_approximately_except_the_identity():
    for hyp in BATCH_007:
        assert (hyp.correctness == "exact") == hyp.is_identity
        assert not hyp.historical_exact_gate


def test_the_007_bars_are_022_s_own_bars_everywhere():
    """One perturbation, so one pair of bars: the head at group-128 int4 and nothing else.

    The conv and the static cache are bit-identical (264/264 and 0.0 nats, measured), and
    a pinned tile changes no arithmetic operation — SPLIT_K stays 1, so even the summation
    order is unchanged. So every slot here should reproduce `022`'s 0.9318 and 0.01674
    nats, and a bar that differed between slots would be claiming otherwise.
    """
    bars = {
        (hyp.top1_threshold, hyp.kl_threshold, hyp.correctness_positions)
        for hyp in BATCH_007
        if hyp.correctness == "approximate"
    }
    assert bars == {(240 / 264, 0.06, 264)}
    flips = (1.0 - 240 / 264) * 264
    assert abs(flips - round(flips)) < 1e-9


def test_007_predictions_are_registered_with_real_rationales():
    for hyp in BATCH_007:
        assert hyp.prediction in ("win", "loss", "inconclusive", "identity")
        assert len(hyp.rationale) > 80, f"{hyp.slug!r} has a label, not a rationale"
