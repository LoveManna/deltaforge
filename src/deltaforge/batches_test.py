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

from .batch import scoped_registry
from .batches import BATCH_001, BATCH_002, BATCH_003, BATCH_004, BATCHES, get_batch
from .config import tiny_config
from .kernels import REGISTRY
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


def test_004_byte_shares_agree_with_the_byte_model():
    """A manifest's byte_share and its weight_bits must describe the same candidate.

    `byte_share` in this repo is the share of per-token bytes a hypothesis *attacks*, not
    the share it saves -- 012 carried 0.779 for the layer projections it quantised, not the
    0.390 it removed. Deriving it from the same arithmetic the bench now uses stops the two
    drifting.
    """
    from .config import qwen3_5_4b_config
    from .harness.bytes_model import decode_bytes_per_token

    config = qwen3_5_4b_config()
    full = decode_bytes_per_token(config, weight_bits={}, context_length=2048)
    for hyp in BATCH_004:
        if not hyp.weight_bits:
            continue
        # Zero-width stand-in: bits=0 removes the attacked regions entirely, so the
        # difference from bf16 is exactly the bytes those regions contribute.
        without = decode_bytes_per_token(
            config, weight_bits=dict.fromkeys(hyp.weight_bits, 0), context_length=2048
        )
        assert abs(hyp.byte_share - (full - without) / full) < 0.01, hyp.slug


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
