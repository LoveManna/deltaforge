"""Tests for the batch model, the outcome arithmetic and the deadline policy.

All of it decidable on a CPU with no checkpoint, which is the point of keeping torch out
of `batch.py`: the parts that decide what a number *means* are the parts worth testing
exhaustively, and they must not need a rented GPU to exercise.
"""

from __future__ import annotations

import pytest

from .batch import (
    COLD_PHASE_ESTIMATES,
    Batch,
    Hypothesis,
    Precondition,
    SlotBudget,
    calibration_holds,
    classify_outcome,
    precondition_holds,
    scoped_registry,
    score_predictions,
    session_fits_one_hypothesis,
)
from .kernels import KernelRegistry, KernelStatus, RegistryError


def make_hypothesis(**overrides) -> Hypothesis:
    base = {
        "slug": "999-test",
        "kernels": ("k1",),
        "category": "B",
        "byte_share": 0.05,
        "mechanism": "moves fewer bytes",
        "prediction": "win",
        "rationale": "because the compiler cannot change the representation",
    }
    base.update(overrides)
    return Hypothesis(**_with_a_gate_it_can_hold(base))


def _with_a_gate_it_can_hold(fields: dict) -> dict:
    """Fill in a gate the hypothesis is allowed to carry, unless the test named one.

    `exact` asserts bit-identity, which only the candidate installing nothing has. So a
    builder that defaults every hypothesis to `exact` would produce an invalid one every
    time it was handed a kernel, and every test using it would fail on the gate rather
    than on what it meant to assert.
    """
    fields = dict(fields)
    fields.setdefault("correctness", "exact" if not fields["kernels"] else "approximate")
    if fields["correctness"] == "approximate":
        fields.setdefault("top1_threshold", 0.9)
        fields.setdefault("kl_threshold", 0.01)
    return fields


def make_registry() -> KernelRegistry:
    registry = KernelRegistry()
    registry.register("k1", impl=lambda: None, replaces="rms_norm", status=KernelStatus.RETIRED)
    registry.register("k2", impl=lambda: None, replaces="swiglu_mlp", status=KernelStatus.RETIRED)
    registry.register("k3", impl=lambda: None, replaces="rms_norm", status=KernelStatus.RETIRED)
    return registry


# -- Hypothesis validation --------------------------------------------------------------


def test_prediction_must_be_known():
    with pytest.raises(ValueError, match="prediction must be one of"):
        make_hypothesis(prediction="probably")


def test_byte_share_is_a_fraction_not_a_percentage():
    # 6.23% is 0.0623. Writing 6.23 would silently claim the hypothesis attacks 623% of
    # per-token bytes and would sort the backlog wrongly.
    with pytest.raises(ValueError, match="fraction"):
        make_hypothesis(byte_share=6.23)


def test_a_hypothesis_installing_nothing_must_predict_identity():
    with pytest.raises(ValueError, match="identity champion"):
        make_hypothesis(kernels=(), prediction="win")


def test_identity_prediction_requires_installing_nothing():
    with pytest.raises(ValueError, match="bit-identical"):
        make_hypothesis(kernels=("k1",), prediction="identity")


def test_identity_hypothesis_is_valid_and_flagged():
    hyp = make_hypothesis(kernels=(), prediction="identity")
    assert hyp.is_identity


def test_mechanism_is_required():
    with pytest.raises(ValueError, match="not a hypothesis"):
        make_hypothesis(mechanism="   ")


def test_rationale_is_required_because_the_prediction_is_the_result():
    with pytest.raises(ValueError, match="unexplained one is worth nothing"):
        make_hypothesis(rationale="")


# -- Batch ------------------------------------------------------------------------------


def test_batch_rejects_duplicate_slugs():
    hyp = make_hypothesis()
    with pytest.raises(ValueError, match="twice"):
        Batch(batch_id="b", hypotheses=(hyp, hyp))


def test_batch_rejects_empty():
    with pytest.raises(ValueError, match="empty"):
        Batch(batch_id="b", hypotheses=())


def test_batch_preserves_manifest_order():
    # Order is load-bearing: calibration first so a broken harness costs three minutes,
    # riskiest last so the cheap results are already on disk when one fails.
    a = make_hypothesis(slug="000-identity", kernels=(), prediction="identity")
    b = make_hypothesis(slug="001-cheap")
    c = make_hypothesis(slug="008-risky")
    batch = Batch(batch_id="b", hypotheses=(a, b, c))
    assert [h.slug for h in batch] == ["000-identity", "001-cheap", "008-risky"]
    assert len(batch) == 3


def test_batch_finds_its_calibration_slot():
    a = make_hypothesis(slug="000-identity", kernels=(), prediction="identity")
    b = make_hypothesis(slug="001-cheap")
    assert Batch(batch_id="b", hypotheses=(a, b)).calibration_slug == "000-identity"
    assert Batch(batch_id="b", hypotheses=(b,)).calibration_slug is None


def test_batch_get_raises_on_unknown_slug():
    batch = Batch(batch_id="b", hypotheses=(make_hypothesis(),))
    with pytest.raises(KeyError):
        batch.get("nope")


# -- scoped_registry --------------------------------------------------------------------


def test_scoped_registry_promotes_only_the_named_kernels():
    source = make_registry()
    scoped = scoped_registry(make_hypothesis(kernels=("k1",)), source)
    assert set(scoped.champions()) == {"rms_norm"}
    assert scoped.champion("rms_norm").name == "k1"


def test_scoped_registry_does_not_mutate_the_source():
    # The whole reason scoped registries exist: a mutation here would leak into the next
    # slot, and a slot that inherited the previous slot's kernels would report a
    # plausible number for the wrong candidate.
    source = make_registry()
    scoped_registry(make_hypothesis(kernels=("k1", "k2")), source)
    assert source.get("k1").status is KernelStatus.RETIRED
    assert source.get("k2").status is KernelStatus.RETIRED
    assert source.champions() == {}


def test_scoped_registry_is_empty_for_the_identity_champion():
    scoped = scoped_registry(make_hypothesis(kernels=(), prediction="identity"), make_registry())
    assert scoped.champions() == {}
    assert len(scoped) == 0


def test_scoped_registry_rejects_two_kernels_replacing_the_same_operation():
    # Caught here rather than at benchmark time, on the ground, before the money.
    with pytest.raises(RegistryError, match="cannot install"):
        scoped_registry(make_hypothesis(kernels=("k1", "k3")), make_registry())


def test_scoped_registry_rejects_an_unknown_kernel_name():
    with pytest.raises(RegistryError, match="no kernel registered"):
        scoped_registry(make_hypothesis(kernels=("typo",)), make_registry())


def test_scoped_registry_records_the_owning_hypothesis():
    scoped = scoped_registry(make_hypothesis(slug="006-gqa", kernels=("k1",)), make_registry())
    assert scoped.get("k1").hypothesis == "006-gqa"


# -- classify_outcome -------------------------------------------------------------------


def test_a_wrong_candidate_is_incorrect_however_fast_it_was():
    assert classify_outcome(2.5, 0.01, correctness_passed=False) == "incorrect"


def test_margin_beyond_the_noise_band_is_a_win():
    assert classify_outcome(1.10, 0.02, correctness_passed=True) == "win"


def test_margin_inside_the_noise_band_is_inconclusive_not_a_win():
    # AGENT.md section 6: recording a noise-band result as a win is how a leaderboard
    # becomes fiction.
    assert classify_outcome(1.01, 0.02, correctness_passed=True) == "inconclusive"
    assert classify_outcome(0.99, 0.02, correctness_passed=True) == "inconclusive"


def test_margin_exactly_at_the_band_edge_is_inconclusive():
    assert classify_outcome(1.02, 0.02, correctness_passed=True) == "inconclusive"


def test_slower_than_the_band_is_a_loss():
    assert classify_outcome(0.90, 0.02, correctness_passed=True) == "loss"


def test_no_ratio_is_an_error_not_a_null_result():
    assert classify_outcome(None, 0.02, correctness_passed=True) == "error"


# -- calibration_holds ------------------------------------------------------------------


def test_identity_at_one_calibrates():
    assert calibration_holds(1.000, 0.01)


def test_identity_far_from_one_does_not_calibrate():
    assert not calibration_holds(1.15, 0.01)


def test_calibration_band_has_a_floor_so_it_stays_falsifiable():
    # An implausibly tight IQR must not reject a deviation that is plainly just noise.
    assert calibration_holds(1.015, 0.0001)
    assert not calibration_holds(1.15, 0.0001)


def test_calibration_uses_the_measured_iqr_when_it_is_wider_than_the_floor():
    assert calibration_holds(1.05, 0.08)


def test_missing_ratio_does_not_calibrate():
    assert not calibration_holds(None, 0.01)


# -- score_predictions ------------------------------------------------------------------


def batch_of_three() -> Batch:
    return Batch(
        batch_id="b",
        hypotheses=(
            make_hypothesis(slug="000-identity", kernels=(), prediction="identity"),
            make_hypothesis(slug="001-a", prediction="inconclusive"),
            make_hypothesis(slug="002-b", prediction="win"),
        ),
    )


def test_predictions_are_scored_against_measured_outcomes():
    scores = score_predictions(
        batch_of_three(),
        {"000-identity": "inconclusive", "001-a": "inconclusive", "002-b": "loss"},
        calibrated=True,
    )
    by_slug = {s.slug: s for s in scores}
    assert by_slug["000-identity"].correct is True
    assert by_slug["001-a"].correct is True
    assert by_slug["002-b"].correct is False


def test_an_errored_slot_scores_none_not_false():
    # The prediction was never tested. Counting it wrong understates the record exactly
    # as counting it right would flatter it.
    scores = score_predictions(batch_of_three(), {"002-b": "error"}, calibrated=True)
    assert {s.slug: s.correct for s in scores}["002-b"] is None


def test_a_slot_that_never_ran_scores_none():
    scores = score_predictions(batch_of_three(), {}, calibrated=True)
    assert all(s.correct is None for s in scores if s.slug != "000-identity")
    assert all(s.outcome == "not_run" for s in scores if s.slug != "000-identity")


def test_the_identity_slot_is_scored_on_calibration_not_on_its_ratio_band():
    scores = score_predictions(batch_of_three(), {"000-identity": "inconclusive"}, calibrated=False)
    assert {s.slug: s.correct for s in scores}["000-identity"] is False


# -- SlotBudget -------------------------------------------------------------------------


class FakeClock:
    def __init__(self, now: float = 0.0) -> None:
        self.now = now

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds


def test_budget_uses_the_seed_estimate_before_any_slot_has_finished():
    clock = FakeClock()
    budget = SlotBudget(deadline_epoch=1000.0, clock=clock, seed_estimate_s=240.0)
    assert budget.estimate_s() == 240.0
    assert budget.can_start()


def test_budget_refuses_when_the_remaining_time_will_not_fit_a_slot():
    clock = FakeClock()
    budget = SlotBudget(deadline_epoch=200.0, clock=clock, seed_estimate_s=240.0)
    assert not budget.can_start()
    assert "Stopping here" in budget.why_not()


def test_budget_estimate_adapts_to_the_slots_actually_observed():
    clock = FakeClock()
    budget = SlotBudget(deadline_epoch=10_000.0, clock=clock, seed_estimate_s=240.0)
    for duration in (100.0, 110.0, 120.0):
        budget.record(duration)
    assert budget.estimate_s() == 110.0


def test_budget_estimate_is_a_median_so_one_slow_slot_does_not_dominate():
    budget = SlotBudget(deadline_epoch=10_000.0, clock=FakeClock(), seed_estimate_s=240.0)
    for duration in (100.0, 105.0, 110.0, 2000.0):
        budget.record(duration)
    assert budget.estimate_s() == 107.5


def test_budget_applies_a_safety_factor_so_it_stops_early_rather_than_late():
    clock = FakeClock()
    budget = SlotBudget(deadline_epoch=110.0, clock=clock, seed_estimate_s=1.0, safety_factor=1.2)
    budget.record(100.0)
    # 110s left, a slot takes 100s, but 100 * 1.2 = 120 > 110, so it declines.
    assert not budget.can_start()


def test_budget_counts_down_as_the_clock_advances():
    clock = FakeClock()
    budget = SlotBudget(deadline_epoch=1000.0, clock=clock, seed_estimate_s=100.0)
    assert budget.can_start()
    clock.advance(900.0)
    assert not budget.can_start()
    assert budget.remaining_s() == 100.0


def test_budget_rejects_a_negative_duration():
    budget = SlotBudget(deadline_epoch=1.0, clock=FakeClock())
    with pytest.raises(ValueError, match="cannot take"):
        budget.record(-1.0)


def test_the_first_two_slots_are_never_capped_below_the_remaining_budget():
    """The cap exists to stop slot 7 eating slot 8.

    Applying it to the first two slots would defeat the guarantee it protects: the
    identity slot plus one kernel slot is the least a rental may produce and still have
    scored a hypothesis.
    """
    budget = SlotBudget(deadline_epoch=1000.0, clock=lambda: 0.0, slot_cap_s=60.0)

    assert budget.cap_for(0) == pytest.approx(1000.0)
    assert budget.cap_for(1) == pytest.approx(1000.0)
    assert budget.cap_for(2) == pytest.approx(60.0)


def test_a_capped_slot_never_outlives_the_deadline():
    """A cap larger than what is left is not a licence to overrun the session."""
    budget = SlotBudget(deadline_epoch=100.0, clock=lambda: 0.0, slot_cap_s=600.0)

    assert budget.cap_for(5) == pytest.approx(100.0)


def test_a_starved_prediction_is_untested_not_wrong():
    """A slot the clock killed tested nothing. Counting it wrong understates the record
    exactly as counting it right would flatter it."""
    batch = Batch(
        batch_id="b",
        hypotheses=(
            make_hypothesis(slug="000-identity", kernels=(), prediction="identity"),
            make_hypothesis(slug="001-x"),
        ),
    )

    scores = score_predictions(batch, {"000-identity": "win", "001-x": "starved"}, calibrated=True)

    assert scores[1].correct is None
    assert scores[1].outcome == "starved"


# -- can this session finish one hypothesis at all? ---------------------------------------


def test_a_session_with_room_for_one_hypothesis_fits():
    fits, shortfall = session_fits_one_hypothesis(
        remaining_s=7200.0,
        phases={"setup_s": 900.0, "reference_compile_s": 2400.0, "slot_s": 900.0},
        reserve_s=720.0,
    )

    assert fits is True
    assert shortfall == 0.0


def test_a_session_that_cannot_fit_one_hypothesis_says_how_short_it_is():
    """Nine rentals produced no number. Not renting beats renting to produce nothing, and
    the shortfall is what tells the next session how far the gate is from being enough."""
    fits, shortfall = session_fits_one_hypothesis(
        remaining_s=3600.0,
        phases={"setup_s": 900.0, "reference_compile_s": 2400.0, "slot_s": 900.0},
        reserve_s=720.0,
    )

    assert fits is False
    # 900 + 2400 + two slots of 900 + 720 of reserve = 5820, against 3600 available.
    assert shortfall == pytest.approx(2220.0)


def test_the_cold_estimates_are_used_when_nothing_has_been_measured():
    """The first rental after this lands has no measured phases: it must fall back to the
    worst case, not to optimism."""
    fits, _shortfall = session_fits_one_hypothesis(remaining_s=10800.0, phases={}, reserve_s=720.0)

    assert fits is True
    assert set(COLD_PHASE_ESTIMATES) == {"setup_s", "reference_compile_s", "slot_s"}


def test_the_minimum_is_two_slots_because_calibration_alone_scores_nothing():
    """The identity champion proves the harness measures what it says. It is not a
    hypothesis, so a rental that fits only that slot has bought no science."""
    # Nothing is zero: a zero in phases.env means "not measured" and falls back to the
    # cold estimate, so a test that wants a phase ignored must make it negligible instead.
    phases = {"setup_s": 1.0, "reference_compile_s": 1.0, "slot_s": 600.0}

    one_slot, _ = session_fits_one_hypothesis(remaining_s=602.0, phases=phases, reserve_s=0.0)
    two_slots, _ = session_fits_one_hypothesis(remaining_s=1202.0, phases=phases, reserve_s=0.0)

    assert one_slot is False
    assert two_slots is True


# -- the approximate correctness policy ------------------------------------------------


def _hypothesis(**overrides):
    fields = dict(
        slug="x",
        kernels=("k",),
        category="B",
        byte_share=0.5,
        mechanism="m",
        prediction="win",
        rationale="r",
    )
    fields.update(overrides)
    return Hypothesis(**_with_a_gate_it_can_hold(fields))


def _raw_hypothesis(**overrides):
    """`_hypothesis` without the gate defaults, for tests about the gate fields themselves."""
    fields = dict(
        slug="x",
        kernels=("k",),
        category="B",
        byte_share=0.5,
        mechanism="m",
        prediction="win",
        rationale="r",
    )
    fields.update(overrides)
    return Hypothesis(**fields)


def test_a_hypothesis_is_gated_exactly_by_default():
    """And `exact` is now reachable only by the candidate that installs nothing."""
    identity = _hypothesis(kernels=(), prediction="identity")

    assert identity.correctness == "exact"
    assert Hypothesis.__dataclass_fields__["correctness"].default == "exact"


def test_an_unknown_correctness_policy_is_refused():
    with pytest.raises(ValueError, match="correctness must be one of"):
        _hypothesis(correctness="vibes")


def test_an_approximate_hypothesis_must_register_both_bars():
    """An approximate gate with no bar passes everything, including a broken kernel."""
    with pytest.raises(ValueError, match="before the rental"):
        _raw_hypothesis(correctness="approximate")
    with pytest.raises(ValueError, match="before the rental"):
        _raw_hypothesis(correctness="approximate", top1_threshold=0.98)
    with pytest.raises(ValueError, match="before the rental"):
        _raw_hypothesis(correctness="approximate", kl_threshold=0.01)


def test_an_exact_hypothesis_may_not_carry_bars_nothing_would_read():
    with pytest.raises(ValueError, match="scored exactly but carries approximate thresholds"):
        _raw_hypothesis(top1_threshold=0.98)


def test_the_bars_are_range_checked():
    with pytest.raises(ValueError, match="top1_threshold is a fraction"):
        _raw_hypothesis(correctness="approximate", top1_threshold=1.5, kl_threshold=0.01)
    with pytest.raises(ValueError, match="cannot be negative"):
        _hypothesis(correctness="approximate", top1_threshold=0.98, kl_threshold=-1.0)


def test_the_identity_slot_may_not_be_gated_approximately():
    """It is bit-identical to the reference. A calibration slot that could not match tokens
    would calibrate nothing, and would hide a harness fault behind a tolerance."""
    with pytest.raises(ValueError, match="must be gated exactly"):
        Hypothesis(
            slug="000-identity",
            kernels=(),
            category="calibration",
            byte_share=0.0,
            mechanism="install nothing",
            prediction="identity",
            rationale="r",
            correctness="approximate",
            top1_threshold=0.9,
            kl_threshold=0.1,
        )


def test_an_approximate_hypothesis_with_both_bars_is_accepted():
    hypothesis = _hypothesis(correctness="approximate", top1_threshold=0.98, kl_threshold=0.01)
    assert hypothesis.top1_threshold == 0.98
    assert hypothesis.kl_threshold == 0.01


# -- preconditions --------------------------------------------------------------------


def test_a_slot_can_require_an_earlier_slots_ratio():
    p = Precondition(slug="015-gemv-bf16", floor=0.56, reason="quantisation cannot tie below this")
    hypothesis = make_hypothesis(slug="017-fp8", requires=p)

    assert hypothesis.requires.floor == 0.56


def test_a_precondition_naming_a_slot_that_is_not_earlier_in_the_batch_is_refused():
    """A forward reference would silently never fire, which is worse than not having one."""
    with pytest.raises(ValueError, match="must name an earlier slot"):
        Batch(
            batch_id="x",
            hypotheses=(
                make_hypothesis(slug="a", requires=Precondition(slug="b", floor=0.5, reason="r")),
                make_hypothesis(slug="b"),
            ),
        )


def test_a_precondition_naming_a_slot_the_batch_does_not_hold_is_refused():
    """A typo'd slug is a precondition that can never hold, so the slot would never run."""
    with pytest.raises(ValueError, match="must name an earlier slot"):
        Batch(
            batch_id="x",
            hypotheses=(make_hypothesis(slug="a", requires=Precondition("z", 0.5, "r")),),
        )


def test_precondition_holds_when_the_named_slot_cleared_the_floor():
    assert precondition_holds(Precondition("a", 0.56, "r"), {"a": 0.60}) is True


def test_precondition_fails_when_it_did_not():
    assert precondition_holds(Precondition("a", 0.56, "r"), {"a": 0.28}) is False


def test_a_precondition_on_a_slot_that_errored_fails_closed():
    """No ratio is not the same as a good ratio. Failing open would run the batch anyway."""
    assert precondition_holds(Precondition("a", 0.56, "r"), {"a": None}) is False


def test_no_precondition_always_holds():
    assert precondition_holds(None, {}) is True


def test_a_precondition_needs_a_reason_a_reader_can_act_on():
    """`precondition_failed` is recorded with its reason and nothing else explains the skip."""
    with pytest.raises(ValueError, match="reason"):
        make_hypothesis(slug="b", requires=Precondition("a", 0.5, "  "))


def test_a_precondition_skipped_slot_scores_no_prediction():
    """It tested nothing. Counting it wrong understates the record exactly as counting it
    right would flatter it -- the rule `error` and `not_run` already follow."""
    batch = Batch(
        batch_id="x",
        hypotheses=(
            make_hypothesis(slug="a", prediction="win"),
            make_hypothesis(slug="b", prediction="win", requires=Precondition("a", 0.5, "r")),
        ),
    )

    scores = score_predictions(batch, {"a": "loss", "b": "precondition_failed"})

    assert [s.correct for s in scores] == [False, None]


# -- gates that can resolve what they claim -------------------------------------------


def test_only_the_identity_champion_may_be_gated_exactly():
    """009 was gated `exact` because it computes the same function as the reference. It does;
    it does not compute the same *bits*. fp32 accumulation in a different order from cuBLAS
    lands one bf16 ULP away, and one ULP flips an argmax on this model -- it matched 1 of 5
    prompts. Bit-identity is a property of the implementation, and only installing nothing
    has it. This is the third time the project has paid for the distinction."""
    with pytest.raises(ValueError, match="only the identity champion is bit-identical"):
        make_hypothesis(slug="015-gemv-bf16", kernels=("tiled_gemv_bf16",), correctness="exact")


def test_the_identity_champion_is_still_gated_exactly():
    """The rule cuts one way only: installing nothing must still be held to the tokens."""
    hypothesis = make_hypothesis(slug="000-identity", kernels=(), prediction="identity")

    assert hypothesis.correctness == "exact"


def test_a_top1_bar_finer_than_the_sample_can_resolve_is_refused():
    """013 missed its bar by 0.000303 at n=264, where agreement quantises to 1/264 = 0.0038.
    A threshold an order of magnitude below one sample is not a decision procedure."""
    with pytest.raises(ValueError, match="finer than one sample"):
        make_hypothesis(
            correctness="approximate",
            top1_threshold=0.97,
            kl_threshold=0.02,
            correctness_positions=264,
        )


def test_a_top1_bar_that_lands_on_an_achievable_count_is_accepted():
    """256/264 is 0.969697. A bar a sample can actually land on is a decision procedure."""
    hypothesis = make_hypothesis(
        correctness="approximate",
        top1_threshold=256 / 264,
        kl_threshold=0.02,
        correctness_positions=264,
    )

    assert hypothesis.correctness_positions == 264


def test_a_bar_with_no_declared_sample_size_is_not_checked_for_resolution():
    """`correctness_positions` is what the batch expects to score; without it there is
    nothing to compare the bar against, and inventing an n would be worse than not checking."""
    hypothesis = make_hypothesis(correctness="approximate", top1_threshold=0.97, kl_threshold=0.02)

    assert hypothesis.correctness_positions is None


# -- what a hypothesis re-encodes -----------------------------------------------------


def test_a_hypothesis_declares_which_weight_regions_it_re_encodes():
    """The bench divides bytes by time. A hypothesis that did not say what it re-encodes
    would be scored against bf16 byte counts and report a bandwidth it never achieved."""
    hypothesis = make_hypothesis(weight_bits={"layers": 8})

    assert hypothesis.weight_bits == {"layers": 8}


def test_a_hypothesis_that_re_encodes_nothing_declares_nothing():
    assert make_hypothesis().weight_bits == {}


def test_a_typo_in_weight_bits_fails_on_a_laptop_rather_than_on_a_rented_box():
    with pytest.raises(ValueError, match="unknown weight region"):
        make_hypothesis(weight_bits={"mpl": 8})
