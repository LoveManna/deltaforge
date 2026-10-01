"""The pre-flight rental 43 needed and did not have.

These assert on the *text*, because the text is the deliverable: this module exists to
say something at minute 27 that batch 007 only said in its writeup.
"""

from __future__ import annotations

from .card_baseline import (
    RECORDED_REFERENCE_GBPS,
    SLOW_FRACTION,
    card_report,
    recorded_for,
)

RTX_5090 = "NVIDIA GeForce RTX 5090"


def test_rental_43s_card_would_have_been_called_out_loudly():
    """845 GB/s against a recorded best of 1282. The whole point of the file."""
    report = card_report(RTX_5090, 845.3)

    assert "WARNING" in report
    assert "SLOW" in report
    assert "845" in report and "1283" in report


def test_rental_40s_card_reads_as_in_family():
    report = card_report(RTX_5090, 1282.6)

    assert "WARNING" not in report
    assert "In family" in report


def test_rental_42s_card_reads_as_in_family_too():
    """0.95 of the best. The threshold has to sit above rental 43 and below this."""
    assert SLOW_FRACTION < 1223.2 / 1282.6
    assert SLOW_FRACTION > 845.3 / 1282.6
    assert "WARNING" not in card_report(RTX_5090, 1223.2)


def test_an_unrecorded_gpu_model_says_so_rather_than_comparing_against_nothing():
    # An RTX 4090 stopped being unknown when rental 54 ran batch 010 on one.
    report = card_report("NVIDIA H100 SXM", 700.0)

    assert "never recorded this GPU model" in report
    assert "WARNING" not in report
    assert not recorded_for("NVIDIA H100 SXM")
    # And the 4090 is no longer the example of an unrecorded model: rental 54 ran batch 010
    # on one, so it has a row now, which is exactly what this table is for.
    assert recorded_for("NVIDIA GeForce RTX 4090")


def test_a_slot_that_declared_no_bytes_is_reported_as_uncharacterised():
    report = card_report(RTX_5090, None)

    assert "uncharacterised" in report
    assert "WARNING" not in report


def test_every_report_says_a_slow_card_does_not_void_a_ratio():
    """The one thing a reader could get wrong from a loud warning.

    Ratios are measured against a reference timed in the same interleaved rounds, so they
    survive a slow card. Absolute predictions and cross-rental comparisons do not.
    """
    for measured in (None, 700.0, 845.3, 1282.6):
        report = card_report(RTX_5090, measured)
        assert "does not void a ratio" in report or "uncharacterised" in report


def test_the_recorded_observations_are_the_rentals_the_writeups_name():
    seen = {o.rental: o.gbps for o in RECORDED_REFERENCE_GBPS[RTX_5090]}

    assert seen == {40: 1282.6, 42: 1223.2, 43: 845.3, 45: 1197.0, 46: 1214.0, 56: 440.8}


# -- rental 56: the field that was always in the record and never compared ---------------


def test_an_unrecognised_driver_is_called_out_at_slot_zero():
    """Rental 56's whole lesson. Both cards that ever ran slow here were on a driver this
    project had seen once, and the field was in every record from the first rental."""
    report = card_report(RTX_5090, 1210.0, "999.99.99")

    assert "new to this project" in report
    assert "999.99.99" in report


def test_a_driver_with_a_slow_history_is_still_called_out_after_it_is_recorded():
    """The warning has to survive being written down.

    `610.43.02` warned as *unknown* until rental 56 was added to the table — and adding it
    is the correct response to measuring it. If that were the only branch, recording an
    observation would silence the warning it earned, which is the shape of a table that
    "silently turns a normal card into an outlier" in reverse.
    """
    report = card_report(RTX_5090, 441.0, "610.43.02")

    assert "has run this model slowly before" in report
    assert "610.43.02" in report
    assert "WARNING" in report  # 0.34x — it is also slow, and both facts are said


def test_a_known_driver_is_reported_with_what_it_has_run_before():
    report = card_report(RTX_5090, 1210.0, "580.173.02")

    assert "580.173.02 has previously run this model at" in report
    assert "1197" in report and "1214" in report
    assert "WARNING" not in report


def test_the_report_still_works_with_no_driver_and_says_nothing_about_one():
    report = card_report(RTX_5090, 1210.0)

    assert "driver" not in report.lower()


def test_every_recorded_observation_names_its_driver():
    """A row without one re-creates the gap: the comparison cannot be made from memory."""
    from .card_baseline import RECORDED_REFERENCE_GBPS

    for gpu, observations in RECORDED_REFERENCE_GBPS.items():
        for observation in observations:
            assert observation.driver, f"{gpu} rental {observation.rental} has no driver recorded"


def test_the_drivers_seen_twice_are_the_fast_ones():
    """The actual finding, asserted so that a new row cannot quietly contradict it.

    Every RTX 5090 driver this project has seen more than once ran the reference between
    1197 and 1283 GB/s. The two outliers — 845 and 441 — are the two drivers seen once.
    """
    from .card_baseline import drivers_seen

    seen = drivers_seen(RTX_5090)
    repeated = [values for values in seen.values() if len(values) > 1]
    once = [values[0] for values in seen.values() if len(values) == 1]

    assert repeated, "the claim needs at least one driver seen twice"
    assert min(min(v) for v in repeated) > 1100
    assert once and max(once) < 900
