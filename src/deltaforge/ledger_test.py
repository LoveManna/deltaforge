"""Budget gate tests against synthetic ledger fixtures.

These gates are the only thing standing between an autonomous agent and a forgotten
$0.34/hr instance, so they are tested for the awkward cases rather than the happy one:
an un-reconciled instance, a run that spans a month boundary, a session that never tore
down. Both gates must **fail closed**.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from .ledger import (
    MONTH_TO_DATE_LIMIT_USD,
    SESSION_MINUTES_LIMIT,
    LedgerError,
    LedgerRow,
    append_destroy,
    append_provision,
    month_to_date_gate,
    month_to_date_usd,
    read_rows,
    session_gate,
    session_minutes,
    write_fixture,
)

EPOCH = int(datetime(2026, 8, 30, 12, 0, tzinfo=timezone.utc).timestamp())


def row(**kwargs) -> LedgerRow:
    base = {
        "ts": "2026-08-30T12:00:00Z",
        "ts_epoch": EPOCH,
        "event": "provision",
        "session_id": "s1",
        "instance_id": "i1",
        "gpu_model": "RTX 5090",
        "hourly_rate_usd": 0.324,
        "estimated_ceiling_usd": 0.486,
        "estimated_minutes": 90.0,
    }
    base.update(kwargs)
    return LedgerRow(**base)


def provision(instance, session="s1", ts="2026-08-30T12:00:00Z", epoch=EPOCH, ceiling=0.486):
    return row(
        instance_id=instance,
        session_id=session,
        ts=ts,
        ts_epoch=epoch,
        estimated_ceiling_usd=ceiling,
    )


def destroy(instance, session="s1", ts="2026-08-30T13:00:00Z", minutes=30.0, cost=0.162):
    return row(
        event="destroy",
        instance_id=instance,
        session_id=session,
        ts=ts,
        ts_epoch=EPOCH + 3600,
        estimated_ceiling_usd=0.0,
        estimated_minutes=0.0,
        actual_minutes=minutes,
        actual_cost_usd=cost,
    )


# -- format ---------------------------------------------------------------------------


def test_rows_round_trip_through_the_file(tmp_path):
    path = tmp_path / "spend.jsonl"
    write_fixture(path, [provision("i1"), destroy("i1")])

    rows = read_rows(path)

    assert len(rows) == 2
    assert rows[0].event == "provision"
    assert rows[1].actual_minutes == 30.0
    assert rows[0].month == "2026-08"


def test_a_missing_ledger_is_an_empty_ledger_not_an_error(tmp_path):
    assert read_rows(tmp_path / "nope.jsonl") == []
    allowed, spend = month_to_date_gate(tmp_path / "nope.jsonl")
    assert allowed and spend == 0.0


def test_rows_stay_flat_so_awk_can_parse_them():
    """remote/provision.sh reads this file with awk before any Python exists. Nested
    objects or braces inside strings would silently break the shell gate."""
    text = provision("i1").to_json()

    assert text.count("{") == 1
    assert text.count("}") == 1
    assert "\n" not in text


@pytest.mark.parametrize("bad", ['a"b', "a{b", "a}b", "a,b", "a\\b"])
def test_characters_that_would_break_the_shell_reader_are_rejected(bad):
    with pytest.raises(LedgerError, match="which the shell reader cannot parse"):
        row(session_id=bad).to_json()


def test_malformed_json_is_a_loud_error(tmp_path):
    path = tmp_path / "spend.jsonl"
    path.write_text('{"ts":"2026-08-30T12:00:00Z"\n')
    with pytest.raises(LedgerError, match="not valid JSON"):
        read_rows(path)


def test_missing_required_field_is_a_loud_error(tmp_path):
    path = tmp_path / "spend.jsonl"
    path.write_text('{"ts":"2026-08-30T12:00:00Z","ts_epoch":1,"event":"provision"}\n')
    with pytest.raises(LedgerError, match="missing required field 'instance_id'"):
        read_rows(path)


# -- month-to-date gate ---------------------------------------------------------------


def test_month_to_date_sums_reconciled_costs():
    rows = [
        provision("i1"),
        destroy("i1", cost=0.162),
        provision("i2"),
        destroy("i2", cost=0.500),
    ]
    assert month_to_date_usd(rows, month="2026-08") == pytest.approx(0.662)


def test_an_unreconciled_instance_counts_at_its_estimated_ceiling():
    """The gate must fail closed. Counting a still-running instance at zero is how a
    budget control fails open exactly when money is being spent."""
    rows = [provision("i1", ceiling=0.486)]
    assert month_to_date_usd(rows, month="2026-08") == pytest.approx(0.486)


def test_reconciliation_replaces_the_estimate_rather_than_adding_to_it():
    rows = [provision("i1", ceiling=0.486), destroy("i1", cost=0.162)]
    assert month_to_date_usd(rows, month="2026-08") == pytest.approx(0.162)


def test_previous_months_do_not_count_against_this_month():
    rows = [
        provision("old", ts="2026-07-15T09:00:00Z"),
        destroy("old", ts="2026-07-15T10:00:00Z", cost=40.0),
        provision("new"),
        destroy("new", cost=1.0),
    ]
    assert month_to_date_usd(rows, month="2026-08") == pytest.approx(1.0)
    assert month_to_date_usd(rows, month="2026-07") == pytest.approx(40.0)


def test_gate_refuses_at_the_threshold_not_merely_above_it(tmp_path):
    path = tmp_path / "spend.jsonl"
    write_fixture(path, [provision("i1"), destroy("i1", cost=MONTH_TO_DATE_LIMIT_USD)])

    allowed, spend = month_to_date_gate(path, month="2026-08")

    assert not allowed
    assert spend == pytest.approx(45.0)


def test_gate_allows_just_below_the_threshold(tmp_path):
    path = tmp_path / "spend.jsonl"
    write_fixture(path, [provision("i1"), destroy("i1", cost=44.99)])

    allowed, spend = month_to_date_gate(path, month="2026-08")

    assert allowed
    assert spend == pytest.approx(44.99)


def test_headroom_is_left_under_the_fifty_dollar_budget():
    assert MONTH_TO_DATE_LIMIT_USD == 45.0
    assert MONTH_TO_DATE_LIMIT_USD < 50.0


# -- session GPU-time gate ------------------------------------------------------------


def test_session_minutes_sums_reconciled_runs():
    rows = [
        provision("i1"),
        destroy("i1", minutes=25.0),
        provision("i2"),
        destroy("i2", minutes=20.0),
    ]
    assert session_minutes(rows, "s1", now_epoch=EPOCH + 99999) == pytest.approx(45.0)


def test_only_this_session_counts():
    rows = [
        provision("i1", session="s1"),
        destroy("i1", session="s1", minutes=25.0),
        provision("i2", session="s2"),
        destroy("i2", session="s2", minutes=50.0),
    ]
    assert session_minutes(rows, "s1", now_epoch=EPOCH + 99999) == pytest.approx(25.0)
    assert session_minutes(rows, "s2", now_epoch=EPOCH + 99999) == pytest.approx(50.0)


def test_a_still_running_instance_counts_at_wall_clock_elapsed():
    """A session must not be able to dodge the gate by simply not tearing down."""
    rows = [provision("i1")]
    assert session_minutes(rows, "s1", now_epoch=EPOCH + 1800) == pytest.approx(30.0)


def test_session_gate_refuses_at_three_hours(tmp_path):
    path = tmp_path / "spend.jsonl"
    write_fixture(path, [provision("i1"), destroy("i1", minutes=180.0)])

    allowed, minutes = session_gate(path, "s1", now_epoch=EPOCH + 99999)

    assert not allowed
    assert minutes == pytest.approx(180.0)
    assert SESSION_MINUTES_LIMIT == 180.0


def test_session_gate_allows_a_session_just_under_the_limit(tmp_path):
    path = tmp_path / "spend.jsonl"
    write_fixture(path, [provision("i1"), destroy("i1", minutes=179.0)])

    allowed, minutes = session_gate(path, "s1", now_epoch=EPOCH + 99999)

    assert allowed
    assert minutes == pytest.approx(179.0)


def test_an_unknown_session_has_used_nothing(tmp_path):
    path = tmp_path / "spend.jsonl"
    write_fixture(path, [provision("i1"), destroy("i1", minutes=60.0)])

    allowed, minutes = session_gate(path, "never-seen", now_epoch=EPOCH)

    assert allowed and minutes == 0.0


def test_clock_skew_cannot_produce_negative_minutes():
    rows = [provision("i1")]
    assert session_minutes(rows, "s1", now_epoch=EPOCH - 10_000) == 0.0


# -- appending ------------------------------------------------------------------------


def test_append_provision_computes_the_ceiling_from_rate_and_minutes(tmp_path):
    path = tmp_path / "spend.jsonl"

    written = append_provision(
        path,
        session_id="s1",
        instance_id="42",
        gpu_model="RTX 5090",
        hourly_rate_usd=0.324,
        estimated_minutes=90,
    )

    assert written.estimated_ceiling_usd == pytest.approx(0.324 * 1.5)
    assert written.actual_cost_usd is None, "not reconciled until destroy"
    assert read_rows(path)[0].instance_id == "42"


def test_append_destroy_computes_the_actual_cost(tmp_path):
    path = tmp_path / "spend.jsonl"
    append_provision(
        path,
        session_id="s1",
        instance_id="42",
        gpu_model="RTX 5090",
        hourly_rate_usd=0.324,
        estimated_minutes=90,
    )

    written = append_destroy(
        path,
        session_id="s1",
        instance_id="42",
        hourly_rate_usd=0.324,
        actual_minutes=30.0,
    )

    assert written.actual_cost_usd == pytest.approx(0.162)
    assert len(read_rows(path)) == 2


def test_the_ledger_is_append_only(tmp_path):
    """A session that dies mid-run must still leave a record of what it started."""
    path = tmp_path / "spend.jsonl"
    append_provision(
        path,
        session_id="s1",
        instance_id="1",
        gpu_model="RTX 5090",
        hourly_rate_usd=0.3,
        estimated_minutes=90,
    )
    first = path.read_text()

    append_provision(
        path,
        session_id="s1",
        instance_id="2",
        gpu_model="RTX 5090",
        hourly_rate_usd=0.3,
        estimated_minutes=90,
    )

    assert path.read_text().startswith(first)
    assert len(read_rows(path)) == 2


def test_a_realistic_session_stays_well_under_budget(tmp_path):
    """The economics the spec claims: the 60-minute gate caps a session near $0.35 at the
    observed market rate, giving well over 75 sessions inside $50."""
    path = tmp_path / "spend.jsonl"
    now = datetime.now(timezone.utc)
    for i in range(10):
        stamp = (now + timedelta(minutes=i)).strftime("%Y-%m-%dT%H:%M:%SZ")
        epoch = int((now + timedelta(minutes=i)).timestamp())
        append_provision(
            path,
            session_id=f"s{i}",
            instance_id=str(i),
            gpu_model="RTX 5090",
            hourly_rate_usd=0.324,
            estimated_minutes=60,
            now=(stamp, epoch),
        )
        append_destroy(
            path,
            session_id=f"s{i}",
            instance_id=str(i),
            hourly_rate_usd=0.324,
            actual_minutes=55.0,
            now=(stamp, epoch),
        )

    spend = month_to_date_usd(read_rows(path), month=now.strftime("%Y-%m"))

    assert spend == pytest.approx(10 * 0.324 * 55 / 60, rel=1e-3)
    assert spend < 3.0


# -- double reconciliation ------------------------------------------------------------
#
# The 90-minute watchdog (spec 8.3) destroys the instance and writes its own destroy row;
# that kills the run, which fires the teardown trap (spec 8.5), which writes a second
# destroy row for the same instance. The ledger is append-only, so neither row can be
# retracted and the readers must dedup. Both gates count one reconciled destroy per
# instance, at the LARGEST reported value — over-counting refuses to spend, which is
# safe; under-counting lets spend escape, which is not.


def test_a_double_reconciled_instance_is_counted_once_at_the_larger_cost():
    rows = [
        provision("i1"),
        destroy("i1", ts="2026-08-30T13:30:00Z", minutes=90.0, cost=0.486),  # watchdog
        destroy("i1", ts="2026-08-30T13:30:04Z", minutes=89.5, cost=0.4833),  # teardown trap
    ]

    assert month_to_date_usd(rows, month="2026-08") == pytest.approx(0.486)


def test_a_double_reconciled_instance_bills_minutes_once_at_the_larger_duration():
    rows = [
        provision("i1"),
        destroy("i1", ts="2026-08-30T13:30:00Z", minutes=90.0, cost=0.486),
        destroy("i1", ts="2026-08-30T13:30:04Z", minutes=89.5, cost=0.4833),
    ]

    assert session_minutes(rows, "s1", now_epoch=EPOCH + 7200) == pytest.approx(90.0)


def test_dedup_takes_the_maximum_regardless_of_which_row_came_first():
    """Not first-wins, not last-wins, not the mean: the largest. Whichever writer got
    there first, the gate must see the more expensive reading."""
    ascending = [
        provision("i1"),
        destroy("i1", minutes=10.0, cost=0.1),
        destroy("i1", minutes=40.0, cost=0.4),
    ]
    descending = [
        provision("i1"),
        destroy("i1", minutes=40.0, cost=0.4),
        destroy("i1", minutes=10.0, cost=0.1),
    ]

    for rows in (ascending, descending):
        assert month_to_date_usd(rows, month="2026-08") == pytest.approx(0.4)
        assert session_minutes(rows, "s1", now_epoch=EPOCH + 7200) == pytest.approx(40.0)


def test_a_single_reconciled_destroy_is_unaffected_by_the_dedup():
    rows = [provision("i1"), destroy("i1", minutes=30.0, cost=0.162)]

    assert month_to_date_usd(rows, month="2026-08") == pytest.approx(0.162)
    assert session_minutes(rows, "s1", now_epoch=EPOCH + 7200) == pytest.approx(30.0)


def test_a_crash_between_the_two_writes_still_falls_back_to_the_estimate():
    """Provision row written, instance never reconciled: unchanged fail-closed behaviour
    — the ceiling for spend, wall-clock elapsed for session minutes."""
    rows = [provision("i1", ceiling=0.486)]

    assert month_to_date_usd(rows, month="2026-08") == pytest.approx(0.486)
    assert session_minutes(rows, "s1", now_epoch=EPOCH + 1800) == pytest.approx(30.0)


def test_two_different_instances_are_still_summed_separately():
    rows = [
        provision("i1"),
        destroy("i1", minutes=30.0, cost=0.162),
        provision("i2"),
        destroy("i2", minutes=20.0, cost=0.108),
    ]

    assert month_to_date_usd(rows, month="2026-08") == pytest.approx(0.27)
    assert session_minutes(rows, "s1", now_epoch=EPOCH + 7200) == pytest.approx(50.0)
