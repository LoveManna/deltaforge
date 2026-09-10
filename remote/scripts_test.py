"""Dry-run tests for the cost machinery.

The scripts in `remote/` are the only thing preventing an autonomous agent from leaving a
GPU running. That makes them safety-critical *and* awkward to test, because exercising
them for real costs money. Hence the dry-run mode: every branch of the logic runs, the
create and destroy endpoints are never contacted, and nothing is spent.

Two properties get the most attention here:

* **The gates fail closed.** Both the month-to-date gate and the session GPU-time gate
  must refuse, and refuse *before* anything is created.
* **Teardown is unconditional.** The trap must destroy the instance and reconcile the
  ledger even when the run fails part way through, which is exactly when a forgotten
  instance is most likely.

The shell gates are also cross-checked against `deltaforge.ledger`: two implementations
of the same arithmetic that must agree on shared fixtures, so a change to one that
diverges from the other fails here.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
REMOTE = REPO_ROOT / "remote"
EPOCH = int(datetime(2026, 8, 30, 12, 0, tzinfo=timezone.utc).timestamp())

pytestmark = pytest.mark.skipif(shutil.which("jq") is None, reason="the offer-selection path requires jq")


def run(script: str, *args: str, expect: int | None = 0, env=None) -> subprocess.CompletedProcess:
    result = subprocess.run(
        ["sh", str(REMOTE / script), *args],
        capture_output=True,
        text=True,
        timeout=180,
        cwd=str(REPO_ROOT),
        env=env,
    )
    if expect is not None and result.returncode != expect:
        raise AssertionError(
            f"{script} exited {result.returncode}, expected {expect}\n"
            f"--- stdout ---\n{result.stdout}\n--- stderr ---\n{result.stderr}"
        )
    return result


def ledger_line(**kwargs) -> str:
    base = {
        "ts": "2026-08-30T12:00:00Z",
        "ts_epoch": EPOCH,
        "event": "provision",
        "session_id": "s1",
        "instance_id": "i1",
        "gpu_model": "RTX 5090",
        "hourly_rate_usd": 0.324,
        "estimated_ceiling_usd": 0.486,
        "estimated_minutes": 90,
        "actual_minutes": None,
        "actual_cost_usd": None,
        "hypothesis": "",
        "note": "",
    }
    base.update(kwargs)
    return json.dumps(base, separators=(",", ":"))


def write_ledger(path: Path, lines: list[str]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(line + "\n" for line in lines))
    return path


@pytest.fixture
def workdir(tmp_path):
    (tmp_path / "ledger").mkdir()
    return tmp_path


# =====================================================================================
# The month-to-date budget gate
# =====================================================================================


def test_provision_refuses_when_month_to_date_spend_is_at_the_limit(workdir):
    ledger = write_ledger(
        workdir / "ledger" / "spend.jsonl",
        [
            ledger_line(instance_id="a"),
            ledger_line(
                event="destroy",
                instance_id="a",
                ts=f"{datetime.now(timezone.utc):%Y-%m}-01T00:00:00Z",
                actual_minutes=600,
                actual_cost_usd=45.0,
            ),
        ],
    )

    result = run(
        "provision.sh",
        "--dry-run",
        "--session-id",
        "gate-test",
        "--ledger",
        str(ledger),
        "--state-file",
        str(workdir / "state"),
        expect=1,
    )

    assert "REFUSED by month-to-date budget gate" in result.stderr
    assert "No instance was created" in result.stderr
    # The refusal happened before any selection: nothing was chosen and no row written.
    assert "selected offer" not in result.stderr
    assert not (workdir / "state").exists()
    assert len(ledger.read_text().strip().splitlines()) == 2, "no new ledger row"


def test_provision_proceeds_when_month_to_date_spend_is_below_the_limit(workdir):
    ledger = write_ledger(
        workdir / "ledger" / "spend.jsonl",
        [
            ledger_line(instance_id="a"),
            ledger_line(
                event="destroy",
                instance_id="a",
                ts=f"{datetime.now(timezone.utc):%Y-%m}-01T00:00:00Z",
                actual_minutes=60,
                actual_cost_usd=10.0,
            ),
        ],
    )

    result = run(
        "provision.sh",
        "--dry-run",
        "--session-id",
        "gate-test",
        "--ledger",
        str(ledger),
        "--state-file",
        str(workdir / "state"),
    )

    assert "selected offer" in result.stderr
    assert (workdir / "state").exists()


def test_spend_from_a_previous_month_does_not_block_provisioning(workdir):
    ledger = write_ledger(
        workdir / "ledger" / "spend.jsonl",
        [
            ledger_line(instance_id="old", ts="2026-01-05T12:00:00Z"),
            ledger_line(
                event="destroy",
                instance_id="old",
                ts="2026-01-05T13:00:00Z",
                actual_minutes=600,
                actual_cost_usd=49.0,
            ),
        ],
    )

    result = run(
        "provision.sh",
        "--dry-run",
        "--session-id",
        "gate-test",
        "--ledger",
        str(ledger),
        "--state-file",
        str(workdir / "state"),
    )

    assert "selected offer" in result.stderr


def test_an_unreconciled_instance_counts_toward_the_gate(workdir):
    """The gate must fail closed: a running instance counts at its estimated ceiling."""
    month = f"{datetime.now(timezone.utc):%Y-%m}"
    ledger = write_ledger(
        workdir / "ledger" / "spend.jsonl",
        [
            ledger_line(instance_id=f"i{i}", ts=f"{month}-02T00:00:00Z", estimated_ceiling_usd=9.0)
            for i in range(5)
        ],
    )

    result = run(
        "provision.sh",
        "--dry-run",
        "--session-id",
        "gate-test",
        "--ledger",
        str(ledger),
        "--state-file",
        str(workdir / "state"),
        expect=1,
    )

    assert "REFUSED by month-to-date budget gate" in result.stderr


# =====================================================================================
# The session GPU-time soft gate
# =====================================================================================


def test_run_remote_refuses_a_session_that_has_reached_its_gate(workdir):
    """The gate mechanism, with the limit passed explicitly.

    The default moved 60 -> 90 when batches arrived and 90 -> 120 once a cold compile was
    measured in tens of minutes; this test is about the gate firing at whatever limit it is
    given, and `test_the_session_gate_defaults_to_two_hours` covers the default.
    """
    ledger = write_ledger(
        workdir / "ledger" / "spend.jsonl",
        [
            ledger_line(session_id="spent", instance_id="a"),
            ledger_line(
                event="destroy",
                session_id="spent",
                instance_id="a",
                actual_minutes=60.0,
                actual_cost_usd=0.324,
            ),
        ],
    )

    result = run(
        "run_remote.sh",
        "--dry-run",
        "--session-id",
        "spent",
        "--session-limit",
        "60",
        "--ledger",
        str(ledger),
        "--state-file",
        str(workdir / "state"),
        expect=3,
    )

    assert "REFUSED by the session GPU-time gate" in result.stderr
    assert "Do not start another hypothesis" in result.stderr
    # It refused before provisioning, so nothing was created and nothing recorded.
    assert "provisioning..." not in result.stderr
    assert len(ledger.read_text().strip().splitlines()) == 2


def test_run_remote_proceeds_for_a_session_under_the_gate(workdir):
    ledger = write_ledger(
        workdir / "ledger" / "spend.jsonl",
        [
            ledger_line(session_id="fresh", instance_id="a"),
            ledger_line(
                event="destroy",
                session_id="fresh",
                instance_id="a",
                actual_minutes=30.0,
                actual_cost_usd=0.162,
            ),
        ],
    )

    result = run(
        "run_remote.sh",
        "--dry-run",
        "--session-id",
        "fresh",
        "--ledger",
        str(ledger),
        "--state-file",
        str(workdir / "state"),
    )

    assert "has used 30.000000 billed GPU minutes" in result.stderr
    assert "provisioning..." in result.stderr


def test_the_session_gate_is_checked_before_the_run_not_during_it(workdir):
    """A benchmark executing at minute 59 must finish: killing it halfway would waste the
    money already spent and leave nothing recorded in exchange. So the gate appears once,
    at the start, and never again."""
    ledger = workdir / "ledger" / "spend.jsonl"
    result = run(
        "run_remote.sh",
        "--dry-run",
        "--session-id",
        "fresh",
        "--ledger",
        str(ledger),
        "--state-file",
        str(workdir / "state"),
    )

    lines = result.stderr.splitlines()
    gate_lines = [i for i, line in enumerate(lines) if "billed GPU minutes" in line]
    bench_lines = [i for i, line in enumerate(lines) if "running the benchmark" in line]

    assert len(gate_lines) == 1, "the gate is evaluated exactly once"
    assert gate_lines[0] < bench_lines[0], "and before the run starts"


# =====================================================================================
# Offer selection
# =====================================================================================


def test_selection_applies_every_filter_and_takes_the_cheapest_survivor(workdir):
    """The fixture is built so that most offers must be rejected: bid-only, multi-GPU,
    low reliability, over-rate, and not-rentable. Only 9007 and 9008 survive, and 9008 is
    cheaper."""
    result = run(
        "provision.sh",
        "--dry-run",
        "--session-id",
        "select",
        "--ledger",
        str(workdir / "ledger" / "spend.jsonl"),
        "--state-file",
        str(workdir / "state"),
    )

    assert "selected offer 9008" in result.stderr
    assert "RTX 5090" in result.stderr
    for rejected in ("9001", "9002", "9003", "9004", "9005"):
        assert f"selected offer {rejected}" not in result.stderr


def test_interruptible_offers_are_never_selected(workdir):
    """A run that dies mid-sweep wastes more than the discount saves."""
    offers = json.loads((REMOTE / "fixtures" / "offers.json").read_text())
    bid_only = [o for o in offers["offers"] if o.get("is_bid_only")]
    assert bid_only, "the fixture must contain an interruptible offer to reject"
    assert min(o["dph_total"] for o in bid_only) < min(
        o["dph_total"] for o in offers["offers"] if not o.get("is_bid_only")
    ), "and it must be the cheapest, so selecting on price alone would pick it"

    result = run(
        "provision.sh",
        "--dry-run",
        "--session-id",
        "select",
        "--ledger",
        str(workdir / "ledger" / "spend.jsonl"),
        "--state-file",
        str(workdir / "state"),
    )

    assert f"selected offer {bid_only[0]['id']}" not in result.stderr


def test_falls_back_to_the_second_generation_card_when_none_match(workdir):
    result = run(
        "provision.sh",
        "--dry-run",
        "--session-id",
        "fallback",
        "--gpu",
        "RTX 6090",  # nothing in the fixture matches
        "--fallback-gpu",
        "RTX 4090",
        "--ledger",
        str(workdir / "ledger" / "spend.jsonl"),
        "--state-file",
        str(workdir / "state"),
    )

    assert "no RTX 6090 offer met the filters" in result.stderr
    assert "selected offer 9006" in result.stderr
    assert "RTX 4090" in result.stderr


def test_exits_cleanly_when_nothing_meets_the_filters(workdir):
    result = run(
        "provision.sh",
        "--dry-run",
        "--session-id",
        "none",
        "--gpu",
        "RTX 6090",
        "--fallback-gpu",
        "",
        "--ledger",
        str(workdir / "ledger" / "spend.jsonl"),
        "--state-file",
        str(workdir / "state"),
        expect=4,
    )

    assert "no offer met the filters" in result.stderr


def test_the_rate_ceiling_is_enforced(workdir):
    result = run(
        "provision.sh",
        "--dry-run",
        "--session-id",
        "cheap",
        "--max-rate",
        "0.10",
        "--ledger",
        str(workdir / "ledger" / "spend.jsonl"),
        "--state-file",
        str(workdir / "state"),
        expect=4,
    )

    assert "no offer met the filters" in result.stderr


# =====================================================================================
# The ledger is written before the instance is used
# =====================================================================================


def test_the_provision_row_is_written_before_the_instance_is_used(workdir):
    ledger = workdir / "ledger" / "spend.jsonl"

    run(
        "provision.sh",
        "--dry-run",
        "--session-id",
        "record",
        "--hypothesis",
        "001-fused-rmsnorm",
        "--ledger",
        str(ledger),
        "--state-file",
        str(workdir / "state"),
    )

    rows = [json.loads(line) for line in ledger.read_text().splitlines()]
    assert len(rows) == 1
    assert rows[0]["event"] == "provision"
    assert rows[0]["session_id"] == "record"
    assert rows[0]["hypothesis"] == "001-fused-rmsnorm"
    assert rows[0]["gpu_model"] == "RTX 5090"
    assert rows[0]["hourly_rate_usd"] == 0.3240
    # The --max-minutes default, 210 minutes, at $0.324/hr. It tracks the hard watchdog
    # rather than the session gate: the ceiling written to the ledger is what this rental
    # could cost at worst, and the watchdog is what bounds that.
    assert rows[0]["estimated_ceiling_usd"] == pytest.approx(1.134)
    assert rows[0]["actual_minutes"] is None, "not reconciled until destroy"


def test_the_ledger_row_is_parseable_by_the_python_reader(workdir):
    """The two readers must agree on the format, or the shell gate and the Python gate
    drift apart silently."""
    from deltaforge.ledger import read_rows

    ledger = workdir / "ledger" / "spend.jsonl"
    run(
        "provision.sh",
        "--dry-run",
        "--session-id",
        "compat",
        "--ledger",
        str(ledger),
        "--state-file",
        str(workdir / "state"),
    )

    rows = read_rows(ledger)

    assert len(rows) == 1
    assert rows[0].event == "provision"
    assert rows[0].gpu_model == "RTX 5090"


# =====================================================================================
# Teardown
# =====================================================================================


def test_teardown_runs_on_the_happy_path(workdir):
    ledger = workdir / "ledger" / "spend.jsonl"

    result = run(
        "run_remote.sh",
        "--dry-run",
        "--session-id",
        "happy",
        "--ledger",
        str(ledger),
        "--state-file",
        str(workdir / "state"),
    )

    assert "[teardown] destroying instance" in result.stderr
    assert "[teardown] complete" in result.stderr
    rows = [json.loads(line) for line in ledger.read_text().splitlines()]
    assert [r["event"] for r in rows] == ["provision", "destroy"]
    assert not (workdir / "state").exists(), "the state file is cleaned up"


@pytest.mark.parametrize("stage", ["sync", "gputests", "correctness", "bench", "pull"])
def test_teardown_still_runs_when_the_run_fails_part_way_through(workdir, stage):
    """The trap is the whole point: a crash, a failed benchmark or an interrupt must
    still destroy the instance and reconcile the ledger."""
    ledger = workdir / "ledger" / "spend.jsonl"

    result = run(
        "run_remote.sh",
        "--dry-run",
        "--session-id",
        f"fail-{stage}",
        "--simulate-failure",
        stage,
        "--ledger",
        str(ledger),
        "--state-file",
        str(workdir / "state"),
        expect=1,
    )

    assert f"simulated failure at stage: {stage}" in result.stderr
    assert "[teardown] destroying instance" in result.stderr
    assert "[teardown] complete" in result.stderr

    rows = [json.loads(line) for line in ledger.read_text().splitlines()]
    assert [r["event"] for r in rows] == ["provision", "destroy"]
    assert rows[1]["actual_minutes"] is not None, "the ledger is reconciled even on failure"
    assert not (workdir / "state").exists()


def test_teardown_works_when_paths_are_given_relative(tmp_path):
    """Regression: a relative `--state-file` used to leave the instance running.

    POSIX `.` searches $PATH when its argument contains no slash, so sourcing a relative
    state file failed with "not found" — *after* the instance had been created. Teardown
    then saw an empty instance id, decided nothing had been provisioned, and destroyed
    nothing. Every other test here passes absolute paths, which is exactly why this
    slipped through.
    """
    result = subprocess.run(
        [
            "sh",
            str(REMOTE / "run_remote.sh"),
            "--dry-run",
            "--session-id",
            "relative-paths",
            "--ledger",
            "scratch.jsonl",
            "--state-file",
            "scratch-state",
        ],
        capture_output=True,
        text=True,
        timeout=180,
        cwd=str(tmp_path),  # relative paths resolve against here, not the repo
    )

    assert result.returncode == 0, result.stderr
    assert "not found" not in result.stderr
    assert "[teardown] destroying instance" in result.stderr
    assert "no instance was created" not in result.stderr

    rows = [json.loads(line) for line in (tmp_path / "scratch.jsonl").read_text().splitlines()]
    assert [r["event"] for r in rows] == ["provision", "destroy"], (
        "the instance must be destroyed and reconciled even with relative paths"
    )


def test_a_failure_before_provisioning_leaves_nothing_to_tear_down(workdir):
    ledger = workdir / "ledger" / "spend.jsonl"

    result = run(
        "run_remote.sh",
        "--dry-run",
        "--session-id",
        "early",
        "--simulate-failure",
        "provision",
        "--ledger",
        str(ledger),
        "--state-file",
        str(workdir / "state"),
        expect=1,
    )

    assert "no instance was created; nothing to destroy" in result.stderr
    assert not ledger.exists() or ledger.read_text().strip() == ""


def test_teardown_never_contacts_the_destroy_endpoint_in_dry_run(workdir):
    result = run(
        "run_remote.sh",
        "--dry-run",
        "--session-id",
        "safe",
        "--ledger",
        str(workdir / "ledger" / "spend.jsonl"),
        "--state-file",
        str(workdir / "state"),
    )

    assert "(dry-run) would DELETE /instances/" in result.stderr


def test_the_watchdog_is_armed_before_any_work_is_attempted(workdir):
    result = run(
        "run_remote.sh",
        "--dry-run",
        "--session-id",
        "watch",
        "--ledger",
        str(workdir / "ledger" / "spend.jsonl"),
        "--state-file",
        str(workdir / "state"),
    )

    lines = result.stderr.splitlines()
    armed = next(i for i, line in enumerate(lines) if "watchdog started" in line)
    work = next(i for i, line in enumerate(lines) if "would rsync" in line)
    assert armed < work


# =====================================================================================
# The watchdog
# =====================================================================================


def test_the_watchdog_fires_and_destroys_after_its_timeout(workdir):
    ledger = workdir / "ledger" / "spend.jsonl"

    result = run(
        "watchdog.sh",
        "--dry-run",
        "--instance-id",
        "test-instance",
        "--timeout-seconds",
        "1",
        "--poll-seconds",
        "1",
        "--ledger",
        str(ledger),
        "--session-id",
        "wd",
        "--rate",
        "0.324",
        expect=2,
    )

    assert "WATCHDOG FIRED" in result.stderr
    assert "This is a fault, not a normal ending" in result.stderr
    assert "(dry-run) would DELETE /instances/test-instance/" in result.stderr


def test_the_watchdog_stands_down_when_the_run_finishes(workdir):
    cancel = workdir / "cancel"
    cancel.touch()

    result = run(
        "watchdog.sh",
        "--dry-run",
        "--instance-id",
        "test-instance",
        "--timeout-seconds",
        "600",
        "--poll-seconds",
        "1",
        "--cancel-file",
        str(cancel),
        expect=0,
    )

    assert "watchdog cancelled" in result.stderr
    assert "WATCHDOG FIRED" not in result.stderr


def test_the_watchdog_requires_an_instance_id():
    result = run("watchdog.sh", "--dry-run", expect=1)
    assert "--instance-id is required" in result.stderr


# =====================================================================================
# Shell and Python gate implementations must agree
# =====================================================================================


AGREEMENT_FIXTURES = {
    "empty": [],
    "single_unreconciled": [ledger_line(instance_id="a")],
    "single_reconciled": [
        ledger_line(instance_id="a"),
        ledger_line(event="destroy", instance_id="a", actual_minutes=30.0, actual_cost_usd=0.162),
    ],
    "mixed": [
        ledger_line(instance_id="a"),
        ledger_line(event="destroy", instance_id="a", actual_minutes=12.5, actual_cost_usd=0.0675),
        ledger_line(instance_id="b", estimated_ceiling_usd=1.25),
        ledger_line(instance_id="c", session_id="s2"),
        ledger_line(
            event="destroy",
            instance_id="c",
            session_id="s2",
            actual_minutes=44.0,
            actual_cost_usd=0.2376,
        ),
    ],
    "double_reconciled": [
        # Watchdog destroy + teardown-trap destroy for one instance. Both readers must
        # count it once, at the larger value.
        ledger_line(instance_id="a"),
        ledger_line(event="destroy", instance_id="a", actual_minutes=90.0, actual_cost_usd=0.486),
        ledger_line(event="destroy", instance_id="a", actual_minutes=89.5, actual_cost_usd=0.4833),
    ],
    "spans_months": [
        ledger_line(instance_id="old", ts="2026-07-31T23:00:00Z"),
        ledger_line(
            event="destroy",
            instance_id="old",
            ts="2026-08-01T00:30:00Z",
            actual_minutes=90.0,
            actual_cost_usd=0.486,
        ),
    ],
}


@pytest.mark.parametrize("name", sorted(AGREEMENT_FIXTURES))
def test_shell_and_python_month_to_date_agree(workdir, name):
    from deltaforge.ledger import month_to_date_usd, read_rows

    ledger = write_ledger(workdir / "ledger" / "spend.jsonl", AGREEMENT_FIXTURES[name])
    month = "2026-08"

    shell = subprocess.run(
        [
            "sh",
            "-c",
            f'DF_REPO_ROOT="{REPO_ROOT}"; . "{REMOTE}/lib.sh"; df_ledger_month_to_date "{ledger}" "{month}"',
        ],
        capture_output=True,
        text=True,
        timeout=60,
        check=True,
    )

    assert float(shell.stdout.strip()) == pytest.approx(
        month_to_date_usd(read_rows(ledger), month=month), abs=1e-6
    )


@pytest.mark.parametrize("name", sorted(AGREEMENT_FIXTURES))
def test_shell_and_python_session_minutes_agree(workdir, name):
    from deltaforge.ledger import read_rows, session_minutes

    ledger = write_ledger(workdir / "ledger" / "spend.jsonl", AGREEMENT_FIXTURES[name])
    now = EPOCH + 3600

    shell = subprocess.run(
        [
            "sh",
            "-c",
            f'DF_REPO_ROOT="{REPO_ROOT}"; . "{REMOTE}/lib.sh"; '
            f'df_ledger_session_minutes "{ledger}" "s1" {now}',
        ],
        capture_output=True,
        text=True,
        timeout=60,
        check=True,
    )

    assert float(shell.stdout.strip()) == pytest.approx(
        session_minutes(read_rows(ledger), "s1", now_epoch=now), abs=1e-6
    )


# =====================================================================================
# The API key must never leak
# =====================================================================================


def test_the_api_key_never_appears_in_output(workdir, monkeypatch):
    """The key is passed to curl through a config file on stdin, so it is never in argv,
    never on disk, and must never reach a log, a results file or a PR body."""
    secret = "vast-secret-value-that-must-not-appear-anywhere"
    env = {
        **dict(__import__("os").environ),
        "VAST_API_KEY": secret,
    }

    result = run(
        "provision.sh",
        "--dry-run",
        "--session-id",
        "secret-test",
        "--ledger",
        str(workdir / "ledger" / "spend.jsonl"),
        "--state-file",
        str(workdir / "state"),
        env=env,
    )

    assert secret not in result.stdout
    assert secret not in result.stderr
    assert secret not in (workdir / "state").read_text()
    assert secret not in (workdir / "ledger" / "spend.jsonl").read_text()
    # It confirms the key loaded without disclosing it.
    assert "API key loaded" in result.stderr
    assert str(len(secret)) in result.stderr


def test_the_api_key_is_read_from_a_dotenv_file(workdir):
    env_file = workdir / ".env"
    env_file.write_text('VAST_API_KEY="dotenv-secret-0123456789"\n')

    probe = subprocess.run(
        [
            "sh",
            "-c",
            f'DF_REPO_ROOT="{workdir}"; . "{REMOTE}/lib.sh"; '
            'df_load_api_key && printf "%s" "${#DF_API_KEY}"',
        ],
        capture_output=True,
        text=True,
        timeout=60,
        check=True,
        env={k: v for k, v in __import__("os").environ.items() if k != "VAST_API_KEY"},
    )

    assert probe.stdout.strip() == str(len("dotenv-secret-0123456789"))
    assert "dotenv-secret" not in probe.stderr


def test_dotenv_is_gitignored():
    assert ".env" in (REPO_ROOT / ".gitignore").read_text().splitlines()


def test_sync_never_transfers_the_dotenv_file():
    assert "--exclude=.env" in (REMOTE / "sync.sh").read_text()


def test_sync_dry_run_reports_both_directions(workdir):
    up = run("sync.sh", "up", "--dry-run")
    down = run("sync.sh", "down", "--dry-run")

    assert "would rsync" in up.stderr
    assert "excluding .env" in up.stderr
    assert "results/" in down.stderr


# =====================================================================================
# Shell hygiene
# =====================================================================================


@pytest.mark.parametrize("script", ["lib.sh", "provision.sh", "watchdog.sh", "sync.sh", "run_remote.sh"])
def test_scripts_parse_as_posix_sh(script):
    result = subprocess.run(["sh", "-n", str(REMOTE / script)], capture_output=True, text=True, timeout=60)
    assert result.returncode == 0, result.stderr


@pytest.mark.parametrize("script", ["provision.sh", "watchdog.sh", "sync.sh", "run_remote.sh"])
def test_scripts_are_executable_and_support_help(script):
    assert (REMOTE / script).stat().st_mode & 0o111, f"{script} is not executable"
    result = run(script, "--help")
    assert "Usage:" in result.stdout


@pytest.mark.parametrize("script", ["provision.sh", "watchdog.sh", "sync.sh", "run_remote.sh"])
def test_every_script_supports_dry_run(script):
    assert "--dry-run" in (REMOTE / script).read_text()


@pytest.mark.parametrize("script", ["lib.sh", "provision.sh", "watchdog.sh", "sync.sh", "run_remote.sh"])
def test_no_script_enables_shell_tracing(script):
    """`set -x` would print the API key. The scripts must never turn it on."""
    body = (REMOTE / script).read_text()
    for line in body.splitlines():
        stripped = line.strip()
        assert not stripped.startswith("set -x")
        assert "set -eux" not in stripped


def _shell(expr):
    return subprocess.run(
        ["sh", "-c", f'DF_REPO_ROOT="{REPO_ROOT}"; . "{REMOTE}/lib.sh"; {expr}'],
        capture_output=True,
        text=True,
        timeout=60,
        check=True,
    ).stdout.strip()


def test_the_shell_gate_counts_a_double_reconciled_instance_once(workdir):
    """The awk readers are what provision.sh and run_remote.sh actually gate on, so a
    Python-only dedup would leave the real gate double-counting."""
    ledger = write_ledger(workdir / "ledger" / "spend.jsonl", AGREEMENT_FIXTURES["double_reconciled"])

    spend = float(_shell(f'df_ledger_month_to_date "{ledger}" "2026-08"'))
    minutes = float(_shell(f'df_ledger_session_minutes "{ledger}" "s1" {EPOCH + 7200}'))

    assert spend == pytest.approx(0.486, abs=1e-6)
    assert minutes == pytest.approx(90.0, abs=1e-6)


def test_a_dry_run_never_writes_to_the_real_spend_ledger(workdir):
    """A rehearsal must read the real budget record and write to a scratch copy.

    Synthetic provision/destroy rows in the real ledger would inflate month-to-date spend
    for every later session, so the gate that exists to stop a runaway bill would start
    refusing runs on money that was never spent.
    """
    real = REPO_ROOT / "ledger" / "spend.jsonl"
    before = real.read_text() if real.exists() else ""
    scratch = workdir / "dryrun-spend.jsonl"

    run(
        "run_remote.sh",
        "--dry-run",
        "--session-id",
        "dryrun-isolation",
        env={**os.environ, "DF_DRYRUN_LEDGER": str(scratch)},
    )

    after = real.read_text() if real.exists() else ""
    assert after == before, "the rehearsal wrote synthetic rows into the real spend record"

    rows = [line for line in scratch.read_text().splitlines() if line.strip()]
    assert any('"event":"provision"' in row for row in rows), "no provision row was recorded at all"
    assert any('"event":"destroy"' in row for row in rows), "teardown did not reconcile"
    # Seeded from the real record, so the gates still read real spend.
    assert scratch.read_text().startswith(before)


def test_an_explicit_ledger_path_is_honoured_even_in_a_dry_run(workdir):
    """The scratch redirect must not hijack a ledger the caller named — the fixture-driven
    gate tests above depend on their rows landing where they said."""
    ledger = write_ledger(workdir / "ledger" / "spend.jsonl", [])

    run(
        "run_remote.sh",
        "--dry-run",
        "--session-id",
        "explicit-ledger",
        "--ledger",
        str(ledger),
        "--state-file",
        str(workdir / "state"),
    )

    rows = [line for line in ledger.read_text().splitlines() if line.strip()]
    assert any('"event":"provision"' in row for row in rows)


# =====================================================================================
# Instance readiness polling
#
# `GET /api/v0/instances/<id>/` answers `{"instances": null}` for a live instance instead
# of failing, so a poller built on it burns its entire timeout reporting an unknown
# status — which is exactly what cost one provisioning cycle. The listing moved to v1 and
# returns a list; these pin the shape the poller depends on.
# =====================================================================================


INSTANCES_FIXTURE = REMOTE / "fixtures" / "instances.json"


def _row(instance_id: str) -> str:
    return _shell(
        f'df_instance_row "$(cat {INSTANCES_FIXTURE})" "{instance_id}"',
    )


def test_the_running_instance_is_found_by_id_in_a_v1_listing():
    row = json.loads(_row("49700454"))

    assert row["ssh_host"] == "ssh5.vast.ai"
    assert row["ssh_port"] == 41234


def test_readiness_is_the_container_state_and_never_falls_back_to_cur_state():
    """`cur_state` is the *contract* state and says "running" from the moment the instance
    is created; `actual_status` is the *container* state and goes null -> loading ->
    running. The poller used to read `.actual_status // .cur_state`, so the fallback fired
    exactly when `actual_status` had not been populated yet -- precisely when the instance
    is not ready. Readiness was therefore declared on the first poll of every rental and
    the ssh probe ran against a container that did not exist.

    Five rentals were written off as "the host never answered sshd" before the log showed
    `status: loading` inside the probe loop. The fixture below is the trap in miniature:
    `actual_status: null` with `cur_state: "running"` is a freshly created instance."""
    listing = INSTANCES_FIXTURE.read_text()
    row = json.loads(_row("49700454"))

    # The fixture is a not-yet-ready instance, whatever cur_state claims.
    assert row["actual_status"] is None
    assert row["cur_state"] == "running"

    gated = _shell(
        "row=$(df_instance_row '" + listing.replace("'", "") + "' 49700454); "
        "printf '%s' \"$row\" | jq -r '.actual_status // empty'"
    )
    assert gated == "", "an unpopulated actual_status must read as not-ready, not ready"

    source = (REMOTE / "run_remote.sh").read_text()
    assert ".actual_status // .cur_state" not in source, (
        "the backwards fallback is the bug; it must not come back"
    )
    assert 'if [ "$_actual" = "running" ]; then' in source


def test_another_instance_in_the_same_listing_is_not_confused_for_ours():
    """Two instances on the account is the ordinary case once anything runs in parallel;
    picking the wrong row would ssh into someone else's work."""
    row = json.loads(_row("49700999"))

    assert row["actual_status"] == "loading"
    assert row["gpu_name"] == "RTX 4090"


def test_an_instance_absent_from_the_listing_yields_no_row():
    assert _row("1234") == ""


def test_a_deprecated_or_broken_response_yields_no_row_rather_than_a_crash():
    for payload in ('{"instances": null}', '{"success": false, "error": "deprecated_endpoint"}', "not json"):
        assert _shell(f"df_instance_row '{payload}' 49700454") == "", (
            f"{payload!r} should parse to no row, not to a false positive"
        )


def test_a_slow_link_is_rejected_however_cheap_it_is(workdir):
    """A run downloads a ~9 GB image and a ~9 GB checkpoint before it computes anything,
    so link speed is a cost input. Offer 9009 is the cheapest in the fixture and has a
    42 Mbit link; it must never be selected."""
    result = run(
        "provision.sh",
        "--dry-run",
        "--session-id",
        "slow-link",
        "--ledger",
        str(write_ledger(workdir / "ledger" / "spend.jsonl", [])),
        "--state-file",
        str(workdir / "state"),
    )

    assert "selected offer 9009" not in result.stderr


def test_an_unverified_host_is_rejected_however_cheap_it_is(workdir):
    """Three provisioning attempts were spent on an unverified consumer host that never
    finished pulling the image and whose ssh proxy was unreachable. Offer 9010 stands in
    for it."""
    result = run(
        "provision.sh",
        "--dry-run",
        "--session-id",
        "unverified",
        "--ledger",
        str(write_ledger(workdir / "ledger" / "spend.jsonl", [])),
        "--state-file",
        str(workdir / "state"),
    )

    assert "selected offer 9010" not in result.stderr


def test_a_machine_that_already_cost_a_rental_can_be_excluded(workdir):
    """The market is ordered by price and deterministic, so a host that burns a rental
    without producing a result gets selected again on the next run unless it is named."""
    args = (
        "--dry-run",
        "--session-id",
        "excluded-machine",
        "--ledger",
        str(write_ledger(workdir / "ledger" / "spend.jsonl", [])),
        "--state-file",
        str(workdir / "state"),
    )

    normal = run("provision.sh", *args)
    assert "selected offer 9008" in normal.stderr

    excluded = run(
        "provision.sh",
        *args,
        env={**os.environ, "DF_EXCLUDE_MACHINES": "100008"},
    )

    assert "selected offer 9008" not in excluded.stderr
    assert "selected offer" in excluded.stderr, "excluding one machine must not empty the market"


# ---------------------------------------------------------------------------
# Registry credentials and stalled image pulls
#
# Every container image this project has failed to pull came from Docker Hub, pulled
# anonymously. These cover the two changes made in response: authenticate when we can,
# and stop paying for a pull that is stuck rather than slow.
# ---------------------------------------------------------------------------


def _env_file(workdir, **values):
    path = workdir / ".env"
    path.write_text("".join(f"{k}={v}\n" for k, v in values.items()))
    path.chmod(0o600)
    return path


def test_registry_login_is_absent_from_the_create_body_when_no_credentials_exist(workdir):
    """The change must be inert for anyone without a Docker Hub account."""
    env = _env_file(workdir, VAST_API_KEY="not-a-real-key")

    body = _shell(
        f'df_load_registry_login "{env}" >/dev/null 2>&1 || true; '
        "DF_IMAGE=img DF_DISK_GB=40 DF_MAX_MINUTES=90; "
        'if [ -n "${DF_REGISTRY_LOGIN:-}" ]; then echo HAS_LOGIN; else echo NO_LOGIN; fi'
    )

    assert body == "NO_LOGIN"


def test_registry_login_is_read_from_env_and_formatted_for_vast(workdir):
    env = _env_file(
        workdir,
        VAST_API_KEY="not-a-real-key",
        DOCKER_LOGIN_USER="someuser",
        DOCKER_LOGIN_TOKEN="dckr_pat_TOPSECRET",
    )

    login = _shell(f'df_load_registry_login "{env}" >/dev/null 2>&1; printf "%s" "$DF_REGISTRY_LOGIN"')

    assert login == "-u someuser -p dckr_pat_TOPSECRET"


def test_loading_registry_credentials_never_prints_the_token(workdir):
    """The whole secret discipline of this repo is that a token reaches curl and nothing
    else. A log line with the token in it would be committed by whoever pastes a session
    transcript into a PR."""
    env = _env_file(
        workdir,
        VAST_API_KEY="not-a-real-key",
        DOCKER_LOGIN_USER="someuser",
        DOCKER_LOGIN_TOKEN="dckr_pat_TOPSECRET",
    )

    proc = subprocess.run(
        ["sh", "-c", f'DF_REPO_ROOT="{REPO_ROOT}"; . "{REMOTE}/lib.sh"; df_load_registry_login "{env}"'],
        capture_output=True,
        text=True,
        timeout=60,
    )

    assert "dckr_pat_TOPSECRET" not in proc.stdout + proc.stderr
    assert "someuser" in proc.stdout + proc.stderr  # the username is fine, and useful


def test_a_request_body_never_reaches_argv(workdir):
    """`--data "$body"` would put a registry token in argv, which `ps` exposes to every
    user on the box. The body must travel through a 0600 file instead."""
    source = (REMOTE / "lib.sh").read_text()
    # Comments are allowed to quote the bad form; only real invocations matter.
    invocations = [
        line for line in source.splitlines() if "--data" in line and not line.lstrip().startswith("#")
    ]

    assert invocations, "no --data invocation found; has the API helper been rewritten?"
    for line in invocations:
        assert '--data "@' in line, f"request body passed through argv: {line.strip()}"
    assert "umask 077; mktemp" in source


def test_the_create_body_carries_image_login_only_when_credentials_are_present(workdir):
    """provision.sh builds the body; both branches must be well-formed JSON."""
    script = (REMOTE / "provision.sh").read_text()
    start = script.index("create_body() {")
    end = script.index("\n}", start) + 2
    create_body = script[start:end]

    with_creds = _shell(
        f"{create_body}\nDF_IMAGE=img DF_DISK_GB=40 DF_MAX_MINUTES=90 "
        'DF_REGISTRY_LOGIN="-u u -p t"; create_body'
    )
    without = _shell(f"{create_body}\nDF_IMAGE=img DF_DISK_GB=40 DF_MAX_MINUTES=90; create_body")

    assert json.loads(with_creds)["image_login"] == "-u u -p t"
    assert "image_login" not in json.loads(without)
    assert json.loads(without)["image"] == "img"


def test_a_stalled_image_pull_aborts_long_before_the_readiness_timeout():
    """A pull that is slow rewrites status_msg with new byte counts; a stuck one repeats
    the same line. Eight rentals were billed for the full readiness timeout because
    nothing told them apart. The stall budget must be well under that timeout, or the
    check cannot fire before the money is already spent."""
    source = (REMOTE / "run_remote.sh").read_text()

    stall = int(re.search(r"DF_PULL_STALL_SECONDS:-(\d+)", source).group(1))
    ready = int(re.search(r"DF_SSH_READY_TIMEOUT:-(\d+)", source).group(1))

    assert stall < ready / 2, "stall detection must fire well before the readiness timeout"
    assert "_msg_changed_at" in source
    assert "Image pull is stuck, not slow" in source


def _create_body(**env):
    """Run provision.sh's create_body in isolation, with lib.sh loaded."""
    script = (REMOTE / "provision.sh").read_text()
    funcs = ""
    for name in ("df_public_key", "create_body"):
        start = script.index(f"{name}() {{")
        funcs += script[start : script.index("\n}", start) + 2] + "\n"
    assignments = " ".join(f'{k}="{v}"' for k, v in env.items())
    return json.loads(_shell(f"{funcs}\n{assignments}; create_body"))


def test_the_create_body_injects_the_public_key_two_ways(workdir):
    """Vast associates the account key with the instance, but that only reaches images
    built to its conventions. A ghcr.io/ai-dock image pulled, ran, and then answered ssh
    with "Permission denied (publickey)" -- one rental to learn that the key must be
    injected explicitly. `PUBLIC_KEY` covers the ai-dock and RunPod families;
    authorized_keys covers everything else."""
    key = workdir / "id.pub"
    key.write_text("ssh-ed25519 AAAATESTKEY someone@example\n")

    body = _create_body(DF_IMAGE="img", DF_DISK_GB="40", DF_MAX_MINUTES="90", DF_SSH_KEY=str(workdir / "id"))

    assert "ssh-ed25519 AAAATESTKEY" in body["env"]
    assert "authorized_keys" in body["onstart"]
    assert "ssh-ed25519 AAAATESTKEY" in body["onstart"]
    # Still shuts itself down: the remote-side backstop must survive the new prefix.
    assert body["onstart"].rstrip().endswith("shutdown -h +90")


def test_the_create_body_is_valid_json_without_a_public_key(workdir):
    """A checkout with no generated key must still produce a well-formed request."""
    body = _create_body(
        DF_IMAGE="img", DF_DISK_GB="40", DF_MAX_MINUTES="90", DF_SSH_KEY=str(workdir / "absent")
    )

    assert "env" not in body
    assert "authorized_keys" not in body["onstart"]
    assert body["onstart"] == "touch ~/.no_auto_tmux; shutdown -h +90"


def test_a_rejected_ssh_key_fails_fast_instead_of_waiting_out_the_timeout():
    """An image answering on the ssh port and refusing the key is a permanent failure.
    Waiting cannot fix it, and the readiness timeout is 20 billed minutes."""
    source = (REMOTE / "run_remote.sh").read_text()

    assert '*"Permission denied"*' in source
    assert "refused the ssh key" in source


def test_the_sshd_probe_loop_has_its_own_stall_budget():
    """The first version of the stall guard covered only the outer poll. A rental then sat
    in the inner sshd probe for the full deadline -- exactly the case the guard was for."""
    source = (REMOTE / "run_remote.sh").read_text()
    probe = source[source.index("_ssh_started=") : source.index("did not become reachable")]

    assert "DF_PULL_STALL_SECONDS" in probe
    assert "without sshd answering" in probe


def test_the_sshd_probe_loop_tracks_progress_rather_than_counting_blindly():
    """`cur_state` can report `running` on the very first poll, before the image has
    landed. A flat countdown from that moment destroys healthy hosts that are simply
    still starting -- it cost two rentals on 2026-09-08, on two different machines,
    each killed at exactly the stall budget with zero outer poll iterations.

    The fix is the one the outer loop already had: watch `status_msg`. A container still
    rewriting it is progressing and keeps the full readiness timeout; only a static one
    is destroyed early."""
    source = (REMOTE / "run_remote.sh").read_text()
    probe = source[source.index("_ssh_started=") : source.index("did not become reachable")]

    assert "_ssh_msg_changed_at" in probe, "the sshd loop must track status_msg progress"
    assert "status_msg" in probe, "the sshd loop must read status_msg, not just the clock"
    # The budget must reset on progress, not run from _ssh_started unconditionally.
    assert "_ssh_msg_changed_at=$(df_now_epoch)" in probe
    assert "$(( $(df_now_epoch) - _ssh_started )) -ge" not in probe, (
        "a flat countdown from _ssh_started is the blind guard this replaced"
    )
    # The waiting line must surface what the instance is actually doing, so a future
    # failure is diagnosable from the log alone rather than costing another rental.
    assert "waiting for sshd (status:" in probe


# ---------------------------------------------------------------------------
# Batch mode
# ---------------------------------------------------------------------------


def test_run_remote_refuses_batch_and_hypothesis_together(workdir):
    """Guessing which one the run means would mislabel every record it writes."""
    result = run(
        "run_remote.sh",
        "--dry-run",
        "--session-id",
        "ambiguous",
        "--batch",
        "001-calibration",
        "--hypothesis",
        "006-gqa-no-expand",
        "--ledger",
        str(workdir / "ledger" / "spend.jsonl"),
        "--state-file",
        str(workdir / "state"),
        expect=1,
    )

    assert "mutually exclusive" in result.stderr
    # It refused before provisioning: an ambiguous run must not cost a rental.
    assert "provisioning..." not in result.stderr


def test_a_batch_run_invokes_the_batch_command_with_a_deadline(workdir):
    result = run(
        "run_remote.sh",
        "--dry-run",
        "--session-id",
        "batchrun",
        "--batch",
        "001-calibration",
        "--ledger",
        str(workdir / "ledger" / "spend.jsonl"),
        "--state-file",
        str(workdir / "state"),
    )

    assert "deltaforge.cli batch" in result.stderr
    assert "--batch '001-calibration'" in result.stderr
    assert "--deadline-epoch" in result.stderr
    # The batch replaces both single-hypothesis steps rather than running alongside them.
    assert "deltaforge.cli correctness" not in result.stderr
    assert "deltaforge.cli bench" not in result.stderr


def test_a_single_hypothesis_run_still_takes_the_old_path(workdir):
    result = run(
        "run_remote.sh",
        "--dry-run",
        "--session-id",
        "solo",
        "--hypothesis",
        "001-fused-rmsnorm-residual",
        "--ledger",
        str(workdir / "ledger" / "spend.jsonl"),
        "--state-file",
        str(workdir / "state"),
    )

    assert "deltaforge.cli correctness" in result.stderr
    assert "deltaforge.cli bench" in result.stderr
    assert "deltaforge.cli batch" not in result.stderr


def test_the_batch_deadline_leaves_the_session_gate_room_for_teardown(workdir):
    """The batch must stop itself before the watchdog does.

    AGENT.md treats a watchdog firing as a reportable fault, so the deadline handed to the
    batch is the session gate minus what is already spent minus a teardown reserve — never
    the raw gate.
    """
    result = run(
        "run_remote.sh",
        "--dry-run",
        "--session-id",
        "reserved",
        "--batch",
        "001-calibration",
        "--session-limit",
        "180",
        "--ledger",
        str(workdir / "ledger" / "spend.jsonl"),
        "--state-file",
        str(workdir / "state"),
    )

    line = next(ln for ln in result.stderr.splitlines() if "deadline in" in ln)
    minutes = float(line.split("deadline in")[1].split("minutes")[0].strip())
    # 180 minute gate, nothing spent, 12 minute reserve. The limit cannot be dropped below
    # the gate's default here: the pre-flight check refuses a session that cannot fit one
    # hypothesis, and on cold estimates that needs ~143 minutes.
    assert 167.0 <= minutes <= 168.5, line


def test_the_session_gate_defaults_to_three_hours(workdir):
    ledger = write_ledger(
        workdir / "ledger" / "spend.jsonl",
        [
            ledger_line(session_id="spent", instance_id="a"),
            ledger_line(
                event="destroy",
                session_id="spent",
                instance_id="a",
                actual_minutes=180.0,
                actual_cost_usd=1.068,
            ),
        ],
    )

    result = run(
        "run_remote.sh",
        "--dry-run",
        "--session-id",
        "spent",
        "--ledger",
        str(ledger),
        "--state-file",
        str(workdir / "state"),
        expect=3,
    )

    assert "REFUSED by the session GPU-time gate" in result.stderr


def test_the_hard_watchdog_stays_above_the_session_gate():
    """The gate ends a run; the watchdog only catches a hang.

    A watchdog firing is a reportable fault (AGENT.md §5), so its default must sit above
    the session gate's. Raising the gate to 120 without moving the watchdog would have
    made the backstop the routine control and turned every long batch into a fault.
    """
    script = (REPO_ROOT / "remote" / "run_remote.sh").read_text()

    def default_of(name):
        line = next(ln for ln in script.splitlines() if ln.startswith(f"{name}="))
        return float(line.split(":-")[1].split("}")[0])

    gate = default_of("DF_SESSION_LIMIT_MINUTES")
    watchdog = default_of("DF_WATCHDOG_MINUTES")

    assert gate == 180.0
    assert watchdog > gate


def test_the_two_refusals_are_distinguishable(workdir):
    """A session with 90 of its 180 minutes spent passes the GPU-time gate and is still
    refused, because 90 minutes cannot fit one hypothesis on a cold cache.

    The two refusals mean different things and a session must be able to tell them apart:
    the gate says "this session is spent", the pre-flight says "what is left cannot buy a
    measurement". Reporting either as the other sends the next session after the wrong fix.
    """
    ledger = write_ledger(
        workdir / "ledger" / "spend.jsonl",
        [
            ledger_line(session_id="half", instance_id="a"),
            ledger_line(
                event="destroy",
                session_id="half",
                instance_id="a",
                actual_minutes=90.0,
                actual_cost_usd=0.534,
            ),
        ],
    )

    result = run(
        "run_remote.sh",
        "--dry-run",
        "--session-id",
        "half",
        "--batch",
        "001-calibration",
        "--ledger",
        str(ledger),
        "--state-file",
        str(workdir / "state"),
        expect=5,
    )

    assert "REFUSED by the session GPU-time gate" not in result.stderr
    assert "cannot fit one hypothesis" in result.stderr


def test_teardown_pulls_results_before_destroying(workdir):
    """A failure late in a 90-minute batch must not take the slots that succeeded.

    The trap could only ever destroy, and it cannot rsync from a dead box, so every
    measurement after the last sync was lost. Dry-run skips the real pull, so what is
    pinned here is the ordering in the teardown path itself.
    """
    script = (REPO_ROOT / "remote" / "run_remote.sh").read_text()
    teardown = script[script.index("df_teardown() {") : script.index("trap 'df_teardown' EXIT")]

    pull_at = teardown.index("pulling results before destroying")
    destroy_at = teardown.index("df_vast_destroy")
    assert pull_at < destroy_at, "teardown must pull results before it destroys the instance"
    # And the pull must never be able to block the destroy.
    assert "destroying anyway" in teardown


def test_a_python_interpreter_is_established_before_anything_uses_one():
    """Cost one rental (50119910, 2026-09-07, $0.0557).

    The image ships `python3` and `pip` but no `python`, so every remote step after the
    installs died with `command not found` — after the container pull, the torch download
    and the pip installs had all been paid for. The interpreter has to be the first thing
    established, not the twentieth thing discovered.
    """
    script = (REPO_ROOT / "remote" / "run_remote.sh").read_text()
    body = script[script.index("preparing the remote environment") :]

    link_at = body.index("/usr/local/bin/python")
    version_at = body.index("python --version")
    first_use = min(
        body.index("python -m pip install"),
        body.index("python -m deltaforge.cli"),
    )
    assert link_at < version_at < first_use, (
        "the python symlink and its verification must come before the first use of python"
    )


def test_pip_runs_through_the_same_interpreter_everything_else_uses():
    """A bare `pip` can belong to a different interpreter than `python`, which installs
    torch somewhere the benchmark cannot import it — a failure that appears only after the
    3 GB download has been paid for."""
    script = (REPO_ROOT / "remote" / "run_remote.sh").read_text()
    body = script[script.index("preparing the remote environment") :]

    for line in body.splitlines():
        if "remote_sh" in line and "pip install" in line:
            assert "python -m pip install" in line, f"bare pip invocation: {line.strip()}"


def test_the_fast_checkpoint_download_is_not_requested_through_a_dropped_extra():
    """Two deprecations deep, so both ends are pinned.

    huggingface-hub 1.30 dropped the `hf_transfer` extra, so `huggingface_hub[hf_transfer]`
    warns and installs nothing — silently leaving a 9.32 GB download on the
    single-connection path, on billed wall-clock time. Then the box told us the rest:
    hf_transfer is itself superseded by Xet, which ignores `HF_HUB_ENABLE_HF_TRANSFER` and
    reads `HF_XET_HIGH_PERFORMANCE`. Both variables are set, so whichever the installed
    version honours, one of them wins."""
    script = (REPO_ROOT / "remote" / "run_remote.sh").read_text()
    cli = (REPO_ROOT / "src" / "deltaforge" / "cli.py").read_text()

    # Scoped to the install commands: the comment above them names the dropped extra on
    # purpose, to explain why it is not used.
    install_lines = [ln for ln in script.splitlines() if "pip install" in ln and "remote_sh" in ln]
    assert install_lines
    for line in install_lines:
        assert "huggingface_hub[hf_transfer]" not in line, f"the dropped extra is back: {line.strip()}"
    assert "HF_HUB_ENABLE_HF_TRANSFER" in cli
    assert "HF_XET_HIGH_PERFORMANCE" in cli


def test_the_teardown_pull_is_skipped_when_ssh_never_answered():
    """Rental 50121263: sshd never answered, so there was nothing on the box to pull, and
    trying anyway spent the connect timeout on an instance that was still billing."""
    script = (REPO_ROOT / "remote" / "run_remote.sh").read_text()
    teardown = script[script.index("df_teardown() {") : script.index("trap 'df_teardown' EXIT")]

    assert "DF_SSH_READY" in teardown, "the pull must be gated on ssh having actually answered"
    # And bounded, because it runs while the instance is still billing.
    assert "timeout" in teardown[: teardown.index("df_vast_destroy")]


def test_ssh_readiness_is_recorded_when_ssh_answers():
    script = (REPO_ROOT / "remote" / "run_remote.sh").read_text()
    probe = script[script.index("ssh is answering") - 400 : script.index("ssh is answering") + 200]
    assert "DF_SSH_READY=1" in probe


def test_provisioning_logs_the_machine_id_not_only_the_offer_id():
    """--exclude-machines takes a MACHINE id. Logging only the offer id made the remedy
    docs/GPU-ACCESS.md prescribes for a host that burns a rental impossible to carry out."""
    script = (REPO_ROOT / "remote" / "provision.sh").read_text()

    assert "machine_id" in script
    assert "--exclude-machines $OFFER_MACHINE" in script


def test_the_selected_offer_records_its_machine_in_the_ledger(workdir):
    result = run(
        "provision.sh",
        "--dry-run",
        "--session-id",
        "machineid",
        "--ledger",
        str(workdir / "ledger" / "spend.jsonl"),
        "--state-file",
        str(workdir / "state"),
    )

    assert "on machine" in result.stderr
    assert "--exclude-machines" in result.stderr
    row = json.loads((workdir / "ledger" / "spend.jsonl").read_text().strip().splitlines()[-1])
    assert "machine" in row["note"]


def test_accelerate_is_installed_because_the_oracle_cannot_load_without_it():
    """Cost rental 50121911: 10.70 minutes and a full 9.32 GB checkpoint download to reach
    `ValueError: Using a device_map ... requires accelerate`.

    Without it `transformers.from_pretrained` refuses any `device_map`, so the weight-value
    oracle cannot be constructed at all -- and until that oracle passes, the reference is
    proven structurally correct but not proven to read weight *values* correctly, which
    makes every number downstream of it a measurement of an unvalidated model."""
    script = (REPO_ROOT / "remote" / "run_remote.sh").read_text()

    install_lines = [ln for ln in script.splitlines() if "pip install" in ln and "remote_sh" in ln]
    assert any("accelerate" in ln for ln in install_lines), "accelerate is not installed"
    # And verified out loud, so a resolver that drops it is visible in the log.
    assert "import accelerate" in script


# =====================================================================================
# The compile cache
# =====================================================================================


def test_the_remote_steps_point_torch_at_a_cache_that_is_pulled_home(workdir):
    """Torch's fx-graph and autotune caches are on by default and write to /tmp on a box we
    destroy, so every rental this project has run compiled cold."""
    result = run(
        "run_remote.sh",
        "--dry-run",
        "--session-id",
        "cache",
        "--batch",
        "001-calibration",
        "--ledger",
        str(workdir / "ledger" / "spend.jsonl"),
        "--state-file",
        str(workdir / "state"),
    )

    assert "TORCHINDUCTOR_CACHE_DIR=/workspace/df-cache/inductor" in result.stderr
    assert "TRITON_CACHE_DIR=/workspace/df-cache/triton" in result.stderr
    assert "TORCHINDUCTOR_COMPILE_THREADS=" in result.stderr


def test_the_compile_cache_is_pulled_after_the_results_and_before_the_destroy():
    """Same ordering rule as the results pull, for the same reason: the trap cannot rsync
    from a dead box. Results first, because a cold compile costs 40 minutes and a lost
    measurement costs the rental."""
    script = (REPO_ROOT / "remote" / "run_remote.sh").read_text()
    teardown = script[script.index("df_teardown() {") : script.index("trap 'df_teardown' EXIT")]

    results_at = teardown.index("pulling results before destroying")
    cache_at = teardown.index("pulling the compile cache")
    destroy_at = teardown.index("destroying instance")

    assert results_at < cache_at < destroy_at


def test_the_compile_cache_pull_cannot_block_the_destroy():
    """A leaked instance costs about $13/day. The cache is worth 40 minutes, once."""
    script = (REPO_ROOT / "remote" / "run_remote.sh").read_text()
    teardown = script[script.index("df_teardown() {") : script.index("trap 'df_teardown' EXIT")]
    start = teardown.index("pulling the compile cache")
    cache_block = teardown[start : teardown.index("destroying instance")]

    assert "timeout" in cache_block


def test_a_cache_pull_with_no_cache_on_the_box_is_not_a_failure(workdir):
    """The first rental after this lands has nothing to bring home, and must not report a
    teardown failure for it."""
    result = run(
        "sync.sh",
        "cache-down",
        "--dry-run",
        "--cache-key",
        "RTX5090-2.14-cu128",
    )

    assert result.returncode == 0


def test_a_session_that_cannot_fit_one_hypothesis_is_refused_before_renting(workdir):
    """The other gate refuses a session that is spent. This one refuses a session that is
    not spent enough: time to rent and compile, but not to score anything.

    Nine rentals were billed on this project without producing a number, and the cheapest
    of those failures would have been not renting.
    """
    ledger = write_ledger(
        workdir / "ledger" / "spend.jsonl",
        [
            ledger_line(session_id="tight", instance_id="a"),
            ledger_line(
                event="destroy",
                session_id="tight",
                instance_id="a",
                actual_minutes=170.0,
                actual_cost_usd=1.008,
            ),
        ],
    )

    result = run(
        "run_remote.sh",
        "--dry-run",
        "--session-id",
        "tight",
        "--batch",
        "001-calibration",
        "--ledger",
        str(ledger),
        "--state-file",
        str(workdir / "state"),
        expect=5,
    )

    assert "cannot fit one hypothesis" in result.stderr


def test_a_fresh_session_passes_the_pre_flight_check(workdir):
    """The cold estimates must fit the gate they are checked against, or the change that
    raised the gate and the change that added the check disagree and nothing ever runs."""
    result = run(
        "run_remote.sh",
        "--dry-run",
        "--session-id",
        "fresh",
        "--batch",
        "001-calibration",
        "--ledger",
        str(workdir / "ledger" / "spend.jsonl"),
        "--state-file",
        str(workdir / "state"),
    )

    assert "pre-flight:" in result.stderr
    assert "REFUSED" not in result.stderr


def test_a_finished_download_is_not_a_stall():
    """Rental 24 was destroyed at 300s on `2741c81b500d: Verifying Checksum ... Download
    complete` -- a message that had stopped changing *because the pull succeeded*.

    The stall budget's premise is that a live pull keeps rewriting status_msg with new
    byte counts. That premise expires the moment the last layer finishes downloading:
    checksum verification, extraction and container start emit no further updates, so a
    healthy instance goes quiet in exactly the window before it becomes reachable. This is
    the same reasoning the loop already applies to an empty status_msg."""
    for msg in (
        "2741c81b500d: Verifying Checksum2741c81b500d: Download complete",
        "4117260ffdf2: Download complete",
        "a1b2c3: Extracting [====>   ]  1.2GB/2.5GB",
        "a1b2c3: Pull complete",
        "a1b2c3: Already exists",
    ):
        assert _shell(f'df_pull_settled "{msg}" && echo settled || echo moving') == "settled", (
            f"a finished/extracting layer must not be read as a stall: {msg!r}"
        )


def test_a_transferring_or_stuck_pull_is_still_subject_to_the_stall_budget():
    """The carve-out must not swallow the case it was built for. `Pulling fs layer`
    repeated forever is what eight rentals were billed for, and a live byte count is the
    signal the budget is meant to reset on -- neither is 'settled'."""
    for msg in (
        "2741c81b500d: Pulling fs layer",
        "2741c81b500d: Downloading [===>     ]  512MB/2.5GB",
        'Error response from daemon: failed to resolve reference "docker.io/vastai/base-image"',
    ):
        assert _shell(f"df_pull_settled '{msg}' && echo settled || echo moving") == "moving", (
            f"this message must still be bounded by the stall budget: {msg!r}"
        )


def test_the_outer_poll_exempts_a_settled_pull_from_the_stall_budget():
    """The classifier is only worth anything if the poll loop consults it."""
    source = (REMOTE / "run_remote.sh").read_text()
    loop = source[source.index("_progress=") : source.index("Image pull is stuck")]

    assert "df_pull_settled" in loop, "the outer poll must exempt a settled pull"


def _run_provision_func(names, script_body):
    """Run named provision.sh functions against a stub, the way _create_body does."""
    script = (REMOTE / "provision.sh").read_text()
    funcs = ""
    for name in names:
        start = script.index(f"{name}() {{")
        funcs += script[start : script.index("\n}", start) + 2] + "\n"
    return subprocess.run(
        ["sh", "-c", f'set -eu; DF_REPO_ROOT="{REPO_ROOT}"; . "{REMOTE}/lib.sh"; {funcs}\n{script_body}'],
        capture_output=True,
        text=True,
        timeout=60,
    )


def test_a_rejected_create_reports_what_the_api_said():
    """The create endpoint returned HTTP 400 and the run died with nothing but
    `curl: (22)`. The diagnostic was already written -- `instance creation failed: ...`
    with the response body -- but it was unreachable: `RESPONSE=$(df_api ...)` is a plain
    assignment, so under `set -e` a curl exit of 22 killed the script one line before the
    message that would have explained it.

    curl runs with --fail-with-body precisely so the body survives an HTTP error. Losing
    it to `set -e` wastes the one thing that says why the offer was refused."""
    result = _run_provision_func(
        ["create_instance"],
        'df_api() { printf \'{"error":"no_such_ask","msg":"ask 48657232 is gone"}\'; return 22; }\n'
        "create_instance 48657232",
    )

    assert result.returncode != 0, "a refused create must still fail the run"
    combined = result.stdout + result.stderr
    assert "no_such_ask" in combined or "ask 48657232 is gone" in combined, (
        f"the API's explanation must reach the log; got: {combined!r}"
    )
