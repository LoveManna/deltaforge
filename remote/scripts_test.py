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


def test_run_remote_refuses_a_session_that_has_used_sixty_minutes(workdir):
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
    # 90 minutes at $0.324/hr.
    assert rows[0]["estimated_ceiling_usd"] == pytest.approx(0.486)
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


def test_the_readiness_state_falls_back_to_cur_state():
    """A live instance comes back with `actual_status: null` and the state in
    `cur_state`. Reading only the documented field leaves the poller unable to tell
    "not ready yet" from "asking the wrong question", which cost two rented cards."""
    listing = INSTANCES_FIXTURE.read_text()
    state = _shell(
        "row=$(df_instance_row '" + listing.replace("'", "") + "' 49700454); "
        "printf '%s' \"$row\" | jq -r '.actual_status // .cur_state // empty'"
    )

    assert state == "running"


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
