"""The append-only spend ledger, and the two gates that read it.

One row per lifecycle event, newline-delimited JSON, never rewritten. A ``provision`` row
is written **before the instance is used**, so a session that dies mid-run still leaves a
record of what it started — the failure mode this guards against is an agent that
provisions, crashes, and leaves a $0.34/hr instance running with nothing in the ledger to
say it exists.

Row format is deliberately **flat**: no nested objects, and no braces or commas inside
string values. `remote/provision.sh` has to read this file with `awk` before any Python
environment exists, and a flat format is one that awk can parse correctly rather than
approximately. ``ledger_test.py`` enforces the flatness, and cross-checks every gate here
against the shell implementation on the same fixtures.

Two gates read it:

* **Month-to-date** — provisioning refuses at >= $45 against a $50 ceiling.
* **Session GPU-time** — at 60 cumulative billed minutes a session may not start another
  hypothesis. Checked *before* a run starts, never during one: a benchmark executing at
  minute 59 finishes normally, because killing it would waste the money already spent.

Both gates count an un-reconciled instance at its **estimated ceiling**, not at zero.
Guessing low on an instance that may still be running is how a budget gate fails open.
"""

from __future__ import annotations

import json
import os
import tempfile
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

__all__ = [
    "MONTH_TO_DATE_LIMIT_USD",
    "SESSION_MINUTES_LIMIT",
    "LedgerError",
    "LedgerRow",
    "append_destroy",
    "append_provision",
    "default_ledger_path",
    "month_to_date_usd",
    "read_rows",
    "session_minutes",
]

#: Refuse to provision at or above this month-to-date spend. The budget is $50; the $5
#: of headroom absorbs an in-flight instance whose actual cost is not yet known.
MONTH_TO_DATE_LIMIT_USD = 45.0

#: Cumulative billed minutes after which a session may not start another batch. Two hours:
#: a cold `max-autotune` compile alone has cost ~40 minutes of a rental, so a shorter gate
#: ends runs on the clock rather than on the measurement. The hard watchdog sits above it.
SESSION_MINUTES_LIMIT = 120.0

PROVISION = "provision"
DESTROY = "destroy"

#: Values that must never appear inside a ledger string field, because the shell reader
#: parses this file with awk.
_FORBIDDEN_IN_STRINGS = ('"', "{", "}", ",", "\n", "\\")


class LedgerError(RuntimeError):
    pass


def default_ledger_path(repo_root: Path | str | None = None) -> Path:
    root = Path(repo_root) if repo_root else Path(__file__).resolve().parents[2]
    return root / "ledger" / "spend.jsonl"


@dataclass(frozen=True)
class LedgerRow:
    ts: str
    ts_epoch: int
    event: str
    session_id: str
    instance_id: str
    gpu_model: str = ""
    hourly_rate_usd: float = 0.0
    estimated_ceiling_usd: float = 0.0
    estimated_minutes: float = 0.0
    actual_minutes: float | None = None
    actual_cost_usd: float | None = None
    hypothesis: str = ""
    note: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "ts": self.ts,
            "ts_epoch": self.ts_epoch,
            "event": self.event,
            "session_id": self.session_id,
            "instance_id": self.instance_id,
            "gpu_model": self.gpu_model,
            "hourly_rate_usd": self.hourly_rate_usd,
            "estimated_ceiling_usd": self.estimated_ceiling_usd,
            "estimated_minutes": self.estimated_minutes,
            "actual_minutes": self.actual_minutes,
            "actual_cost_usd": self.actual_cost_usd,
            "hypothesis": self.hypothesis,
            "note": self.note,
        }

    def to_json(self) -> str:
        for key, value in self.to_dict().items():
            if isinstance(value, str):
                for bad in _FORBIDDEN_IN_STRINGS:
                    if bad in value:
                        raise LedgerError(
                            f"ledger field {key!r} contains {bad!r}, which the shell reader "
                            "cannot parse; ledger string values must stay flat"
                        )
        return json.dumps(self.to_dict(), separators=(",", ":"))

    @property
    def month(self) -> str:
        return self.ts[:7]

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> LedgerRow:
        try:
            return cls(
                ts=data["ts"],
                ts_epoch=int(data["ts_epoch"]),
                event=data["event"],
                session_id=data.get("session_id", ""),
                instance_id=str(data["instance_id"]),
                gpu_model=data.get("gpu_model", ""),
                hourly_rate_usd=float(data.get("hourly_rate_usd") or 0.0),
                estimated_ceiling_usd=float(data.get("estimated_ceiling_usd") or 0.0),
                estimated_minutes=float(data.get("estimated_minutes") or 0.0),
                actual_minutes=_opt_float(data.get("actual_minutes")),
                actual_cost_usd=_opt_float(data.get("actual_cost_usd")),
                hypothesis=data.get("hypothesis", ""),
                note=data.get("note", ""),
            )
        except KeyError as exc:
            raise LedgerError(f"ledger row is missing required field {exc.args[0]!r}") from None


def _opt_float(value: Any) -> float | None:
    if value is None or value == "":
        return None
    return float(value)


def read_rows(path: Path | str) -> list[LedgerRow]:
    """Parse the ledger. A missing file is an empty ledger, not an error."""
    path = Path(path)
    if not path.exists():
        return []
    rows: list[LedgerRow] = []
    for line_no, line in enumerate(path.read_text().splitlines(), start=1):
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        try:
            data = json.loads(line)
        except json.JSONDecodeError as exc:
            raise LedgerError(f"{path}:{line_no}: not valid JSON: {exc}") from None
        rows.append(LedgerRow.from_dict(data))
    return rows


def _append(path: Path | str, row: LedgerRow) -> LedgerRow:
    """Append one row durably. Written with O_APPEND so concurrent writers interleave
    whole lines rather than corrupting each other."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = (row.to_json() + "\n").encode("utf-8")
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o644)
    try:
        os.write(fd, payload)
        os.fsync(fd)
    finally:
        os.close(fd)
    return row


def _now() -> tuple[str, int]:
    now = datetime.now(timezone.utc)
    return now.strftime("%Y-%m-%dT%H:%M:%SZ"), int(now.timestamp())


def append_provision(
    path: Path | str,
    *,
    session_id: str,
    instance_id: str,
    gpu_model: str,
    hourly_rate_usd: float,
    estimated_minutes: float,
    hypothesis: str = "",
    note: str = "",
    now: tuple[str, int] | None = None,
) -> LedgerRow:
    """Record an instance at provision time, before it is used."""
    ts, ts_epoch = now or _now()
    row = LedgerRow(
        ts=ts,
        ts_epoch=ts_epoch,
        event=PROVISION,
        session_id=session_id,
        instance_id=str(instance_id),
        gpu_model=gpu_model,
        hourly_rate_usd=float(hourly_rate_usd),
        estimated_ceiling_usd=round(float(hourly_rate_usd) * float(estimated_minutes) / 60.0, 6),
        estimated_minutes=float(estimated_minutes),
        hypothesis=hypothesis,
        note=note,
    )
    return _append(path, row)


def append_destroy(
    path: Path | str,
    *,
    session_id: str,
    instance_id: str,
    gpu_model: str = "",
    hourly_rate_usd: float = 0.0,
    actual_minutes: float,
    hypothesis: str = "",
    note: str = "",
    now: tuple[str, int] | None = None,
) -> LedgerRow:
    """Reconcile an instance at destroy time with what it actually cost."""
    ts, ts_epoch = now or _now()
    row = LedgerRow(
        ts=ts,
        ts_epoch=ts_epoch,
        event=DESTROY,
        session_id=session_id,
        instance_id=str(instance_id),
        gpu_model=gpu_model,
        hourly_rate_usd=float(hourly_rate_usd),
        actual_minutes=float(actual_minutes),
        actual_cost_usd=round(float(hourly_rate_usd) * float(actual_minutes) / 60.0, 6),
        hypothesis=hypothesis,
        note=note,
    )
    return _append(path, row)


@dataclass
class _InstanceState:
    provision: LedgerRow | None = None
    destroys: list[LedgerRow] = field(default_factory=list)


def _by_instance(rows: list[LedgerRow]) -> dict[str, _InstanceState]:
    states: dict[str, _InstanceState] = {}
    for row in rows:
        state = states.setdefault(row.instance_id, _InstanceState())
        if row.event == PROVISION:
            if state.provision is None:
                state.provision = row
        elif row.event == DESTROY:
            state.destroys.append(row)
    return states


def month_to_date_usd(rows: list[LedgerRow], month: str | None = None) -> float:
    """Spend for ``month`` (``YYYY-MM``, default: the current UTC month).

    An instance that was provisioned but never reconciled counts at its estimated
    ceiling. A gate that counted it at zero would fail open exactly when an instance is
    still running and burning money.
    """
    month = month or datetime.now(timezone.utc).strftime("%Y-%m")
    total = 0.0
    for instance_id, state in _by_instance(rows).items():
        del instance_id
        reconciled = [d for d in state.destroys if d.actual_cost_usd is not None]
        if reconciled:
            # One instance can be reconciled twice: the local watchdog destroys it and
            # writes its row (spec 8.3), which kills the run and fires the teardown trap,
            # which writes another (spec 8.5). The ledger is append-only, so neither row
            # can be retracted and the readers are the only correct place to dedup.
            # Count the LARGEST reported cost, never the first, last, or mean — this is a
            # budget gate, and over-counting refuses to spend while under-counting lets
            # spend escape. Mirrored in df_ledger_month_to_date in remote/lib.sh.
            destroy = max(reconciled, key=lambda d: d.actual_cost_usd or 0.0)
            if destroy.month == month:
                total += destroy.actual_cost_usd or 0.0
        elif state.provision is not None and state.provision.month == month:
            total += state.provision.estimated_ceiling_usd
    return round(total, 6)


def session_minutes(rows: list[LedgerRow], session_id: str, now_epoch: int | None = None) -> float:
    """Billed minutes this session has consumed.

    A provisioned-but-not-destroyed instance counts at its wall-clock elapsed time, so a
    session cannot dodge the gate by simply not tearing down.
    """
    now_epoch = now_epoch if now_epoch is not None else int(datetime.now(timezone.utc).timestamp())
    total = 0.0
    for state in _by_instance([r for r in rows if r.session_id == session_id]).values():
        reconciled = [d for d in state.destroys if d.actual_minutes is not None]
        if reconciled:
            # Deduped at read time for the same reason as month_to_date_usd, at the
            # largest reported duration. Mirrored in df_ledger_session_minutes.
            total += max(d.actual_minutes or 0.0 for d in reconciled)
        elif state.provision is not None:
            total += max(0.0, (now_epoch - state.provision.ts_epoch) / 60.0)
    return round(total, 6)


def month_to_date_gate(
    path: Path | str, limit: float = MONTH_TO_DATE_LIMIT_USD, month: str | None = None
) -> tuple[bool, float]:
    """``(allowed, month_to_date_usd)``. Refuses at or above the limit."""
    spend = month_to_date_usd(read_rows(path), month=month)
    return spend < limit, spend


def session_gate(
    path: Path | str,
    session_id: str,
    limit: float = SESSION_MINUTES_LIMIT,
    now_epoch: int | None = None,
) -> tuple[bool, float]:
    """``(allowed, minutes_used)``. Refuses at or above the limit."""
    minutes = session_minutes(read_rows(path), session_id, now_epoch=now_epoch)
    return minutes < limit, minutes


def write_fixture(path: Path | str, rows: list[LedgerRow]) -> Path:
    """Write a complete ledger atomically. For tests and fixtures only — the real
    ledger is only ever appended to."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    body = "".join(row.to_json() + "\n" for row in rows)
    with tempfile.NamedTemporaryFile("w", dir=path.parent, delete=False, encoding="utf-8") as handle:
        handle.write(body)
        temp = Path(handle.name)
    temp.replace(path)
    return path
