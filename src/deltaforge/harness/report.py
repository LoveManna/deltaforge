"""Results records: the JSON of spec section 13, plus a markdown fragment.

Provenance is the point. A ratio with no record of which card produced it, at what
clocks, against which git SHA, on which torch and Triton, is not evidence — it is a
number someone typed. Everything needed to argue with the result is captured here.

Environment capture degrades gracefully: on a machine with no CUDA every GPU field comes
back ``None`` rather than raising, so the record format can be tested without a GPU.
"""

from __future__ import annotations

import json
import platform
import subprocess
import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

__all__ = [
    "SCHEMA_VERSION",
    "ResultRecord",
    "capture_environment",
    "git_info",
    "render_markdown",
    "write_record",
]

SCHEMA_VERSION = 1

#: Outcomes a run may record. `inconclusive` exists because a margin inside the noise
#: band is not a win: it neither promotes nor enters the graveyard, and the hypothesis
#: stays open for a cleaner measurement.
OUTCOMES = ("baseline", "win", "loss", "inconclusive", "incorrect", "error")


def _run(cmd: list[str]) -> str | None:
    try:
        out = subprocess.run(cmd, capture_output=True, text=True, timeout=15, check=False)
    except (OSError, subprocess.SubprocessError):
        return None
    if out.returncode != 0:
        return None
    return out.stdout.strip() or None


def git_info(repo: Path | str | None = None) -> dict[str, Any]:
    """Git SHA, branch and dirty flag. All ``None`` outside a git checkout."""
    cwd = ["-C", str(repo)] if repo else []
    sha = _run(["git", *cwd, "rev-parse", "HEAD"])
    branch = _run(["git", *cwd, "rev-parse", "--abbrev-ref", "HEAD"])
    status = _run(["git", *cwd, "status", "--porcelain"])
    return {
        "sha": sha,
        "branch": branch,
        # `status` is None on failure and "" when clean; only a non-empty result is dirty.
        "dirty": None if status is None and sha is None else bool(status),
    }


def _nvidia_smi(query: str) -> str | None:
    return _run(["nvidia-smi", f"--query-gpu={query}", "--format=csv,noheader,nounits"])


def capture_environment() -> dict[str, Any]:
    """Hardware and software provenance. Every GPU field is ``None`` without CUDA."""
    env: dict[str, Any] = {
        "python_version": sys.version.split()[0],
        "platform": platform.platform(),
        "hostname": platform.node(),
        "torch_version": None,
        "triton_version": None,
        "cuda_available": False,
        "cuda_version": None,
        "gpu_name": None,
        "gpu_count": 0,
        "driver_version": None,
        "gpu_memory_total_mb": None,
        "gpu_clocks_mhz": None,
        "gpu_clocks_note": (
            "Observed at capture time, not locked. Clock locking needs privileged "
            "container access we do not reliably have, so the methodology never depends "
            "on it; interleaved rounds absorb drift instead."
        ),
    }

    try:
        import torch  # noqa: PLC0415
    except ImportError:  # pragma: no cover - torch is a core dependency
        return env

    env["torch_version"] = torch.__version__
    env["cuda_available"] = bool(torch.cuda.is_available())
    if not env["cuda_available"]:
        return env

    env["cuda_version"] = torch.version.cuda
    env["gpu_count"] = torch.cuda.device_count()
    env["gpu_name"] = torch.cuda.get_device_name(0)
    props = torch.cuda.get_device_properties(0)
    env["gpu_memory_total_mb"] = int(props.total_memory // (1024 * 1024))
    env["compute_capability"] = f"{props.major}.{props.minor}"

    try:
        import triton  # noqa: PLC0415

        env["triton_version"] = triton.__version__
    except ImportError:
        env["triton_version"] = None

    env["driver_version"] = _nvidia_smi("driver_version")
    sm_clock = _nvidia_smi("clocks.current.sm")
    mem_clock = _nvidia_smi("clocks.current.memory")
    if sm_clock or mem_clock:
        env["gpu_clocks_mhz"] = {
            "sm": _maybe_int(sm_clock),
            "memory": _maybe_int(mem_clock),
            "observed_at": datetime.now(timezone.utc).isoformat(),
        }
    return env


def _maybe_int(value: str | None) -> int | None:
    if value is None:
        return None
    try:
        return int(float(value.splitlines()[0].strip()))
    except (ValueError, IndexError):
        return None


@dataclass
class ResultRecord:
    """One benchmark run, win or lose.

    ``bench`` and ``correctness`` take the ``to_dict()`` output of their respective
    harness objects, so this module stays a serialiser and never re-derives a score.
    """

    kind: str  # "baseline" or "hypothesis"
    outcome: str
    config_name: str
    workload: dict[str, Any] = field(default_factory=dict)
    hypothesis: dict[str, Any] | None = None
    bench: dict[str, Any] | None = None
    correctness: dict[str, Any] | None = None
    cost: dict[str, Any] | None = None
    environment: dict[str, Any] = field(default_factory=dict)
    git: dict[str, Any] = field(default_factory=dict)
    notes: str = ""
    timestamp: str = ""
    schema_version: int = SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.outcome not in OUTCOMES:
            raise ValueError(f"outcome must be one of {OUTCOMES}, got {self.outcome!r}")
        if self.kind not in ("baseline", "hypothesis"):
            raise ValueError(f"kind must be 'baseline' or 'hypothesis', got {self.kind!r}")
        if not self.timestamp:
            self.timestamp = datetime.now(timezone.utc).isoformat()
        if not self.environment:
            self.environment = capture_environment()
        if not self.git:
            self.git = git_info()

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "kind": self.kind,
            "outcome": self.outcome,
            "timestamp": self.timestamp,
            "config_name": self.config_name,
            "hypothesis": self.hypothesis,
            "workload": self.workload,
            "git": self.git,
            "environment": self.environment,
            "bench": self.bench,
            "correctness": self.correctness,
            "cost": self.cost,
            "notes": self.notes,
        }

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, sort_keys=False)


def write_record(record: ResultRecord, path: Path | str) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(record.to_json() + "\n")
    return path


def _fmt(value: Any, spec: str = "", missing: str = "—") -> str:
    if value is None:
        return missing
    return format(value, spec) if spec else str(value)


def render_markdown(record: ResultRecord) -> str:
    """A markdown fragment suitable for a PR body or a LEADERBOARD row."""
    data = record.to_dict()
    env = data["environment"]
    bench = data["bench"]
    correctness = data["correctness"]
    cost = data["cost"]

    title = "Baseline" if record.kind == "baseline" else "Hypothesis"
    if record.hypothesis:
        title = f"Hypothesis {record.hypothesis.get('id', '?')} — {record.hypothesis.get('slug', '')}"

    lines = [
        f"## {title}",
        "",
        f"**Outcome:** `{record.outcome}`  ",
        f"**When:** {data['timestamp']}  ",
        f"**Commit:** `{_fmt(data['git'].get('sha'))}`"
        + (" *(working tree dirty)*" if data["git"].get("dirty") else ""),
        "",
    ]

    if record.hypothesis and record.hypothesis.get("statement"):
        lines += ["> " + record.hypothesis["statement"], ""]

    lines += [
        "### Environment",
        "",
        f"- GPU: {_fmt(env.get('gpu_name'))} "
        f"({_fmt(env.get('gpu_memory_total_mb'))} MB, driver {_fmt(env.get('driver_version'))})",
        f"- torch {_fmt(env.get('torch_version'))}, "
        f"triton {_fmt(env.get('triton_version'))}, CUDA {_fmt(env.get('cuda_version'))}",
    ]
    clocks = env.get("gpu_clocks_mhz")
    if clocks:
        lines.append(
            f"- Observed clocks: SM {_fmt(clocks.get('sm'))} MHz, "
            f"memory {_fmt(clocks.get('memory'))} MHz (not locked)"
        )
    lines.append("")

    if record.workload:
        pairs = ", ".join(f"{k}={v}" for k, v in record.workload.items())
        lines += ["### Workload", "", f"- {pairs}", ""]

    if bench:
        lines += [
            "### Timing",
            "",
            f"Interleaved, R={bench.get('rounds')} with the first "
            f"{bench.get('warmup_rounds')} rounds discarded. "
            f"Score is the median per-round ratio against `{bench.get('baseline')}`.",
            "",
            "| Column | Median ms | Ratio vs baseline | IQR of ratio |",
            "|---|---:|---:|---:|",
        ]
        median_ms = bench.get("median_ms", {})
        median_ratio = bench.get("median_ratio", {})
        iqr_ratio = bench.get("iqr_ratio", {})
        for label in bench.get("labels", []):
            ratio = median_ratio.get(label)
            iqr = iqr_ratio.get(label)
            lines.append(
                f"| `{label}` | {_fmt(median_ms.get(label), '.3f')} "
                f"| {_fmt(ratio, '.4f') if ratio is not None else '— (baseline)'} "
                f"| {_fmt(iqr, '.4f')} |"
            )
        lines.append("")

    if correctness:
        passed = correctness.get("passed")
        lines += [
            "### Correctness",
            "",
            f"**{'PASS' if passed else 'FAIL'}** — "
            f"worst max abs err {_fmt(correctness.get('worst_max_abs_err'), '.3e')}, "
            f"worst max rel err {_fmt(correctness.get('worst_max_rel_err'), '.3e')}",
            "",
        ]
        checks = correctness.get("layer1_kernel_checks") or []
        if checks:
            lines += [
                "| Kernel | Replaces | Pass | Max abs err | Max rel err |",
                "|---|---|---|---:|---:|",
            ]
            for check in checks:
                lines.append(
                    f"| `{check['name']}` | `{check['replaces']}` "
                    f"| {'yes' if check['passed'] else 'NO'} "
                    f"| {check['max_abs_err']:.3e} | {check['max_rel_err']:.3e} |"
                )
            lines.append("")
        layer2 = correctness.get("layer2_end_to_end")
        if layer2:
            failed = [m for m in layer2["per_prompt"] if not m["matched"]]
            if failed:
                detail = ", ".join(
                    f"prompt {m['prompt_index']} at token {m['first_divergence']}" for m in failed
                )
                lines += [f"End-to-end token match **failed**: {detail}.", ""]
            else:
                lines += [
                    f"End-to-end: all {layer2['num_prompts']} prompts matched eager exactly "
                    f"over {layer2['max_new_tokens']} greedy tokens.",
                    "",
                ]

    if cost:
        lines += [
            "### Cost",
            "",
            f"- Instance `{_fmt(cost.get('instance_id'))}` "
            f"({_fmt(cost.get('gpu_model'))}) at ${_fmt(cost.get('hourly_rate_usd'), '.4f')}/hr",
            f"- {_fmt(cost.get('actual_minutes'), '.1f')} minutes, "
            f"${_fmt(cost.get('actual_cost_usd'), '.4f')}",
            "",
        ]

    if record.notes:
        lines += ["### Notes", "", record.notes, ""]

    return "\n".join(lines)
