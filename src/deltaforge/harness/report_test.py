"""Results-record tests.

Provenance is the product here: a ratio with no record of the card, the clocks, the git
SHA and the library versions is a number someone typed, not evidence. These tests also
pin that environment capture **degrades to nulls** without CUDA rather than raising or
inventing values.
"""

from __future__ import annotations

import json

import pytest

from .bench import BenchConfig, BenchResult
from .report import (
    SCHEMA_VERSION,
    ResultRecord,
    capture_environment,
    git_info,
    render_markdown,
    write_record,
)


@pytest.fixture
def bench_result():
    return BenchResult(
        labels=("eager", "compiled", "compiled_nocudagraphs", "candidate"),
        config=BenchConfig(rounds=7, warmup_rounds=2),
        timings_ms={
            "eager": [99.0, 88.0] + [20.0] * 5,
            "compiled": [99.0, 88.0] + [10.0] * 5,
            "compiled_nocudagraphs": [99.0, 88.0] + [12.0] * 5,
            "candidate": [99.0, 88.0] + [8.0] * 5,
        },
        call_order=("eager", "compiled", "compiled_nocudagraphs", "candidate") * 7,
    )


# -- environment capture --------------------------------------------------------------


def test_environment_capture_never_raises_without_cuda():
    env = capture_environment()

    assert env["torch_version"], "torch is a core dependency and must be reported"
    assert env["python_version"]
    assert isinstance(env["cuda_available"], bool)


def test_gpu_fields_are_null_rather_than_fabricated_without_cuda():
    env = capture_environment()
    if env["cuda_available"]:
        pytest.skip("a GPU is present; this pins the CPU-only behaviour")

    assert env["gpu_name"] is None
    assert env["driver_version"] is None
    assert env["gpu_clocks_mhz"] is None
    assert env["gpu_count"] == 0


def test_clock_note_states_that_clocks_are_not_locked():
    """The methodology must never depend on clock locking, and the record must say so
    rather than implying the numbers were taken at fixed clocks."""
    assert "not locked" in capture_environment()["gpu_clocks_note"]


def test_git_info_reports_sha_branch_and_dirtiness():
    info = git_info()

    assert set(info) == {"sha", "branch", "dirty"}
    if info["sha"] is not None:
        assert len(info["sha"]) == 40
        assert isinstance(info["dirty"], bool)


def test_git_info_outside_a_repository_is_all_null(tmp_path):
    info = git_info(repo=tmp_path)
    assert info["sha"] is None
    assert info["branch"] is None


# -- the record -----------------------------------------------------------------------


def test_record_carries_everything_needed_to_argue_with_the_result(bench_result):
    record = ResultRecord(
        kind="hypothesis",
        outcome="win",
        config_name="qwen3.5-4b",
        hypothesis={"id": "001", "slug": "fused-rmsnorm", "statement": "Fusing helps."},
        workload={"batch_size": 1, "context_length": 2048, "decode_tokens": 128},
        bench=bench_result.to_dict(),
        cost={"instance_id": "42", "hourly_rate_usd": 0.324, "actual_minutes": 31.5},
    )
    data = record.to_dict()

    assert data["schema_version"] == SCHEMA_VERSION
    assert data["outcome"] == "win"
    assert data["timestamp"], "auto-filled"
    assert data["environment"]["torch_version"]
    assert set(data["git"]) == {"sha", "branch", "dirty"}
    assert data["bench"]["median_ratio"]["candidate"] == pytest.approx(1.25)
    # Raw per-round timings survive, including the discarded warmup rounds.
    assert data["bench"]["timings_ms"]["candidate"][:2] == [99.0, 88.0]
    assert data["workload"]["context_length"] == 2048
    assert data["cost"]["actual_minutes"] == 31.5


def test_record_is_json_serialisable(bench_result):
    record = ResultRecord(
        kind="baseline", outcome="baseline", config_name="qwen3.5-4b", bench=bench_result.to_dict()
    )
    reloaded = json.loads(record.to_json())

    assert reloaded["kind"] == "baseline"
    assert reloaded["bench"]["baseline"] == "compiled"


def test_unknown_outcome_is_rejected():
    with pytest.raises(ValueError, match="outcome must be one of"):
        ResultRecord(kind="baseline", outcome="great", config_name="x")


def test_inconclusive_is_a_first_class_outcome():
    """A margin inside the noise band is not a win: it neither promotes nor enters the
    graveyard, and the hypothesis stays open for a cleaner measurement."""
    record = ResultRecord(kind="hypothesis", outcome="inconclusive", config_name="x")
    assert record.to_dict()["outcome"] == "inconclusive"


def test_incorrect_is_recorded_rather_than_discarded():
    record = ResultRecord(kind="hypothesis", outcome="incorrect", config_name="x")
    assert record.to_dict()["outcome"] == "incorrect"


def test_unknown_kind_is_rejected():
    with pytest.raises(ValueError, match="kind must be"):
        ResultRecord(kind="experiment", outcome="win", config_name="x")


def test_write_record_creates_parent_directories(tmp_path, bench_result):
    target = tmp_path / "results" / "hypotheses" / "001-fused.json"

    written = write_record(
        ResultRecord(
            kind="hypothesis",
            outcome="win",
            config_name="x",
            bench=bench_result.to_dict(),
        ),
        target,
    )

    assert written.exists()
    assert json.loads(written.read_text())["outcome"] == "win"


# -- markdown -------------------------------------------------------------------------


def test_markdown_reports_the_ratio_and_the_noise_band(bench_result):
    record = ResultRecord(
        kind="hypothesis",
        outcome="win",
        config_name="qwen3.5-4b",
        hypothesis={"id": "001", "slug": "fused-rmsnorm", "statement": "Fusing helps."},
        workload={"batch_size": 1, "context_length": 2048},
        bench=bench_result.to_dict(),
    )
    markdown = render_markdown(record)

    assert "## Hypothesis 001 — fused-rmsnorm" in markdown
    assert "> Fusing helps." in markdown
    assert "**Outcome:** `win`" in markdown
    assert "1.2500" in markdown, "the median ratio"
    assert "— (baseline)" in markdown, "the baseline column has no ratio against itself"
    assert "R=7 with the first 2 rounds discarded" in markdown
    assert "batch_size=1" in markdown


def test_markdown_shows_missing_environment_fields_as_dashes(bench_result):
    record = ResultRecord(
        kind="baseline",
        outcome="baseline",
        config_name="x",
        bench=bench_result.to_dict(),
        environment={"gpu_name": None, "torch_version": "2.11.0", "triton_version": None},
        git={"sha": "abc", "branch": "main", "dirty": False},
    )
    markdown = render_markdown(record)

    assert "GPU: —" in markdown
    assert "torch 2.11.0" in markdown
    assert "triton —" in markdown


def test_markdown_flags_a_dirty_working_tree(bench_result):
    record = ResultRecord(
        kind="baseline",
        outcome="baseline",
        config_name="x",
        bench=bench_result.to_dict(),
        git={"sha": "deadbeef", "branch": "main", "dirty": True},
    )
    assert "*(working tree dirty)*" in render_markdown(record)


def test_markdown_reports_correctness_magnitudes():
    record = ResultRecord(
        kind="hypothesis",
        outcome="incorrect",
        config_name="x",
        correctness={
            "passed": False,
            "worst_max_abs_err": 0.25,
            "worst_max_rel_err": 0.5,
            "layer1_kernel_checks": [
                {
                    "name": "fused_rmsnorm",
                    "replaces": "rms_norm",
                    "passed": False,
                    "max_abs_err": 0.25,
                    "max_rel_err": 0.5,
                }
            ],
            "layer2_end_to_end": {
                "passed": False,
                "num_prompts": 5,
                "max_new_tokens": 128,
                "per_prompt": [{"prompt_index": 2, "matched": False, "first_divergence": 17}],
            },
        },
    )
    markdown = render_markdown(record)

    assert "**FAIL**" in markdown
    assert "2.500e-01" in markdown
    assert "prompt 2 at token 17" in markdown


def test_markdown_reports_a_clean_correctness_pass():
    record = ResultRecord(
        kind="baseline",
        outcome="baseline",
        config_name="x",
        correctness={
            "passed": True,
            "worst_max_abs_err": 0.0,
            "worst_max_rel_err": 0.0,
            "layer1_kernel_checks": [],
            "layer2_end_to_end": {
                "passed": True,
                "num_prompts": 5,
                "max_new_tokens": 128,
                "per_prompt": [],
            },
        },
    )
    markdown = render_markdown(record)

    assert "**PASS**" in markdown
    assert "all 5 prompts matched eager exactly over 128 greedy tokens" in markdown


def test_markdown_reports_cost():
    record = ResultRecord(
        kind="hypothesis",
        outcome="loss",
        config_name="x",
        cost={
            "instance_id": "9008",
            "gpu_model": "RTX 5090",
            "hourly_rate_usd": 0.324,
            "actual_minutes": 31.5,
            "actual_cost_usd": 0.1701,
        },
    )
    markdown = render_markdown(record)

    assert "Instance `9008`" in markdown
    assert "RTX 5090" in markdown
    assert "31.5 minutes" in markdown
