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


# --------------------------------------------------------------------------------------
# Batch records
# --------------------------------------------------------------------------------------


def _slot(slug, outcome, prediction, *, ratio=None, iqr=None, correct=None, error=None):
    return {
        "slug": slug,
        "outcome": outcome,
        "prediction": prediction,
        "median_ratio": ratio,
        "iqr_ratio": iqr,
        "byte_share": 0.0623,
        "replaces": ["gqa_attention"],
        "prediction_correct": correct,
        "error": error,
    }


def make_batch_record(**overrides):
    from .report import BatchRecord

    base = {
        "batch_id": "001-calibration",
        "session_id": "sess",
        "config_name": "Qwen/Qwen3.5-4B",
        "slots": [
            _slot("000-identity", "inconclusive", "identity", ratio=1.001, iqr=0.01, correct=True),
            _slot("006-gqa", "win", "win", ratio=1.06, iqr=0.01, correct=True),
            _slot("008-flash", "error", "inconclusive", error="Triton compile failed"),
            _slot("009-late", "not_run", "inconclusive"),
        ],
        "calibrated": True,
        "predictions": [
            {"slug": "000-identity", "predicted": "identity", "outcome": "inconclusive", "correct": True},
            {"slug": "006-gqa", "predicted": "win", "outcome": "win", "correct": True},
            {"slug": "008-flash", "predicted": "inconclusive", "outcome": "error", "correct": None},
            {"slug": "009-late", "predicted": "inconclusive", "outcome": "not_run", "correct": None},
        ],
    }
    base.update(overrides)
    return BatchRecord(**base)


def test_batch_record_counts_outcomes():
    assert make_batch_record().counts == {
        "inconclusive": 1,
        "win": 1,
        "error": 1,
        "not_run": 1,
    }


def test_batch_prediction_record_excludes_unscored_slots():
    # Two of four slots never produced a verdict, so the record is 2/2, not 2/4.
    assert make_batch_record().prediction_record == (2, 2)


def test_batch_record_round_trips_through_json():
    import json

    record = make_batch_record()
    data = json.loads(record.to_json())
    assert data["kind"] == "batch"
    assert data["batch_id"] == "001-calibration"
    assert data["prediction_record"] == {"correct": 2, "scored": 2}
    assert len(data["slots"]) == 4


def test_batch_markdown_lists_every_slot_including_the_ones_that_never_ran():
    from .report import render_batch_markdown

    text = render_batch_markdown(make_batch_record())
    for slug in ("000-identity", "006-gqa", "008-flash", "009-late"):
        assert slug in text
    assert "did not run" in text
    assert "Triton compile failed" in text


def test_failed_calibration_voids_the_batch_loudly():
    from .report import render_batch_markdown

    text = render_batch_markdown(make_batch_record(calibrated=False))
    assert "CALIBRATION FAILED" in text
    assert "void" in text


def test_a_batch_with_no_calibration_slot_says_so():
    from .report import render_batch_markdown

    text = render_batch_markdown(make_batch_record(calibrated=None))
    assert "No calibration slot" in text


def test_write_batch_record_creates_parent_directories(tmp_path):
    from .report import write_batch_record

    out = tmp_path / "batches" / "001" / "summary.json"
    assert write_batch_record(make_batch_record(), out) == out
    assert out.exists()


def test_not_run_is_a_valid_outcome_for_a_result_record():
    from .report import ResultRecord

    record = ResultRecord(kind="hypothesis", outcome="not_run", config_name="m")
    assert record.outcome == "not_run"


def test_a_record_carries_its_phase_timings():
    """The phase split is the calibration rental's product.

    `docs/BATCHES.md` costs a rental at ~15 minutes fixed plus 2-4 per slot, and rental 22
    contradicted that with a ~40-minute compile. A record that carries where its own time
    went is what lets an estimate be replaced by a measurement.
    """
    record = ResultRecord(
        kind="hypothesis",
        outcome="win",
        config_name="Qwen/Qwen3.5-4B",
        phases={"candidate_build": 3.5, "compile_candidate_compiled": 2400.0},
    )

    assert record.to_dict()["phases"]["compile_candidate_compiled"] == 2400.0


def test_a_batch_record_carries_phase_timings_too():
    from .report import BatchRecord

    record = BatchRecord(
        batch_id="002-compile-cost",
        session_id="s",
        config_name="Qwen/Qwen3.5-4B",
        phases={"000-identity.bench": 180.0},
    )

    assert record.to_dict()["phases"]["000-identity.bench"] == 180.0


def test_an_approximate_gate_renders_its_numbers_beside_the_bars_it_had_to_clear():
    """A result that lives only in JSON is a result nobody reads.

    The bars were registered before the rental, so the rendered report must show both the
    measurement and what it was asked to clear — a bare "passed" hides which of the two
    was generous.
    """
    record = ResultRecord(
        kind="hypothesis",
        outcome="win",
        config_name="qwen3.5-4b",
        correctness={
            "passed": True,
            "worst_max_abs_err": 1e-3,
            "worst_max_rel_err": 1e-3,
            "layer1_kernel_checks": [],
            "layer2_end_to_end": None,
            "layer2_policy": "approximate",
            "layer2_distribution": {
                "policy": "approximate",
                "passed": True,
                "top1_agreement": 0.9934,
                "mean_kl": 0.00241,
                "max_kl": 0.0181,
                "num_positions": 765,
                "top1_threshold": 0.98,
                "kl_threshold": 0.01,
            },
        },
    )

    rendered = render_markdown(record)

    assert "Approximate gate passed" in rendered
    assert "765 positions" in rendered
    assert "0.9934" in rendered
    assert ">= 0.98" in rendered
    assert "<= 0.01" in rendered


def test_a_failed_approximate_gate_says_so_loudly():
    record = ResultRecord(
        kind="hypothesis",
        outcome="incorrect",
        config_name="qwen3.5-4b",
        correctness={
            "passed": False,
            "worst_max_abs_err": 0.0,
            "worst_max_rel_err": 0.0,
            "layer1_kernel_checks": [],
            "layer2_end_to_end": None,
            "layer2_policy": "approximate",
            "layer2_distribution": {
                "policy": "approximate",
                "passed": False,
                "top1_agreement": 0.71,
                "mean_kl": 0.44,
                "max_kl": 2.1,
                "num_positions": 765,
                "top1_threshold": 0.85,
                "kl_threshold": 0.15,
            },
        },
    )

    assert "Approximate gate **FAILED**" in render_markdown(record)
