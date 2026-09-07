"""Running a batch on the rented box: the part that needs a GPU.

The policy — what a measurement means, whether a slot fits before the deadline, how
predictions are scored — lives in `batch.py` and is tested on a CPU. This module is the
orchestration around it, and it has exactly three jobs:

1. **Compile the reference once** and keep it for the whole batch. That is the entire
   saving: a `max-autotune` compile of a 32-layer model is 3-4 minutes, and paying it per
   hypothesis would eat the batch.

2. **Isolate failures.** Every slot runs inside its own try/except. Six of batch 001's
   nine hypotheses are Triton kernels written without a GPU to test them on; some will be
   wrong. A wrong kernel must cost a slot, not the rental.

3. **Write as it goes.** Each slot's record hits disk the moment it finishes, so a hard
   crash at slot 8 leaves slots 0-7 recoverable.

**Only the compilation is amortised, not the measurement.** The reference `compiled`
column is re-timed inside every hypothesis's own interleaved rounds, so per-round thermal
and clock drift divides out exactly as it does in the single-hypothesis harness. Reusing a
*timing* across slots would break that, and is not done. A sceptical reader will ask about
this, so `_slot_metadata` records it in every result.
"""

from __future__ import annotations

import gc
import time
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .batch import (
    Batch,
    Hypothesis,
    SlotBudget,
    calibration_holds,
    classify_outcome,
    scoped_registry,
    score_predictions,
)

__all__ = ["BatchRunner", "SlotResult", "run_batch"]


#: The scoring comparison, and the reason a batch is worth running at all: identical
#: `max-autotune` treatment on both sides, the only difference being who wrote the kernel.
BASELINE_COLUMN = "compiled"
CANDIDATE_COLUMN = "candidate_compiled"


@dataclass
class SlotResult:
    """What one hypothesis produced. Always has an outcome, even when it failed."""

    hypothesis: Hypothesis
    outcome: str
    median_ratio: float | None = None
    iqr_ratio: float | None = None
    correctness: dict[str, Any] | None = None
    bench: dict[str, Any] | None = None
    error: str | None = None
    duration_s: float | None = None
    peak_memory_mb: int | None = None

    def to_slot_dict(self) -> dict[str, Any]:
        return {
            "slug": self.hypothesis.slug,
            "outcome": self.outcome,
            "prediction": self.hypothesis.prediction,
            "category": self.hypothesis.category,
            "byte_share": self.hypothesis.byte_share,
            "replaces": list(self.hypothesis.replaces),
            "kernels": list(self.hypothesis.kernels),
            "mechanism": self.hypothesis.mechanism,
            "rationale": self.hypothesis.rationale,
            "median_ratio": self.median_ratio,
            "iqr_ratio": self.iqr_ratio,
            "error": self.error,
            "duration_s": self.duration_s,
            "peak_memory_mb": self.peak_memory_mb,
        }


class BatchRunner:
    """Owns the reference model and its compiled columns for the life of a batch."""

    def __init__(
        self,
        *,
        config,
        reference,
        prompt,
        prompt_ids,
        workload: dict[str, int],
        weights_dtype,
        bench_config,
        max_new_tokens: int,
        columns: tuple[str, ...],
        log=print,
    ) -> None:
        self.config = config
        self.reference = reference
        self.prompt = prompt
        self.prompt_ids = prompt_ids
        self.workload = workload
        self.weights_dtype = weights_dtype
        self.bench_config = bench_config
        self.max_new_tokens = max_new_tokens
        self.columns = columns
        self.log = log
        self._reference_columns: dict[str, tuple] = {}

    # -- reference columns, built once ---------------------------------------------

    def _make_column(self, model, compile_mode):
        import torch

        from .model import greedy_decode

        runnable = model if compile_mode is None else torch.compile(model, mode=compile_mode)
        batch = self.workload["batch_size"]
        cache = model.new_cache(batch, self.workload["context_length"] + self.workload["decode_tokens"])
        tokens = self.workload["decode_tokens"]
        prompt = self.prompt

        def setup() -> None:
            # Prefill and cache restoration are excluded from the timed region. Inside it
            # they would add the same constant to every column, which does not cancel in a
            # ratio: it drags every ratio toward 1 and hides whatever win is really there.
            cache.reset()
            runnable(prompt, cache, num_logits_to_keep=1)

        def run() -> None:
            greedy_decode(runnable, prompt[:, -1:], tokens, cache=cache)

        return setup, run

    def prepare_reference(self) -> None:
        """Build and warm the reference columns. The batch's one fixed compilation cost."""
        from .cli import BENCH_COLUMNS

        for label in self.columns:
            which, mode = BENCH_COLUMNS[label]
            if which != "reference":
                continue
            self.log(f"[batch] building reference column {label!r} (mode={mode})")
            self._reference_columns[label] = self._make_column(self.reference, mode)

    # -- one slot --------------------------------------------------------------------

    def _build_candidate(self, hypothesis: Hypothesis):
        """A candidate module tree sharing the reference's parameter tensors."""
        import torch

        from .cli import _assert_parameters_are_shared
        from .kernels import REGISTRY
        from .model import apply_champions
        from .reference import ReferenceModel

        # Built on the **meta** device, then bound to the reference's tensors.
        #
        # `ReferenceModel(config).to("cuda")` allocates a full fresh 8.4 GB of parameters
        # and only then does `load_state_dict(assign=True)` rebind them to the reference's
        # and free the duplicates. Peak is therefore 16.8 GB of weights for a model that
        # needs 8.4 -- survivable in the single-hypothesis path, where it happens once
        # before anything is compiled, and fatal in a batch, where the reference's compiled
        # state is already resident. Every slot of batch 001 OOMed here on a 32 GB card,
        # at exactly the line below. A meta-device model allocates nothing.
        with torch.device("meta"):
            candidate = ReferenceModel(self.config)
        candidate.load_state_dict(self.reference.state_dict(), assign=True)
        # Non-persistent buffers -- `rotary_emb.inv_freq` is one -- never appear in a
        # state_dict, so `assign=True` leaves them on meta and the first forward dies with
        # "Cannot copy out of meta tensor". They are shared from the reference explicitly.
        reference_buffers = dict(self.reference.named_buffers())
        for name, buffer in list(candidate.named_buffers()):
            if buffer.is_meta:
                parent, _, leaf = name.rpartition(".")
                setattr(candidate.get_submodule(parent) if parent else candidate, leaf, reference_buffers[name])
        candidate = candidate.eval()

        before = {name: type(module) for name, module in candidate.named_modules()}
        applied = apply_champions(candidate, scoped_registry(hypothesis, REGISTRY))
        after = {name: type(module) for name, module in candidate.named_modules()}

        _assert_parameters_are_shared(self.reference, candidate)

        if hypothesis.is_identity:
            if applied or after != before:
                raise RuntimeError(
                    f"{hypothesis.slug!r} is the identity champion but installed {applied}. "
                    "It must leave the model untouched or it calibrates nothing."
                )
        elif after == before:
            # The failure this check exists for: a candidate identical to the reference
            # measures 1.00 and is indistinguishable from a well-behaved null result.
            raise RuntimeError(
                f"{hypothesis.slug!r} installed {applied} but changed no module class. "
                "Refusing to benchmark the reference while labelling it the candidate."
            )
        del before, after
        torch.cuda.synchronize()
        return candidate

    def _run_correctness(self, hypothesis: Hypothesis, candidate) -> dict[str, Any]:
        from .harness.correctness import CorrectnessReport, check_end_to_end
        from .harness.prompts import PROMPT_DIGEST
        from .kernels import REGISTRY, build_kernel_checks

        registry = scoped_registry(hypothesis, REGISTRY)
        # Layer 1 runs against `self.reference`, whose modules no install has touched, so
        # each check compares the kernel against the operation it claims to replace.
        kernel_checks = build_kernel_checks(self.reference, registry=registry, device="cuda")
        end_to_end = check_end_to_end(
            self.reference,
            candidate,
            self.prompt_ids,
            max_new_tokens=self.max_new_tokens,
            prompt_digest=PROMPT_DIGEST,
        )
        return CorrectnessReport(kernel_checks=kernel_checks, end_to_end=end_to_end).to_dict()

    def _slot_metadata(self, hypothesis: Hypothesis) -> dict[str, Any]:
        return {
            "workload": self.workload,
            "columns": list(self.columns),
            "batch_mode": True,
            "hypothesis": hypothesis.slug,
            "prediction": hypothesis.prediction,
            "reference_compilation": (
                "The reference columns were compiled once for the whole batch and reused. "
                "Their TIMINGS are not reused: the compiled column is re-timed inside this "
                "hypothesis's own interleaved rounds, so per-round drift divides out exactly "
                "as it does in a single-hypothesis run."
            ),
            "cross_slot_comparability": (
                "Ratios are comparable within a slot. Across slots they are only loosely "
                "comparable: the slots share one card and one thermal history, and "
                "interleaving divides out drift within a slot, not between them."
            ),
            "correctness_max_new_tokens": self.max_new_tokens,
        }

    def run_slot(self, hypothesis: Hypothesis) -> SlotResult:
        """Measure one hypothesis. Never raises: a failure becomes an `error` outcome."""
        import torch

        from .cli import BENCH_COLUMNS
        from .harness.bench import CudaEventTimer, run_interleaved

        started = time.monotonic()
        candidate = None
        try:
            torch.cuda.reset_peak_memory_stats()
            self.log(f"[batch] {hypothesis.slug}: building candidate")
            candidate = self._build_candidate(hypothesis)

            self.log(f"[batch] {hypothesis.slug}: correctness gates")
            correctness = self._run_correctness(hypothesis, candidate)

            self.log(f"[batch] {hypothesis.slug}: benchmarking")
            columns: dict[str, Any] = {}
            setups: dict[str, Any] = {}
            for label in self.columns:
                which, mode = BENCH_COLUMNS[label]
                if which == "reference":
                    setups[label], columns[label] = self._reference_columns[label]
                else:
                    setups[label], columns[label] = self._make_column(candidate, mode)

            result = run_interleaved(
                columns,
                self.bench_config,
                timer=CudaEventTimer(),
                setups=setups,
                metadata=self._slot_metadata(hypothesis),
            )

            ratio = result.median_ratio.get(CANDIDATE_COLUMN)
            iqr = result.iqr_ratio.get(CANDIDATE_COLUMN, 0.0)
            outcome = classify_outcome(ratio, iqr, correctness_passed=bool(correctness["passed"]))

            return SlotResult(
                hypothesis=hypothesis,
                outcome=outcome,
                median_ratio=ratio,
                iqr_ratio=iqr,
                correctness=correctness,
                bench=result.to_dict(),
                duration_s=time.monotonic() - started,
                peak_memory_mb=int(torch.cuda.max_memory_allocated() // (1024 * 1024)),
            )
        except Exception as exc:  # noqa: BLE001 - isolating the slot is the whole point
            self.log(f"[batch] {hypothesis.slug}: ERROR {type(exc).__name__}: {exc}")
            return SlotResult(
                hypothesis=hypothesis,
                outcome="error",
                error=f"{type(exc).__name__}: {exc}\n{traceback.format_exc()}",
                duration_s=time.monotonic() - started,
            )
        finally:
            del candidate
            gc.collect()
            torch.cuda.empty_cache()


def run_batch(
    runner: BatchRunner,
    batch: Batch,
    *,
    budget: SlotBudget,
    on_slot=None,
    log=print,
) -> tuple[list[SlotResult], bool | None, tuple]:
    """Run every slot that fits before the deadline.

    Returns ``(results, calibrated, prediction_scores)``. ``calibrated`` is ``None`` when
    the batch has no identity slot, which is itself worth seeing in the record.
    """
    runner.prepare_reference()

    results: list[SlotResult] = []
    for index, hypothesis in enumerate(batch):
        if not budget.can_start():
            log(f"[batch] stopping before {hypothesis.slug}: {budget.why_not()}")
            for remaining in list(batch)[index:]:
                results.append(SlotResult(hypothesis=remaining, outcome="not_run"))
            break

        log(f"[batch] slot {index}/{len(batch) - 1}: {hypothesis.slug} (predicted {hypothesis.prediction})")
        result = runner.run_slot(hypothesis)
        results.append(result)
        if result.duration_s is not None and result.outcome != "not_run":
            budget.record(result.duration_s)
        log(
            f"[batch] slot {index} {hypothesis.slug}: {result.outcome}"
            + (f" ratio={result.median_ratio:.4f}" if result.median_ratio is not None else "")
            + (f" in {result.duration_s:.0f}s" if result.duration_s else "")
        )
        if on_slot is not None:
            on_slot(result)

    by_slug = {r.hypothesis.slug: r for r in results}
    calibrated: bool | None = None
    calibration_slug = batch.calibration_slug
    if calibration_slug is not None:
        slot = by_slug.get(calibration_slug)
        if slot is not None and slot.outcome not in ("not_run", "error"):
            calibrated = calibration_holds(slot.median_ratio, slot.iqr_ratio or 0.0)
        else:
            calibrated = False

    outcomes = {slug: r.outcome for slug, r in by_slug.items()}
    scores = score_predictions(batch, outcomes, calibrated=calibrated)
    return results, calibrated, scores


def slot_record_path(root: Path, batch_id: str, slug: str) -> Path:
    return Path(root) / "batches" / batch_id / f"{slug}.json"
