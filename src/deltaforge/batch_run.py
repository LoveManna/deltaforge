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

import contextlib
import gc
import time
import traceback
from dataclasses import dataclass, field
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

__all__ = [
    "BatchRunner",
    "SlotResult",
    "graphs_compiled_during",
    "recompile_limit_for",
    "release_compiled_state",
    "run_batch",
]


#: Dynamo's default: after this many cache entries on one code object it stops compiling
#: that code object and runs it **eagerly**, for the rest of the process, with a warning.
DEFAULT_RECOMPILE_LIMIT = 8


def recompile_limit_for(slots: int) -> int:
    """How many dynamo cache entries a batch of ``slots`` hypotheses legitimately needs.

    Every slot builds a fresh candidate module and compiles it, and dynamo caches per
    *code object* with guards — so slot N is cache entry N against the same
    `ReferenceModel.forward`. At the default of 8, a batch longer than a handful of slots
    silently stops compiling candidates and times eager ones instead.

    Rental 35 is what that looks like from the outside. The limit tripped inside slot 2;
    from slot 3 on the candidate's first round fell from ~170s of compiling to ~6s of not
    compiling, and six hypotheses attacking six different operations all returned ratios
    between 0.146 and 0.157, because every one of them was measuring eager against
    compiled rather than a kernel against inductor.

    Two entries per slot covers the candidate and its dynamic-shape variant, plus the
    default for the reference and whatever else shares those code objects. Still bounded,
    and deliberately: a limit that grew without end would hide the runaway recompilation
    this setting exists to catch.
    """
    return min(64, max(DEFAULT_RECOMPILE_LIMIT, 2 * slots + DEFAULT_RECOMPILE_LIMIT))


#: The scoring comparison, and the reason a batch is worth running at all: identical
#: `max-autotune` treatment on both sides, the only difference being who wrote the kernel.
BASELINE_COLUMN = "compiled"
CANDIDATE_COLUMN = "candidate_compiled"


def release_compiled_state(torch_module, cudagraph_module=None, log=print) -> list[str]:
    """Free what a finished slot leaves on the card, and say what was freed.

    `del candidate; gc.collect(); empty_cache()` frees the module and its KV cache but
    **not** the CUDA-graph pool inductor recorded for `candidate_compiled`. Those pools are
    held by inductor's graph trees, so slot N stays resident on N of them.

    This used to reclaim them with `reset_cudagraph_trees`, on the stated premise that the
    reference would "re-record on the next slot's first warmup call". **That premise was
    false, and it emptied the batch.** The shutdown is permanent for a callable that has
    already been recorded, and the trees are per *device*, not per model — so releasing the
    candidate tore down the reference columns with it.

    It could only show up once a slot actually finished, which took until rental 34: slot 0
    calibrated at ratio 1.0009, and then all eight scoring slots died in 23s each on
    `AssertionError: Running CUDAGraph after shutdown`, none of them reaching a timing.

    The reclaim is also worth less than it looks. On rental 34, slot 0 recorded CUDA graphs
    for both columns and sat at **8.07 GiB allocated / 8.08 reserved of 31.36, before and
    after the release alike** — at two columns with autograd off, what the tree reset gave
    back was below the resolution of the number being logged. (The 8.85 GiB reserved seen
    later was slot 1's transient, and `empty_cache` is what returned it.) So the pools now
    stay. `_log_memory` prints allocated and reserved after every slot, so if they do
    accumulate the record will say so, rather than a slot dying to prevent it.

    `cudagraph_module` is still accepted so the call site keeps saying what it is choosing
    not to do.
    """
    if not torch_module.cuda.is_available():
        return []

    steps: list[str] = []
    torch_module.cuda.synchronize()
    steps.append("synchronize")
    torch_module.cuda.empty_cache()
    steps.append("empty_cache")
    return steps


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
    #: How many graphs dynamo compiled during this slot's benchmark. `0` means the ratio
    #: compares eager against compiled; `None` means the counter could not be read.
    graphs_compiled: int | None = None
    #: Wall-clock seconds per phase. Recorded even when the slot failed, because a slot
    #: that died 40 minutes into a compile is itself the measurement worth having.
    phases_s: dict[str, float] = field(default_factory=dict)

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
            "graphs_compiled": self.graphs_compiled,
            "phases_s": self.phases_s,
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
            #
            # And because it is excluded, the prefill runs on `model` rather than
            # `runnable`: compiling it costs everything and buys nothing. `gated_delta_rule`
            # unrolls its scan over the sequence, so a 2048-token prefill hands inductor
            # ~22 nodes x 2048 tokens x 24 linear-attention layers -- about a million FX
            # nodes -- under `max-autotune`, once per column. That graph is where rentals
            # 22, 30, 31 and 32 all died; rental 32's slot 0 spent its entire 6980.9s cap
            # inside it. The decode steps below, which are the measurement, run the scan
            # once per token and still go through the compiled wrapper.
            cache.reset()
            model(prompt, cache, num_logits_to_keep=1)

        def run() -> None:
            greedy_decode(runnable, prompt[:, -1:], tokens, cache=cache)

        return setup, run

    def _log_memory(self, where: str) -> None:
        """Say how much of the card is gone, and where it went.

        Rental 21 lost all nine slots to an OOM at 30.71 GiB of 31.36, and the logs could
        not say what was holding it: the batch reported memory only on slots that
        *succeeded*, which is exactly the set that is empty when memory is the problem.
        A number that only prints on the happy path is not instrumentation.
        """
        import torch

        if not torch.cuda.is_available():
            return
        allocated = torch.cuda.memory_allocated() / (1024**3)
        reserved = torch.cuda.memory_reserved() / (1024**3)
        total = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        self.log(
            f"[batch] memory after {where}: {allocated:.2f} GiB allocated, "
            f"{reserved:.2f} GiB reserved, of {total:.2f} GiB"
        )

    def prepare_reference(self) -> None:
        """Build and warm the reference columns. The batch's one fixed compilation cost."""
        from .cli import BENCH_COLUMNS

        self._log_memory("loading weights")
        for label in self.columns:
            which, mode = BENCH_COLUMNS[label]
            if which != "reference":
                continue
            self.log(f"[batch] building reference column {label!r} (mode={mode})")
            self._reference_columns[label] = self._make_column(self.reference, mode)
            self._log_memory(f"reference column {label!r}")

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
                setattr(
                    candidate.get_submodule(parent) if parent else candidate, leaf, reference_buffers[name]
                )
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

    def run_slot(self, hypothesis: Hypothesis, cap_s: float = 0.0) -> SlotResult:
        """Measure one hypothesis. Never raises: a failure becomes an `error` outcome.

        ``cap_s`` bounds the slot's wall clock. `SlotTimeout` is an `Exception`, so a slot
        that overruns is recorded as `error` by the same handler that isolates a wrong
        kernel — which is the point: a runaway slot costs a slot, not the rental.
        """
        import torch

        from .cli import BENCH_COLUMNS
        from .harness.bench import CudaEventTimer, run_interleaved
        from .slot_timer import slot_deadline

        started = time.monotonic()
        phases: dict[str, float] = {}
        candidate = None

        def phase(name: str, mark: float) -> float:
            now = time.monotonic()
            phases[name] = now - mark
            return now

        try:
            with slot_deadline(cap_s):
                torch.cuda.reset_peak_memory_stats()
                mark = time.monotonic()

                self.log(f"[batch] {hypothesis.slug}: building candidate")
                candidate = self._build_candidate(hypothesis)
                mark = phase("candidate_build", mark)

                self.log(f"[batch] {hypothesis.slug}: correctness gates")
                correctness = self._run_correctness(hypothesis, candidate)
                mark = phase("correctness", mark)

                self.log(f"[batch] {hypothesis.slug}: benchmarking")
                self._log_memory(f"{hypothesis.slug} candidate build")
                columns: dict[str, Any] = {}
                setups: dict[str, Any] = {}
                for label in self.columns:
                    which, mode = BENCH_COLUMNS[label]
                    if which == "reference":
                        setups[label], columns[label] = self._reference_columns[label]
                    else:
                        setups[label], columns[label] = self._make_column(candidate, mode)
                        mark = phase(f"compile_{label}", mark)
                        self._log_memory(f"{hypothesis.slug} column {label!r}")

                bench_started = time.monotonic()

                def say(round_index: int, label: str, elapsed_ms: float) -> None:
                    # Round 0 is warmup, and warmup is where `max-autotune` compiles. A
                    # slot that dies mid-compile now says which column it was on and how
                    # long it had been there; rental 32's could only say "cap exceeded".
                    since = time.monotonic() - bench_started
                    self.log(
                        f"[batch] {hypothesis.slug}: round {round_index} {label} "
                        f"{elapsed_ms:.1f} ms ({since:.0f}s into the benchmark)"
                    )

                with graphs_compiled_during() as graphs:
                    result = run_interleaved(
                        columns,
                        self.bench_config,
                        timer=CudaEventTimer(),
                        setups=setups,
                        metadata=self._slot_metadata(hypothesis),
                        progress=say,
                    )
                phase("bench", mark)
                self.log(f"[batch] {hypothesis.slug}: dynamo compiled {graphs.compiled} graph(s)")
                if graphs.compiled == 0:
                    self.log(
                        f"[batch] {hypothesis.slug}: WARNING nothing compiled during this "
                        "benchmark -- the ratio below compares eager against compiled, not a "
                        "kernel against inductor. See recompile_limit_for()."
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
                graphs_compiled=graphs.compiled,
                phases_s=phases,
            )
        except Exception as exc:  # noqa: BLE001 - isolating the slot is the whole point
            self.log(f"[batch] {hypothesis.slug}: ERROR {type(exc).__name__}: {exc}")
            self._log_memory(f"{hypothesis.slug} failure")
            return SlotResult(
                hypothesis=hypothesis,
                outcome="error",
                error=f"{type(exc).__name__}: {exc}\n{traceback.format_exc()}",
                duration_s=time.monotonic() - started,
                phases_s=phases,
            )
        finally:
            del candidate
            gc.collect()
            cudagraphs = None
            # Private API, and absent on some builds. Its absence costs the reclaim, not
            # the batch — `release_compiled_state` handles `None`.
            with contextlib.suppress(ImportError):
                from torch._inductor import cudagraph_trees as cudagraphs
            release_compiled_state(torch, cudagraphs, log=self.log)
            self._log_memory(f"{hypothesis.slug} released")


@dataclass
class GraphCount:
    """How many graphs dynamo compiled inside a region. ``None`` when it could not be read."""

    compiled: int | None = None


@contextlib.contextmanager
def graphs_compiled_during(counters=None):
    """Count dynamo's compilations across a region, so a record can prove one happened.

    A candidate that dynamo has stopped compiling still produces a median, an IQR and a
    ratio, and nothing about them looks wrong. Rental 35 reported six such ratios, clustered
    between 0.146 and 0.157 across six unrelated kernels, and the only evidence that they
    measured eager rather than a kernel was a warning buried in the rental log.

    `torch._dynamo.utils.counters` is private, so failing to read it costs the evidence and
    not the slot — hence ``None`` rather than an exception or a misleading zero.
    """
    if counters is None:  # pragma: no cover - exercised on the GPU path
        try:
            from torch._dynamo.utils import counters as counters  # noqa: PLC0415
        except ImportError:
            yield GraphCount()
            return

    count = GraphCount()
    try:
        before = counters["stats"]["unique_graphs"]
    except (KeyError, TypeError):
        yield count
        return

    yield count
    try:
        count.compiled = counters["stats"]["unique_graphs"] - before
    except (KeyError, TypeError):  # pragma: no cover - defensive
        count.compiled = None


def _raise_recompile_limit(slots: int, log=print) -> None:
    """Give dynamo room for one candidate per slot, and say so in the log.

    Torch renamed `cache_size_limit` to `recompile_limit`; both names are set where they
    exist so this works either side of that. Imported lazily and suppressed, because
    `run_batch` is driven by a fake runner in the CPU tests and must not require torch.
    """
    wanted = recompile_limit_for(slots)
    try:
        from torch._dynamo import config as dynamo_config  # noqa: PLC0415
    except ImportError:  # pragma: no cover - CPU test path has torch, the smoke path may not
        return
    for name in ("recompile_limit", "cache_size_limit"):
        if getattr(dynamo_config, name, None) is not None:
            setattr(dynamo_config, name, wanted)
    log(f"[batch] dynamo recompile limit raised to {wanted} for {slots} slots")


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
    _raise_recompile_limit(len(batch), log=log)
    runner.prepare_reference()

    results: list[SlotResult] = []
    for index, hypothesis in enumerate(batch):
        if not budget.can_start():
            log(f"[batch] stopping before {hypothesis.slug}: {budget.why_not()}")
            # `not_run` means the batch stopped early having already measured something.
            # `starved` means the rental scored nothing at all and the session gate is the
            # reason — the evidence for raising that gate, which the last two rentals
            # needed and did not have. Flattening the two would erase it.
            scored = any(r.outcome in ("win", "loss", "inconclusive", "incorrect") for r in results)
            outcome = "not_run" if scored else "starved"
            for remaining in list(batch)[index:]:
                results.append(SlotResult(hypothesis=remaining, outcome=outcome))
            break

        log(f"[batch] slot {index}/{len(batch) - 1}: {hypothesis.slug} (predicted {hypothesis.prediction})")
        result = runner.run_slot(hypothesis, cap_s=budget.cap_for(index))
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
