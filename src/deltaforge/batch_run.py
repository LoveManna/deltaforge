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
import logging
import time
import traceback
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

from .batch import (
    Batch,
    Hypothesis,
    SlotBudget,
    calibration_holds,
    classify_outcome,
    precondition_holds,
    scoped_registry,
    score_predictions,
)
from .card_baseline import card_report

__all__ = [
    "BatchRunner",
    "SlotResult",
    "cudagraphs_during",
    "device_name",
    "graphs_compiled_during",
    "recompile_limit_for",
    "reference_gbps",
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
    #: How many graphs dynamo compiled during this slot's benchmark. This is a delta on
    #: `unique_graphs`, so **`0` means "no NEW graph", not "no compile"**: a slot whose
    #: candidate differs from an earlier slot's only in launch parameters — a pinned tile
    #: lives in `_TUNED`, not in the traced graph — passes its guards and legitimately
    #: reuses the cached graph. Rental 43's `037` and `038` did exactly that. The eager
    #: fallback this counter exists to catch (blocker 16) looks different and the bench
    #: record distinguishes them: a compiling candidate's first warmup round takes tens of
    #: seconds, a cache hit's takes as long as every other round, and rental 35's real
    #: fallback ran 6.8x slow. `None` means the counter could not be read.
    graphs_compiled: int | None = None
    #: CUDA-graph nodes inductor recorded during this slot's benchmark, and how many times
    #: it declined to record one. Batch 005 is the first hypothesis whose whole claim is
    #: that the candidate gets CUDA-graphed and the reference does not, and without these
    #: a candidate that silently did not engage returns 1.00 and reads as a refutation —
    #: which is blocker 16 in a different costume. `None` means the counter was unreadable.
    cudagraph_nodes: int | None = None
    cudagraph_skips: int | None = None
    #: The distinct reasons inductor gave for each refusal, as it logged them. Batch 005
    #: recorded the *count* and could not say why: `021` returned `nodes: 0, skips: 127`
    #: and the writeup had to rank three suspects it could not separate. A count says the
    #: hypothesis was not tested; the reason says which line to go and fix.
    cudagraph_skip_reasons: list[str] = field(default_factory=list)
    #: The tile `tune_launch_shape` chose per ``(kind, N, K)``, as ``"kind N K"`` ->
    #: ``[BLOCK_N, BLOCK_K, SPLIT_K, num_warps, num_stages]``. For batch 006 this is the
    #: finding as much as the ratio is: a slot that wins with the heuristic's own tile and
    #: one that wins with SPLIT_K=32 are different results about the same kernel.
    launch_shapes: dict[str, list[int]] = field(default_factory=dict)
    #: Wall-clock seconds per phase. Recorded even when the slot failed, because a slot
    #: that died 40 minutes into a compile is itself the measurement worth having.
    phases_s: dict[str, float] = field(default_factory=dict)
    #: Clocks, power draw, power limit and the card's active throttle reasons, sampled as
    #: this slot finished. Rental 45's card downclocked 2910 -> 2400 MHz at slot 4 and held
    #: it for the rest of the batch, and the record could only say so because someone read
    #: the log afterwards. Per slot, this is a column in the results table instead.
    card: dict[str, Any] = field(default_factory=dict)

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
            "cudagraph_nodes": self.cudagraph_nodes,
            "cudagraph_skips": self.cudagraph_skips,
            "cudagraph_skip_reasons": list(self.cudagraph_skip_reasons),
            "launch_shapes": dict(self.launch_shapes),
            "phases_s": self.phases_s,
            "card": dict(self.card),
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

        from .model import greedy_decode, prefill_setup

        runnable = model if compile_mode is None else torch.compile(model, mode=compile_mode)
        batch = self.workload["batch_size"]
        cache = model.new_cache(batch, self.workload["context_length"] + self.workload["decode_tokens"])
        tokens = self.workload["decode_tokens"]
        prompt = self.prompt

        # Prefill and cache restoration are excluded from the timed region. Inside it they
        # would add the same constant to every column, which does not cancel in a ratio: it
        # drags every ratio toward 1 and hides whatever win is really there. Being excluded
        # is also why the prefill may run once and be restored per round rather than re-run
        # -- `prefill_setup` has the arithmetic and the reasons.
        setup = prefill_setup(model, prompt, cache)

        def run() -> None:
            greedy_decode(runnable, prompt[:, -1:], tokens, cache=cache)

        return setup, run

    def _card_telemetry(self, hypothesis) -> dict[str, Any]:
        """Sample the card as this slot ends, and say so when it is throttling.

        The batch already divides clock drift out of every ratio by interleaving rounds, so
        this changes no number. What it changes is that a rental which spent its second half
        on a downclocked card says so in its own record rather than in whoever reads the log.
        """
        from .harness.report import capture_gpu_telemetry  # noqa: PLC0415

        telemetry = capture_gpu_telemetry()
        if not telemetry:
            return {}
        throttle = telemetry.get("throttle_reasons")
        throttled = bool(throttle) and throttle not in ("0x0000000000000000", "0x0", "Not Active")
        self.log(
            f"[batch] {hypothesis.slug}: card {telemetry.get('sm_mhz')} MHz SM, "
            f"{telemetry.get('memory_mhz')} MHz mem, {telemetry.get('power_w')} W of "
            f"{telemetry.get('power_limit_w')} W" + (f", THROTTLING ({throttle})" if throttled else "")
        )
        return telemetry

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
        """Layer 1 always; layer 2 under whichever policy the hypothesis registered.

        A quantised candidate cannot match bf16 tokens exactly however correct it is, so
        forcing it through the exact gate would record `incorrect` for every weight-only
        hypothesis this project will ever run — a gate that cannot distinguish "wrong" from
        "different" reports nothing. `Hypothesis.correctness` chooses, and the thresholds
        it is judged against were committed to `batches.py` before the rental.
        """
        from .harness.correctness import CorrectnessReport, check_distribution, check_end_to_end
        from .harness.prompts import PROMPT_DIGEST
        from .kernels import REGISTRY, build_kernel_checks

        registry = scoped_registry(hypothesis, REGISTRY)
        # Layer 1 runs against `self.reference`, whose modules no install has touched, so
        # each check compares the kernel against the operation it claims to replace.
        kernel_checks = build_kernel_checks(self.reference, registry=registry, device="cuda")

        if hypothesis.correctness == "approximate":
            distribution = check_distribution(
                self.reference,
                candidate,
                self.prompt_ids,
                top1_threshold=hypothesis.top1_threshold,
                kl_threshold=hypothesis.kl_threshold,
                max_new_tokens=self.max_new_tokens,
                prompt_digest=PROMPT_DIGEST,
            )
            self.log(
                f"[batch] {hypothesis.slug}: top-1 agreement {distribution.top1_agreement:.4f} "
                f"(bar {distribution.top1_threshold}), mean KL {distribution.mean_kl:.5f} nats "
                f"(bar {distribution.kl_threshold}) over {distribution.num_positions} positions"
            )
            return CorrectnessReport(kernel_checks=kernel_checks, distribution=distribution).to_dict()

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

    def _bytes_per_token(self, hypothesis: Hypothesis) -> dict[str, float]:
        """MB/token for the two scoring columns, so the slot record carries a bandwidth.

        The reference streams bf16 weights by definition; the candidate streams whatever
        its manifest says it re-encodes. A config with no published tensor manifest -- the
        CPU `tiny_config`, for instance -- gets no byte model rather than an invented one.
        """
        from .harness.bytes_model import decode_bytes_per_token

        shape = dict(
            context_length=self.workload["context_length"],
            decode_tokens=self.workload["decode_tokens"],
            batch=self.workload["batch_size"],
        )
        try:
            return {
                self.bench_config.baseline: decode_bytes_per_token(self.config, weight_bits={}, **shape),
                CANDIDATE_COLUMN: decode_bytes_per_token(
                    self.config, weight_bits=hypothesis.weight_bits, **shape
                ),
            }
        except ValueError as exc:
            self.log(f"[batch] {hypothesis.slug}: no byte model ({exc}); bandwidth not reported")
            return {}

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

                bench_config = replace(
                    self.bench_config,
                    bytes_per_token=self._bytes_per_token(hypothesis),
                    decode_tokens=self.workload["decode_tokens"],
                )

                with graphs_compiled_during() as graphs, cudagraphs_during() as cudagraph_count:
                    result = run_interleaved(
                        columns,
                        bench_config,
                        timer=CudaEventTimer(),
                        setups=setups,
                        metadata=self._slot_metadata(hypothesis),
                        progress=say,
                    )
                phase("bench", mark)
                self.log(f"[batch] {hypothesis.slug}: dynamo compiled {graphs.compiled} graph(s)")
                self.log(
                    f"[batch] {hypothesis.slug}: inductor recorded "
                    f"{cudagraph_count.nodes} cudagraph node(s), skipped {cudagraph_count.skips}"
                )
                for reason in cudagraph_count.reasons:
                    self.log(f"[batch] {hypothesis.slug}: cudagraph note: {reason}")
            if result.achieved_gbps:
                self.log(
                    f"[batch] {hypothesis.slug}: achieved "
                    + ", ".join(f"{k} {v:.0f} GB/s" for k, v in result.achieved_gbps.items())
                )
                if graphs.compiled == 0:
                    self.log(
                        f"[batch] {hypothesis.slug}: NOTE dynamo compiled no NEW graph here. "
                        f"{_no_new_graph_reading(result)} A cache hit is healthy -- a slot that "
                        "differs from an earlier one only in launch parameters reuses its graph. "
                        "An eager fallback is not, and voids the ratio: see recompile_limit_for()."
                    )

            ratio = result.median_ratio.get(CANDIDATE_COLUMN)
            iqr = result.iqr_ratio.get(CANDIDATE_COLUMN, 0.0)
            outcome = classify_outcome(ratio, iqr, correctness_passed=bool(correctness["passed"]))
            card = self._card_telemetry(hypothesis)

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
                cudagraph_nodes=cudagraph_count.nodes,
                cudagraph_skips=cudagraph_count.skips,
                cudagraph_skip_reasons=list(cudagraph_count.reasons),
                launch_shapes=_launch_shapes_now(),
                phases_s=phases,
                card=card,
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


@dataclass
class CudagraphCount:
    """CUDA-graph nodes recorded, and skips declined, inside a region.

    Two numbers rather than one because they answer different questions. ``skips`` rising
    means inductor looked at a region and refused it — the compiled baseline does that 128
    times per compile, for the mutated decode cache. ``nodes`` rising means it actually
    recorded something, which is the only evidence that a candidate claiming to be
    CUDA-graphed really is.
    """

    nodes: int | None = None
    skips: int | None = None
    #: Distinct skip messages logged inside the region, in the order first seen.
    reasons: list[str] = field(default_factory=list)


class _SkipReasonHandler(logging.Handler):
    """Collect ``skipping cudagraphs due to ...`` messages without printing them.

    Inductor's skip path is `log_cudagraph_skip_and_bump_counter`, which bumps the counter
    this context manager already reads and warns on the ``cudagraphs`` artifact logger in
    the same breath. Batch 005 took the counter and dropped the sentence beside it, and
    then spent a section of its writeup ranking suspects the sentence would have named.

    Distinct messages only, and bounded: the baseline logs one per decode step, 128 times
    a compile, and a record is not a transcript.
    """

    LIMIT = 8

    def __init__(self, sink: list[str]) -> None:
        super().__init__(level=logging.DEBUG)
        self.sink = sink

    def emit(self, record: logging.LogRecord) -> None:
        try:
            message = record.getMessage()
        except Exception:  # noqa: BLE001 - a diagnostic must not raise into the slot
            return
        if "cudagraph" not in message.lower():
            return
        message = " ".join(message.split())[:400]
        if message not in self.sink and len(self.sink) < self.LIMIT:
            self.sink.append(message)


@contextlib.contextmanager
def _capturing_skip_reasons(sink: list[str]):
    """Attach `_SkipReasonHandler` to inductor's cudagraph loggers for the region.

    The artifact logger drops records unless its artifact is enabled, so the artifact is
    enabled for the duration and restored afterwards. Everything here is best-effort: a
    torch that has moved the logger costs the evidence, never the slot.
    """
    handler = _SkipReasonHandler(sink)
    loggers = []
    restore = None
    try:
        import torch  # noqa: PLC0415

        restore = torch._logging._internal.log_state  # noqa: SLF001
        torch._logging.set_logs(cudagraphs=True)
    except Exception:  # noqa: BLE001
        restore = None
    for name in (
        "torch._inductor.cudagraph_utils.__cudagraphs",
        "torch._inductor.cudagraph_trees.__cudagraphs",
        "torch._inductor.cudagraph_utils",
    ):
        try:
            logger = logging.getLogger(name)
            logger.addHandler(handler)
            loggers.append(logger)
        except Exception:  # noqa: BLE001
            continue
    try:
        yield
    finally:
        for logger in loggers:
            logger.removeHandler(handler)
        if restore is not None:
            try:
                import torch  # noqa: PLC0415

                torch._logging.set_logs(cudagraphs=False)  # noqa: SLF001
            except Exception:  # noqa: BLE001
                pass


def _launch_shapes_now() -> dict[str, list[int]]:
    """Whatever `tune_launch_shape` has measured so far, flattened for JSON.

    Read after the benchmark rather than passed in, because the tuner runs at install time
    and memoises per ``(kind, N, K)`` for the life of the process: slot N's record
    therefore names every tile in force during slot N, including the ones an earlier slot
    paid to measure. Empty on any rental that did no tuning, which is every rental before
    batch 006.
    """
    try:
        from .kernels.tiled_gemv import tuned_launch_shapes  # noqa: PLC0415

        return {f"{kind} {n} {k}": list(shape) for (kind, n, k), shape in tuned_launch_shapes().items()}
    except Exception:  # noqa: BLE001 - a diagnostic must not be able to fail a slot
        return {}


@contextlib.contextmanager
def cudagraphs_during(counters=None, manager_for=None):
    """Count recorded CUDA-graph nodes and cudagraph skips across a region.

    Both readings are private API, so failing to read either costs the evidence and not
    the slot — the same policy `graphs_compiled_during` follows, and for the same reason:
    a missing diagnostic must not be able to end a measurement.
    """
    if counters is None:  # pragma: no cover - exercised on the GPU path
        try:
            from torch._dynamo.utils import counters as counters  # noqa: PLC0415
        except ImportError:
            yield CudagraphCount()
            return
    if manager_for is None:  # pragma: no cover - exercised on the GPU path

        def manager_for():
            from torch._inductor.cudagraph_trees import get_manager  # noqa: PLC0415

            return get_manager(device_index=None, create_if_none_exists=False)

    def nodes_now() -> int | None:
        try:
            manager = manager_for()
        except Exception:  # noqa: BLE001 - private API; its absence is not a slot failure
            return None
        if manager is None:
            return 0
        try:
            total = 0
            for roots in manager.roots.values():
                stack = list(roots)
                while stack:
                    node = stack.pop()
                    total += 1
                    for children in node.children.values():
                        stack.extend(children)
            return total
        except Exception:  # noqa: BLE001
            return None

    def skips_now() -> int | None:
        try:
            return int(counters["inductor"]["cudagraph_skips"])
        except (KeyError, TypeError, ValueError):
            return None

    count = CudagraphCount()
    before_nodes, before_skips = nodes_now(), skips_now()
    with _capturing_skip_reasons(count.reasons):
        yield count
    after_nodes, after_skips = nodes_now(), skips_now()
    if before_nodes is not None and after_nodes is not None:
        count.nodes = after_nodes - before_nodes
    if before_skips is not None and after_skips is not None:
        count.skips = after_skips - before_skips


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


def _no_new_graph_reading(result) -> str:
    """Which reading of `graphs_compiled == 0` this slot's own timings support.

    The counter cannot tell a guard-passing cache hit from dynamo giving up and running
    the candidate eagerly, but the warmup round can. A candidate that compiled spends tens
    of seconds in its first round (rental 43: 70437 ms and 63639 ms, against ~1370 ms
    steady state); a candidate that reused a graph spends the same as every other round;
    and rental 35's real fallback ran **6.8x slower than the reference** for every round.

    So this reports the evidence rather than a verdict, because a wrong verdict here is
    worse than none: batch 007 spent a writeup arguing back from raw round timings that
    the warning had already thrown away.
    """
    rounds = (result.timings_ms or {}).get(CANDIDATE_COLUMN) or []
    median = (result.median_ms or {}).get(CANDIDATE_COLUMN)
    if not rounds or not median:
        return "No candidate timings to read it against."
    first = rounds[0]
    if first > 4 * median:
        return f"First round {first:.0f} ms against {median:.0f} ms median -- it DID compile."
    return f"First round {first:.0f} ms against {median:.0f} ms median -- no compile happened."


def device_name() -> str:
    """What `torch.cuda.get_device_name()` says, or ``""`` where there is no card.

    Suppressed rather than required: `run_batch` is driven by a fake runner in the CPU
    tests, and a pre-flight that could raise would be a pre-flight that can fail a batch.
    """
    try:
        import torch  # noqa: PLC0415

        if torch.cuda.is_available():
            return str(torch.cuda.get_device_name())
    except Exception:  # noqa: BLE001 - a report is never worth an exception
        return ""
    return ""


def reference_gbps(result: SlotResult) -> float | None:
    """The `compiled` column's achieved bandwidth in a finished slot, if it declared bytes."""
    achieved = (result.bench or {}).get("achieved_gbps") or {}
    value = achieved.get(BASELINE_COLUMN)
    return float(value) if value else None


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
    ratios: dict[str, float | None] = {}
    for index, hypothesis in enumerate(batch):
        # Checked before the budget: a slot an earlier ratio has already settled should
        # be skipped whether or not the clock would have allowed it, and skipping it is
        # what buys the clock for the slots that are still open questions.
        if not precondition_holds(hypothesis.requires, ratios):
            need = hypothesis.requires
            read = need.describe(ratios)
            log(f"[batch] skipping {hypothesis.slug}: {read}")
            results.append(
                SlotResult(
                    hypothesis=hypothesis,
                    outcome="precondition_failed",
                    error=f"{read}: {need.reason}",
                )
            )
            continue
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
        ratios[hypothesis.slug] = result.median_ratio
        if hypothesis.slug == batch.calibration_slug:
            # Said here rather than in the writeup: rental 43's card ran the reference at
            # 0.66 of rental 40's and flattened every effect in the batch, and the figure
            # that would have said so was already in this slot. It reports and never
            # decides — see `card_baseline`.
            log(card_report(device_name(), reference_gbps(result)))
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
