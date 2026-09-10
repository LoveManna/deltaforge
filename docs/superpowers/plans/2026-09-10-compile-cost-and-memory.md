# Compile Cost and Memory Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make a DeltaForge rental able to finish at least one hypothesis — by carrying the compile cache between rentals, freeing the CUDA-graph pools a batch leaks, capping any slot that runs away, and refusing to rent at all when the session gate cannot fit one hypothesis.

**Architecture:** Policy stays in torch-free modules (`batch.py`, a new `slot_timer.py`) so it is decided on CPU and tested without a GPU; `batch_run.py` keeps orchestration only; `remote/*.sh` gains the cache transport and the pre-flight gate. Nothing changes about the measurement itself — both scored columns keep identical `max-autotune` treatment.

**Tech Stack:** Python 3.14, PyTorch 2.14 (CPU locally, CUDA on the box), pytest, POSIX shell, rsync, uv, ruff.

## Global Constraints

- **Never modify `src/deltaforge/reference.py`.** `reference_purity_test.py` enforces it by AST inspection.
- **`uv run pytest` must pass on a CPU-only checkout with no CUDA and no checkpoint.** Anything needing either carries the `gpu` or `weights` marker and skips on a real condition.
- **No torch import in `batch.py`, `batches.py`, `ledger.py`, or the new `slot_timer.py`.** They are the CPU-decidable layer.
- **Both scoring columns get identical treatment.** `compiled` and `candidate_compiled` are both `max-autotune`; any change that touches one must touch the other.
- **Do not fabricate, estimate, or placeholder any measured number.** Estimates that reach a doc are labelled as estimates.
- Style: `uv run ruff check . && uv run ruff format --check .` must pass. Line length follows `pyproject.toml`.
- Tests are colocated as `<module>_test.py`; `testpaths` is `src` and `remote`.
- Commit messages: imperative subject describing the change, body explaining the mechanism. Trailers `Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>` and `Claude-Session: https://claude.ai/code/session_01G5oRrY6Frd9Y48C4UPFUtc`.

---

## File Structure

| File | Responsibility | Task |
|---|---|---|
| `src/deltaforge/slot_timer.py` | **New.** Wall-clock cap for one slot; torch-free, no knowledge of batches. | 2 |
| `src/deltaforge/slot_timer_test.py` | **New.** Tests for the above. | 2 |
| `src/deltaforge/batch.py` | `starved` outcome, per-slot cap policy, `phases.env` budget arithmetic. | 1, 7 |
| `src/deltaforge/batch_run.py` | Arms the cap, releases GPU state between slots, times phases. | 2, 3, 4 |
| `src/deltaforge/cli.py` | Two-column default; writes `phases.env`; carries phases into records. | 4, 5, 7 |
| `src/deltaforge/harness/report.py` | `phases` field on both record types. | 4 |
| `src/deltaforge/batches.py` | Batch 002 manifest, calibration flag. | 9 |
| `remote/run_remote.sh` | Cache env vars, cache push/pull, pre-flight gate, new gate values. | 6, 7, 8 |
| `remote/sync.sh` | `cache-up` / `cache-down` transport. | 6 |
| `remote/watchdog.sh`, `remote/provision.sh` | New timeout defaults. | 8 |
| `.gitignore` | `cache/`. | 6 |
| `AGENT.md`, `README.md`, `docs/BATCHES.md` | The §6.1 close-out pass. | 9 |

---

### Task 1: `starved` outcome and the per-slot cap policy

Pure policy, no torch, no GPU. `SlotBudget` currently decides only whether a slot may *start*; it gains the cap that bounds a slot that has already started, plus the exemption that protects the guarantee.

**Files:**
- Modify: `src/deltaforge/batch.py` (`BATCH_OUTCOMES` at line ~68, `score_predictions` at line ~271, `SlotBudget` at line ~302)
- Test: `src/deltaforge/batch_test.py`

**Interfaces:**
- Consumes: nothing from earlier tasks.
- Produces:
  - `BATCH_OUTCOMES` gains `"starved"`.
  - `SlotBudget.slot_cap_s: float = 1800.0` and `SlotBudget.uncapped_slots: int = 2` fields.
  - `SlotBudget.cap_for(index: int) -> float` — seconds this slot may take.
  - `score_predictions` treats `"starved"` exactly as `"not_run"` (scores `None`).

- [ ] **Step 1: Write the failing tests**

Append to `src/deltaforge/batch_test.py`:

```python
def test_the_first_two_slots_are_never_capped_below_the_remaining_budget():
    """The cap exists to stop slot 7 eating slot 8.

    Applying it to the first two slots would defeat the guarantee it protects: the
    identity slot plus one kernel slot is the minimum a session must be able to finish.
    """
    budget = SlotBudget(deadline_epoch=1000.0, clock=lambda: 0.0, slot_cap_s=60.0)

    assert budget.cap_for(0) == pytest.approx(1000.0)
    assert budget.cap_for(1) == pytest.approx(1000.0)
    assert budget.cap_for(2) == pytest.approx(60.0)


def test_a_capped_slot_never_outlives_the_deadline():
    """A cap larger than what is left is not a licence to overrun the session."""
    budget = SlotBudget(deadline_epoch=100.0, clock=lambda: 0.0, slot_cap_s=600.0)

    assert budget.cap_for(5) == pytest.approx(100.0)


def test_a_starved_prediction_is_untested_not_wrong():
    """A slot the clock killed tested nothing. Counting it wrong understates the record
    exactly as counting it right would flatter it."""
    batch = Batch(batch_id="b", hypotheses=(_identity(), _kernel_hypothesis("001-x")))

    scores = score_predictions(batch, {"000-identity": "win", "001-x": "starved"}, calibrated=True)

    assert scores[1].correct is None
    assert scores[1].outcome == "starved"
```

If `_identity()` / `_kernel_hypothesis()` helpers do not already exist in that file, use whatever fixture the neighbouring tests use to build a `Hypothesis`; read the top of `batch_test.py` first and match it.

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest src/deltaforge/batch_test.py -k "capped or starved" -v`
Expected: FAIL — `SlotBudget.__init__() got an unexpected keyword argument 'slot_cap_s'`, and the `starved` test fails on `PredictionScore` accepting the outcome or on the assertion.

- [ ] **Step 3: Implement**

In `src/deltaforge/batch.py`, extend `BATCH_OUTCOMES` and its comment:

```python
#: `starved` — the session ran out of clock before any scoring slot completed. Distinct
#:           from `not_run` on purpose: `not_run` means the batch stopped early having
#:           already measured something, `starved` means the rental produced nothing and
#:           the gate is the reason. It is the evidence for raising the gate.
BATCH_OUTCOMES = ("win", "loss", "inconclusive", "incorrect", "error", "not_run", "starved")
```

In `score_predictions`, widen the untested branch:

```python
        if outcome in ("error", "not_run", "starved"):
            correct: bool | None = None
```

In `SlotBudget`, add the two fields after `safety_factor` and the method after `can_start`:

```python
    #: Ceiling on a single slot once it has started. `can_start` bounds what a slot may
    #: begin; nothing bounded what it could then do. Rental 22 spent ~40 minutes inside
    #: one slot's cold compile and the session ended with nothing measured.
    slot_cap_s: float = 1800.0
    #: How many leading slots are exempt from that ceiling. The identity champion plus
    #: one kernel slot is the least a rental may produce and still be worth its money, so
    #: capping them would defeat the guarantee the cap exists to protect.
    uncapped_slots: int = 2
```

```python
    def cap_for(self, index: int) -> float:
        """Seconds slot ``index`` may take. Never longer than the session has left."""
        remaining = self.remaining_s()
        if index < self.uncapped_slots:
            return remaining
        return min(self.slot_cap_s, remaining)
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `uv run pytest src/deltaforge/batch_test.py -v`
Expected: PASS, including the existing budget tests.

- [ ] **Step 5: Commit**

```bash
git add src/deltaforge/batch.py src/deltaforge/batch_test.py
git commit -m "Bound a slot that has already started, and record when the clock won"
```

---

### Task 2: The slot timer

A slot's cap has to interrupt work that is inside `torch.compile` — C-level, holding the GIL only intermittently, and not checking any flag we own. `_thread.interrupt_main()` raises `KeyboardInterrupt` in the main thread at the next bytecode boundary, which inductor's Python-level autotuning loop reaches constantly.

**Files:**
- Create: `src/deltaforge/slot_timer.py`
- Test: `src/deltaforge/slot_timer_test.py`

**Interfaces:**
- Consumes: `SlotBudget.cap_for` from Task 1 (the caller supplies seconds; this module does not know about budgets).
- Produces: `SlotTimeout(Exception)`, and `slot_deadline(seconds: float) -> contextmanager` yielding a `_Armed` object with `.fired: bool`.

- [ ] **Step 1: Write the failing test**

Create `src/deltaforge/slot_timer_test.py`:

```python
"""The cap that stops one slot from taking the rental."""

import time

import pytest

from .slot_timer import SlotTimeout, slot_deadline


def test_work_that_finishes_in_time_is_untouched():
    with slot_deadline(5.0) as armed:
        result = sum(range(1000))

    assert result == 499500
    assert armed.fired is False


def test_work_that_overruns_is_interrupted():
    """The interrupt arrives as KeyboardInterrupt and is re-raised as SlotTimeout, so a
    caller catching Exception sees it: a slot that runs away must cost a slot, not the
    rental."""
    started = time.monotonic()

    with pytest.raises(SlotTimeout) as excinfo:
        with slot_deadline(0.1):
            while True:
                time.sleep(0.01)

    assert time.monotonic() - started < 5.0
    assert "0.1" in str(excinfo.value)


def test_a_cap_of_zero_or_less_does_not_arm():
    """A slot with no time left is refused by SlotBudget.can_start, not by a timer that
    fires instantly and turns a clean stop into an error record."""
    with slot_deadline(0.0) as armed:
        time.sleep(0.05)

    assert armed.fired is False


def test_the_timer_is_disarmed_when_the_body_raises():
    with pytest.raises(ValueError):
        with slot_deadline(30.0) as armed:
            raise ValueError("boom")

    assert armed.fired is False
    time.sleep(0.05)  # nothing may arrive after the body has left
```

- [ ] **Step 2: Run it to verify it fails**

Run: `uv run pytest src/deltaforge/slot_timer_test.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'deltaforge.slot_timer'`.

- [ ] **Step 3: Implement**

Create `src/deltaforge/slot_timer.py`:

```python
"""A wall-clock cap for one unit of work.

`SlotBudget.can_start` decides whether a slot may *begin*. Nothing bounded what it could
then do: rental 22 spent ~40 minutes inside slot 0's cold `max-autotune` compile and the
session gate ended the run with no measurement at all.

The work to interrupt is mostly inside inductor — C extensions, subprocess pools, and a
Python-level autotuning loop. A cooperative flag would never be read, so this uses
`_thread.interrupt_main()`, which raises `KeyboardInterrupt` in the main thread at the
next bytecode boundary. That boundary arrives constantly during autotuning.

**Nothing here imports torch.** It is a timer.
"""

from __future__ import annotations

import _thread
import threading
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Iterator

__all__ = ["SlotTimeout", "slot_deadline"]


class SlotTimeout(Exception):
    """The capped work did not finish in time.

    Deliberately an `Exception` rather than a `BaseException`: `BatchRunner.run_slot`
    catches `Exception` to isolate a slot, and a cap that escaped that handler would take
    the rental down — which is the failure it exists to prevent.
    """


@dataclass
class _Armed:
    fired: bool = False


@contextmanager
def slot_deadline(seconds: float) -> Iterator[_Armed]:
    """Interrupt the body after ``seconds``, raising `SlotTimeout`.

    A non-positive cap does not arm: a slot with no time left is refused by
    `SlotBudget.can_start`, and a timer that fired instantly would turn a clean stop into
    an error record.
    """
    armed = _Armed()
    if seconds <= 0:
        yield armed
        return

    def fire() -> None:
        armed.fired = True
        _thread.interrupt_main()

    timer = threading.Timer(seconds, fire)
    timer.daemon = True
    timer.start()
    try:
        yield armed
    except KeyboardInterrupt:
        if armed.fired:
            raise SlotTimeout(f"exceeded its {seconds:.1f}s cap") from None
        raise
    finally:
        timer.cancel()
```

- [ ] **Step 4: Run it to verify it passes**

Run: `uv run pytest src/deltaforge/slot_timer_test.py -v`
Expected: PASS, 4 tests, in well under a second each.

- [ ] **Step 5: Commit**

```bash
git add src/deltaforge/slot_timer.py src/deltaforge/slot_timer_test.py
git commit -m "Give a slot a wall clock it cannot ignore"
```

---

### Task 3: Release the CUDA-graph pools between slots

The leak that ate rental 21. Written as a free function taking the modules it touches, so it is testable with fakes on a CPU.

**Files:**
- Modify: `src/deltaforge/batch_run.py` (imports at top; `run_slot`'s `finally`, line ~330)
- Test: `src/deltaforge/batch_run_test.py`

**Interfaces:**
- Consumes: nothing.
- Produces: `release_compiled_state(torch_module, cudagraph_module=None, log=print) -> list[str]` returning the names of the steps it actually performed.

- [ ] **Step 1: Write the failing tests**

Append to `src/deltaforge/batch_run_test.py`:

```python
def test_releasing_compiled_state_resets_the_cudagraph_pools():
    """Slot N was resident on N graph pools.

    `del candidate; empty_cache()` frees the module and its KV cache but not the pool
    inductor recorded for `candidate_compiled`, which is where rental 21's missing
    ~22 GiB went.
    """
    calls = []
    torch_module = SimpleNamespace(
        cuda=SimpleNamespace(
            is_available=lambda: True,
            empty_cache=lambda: calls.append("empty_cache"),
            synchronize=lambda: calls.append("synchronize"),
        )
    )
    cudagraphs = SimpleNamespace(reset_cudagraph_trees=lambda: calls.append("reset_cudagraph_trees"))

    steps = release_compiled_state(torch_module, cudagraphs, log=lambda _: None)

    assert "reset_cudagraph_trees" in calls
    assert calls.index("reset_cudagraph_trees") < calls.index("empty_cache")
    assert "reset_cudagraph_trees" in steps


def test_releasing_compiled_state_survives_a_torch_without_the_private_api():
    """`reset_cudagraph_trees` is private API. A torch that lacks it must cost us the
    reclaim, not the batch."""
    calls = []
    torch_module = SimpleNamespace(
        cuda=SimpleNamespace(
            is_available=lambda: True,
            empty_cache=lambda: calls.append("empty_cache"),
            synchronize=lambda: calls.append("synchronize"),
        )
    )

    steps = release_compiled_state(torch_module, SimpleNamespace(), log=lambda _: None)

    assert "empty_cache" in calls
    assert "reset_cudagraph_trees" not in steps


def test_releasing_compiled_state_does_nothing_without_cuda():
    torch_module = SimpleNamespace(cuda=SimpleNamespace(is_available=lambda: False))

    assert release_compiled_state(torch_module, None, log=lambda _: None) == []
```

Add `from types import SimpleNamespace` and `release_compiled_state` to that file's imports.

- [ ] **Step 2: Run to verify they fail**

Run: `uv run pytest src/deltaforge/batch_run_test.py -k release -v`
Expected: FAIL — `ImportError: cannot import name 'release_compiled_state'`.

- [ ] **Step 3: Implement**

In `src/deltaforge/batch_run.py`, add to `__all__` and define above `BatchRunner`:

```python
def release_compiled_state(torch_module, cudagraph_module=None, log=print) -> list[str]:
    """Free what a finished slot leaves on the card, and say what was freed.

    `del candidate; gc.collect(); empty_cache()` frees the module and its KV cache but
    **not** the CUDA-graph pool that inductor recorded for `candidate_compiled`. Those
    pools are held by inductor's graph trees, so slot N was resident on N of them — which
    is where rental 21's ~22 GiB went, given that rental 22 measured construction at
    0.11 GiB.

    Deliberately **not** `torch._dynamo.reset()`, which would discard the reference's
    compilation too. That is the batch's entire saving. Resetting the graph trees costs
    the reference a graph re-record on the next slot's first warmup call — seconds, not a
    recompile.

    `reset_cudagraph_trees` is private API, so its absence costs the reclaim and not the
    batch.
    """
    if not torch_module.cuda.is_available():
        return []

    steps: list[str] = []
    reset = getattr(cudagraph_module, "reset_cudagraph_trees", None)
    if reset is not None:
        reset()
        steps.append("reset_cudagraph_trees")
    else:
        log("[batch] torch has no reset_cudagraph_trees; graph pools stay resident")

    torch_module.cuda.synchronize()
    steps.append("synchronize")
    torch_module.cuda.empty_cache()
    steps.append("empty_cache")
    return steps
```

Then use it in `run_slot`'s `finally`, replacing the existing three lines:

```python
        finally:
            del candidate
            gc.collect()
            cudagraphs = None
            try:
                from torch._inductor import cudagraph_trees as cudagraphs
            except ImportError:  # pragma: no cover - depends on the torch build
                pass
            release_compiled_state(torch, cudagraphs, log=self.log)
            self._log_memory(f"{hypothesis.slug} released")
```

- [ ] **Step 4: Run to verify they pass**

Run: `uv run pytest src/deltaforge/batch_run_test.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/deltaforge/batch_run.py src/deltaforge/batch_run_test.py
git commit -m "Give back the graph pool a finished slot was still holding"
```

---

### Task 4: Phase timings, and arming the cap

The calibration rental's actual product. Also wires Task 1's `cap_for` and Task 2's `slot_deadline` into the slot loop.

**Files:**
- Modify: `src/deltaforge/batch_run.py` (`SlotResult`, `run_slot`, `run_batch`)
- Modify: `src/deltaforge/harness/report.py` (`ResultRecord`, `BatchRecord`)
- Modify: `src/deltaforge/cli.py` (`write_slot`, `BatchRecord` construction)
- Test: `src/deltaforge/batch_run_test.py`, `src/deltaforge/harness/report_test.py`

**Interfaces:**
- Consumes: `slot_deadline`, `SlotTimeout` (Task 2); `SlotBudget.cap_for` (Task 1).
- Produces:
  - `SlotResult.phases_s: dict[str, float]`, included in `to_slot_dict()` as `"phases_s"`.
  - `ResultRecord.phases: dict[str, float]` and `BatchRecord.phases: dict[str, float]`, both serialised under `"phases"`.
  - `BatchRunner.run_slot(hypothesis, cap_s: float = 0.0)`.

- [ ] **Step 1: Write the failing tests**

Append to `src/deltaforge/batch_run_test.py`:

```python
def test_a_slot_that_overruns_its_cap_is_an_error_and_the_batch_continues():
    """A runaway slot must cost a slot, not the rental."""
    batch = _two_slot_batch()
    runner = _FakeRunner(behaviour={"001-slow": "hang"})
    budget = SlotBudget(deadline_epoch=1e12, clock=time.time, slot_cap_s=0.2, uncapped_slots=0)

    results, _calibrated, _scores = run_batch(runner, batch, budget=budget, log=lambda _: None)

    assert results[0].outcome == "error"
    assert "SlotTimeout" in results[0].error
    assert results[1].outcome != "not_run", "a capped slot must not end the batch"
```

`_FakeRunner` stands in for `BatchRunner` — give it a `prepare_reference()` no-op and a
`run_slot(hypothesis, cap_s)` that sleeps forever when `behaviour[slug] == "hang"`, wrapping
the sleep in `slot_deadline(cap_s)` and returning `SlotResult(hypothesis=..., outcome="error",
error=f"{type(exc).__name__}: {exc}")` on `SlotTimeout`. Match the fake to whatever
`batch_run_test.py` already uses for its runner if one exists.

Append to `src/deltaforge/harness/report_test.py`:

```python
def test_a_record_carries_its_phase_timings():
    """The phase split is the calibration rental's product: it is what lets
    docs/BATCHES.md replace estimates with measurements."""
    record = ResultRecord(
        kind="hypothesis",
        outcome="win",
        config_name="Qwen/Qwen3.5-4B",
        phases={"candidate_build": 3.5, "compile_candidate_compiled": 2400.0},
    )

    assert record.to_dict()["phases"]["compile_candidate_compiled"] == 2400.0
```

- [ ] **Step 2: Run to verify they fail**

Run: `uv run pytest src/deltaforge/batch_run_test.py src/deltaforge/harness/report_test.py -k "cap or phase" -v`
Expected: FAIL — `TypeError: __init__() got an unexpected keyword argument 'phases'`, and `run_slot()` rejecting `cap_s`.

- [ ] **Step 3: Implement**

`report.py` — add to both dataclasses, before `notes`:

```python
    #: Wall-clock seconds per phase: candidate build, correctness, one entry per column
    #: compiled, and the benchmark. The batch's cost arithmetic was an estimate that a
    #: rental contradicted; this is how it stops being one.
    phases: dict[str, float] = field(default_factory=dict)
```

and in each `to_dict()`, add `"phases": self.phases,` next to `"workload"`.

`batch_run.py` — add the field to `SlotResult`:

```python
    phases_s: dict[str, float] = field(default_factory=dict)
```

and `"phases_s": self.phases_s,` to `to_slot_dict()`. Import `field` from dataclasses if it is not already imported. Then in `run_slot`, take the cap, time each phase, and catch the timeout:

```python
    def run_slot(self, hypothesis: Hypothesis, cap_s: float = 0.0) -> SlotResult:
        """Measure one hypothesis. Never raises: a failure becomes an `error` outcome."""
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

                result = run_interleaved(
                    columns,
                    self.bench_config,
                    timer=CudaEventTimer(),
                    setups=setups,
                    metadata=self._slot_metadata(hypothesis),
                )
                phase("bench", mark)
```

The `except`/`finally` blocks stay as they are, with `phases_s=phases` added to both
`SlotResult(...)` constructions and `duration_s` unchanged. Note that `SlotTimeout` is an
`Exception`, so the existing `except Exception` records it as `error` with its traceback —
no new handler is needed.

`run_batch` — pass the cap, and distinguish a starved rental from an early stop:

```python
        result = runner.run_slot(hypothesis, cap_s=budget.cap_for(index))
```

and in the `break` branch that records the remaining slots, choose the outcome by whether
anything was actually measured:

```python
        if not budget.can_start():
            log(f"[batch] stopping before {hypothesis.slug}: {budget.why_not()}")
            # `starved` when the clock ended the rental with nothing scored, `not_run` when
            # it stopped early having already measured something. The distinction is the
            # evidence for raising the session gate, so it must not be flattened.
            scored = any(r.outcome in ("win", "loss", "inconclusive", "incorrect") for r in results)
            outcome = "not_run" if scored else "starved"
            for remaining in list(batch)[index:]:
                results.append(SlotResult(hypothesis=remaining, outcome=outcome))
            break
```

`cli.py` — carry phases into both records: add `phases=result.phases_s,` to the
`ResultRecord(...)` in `write_slot`, and to the `BatchRecord(...)` add:

```python
        phases={f"{r.hypothesis.slug}.{k}": v for r in results for k, v in r.phases_s.items()},
```

- [ ] **Step 4: Run to verify they pass**

Run: `uv run pytest src/deltaforge -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/deltaforge/batch_run.py src/deltaforge/batch_run_test.py src/deltaforge/cli.py src/deltaforge/harness/report.py src/deltaforge/harness/report_test.py
git commit -m "Time every phase of a slot, and enforce the cap on the slot loop"
```

---

### Task 5: Two scoring columns by default

**Files:**
- Modify: `src/deltaforge/cli.py` (`DEFAULT_COLUMNS`, line ~256)
- Test: `src/deltaforge/cli_test.py`

**Interfaces:**
- Consumes: nothing.
- Produces: `DEFAULT_COLUMNS == SCORING_COLUMNS`.

- [ ] **Step 1: Write the failing test**

Append to `src/deltaforge/cli_test.py`:

```python
def test_the_default_columns_are_the_two_that_score():
    """Rental 21 OOMed at 30.71 GiB of 31.36 during warmup with four columns.

    Construction was measured at 0.11 GiB on rental 22, so the memory went to the two
    eager diagnostic columns being resident through every warmup. They are diagnostics;
    `--columns all` still asks for them.
    """
    from .cli import DEFAULT_COLUMNS, SCORING_COLUMNS

    assert DEFAULT_COLUMNS == SCORING_COLUMNS
```

- [ ] **Step 2: Run to verify it fails**

Run: `uv run pytest src/deltaforge/cli_test.py -k default_columns -v`
Expected: FAIL — `('eager', 'compiled', 'candidate', 'candidate_compiled') != ('compiled', 'candidate_compiled')`.

- [ ] **Step 3: Implement**

Replace `DEFAULT_COLUMNS` in `src/deltaforge/cli.py` and rewrite its comment:

```python
#: What runs unless `--columns` says otherwise: the two that score, and nothing else.
#:
#: `eager` and `candidate` are diagnostics — the gap between them shows the compiler's
#: contribution — but every column is a live model state on the card with its own KV
#: cache, and all of them stay resident through warmup. Rental 21 OOMed there at 30.71
#: GiB of 31.36 while construction accounted for 0.11. `--columns all` asks for the full
#: set when a result is confusing enough to be worth the memory and the compile.
#:
#: `compiled_nocudagraphs` is in `all` and not here: it costs a full `max-autotune`
#: compilation, and the confound it existed to rule out disappeared when the scoring
#: column became `candidate_compiled`, since both sides now have CUDA graphs.
DEFAULT_COLUMNS = SCORING_COLUMNS
```

- [ ] **Step 4: Run to verify it passes**

Run: `uv run pytest src/deltaforge/cli_test.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/deltaforge/cli.py src/deltaforge/cli_test.py
git commit -m "Measure the two columns that score, and nothing else, by default"
```

---

### Task 6: A compile cache that outlives the instance

**Files:**
- Modify: `remote/sync.sh` (new `cache-up` / `cache-down` modes)
- Modify: `remote/run_remote.sh` (env vars on remote steps; cache push after sync-up; cache pull in teardown after the results pull)
- Modify: `.gitignore`
- Test: `remote/scripts_test.py`

**Interfaces:**
- Consumes: nothing.
- Produces:
  - `DF_CACHE_KEY` — `<gpu>-<torch>-<cuda>`, slugified, computed on the box and echoed as `[deltaforge] compile cache key: ...`.
  - `DF_REMOTE_CACHE=/workspace/df-cache`, `DF_LOCAL_CACHE=$DF_REPO_ROOT/cache/compile`.
  - `sh remote/sync.sh cache-down --host H --port P --remote-dir D --cache-key K`.

- [ ] **Step 1: Write the failing tests**

Append to `remote/scripts_test.py`:

```python
def test_the_remote_steps_point_torch_at_a_cache_that_is_pulled_home(workdir):
    """Torch's inductor and autotune caches are on by default and write to /tmp on a box
    we destroy, so every rental this project has run compiled cold."""
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
    from a dead box. Results first — a cold compile costs 40 minutes, a lost measurement
    costs the rental."""
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
    cache_block = teardown[teardown.index("pulling the compile cache") :]

    assert "timeout" in cache_block.split("destroying instance")[0]
```

- [ ] **Step 2: Run to verify they fail**

Run: `uv run pytest remote/scripts_test.py -k cache -v`
Expected: FAIL — the env vars and the teardown marker strings do not exist.

- [ ] **Step 3: Implement**

`.gitignore`, under "Local run state":

```
# Compile cache pulled off rented boxes: a build artefact, not a record
cache/
```

`remote/sync.sh` — add two modes beside `up` and `down`, following the existing rsync
invocation exactly (same `RSYNC_SSH`, same `df_dry` guard). `cache-up` sends
`$DF_LOCAL_CACHE/$DF_CACHE_KEY/` to `$DF_REMOTE_CACHE/`, `cache-down` brings
`$DF_REMOTE_CACHE/` back to `$DF_LOCAL_CACHE/$DF_CACHE_KEY/`. Both take `--cache-key`,
both no-op with a log line when the source does not exist.

`remote/run_remote.sh` — near the other defaults:

```sh
# Torch's fx-graph and autotune caches are on by default but write to /tmp on a box that
# gets destroyed, so every rental this project has run compiled cold -- and rental 22 spent
# ~40 minutes in one. Point them somewhere we can pull home, and key the local copy by GPU
# and toolchain so a 4090's cache is never handed to a 5090.
DF_REMOTE_CACHE="${DF_REMOTE_CACHE:-/workspace/df-cache}"
DF_LOCAL_CACHE="${DF_LOCAL_CACHE:-$DF_REPO_ROOT/cache/compile}"
DF_CACHE_PULL_TIMEOUT="${DF_CACHE_PULL_TIMEOUT:-120}"
DF_COMPILE_ENV="TORCHINDUCTOR_CACHE_DIR=$DF_REMOTE_CACHE/inductor TRITON_CACHE_DIR=$DF_REMOTE_CACHE/triton TORCHINDUCTOR_COMPILE_THREADS=\$(nproc)"
```

Prefix `$DF_COMPILE_ENV` onto the `deltaforge.cli batch`, `deltaforge.cli bench` and
`deltaforge.cli correctness` remote commands, ahead of the existing
`PYTORCH_CUDA_ALLOC_CONF=...`. Log `nproc` once at the start of the remote work so a
container reporting one core is visible rather than silent.

In `df_teardown`, directly after the results-pull block:

```sh
    if [ -n "$DF_INSTANCE_ID" ] && [ "${DF_SSH_READY:-0}" = "1" ] && [ "$DF_DRY_RUN" != "1" ]; then
        df_log "[teardown] pulling the compile cache"
        if timeout "$DF_CACHE_PULL_TIMEOUT" \
            sh "$DF_REPO_ROOT/remote/sync.sh" cache-down \
            --host "$DF_SSH_HOST" --port "${DF_SSH_PORT:-22}" \
            --remote-dir "$DF_REMOTE_CACHE" --cache-key "$DF_CACHE_KEY" 2>&1; then
            df_log "[teardown] compile cache pulled"
        else
            df_warn "[teardown] could not pull the compile cache (exit $?); destroying anyway"
        fi
    fi
```

- [ ] **Step 4: Run to verify they pass**

Run: `uv run pytest remote/scripts_test.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add remote/run_remote.sh remote/sync.sh remote/scripts_test.py .gitignore
git commit -m "Carry the compile cache home instead of destroying it with the box"
```

---

### Task 7: The pre-flight starvation gate

Refuse to rent when the session cannot fit one hypothesis. The box writes its measured
phase costs as shell-sourceable `KEY=VALUE` lines so the check needs no JSON parser and no
Python on the local side.

**Files:**
- Modify: `src/deltaforge/cli.py` (write `phases.env` at the end of `cmd_batch`)
- Modify: `src/deltaforge/batch.py` (the arithmetic, so it is unit-tested)
- Modify: `remote/run_remote.sh` (source the file, run the check before provisioning)
- Test: `src/deltaforge/batch_test.py`, `remote/scripts_test.py`

**Interfaces:**
- Consumes: `SlotResult.phases_s` (Task 4).
- Produces:
  - `batch.session_fits_one_hypothesis(remaining_s, phases, reserve_s) -> tuple[bool, float]` returning `(fits, shortfall_s)`.
  - `batch.COLD_PHASE_ESTIMATES: dict[str, float]` — the §4.1 worst case, in seconds.
  - `cache/compile/<key>/phases.env` with `DF_PHASE_SETUP_S`, `DF_PHASE_REFERENCE_COMPILE_S`, `DF_PHASE_SLOT_S`.
  - `run_remote.sh` exit code **5**, "REFUSED: the session gate cannot fit one hypothesis".

- [ ] **Step 1: Write the failing tests**

Append to `src/deltaforge/batch_test.py`:

```python
def test_a_session_with_room_for_one_hypothesis_fits():
    fits, shortfall = session_fits_one_hypothesis(
        remaining_s=7200.0,
        phases={"setup_s": 900.0, "reference_compile_s": 2400.0, "slot_s": 900.0},
        reserve_s=720.0,
    )

    assert fits is True
    assert shortfall == 0.0


def test_a_session_that_cannot_fit_one_hypothesis_says_how_short_it_is():
    """Nine rentals produced no number. Not renting beats renting to produce nothing, and
    the shortfall is what tells the next session how far to raise the gate."""
    fits, shortfall = session_fits_one_hypothesis(
        remaining_s=3600.0,
        phases={"setup_s": 900.0, "reference_compile_s": 2400.0, "slot_s": 900.0},
        reserve_s=720.0,
    )

    assert fits is False
    assert shortfall == pytest.approx(1620.0)


def test_the_cold_estimates_are_used_when_nothing_has_been_measured():
    """The first rental after this lands has no phases.env: it must fall back to the
    worst case in the spec, not to optimism."""
    fits, _shortfall = session_fits_one_hypothesis(remaining_s=10800.0, phases={}, reserve_s=720.0)

    assert fits is True
    assert set(COLD_PHASE_ESTIMATES) == {"setup_s", "reference_compile_s", "slot_s"}
```

Append to `remote/scripts_test.py`:

```python
def test_a_session_that_cannot_fit_one_hypothesis_is_refused_before_renting(workdir):
    ledger = write_ledger(
        workdir / "ledger" / "spend.jsonl",
        [
            ledger_line(session_id="tight", instance_id="a"),
            ledger_line(
                event="destroy",
                session_id="tight",
                instance_id="a",
                actual_minutes=170.0,
                actual_cost_usd=1.01,
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
```

- [ ] **Step 2: Run to verify they fail**

Run: `uv run pytest src/deltaforge/batch_test.py -k fits remote/scripts_test.py -k "fit one hypothesis" -v`
Expected: FAIL — `ImportError: cannot import name 'session_fits_one_hypothesis'`, and the shell exiting 0 rather than 5.

- [ ] **Step 3: Implement**

`batch.py` — add to `__all__` and define near `SlotBudget`:

```python
#: Worst-case cold-cache phase costs, in seconds, from the spec's §4.1 table. **These are
#: estimates**, used only until a rental has measured the real ones into `phases.env`.
#: `reference_compile_s` is the one number here that was observed: rental 22, ~40 minutes.
COLD_PHASE_ESTIMATES = {
    "setup_s": 1500.0,
    "reference_compile_s": 2400.0,
    "slot_s": 1980.0,
}


def session_fits_one_hypothesis(
    remaining_s: float,
    phases: dict[str, float] | None = None,
    reserve_s: float = 720.0,
) -> tuple[bool, float]:
    """``(fits, shortfall_s)`` for the least a rental may produce and still be worth it.

    That minimum is the identity champion plus one kernel slot: the identity slot alone
    calibrates the harness but scores no hypothesis. Nine rentals have been billed on this
    project without producing a number, so a session that cannot reach that minimum should
    not rent at all — and the shortfall says how far the gate is from being enough.
    """
    known = dict(COLD_PHASE_ESTIMATES)
    known.update({k: v for k, v in (phases or {}).items() if v > 0})
    needed = known["setup_s"] + known["reference_compile_s"] + 2 * known["slot_s"] + reserve_s
    shortfall = needed - remaining_s
    return (shortfall <= 0, max(0.0, shortfall))
```

`cli.py` — at the end of `cmd_batch`, after the summary is written, record what the rental
actually cost per phase:

```python
    if args.phases_env:
        slot_times = [r.duration_s for r in results if r.duration_s and r.outcome != "not_run"]
        measured = {
            "DF_PHASE_REFERENCE_COMPILE_S": sum(
                v for k, v in record.phases.items() if k.endswith(".compile_compiled")
            ),
            "DF_PHASE_SLOT_S": max(slot_times) if slot_times else 0.0,
        }
        Path(args.phases_env).parent.mkdir(parents=True, exist_ok=True)
        Path(args.phases_env).write_text(
            "".join(f"{k}={v:.1f}\n" for k, v in measured.items() if v > 0)
        )
        print(f"[batch] wrote {args.phases_env}")
```

Add the `--phases-env` argument to the `batch` subparser with `default=""`. `setup_s` is
timed by the shell, not by python, so `run_remote.sh` appends `DF_PHASE_SETUP_S` to the
same file after its own setup completes.

`run_remote.sh` — before provisioning, after the existing session gate:

```sh
DF_PHASES_FILE="$DF_LOCAL_CACHE/$DF_CACHE_KEY/phases.env"
DF_PHASE_SETUP_S=1500; DF_PHASE_REFERENCE_COMPILE_S=2400; DF_PHASE_SLOT_S=1980
# shellcheck disable=SC1090
[ -f "$DF_PHASES_FILE" ] && . "$DF_PHASES_FILE"
DF_NEEDED_MIN=$(awk -v s="$DF_PHASE_SETUP_S" -v c="$DF_PHASE_REFERENCE_COMPILE_S" \
    -v t="$DF_PHASE_SLOT_S" -v r="$DF_BATCH_RESERVE_MINUTES" \
    'BEGIN { printf "%.1f", (s + c + 2 * t) / 60 + r }')
DF_AVAILABLE_MIN=$(awk -v l="$DF_SESSION_LIMIT_MINUTES" -v u="$SESSION_MINUTES" \
    'BEGIN { printf "%.1f", l - u }')
if df_ge "$DF_NEEDED_MIN" "$DF_AVAILABLE_MIN"; then
    df_warn "REFUSED: the session gate cannot fit one hypothesis."
    df_warn "needs ${DF_NEEDED_MIN} minutes (identity slot + one kernel slot + reserve),"
    df_warn "has ${DF_AVAILABLE_MIN}. Raise --session-limit or start a new session."
    exit 5
fi
df_log "pre-flight: ${DF_AVAILABLE_MIN} minutes available, one hypothesis needs ${DF_NEEDED_MIN}"
```

Document exit code 5 in `usage()` beside the existing codes.

- [ ] **Step 4: Run to verify they pass**

Run: `uv run pytest src/deltaforge/batch_test.py remote/scripts_test.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/deltaforge/batch.py src/deltaforge/batch_test.py src/deltaforge/cli.py remote/run_remote.sh remote/scripts_test.py
git commit -m "Refuse to rent a session that cannot finish one hypothesis"
```

---

### Task 8: The gate moves to three hours

**Files:**
- Modify: `remote/run_remote.sh` (`DF_SESSION_LIMIT_MINUTES`, `DF_WATCHDOG_MINUTES`, `DF_BATCH_TIMEOUT`, usage text)
- Modify: `remote/watchdog.sh`, `remote/provision.sh` (defaults and help)
- Modify: `src/deltaforge/ledger.py` (`SESSION_MINUTES_LIMIT`)
- Test: `remote/scripts_test.py`, `src/deltaforge/ledger_test.py`

**Interfaces:**
- Consumes: nothing.
- Produces: gate 180, watchdog 210, `DF_BATCH_TIMEOUT` 9600, `provision.sh --max-minutes` 210.

- [ ] **Step 1: Update the tests that pin the numbers**

In `remote/scripts_test.py`, rename `test_the_session_gate_defaults_to_two_hours` to
`test_the_session_gate_defaults_to_three_hours`, change its ledger row to
`actual_minutes=180.0, actual_cost_usd=1.068`, and in
`test_the_hard_watchdog_stays_above_the_session_gate` change `assert gate == 120.0` to
`assert gate == 180.0`. In `test_the_provision_row_is_written_before_the_instance_is_used`
change the ceiling to `pytest.approx(1.134)` (210 minutes at $0.324/hr) and update its
comment. In `src/deltaforge/ledger_test.py`, change the two gate tests from 120/119 to
180/179 and the `SESSION_MINUTES_LIMIT` assertion to `180.0`.

- [ ] **Step 2: Run to verify they fail**

Run: `uv run pytest remote/scripts_test.py -k "gate or watchdog or provision_row" src/deltaforge/ledger_test.py -k gate -v`
Expected: FAIL on the old values.

- [ ] **Step 3: Implement**

`run_remote.sh`:

```sh
# 180 rather than 120: the cold-cache arithmetic in
# docs/superpowers/specs/2026-09-10-compile-cost-and-memory-design.md §4.1 puts one
# hypothesis -- the identity slot plus one kernel slot -- at 83-129 minutes, and 120 minus
# the reserve left 108. The gate is a ceiling, not a spend commitment: vast bills by the
# minute and the run destroys itself when the batch ends, so a warm session still pays for
# the ~40 minutes it uses. The month-to-date $45 gate is the real budget control.
DF_SESSION_LIMIT_MINUTES="${DF_SESSION_LIMIT_MINUTES:-180}"
DF_WATCHDOG_MINUTES="${DF_WATCHDOG_MINUTES:-210}"
```

`DF_BATCH_TIMEOUT` becomes `9600` (160 minutes), keeping it above the largest deadline a
180-minute gate can hand a batch (~155 minutes after setup and reserve); update its comment
to say 180/155. `watchdog.sh`: default `210`, and its two "120-minute session gate" strings
become "180-minute". `provision.sh`: `DF_MAX_MINUTES` default `210` and its help line.

- [ ] **Step 4: Run to verify they pass**

Run: `uv run pytest remote -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add remote/run_remote.sh remote/watchdog.sh remote/provision.sh remote/scripts_test.py src/deltaforge/ledger.py src/deltaforge/ledger_test.py
git commit -m "Give a session three hours, because one hypothesis does not fit in two"
```

---

### Task 9: Batch 002, the calibration exemption, and the docs

**Files:**
- Modify: `src/deltaforge/batches.py` (add `BATCH_002`, register it in `get_batch`)
- Modify: `src/deltaforge/batch.py` (`Batch.is_calibration`)
- Modify: `src/deltaforge/batches_test.py`
- Modify: `AGENT.md`, `README.md`, `docs/BATCHES.md`
- Test: `src/deltaforge/batches_test.py`

**Interfaces:**
- Consumes: everything above.
- Produces: `BATCH_002` (`"002-compile-cost"`), `Batch.is_calibration: bool` field defaulting `False`.

- [ ] **Step 1: Write the failing tests**

In `src/deltaforge/batches_test.py`, replace the size test and add the manifest tests:

```python
def test_a_batch_holds_seven_to_twelve_hypotheses_unless_it_is_calibrating():
    """The floor amortises fixed cost across measurements. A calibration batch's product
    IS the cost measurement, so the floor does not apply to it."""
    assert 7 <= len(BATCH_001) <= 12
    assert BATCH_001.is_calibration is False

    assert len(BATCH_002) < 7
    assert BATCH_002.is_calibration is True


def test_batch_002_opens_with_the_identity_champion():
    assert BATCH_002.hypotheses[0].is_identity
    assert BATCH_002.calibration_slug == "000-identity"


def test_batch_002_can_score_a_kernel_hypothesis():
    """Calibration is necessary but is not a result: the rental must also score a real
    kernel, or it proves only that the harness works."""
    assert any(not h.is_identity for h in BATCH_002)
```

- [ ] **Step 2: Run to verify they fail**

Run: `uv run pytest src/deltaforge/batches_test.py -v`
Expected: FAIL — `ImportError: cannot import name 'BATCH_002'`.

- [ ] **Step 3: Implement**

`batch.py` — add to `Batch`, after `description`:

```python
    #: A batch whose product is the cost measurement itself, not a ranked set of
    #: hypotheses. It is exempt from the 7-12 floor in docs/BATCHES.md, which exists to
    #: amortise fixed cost across measurements.
    is_calibration: bool = False
```

`batches.py` — add `BATCH_002` with the identity champion first, then the **two kernels
with the smallest graph change** among those already written and gated on CPU (read
`REGISTRY` and pick accordingly; do not pick by byte share — this rental is measuring cost,
not ranking ceilings). Each keeps a real `mechanism`, `prediction` and `rationale`, and the
batch's `description` states that its purpose is the phase timings. Register it in
`get_batch` beside `BATCH_001`.

Docs, per `AGENT.md` §6.1:
- `docs/BATCHES.md`: the cost table gains the cold-cache split from spec §4.1 marked as
  estimates; the "7 is the floor" rule gains the calibration exemption; "After the rental"
  gains `phases.env`.
- `AGENT.md` §5: the gate is 180, the watchdog 210, and a session that cannot fit one
  hypothesis is refused before renting (exit 5).
- `README.md` cost control: the same three numbers, plus one line on the compile cache.

- [ ] **Step 4: Run the whole suite and the linters**

Run: `uv run pytest -q && uv run ruff check . && uv run ruff format --check .`
Expected: PASS, with only the usual `gpu`/`weights` skips.

- [ ] **Step 5: Dry-run the money machinery, which spends nothing**

Run: `remote/run_remote.sh --dry-run --session-id smoke --batch 002-compile-cost`
Expected: exit 0; the log shows the pre-flight line, the compile-cache env vars, and the
teardown ordering results → cache → destroy.

- [ ] **Step 6: Commit**

```bash
git add src/deltaforge/batch.py src/deltaforge/batches.py src/deltaforge/batches_test.py AGENT.md README.md docs/BATCHES.md
git commit -m "Add the calibration batch whose product is the cost of a slot"
```

---

## Self-review

**Spec coverage:** §3.1 → Task 6; §3.2 → Task 6; §3.3 → Task 3; §3.4 → Task 5; §3.5 → Tasks 1, 2, 4; §3.6 → Task 4; §4.1 → Task 8; §4.2 → Tasks 1 (`starved`, exemption), 4 (`phases_s`), 7 (`phases.env`, pre-flight); §5 → Task 9; §6 → tests inside each task; §7 → Task 9.

**Gap found and closed:** §4.2 says slots record `starved` when the clock ends the batch, but `run_batch` writes `not_run` for every remaining slot. Task 1 adds the outcome to `BATCH_OUTCOMES` and `score_predictions`; Task 4's `run_batch` change now carries the branch that picks between `starved` and `not_run` by whether any slot scored.

**Type consistency:** `cap_for(index) -> float` (Task 1) feeds `run_slot(hypothesis, cap_s)` (Task 4) and `slot_deadline(seconds)` (Task 2) — all seconds, all floats. `phases_s` is the `SlotResult` field; `phases` is the record field; the two names differ on purpose and Task 4 maps between them.
