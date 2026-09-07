# Batched hypotheses: many measurements per rental

**Status:** approved 2026-09-06. Supersedes the one-hypothesis-per-rental workflow in
`AGENT.md` §4.

---

## 1. The problem, stated as arithmetic

A rental's cost splits into a fixed part and a marginal part.

| | Cost | Paid |
|---|---:|---|
| Container image pull | ~3 min | once per rental |
| torch + CUDA libs from the PyTorch CDN | ~1-3 min | once per rental |
| Checkpoint, 9.32 GB | ~2-5 min | once per rental |
| `transformers`, tokenizers, pytest | <1 min | once per rental |
| GPU test suite + weight-value oracle | ~2 min | once per rental |
| **Fixed subtotal** | **~10-15 min** | |
| Reference `max-autotune` compile | ~3-4 min | once per rental *(new)* |
| Candidate compile + correctness gate + bench | ~2-4 min | **per hypothesis** |

Under the old workflow a session pays 15 minutes of fixed cost to buy 3 minutes of
science, and then destroys the machine. Nine rentals have been billed on this project
and none produced a number, so the fixed cost has been paid nine times for nothing.

Batching changes the denominator. The same 15 minutes buys 7-12 measurements instead of
one, and the per-hypothesis cost falls to roughly a tenth of a rental.

## 2. The consequence that matters more than the money

`docs/HYPOTHESES.md` graveyards five hypotheses as "unmeasurable": their ceiling is below
the harness's declared noise band, so renting a GPU to measure them was judged not worth
it.

**That argument is about cost, not about truth.** It was correct under a workflow where
measuring one of them consumed a whole rental. At 3 minutes a slot it is no longer
correct — and a measured null, carrying a real ratio and a real IQR from a real card, is
a materially stronger record than an arithmetic prediction of a null.

The two claims are also not the same claim. "The ceiling is 0.018%, so no win is
possible" is a statement about bytes. What a measurement returns is the ratio the harness
actually observes, which also contains launch overhead, CUDA-graph behaviour, and
whatever inductor did. Those can differ from the byte arithmetic, and the difference is
exactly the kind of finding this project exists to produce.

So the first batch goes and collects those numbers. The graveyard entries stay, but they
get amended from *predicted* null to *measured* null — or, if any of them surprises us,
to something better.

## 3. Design

### 3.1 A hypothesis becomes data

Today the candidate model is whatever `kernels.REGISTRY.champions()` returns. That is a
process-wide singleton with a hard "at most one champion per operation" invariant, so a
single process can express exactly one candidate. Batch mode needs N.

`src/deltaforge/batch.py` introduces:

```python
@dataclass(frozen=True)
class Hypothesis:
    slug: str                      # "006-gqa-no-expand"
    kernels: tuple[str, ...]       # registry names to install; () is the identity champion
    category: str                  # "A" | "B" | "C" | "calibration"
    byte_share: float              # share of per-token bytes, from docs/roofline.py
    mechanism: str                 # one sentence: how it wins
    prediction: str                # "win" | "loss" | "inconclusive" | "identity"
    rationale: str                 # why that prediction, recorded BEFORE the run
```

`scoped_registry(hypothesis)` returns a fresh `KernelRegistry` holding exactly those
kernels at `CHAMPION` status. The global `REGISTRY` remains the catalogue of what exists;
it stops being the definition of what is under test. The champion invariant still holds —
per scoped registry, which is where it means something.

A `Batch` is an ordered tuple of hypotheses plus an id. Batches live in
`src/deltaforge/batches.py` as Python data: typed, importable, unit-testable, no parser.

### 3.2 The `prediction` field is the point

`AGENT.md` §1 says the finding is a correct mechanistic account registered *before* the
measurement, and that being right in advance is the result. Nothing in the repo currently
records a prediction anywhere it can be scored against an outcome.

`prediction` and `rationale` are set in the manifest, committed before the rental, and the
batch summary scores each one against what was measured. A batch of nine registered
predictions checked against nine measurements is the first evidence this project can
offer for its actual claim — and it is worth more than any single ratio in the table.

### 3.3 Execution: reference compiled once, candidates one at a time

One process, `deltaforge.cli batch`:

1. Load the reference model once. Parameters are shared with every candidate, as today.
2. Build and compile the `eager` and `compiled` reference columns **once**. They stay
   resident for the whole batch.
3. For each hypothesis in manifest order:
   1. Build a candidate module tree sharing the reference's parameter tensors.
   2. Install its scoped kernels. `apply_champions` already refuses a champion with no
      installer; batch mode additionally asserts that a non-identity hypothesis actually
      patched something, because a candidate that silently equals the reference would be
      recorded as a null result rather than as the bug it is.
   3. Run the correctness gates.
   4. Run the interleaved benchmark.
   5. Write `results/batches/<batch_id>/<slug>.json` immediately.
   6. Free the candidate, `gc.collect()`, `torch.cuda.empty_cache()`.
4. Write `results/batches/<batch_id>/summary.json` and a markdown scorecard.

**Only the compilation is amortized, not the measurement.** The reference `compiled`
column is re-timed inside every hypothesis's own interleaved rounds, so per-round thermal
and clock drift still divides out exactly as it does in the single-hypothesis harness.
Reusing a *timing* across hypotheses would break that and is not done. This distinction
is recorded in every result record, because a sceptical reader will ask.

### 3.4 Failure isolation is the load-bearing property

Each hypothesis runs inside its own try/except. An exception becomes `outcome: "error"`
carrying the exception type, message and traceback, and the batch continues to the next
slot.

This is what makes the whole design safe. Six of the nine hypotheses in batch 1 are
Triton kernels written without a GPU to test them on; some will be wrong. Under the old
workflow a wrong kernel cost the rental. Here it costs a slot, and the error record is
itself a result worth reading.

Every result JSON is written the moment its hypothesis finishes, so a hard crash at slot 8
still leaves slots 0-7 on disk and recoverable.

### 3.5 Deadline awareness

The batch takes `--deadline-epoch`, derived by `run_remote.sh` from the session gate.
Before starting each hypothesis it compares the remaining time against a rolling estimate
— the median duration of the slots already completed, seeded at 240 s — and if the
estimate does not fit, it stops cleanly and records the remaining hypotheses as
`not_run`.

`AGENT.md` §5 calls a watchdog firing a reportable fault. This is what keeps it from
happening: the batch ends itself with its results written, rather than being killed with
work in flight.

### 3.6 Budget

Session gate 60 → **90 minutes**; watchdog 90 → **120 minutes**. At the $0.356/hr RTX 5090
the last rentals used, a full 90-minute batch costs about $0.53.

### 3.7 `run_remote.sh` changes

* `--batch NNN` selects a batch. `--hypothesis SLUG` still works and is a batch of one.
* The deadline is computed from the session gate and passed to the batch command.
* **Teardown pulls results before destroying.** The trap currently destroys the instance
  and then cannot rsync from a dead box. That is survivable when a run is 10 minutes and
  one hypothesis; it is not survivable when it is 90 minutes and nine. The teardown now
  attempts one best-effort `sync.sh down` before the destroy call, guarded so a failure
  to pull can never prevent the destroy.

### 3.8 Correctness in batch mode

The full layer-1 per-kernel checks run for every hypothesis, unchanged. The layer-2
end-to-end exact-token gate runs at **32** new tokens rather than 128, and the token count
is recorded in each result. 128 is reserved for re-checking a candidate that is about to
be promoted to champion. This is the one place the design trades rigor for slots, and it
is stated in the record rather than left for a reader to discover.

## 4. Batch 001 — `001-calibration`

Ordered cheapest and most diagnostic first, riskiest last.

| # | slug | replaces | byte share | prediction |
|---|---|---|---:|---|
| 0 | `000-identity` | — | 0% | **identity**: every column 1.00 ± noise |
| 1 | `001-fused-rmsnorm-residual` | `rms_norm_residual` | 0.018% | inconclusive |
| 2 | `002-rmsnorm-only` | `rms_norm` | 0.018% | inconclusive |
| 3 | `003-qk-norm-triton` | `rms_norm` | ~0.001% | inconclusive |
| 4 | `004-fused-swiglu` | `swiglu_mlp` | 0.026% | inconclusive |
| 5 | `005-fused-qkv-rope` | `qkv_projection_rope` | 0.004% | inconclusive |
| 6 | `006-gqa-no-expand` | `gqa_attention` | 6.23% | **win, conditional** |
| 7 | `007-gated-delta-fused-step` | `gated_delta_rule` | 1.10% | inconclusive |
| 8 | `008-flash-decode-splitkv` | `gqa_attention` | 0.78% | inconclusive at ctx 2048 |

Slot 0 is the calibration gate. If the identity champion does not return 1.00 ± noise on
every column, the harness is measuring something other than what it claims and **every
other number in the batch is void** — the writeup must say so rather than reporting the
rest as findings.

Slot 6 is the only predicted win, and the prediction is conditional on a fact nobody has
checked: whether inductor materialises the `repeat_interleave` at `reference.py:579` or
folds the index arithmetic into the consumer. If it folds it, `compiled` already has this
win and there is nothing to take — and that is a genuinely interesting fact about
inductor, recorded as the finding.

**Deliberately excluded from batch 1:**

* *Merged gate/up GEMM.* Needs its own concatenated weight tensor, which breaks the
  shared-parameter invariant `_assert_parameters_are_shared` enforces, and adds ~4.5 GB
  resident. Batch 2, with an explicit weight-owning hypothesis flag.
* *int8/int4 weight-only dequant GEMV.* The 91.85% hypothesis and the only one with a
  ceiling worth anything, but a quantised candidate is not bit-comparable to a bf16
  reference, so it fails the layer-2 exact-token gate by construction and needs a
  different correctness claim (KL or perplexity against bf16). Designing that claim is
  its own piece of work and must not ride along on the batch that is calibrating the
  harness.

## 5. What this design does not do

It does not make a batch atomic. Nine hypotheses on one card share one set of thermal
conditions and one driver state; a card that throttles in slot 3 affects slots 3-8. The
interleaving inside each hypothesis divides out drift *within* a slot but not *between*
slots, so **ratios remain comparable within a hypothesis and only loosely comparable
across them.** Cross-slot comparisons in the writeup must say that.

It does not carry numbers across batches. Every batch rents different physical hardware
and the reigning champion is re-benchmarked in every batch that means to compare against
it, exactly as `AGENT.md` §6 already requires.
