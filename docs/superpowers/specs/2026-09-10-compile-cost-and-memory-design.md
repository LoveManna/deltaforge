# Spec: surviving the compile, and the memory it leaves behind

**Status:** design, not yet implemented.
**Problem:** two rentals reached the batch loop and neither produced a ratio. Rental 21
lost every slot to a CUDA OOM during warmup; rental 22 spent ~40 minutes inside slot 0's
cold `max-autotune` compile and was ended by the session gate.

This spec fixes both, and adds the thing neither fix is worth without: a **guarantee that a
session can finish at least one hypothesis**, enforced before the money is spent rather
than discovered after it.

---

## 1. What the evidence actually says

Two failures with different causes, both visible in `results/batches/001-calibration/`.

**Time.** `BatchRunner._make_column(candidate, "max-autotune")` compiles *per slot*. Only
the reference compile is amortised; every candidate is a fresh compile of the same 32-layer
graph. Rental 22 observed `nvidia-smi` at 0% GPU with python at 129% CPU and one inductor
worker — autotuning, single-threaded, against a cache that had never been warmed and that
`destroy` would throw away. Torch enables `fx_graph_cache` and `autotune_local_cache` by
default, but they write to `/tmp/torchinductor_<user>`, which dies with the instance.
**Every rental this project has ever run compiled cold, and nothing in `remote/` carries
that directory home.**

**Memory.** Each slot builds `candidate_compiled`, which under `max-autotune` records its
own CUDA-graph pool. The slot's `finally` does `del candidate; gc.collect();
empty_cache()` — which frees the module and its KV cache but **not the graph pool**, held
by inductor's graph trees. Slot *N* is therefore resident on *N* pools. That matches
rental 21's profile: construction accounted for ~8 GiB (rental 22 measured it), and the
remaining ~22 GiB accumulated during warmup across slots.

## 2. What this does not change

The win condition. `torch.compile(mode="max-autotune")` is what this project claims to
beat, so it stays on both sides of the comparison, with identical treatment. A cheaper
compile mode would compile faster and rewrite the claim; asymmetric treatment between the
two columns would invalidate the score outright. Neither is on the table.

A warm compile cache changes **how long compilation takes, not what it emits**, and both
columns share one cache, so the comparison stays symmetric. Every result records
`compile_cache: {warm: bool, key: ...}` so a reader can see which it was rather than
take it on trust.

## 3. The six mechanisms

### 3.1 A compile cache that outlives the instance

Every remote python step gets `TORCHINDUCTOR_CACHE_DIR=/workspace/df-cache/inductor` and
`TRITON_CACHE_DIR=/workspace/df-cache/triton`. Teardown pulls that directory **after the
results pull and before destroy**, into a gitignored local
`cache/compile/<gpu>-<torch>-<cuda>/`; provision pushes the matching one up when it exists.

`cache/` joins `.gitignore` — it is a build artefact of a rented box, not a record.
The directory name is the compatibility key: a 4090's cache is never handed to a 5090.
The pull is bounded exactly as the results pull is — its own timeout, its own size ceiling,
best-effort, and **never able to block the destroy**. A leaked instance costs about
$13/day; a cold compile costs 40 minutes.

### 3.2 Compile workers

Export `TORCHINDUCTOR_COMPILE_THREADS=$(nproc)`, and log both it and `nproc` at batch
start. Rental 22 showed *one* inductor worker. If a container reports one core, that alone
explains a 40-minute compile — and nothing currently says so out loud.

### 3.3 Release the CUDA-graph pools between slots

In `run_slot`'s `finally`, after `del candidate`: `reset_cudagraph_trees()` (guarded by
`hasattr`, it is private API), then `gc.collect()`, `empty_cache()`, and log memory.

Deliberately **not** `torch._dynamo.reset()`, which would discard the reference's
compilation — the batch's entire saving. The reference re-records its graphs on the next
slot's first warmup call: seconds, not a recompile.

### 3.4 Two columns by default

`DEFAULT_COLUMNS` becomes `SCORING_COLUMNS`. `eager` and `candidate` become opt-in under
`--columns all`. They are diagnostics, they stayed resident through every warmup, and the
OOM happened during warmup. `run_remote.sh --columns` already exists; nothing passed it.

### 3.5 A per-slot wall-clock cap

`SlotBudget` decides whether to *start* a slot, so a slot that starts and then runs for 40
minutes takes the rental. A timer thread armed at slot start calls `_thread.interrupt_main()`
at the cap; `run_slot` catches `KeyboardInterrupt` alongside `Exception` and records
`error` with `slot_timeout`.

This makes "a failing slot costs a slot, not the rental" true for a **slow** slot and not
only a crashing one. **Slots 0 and 1 are exempt** — see §4.

### 3.6 Phase timings in the record

Time each phase — candidate build, correctness, per-column compile+warmup, bench — and
record memory after each. This is the calibration rental's actual product: the numbers that
let `docs/BATCHES.md` replace its estimates with measurements.

## 4. The guarantee: one hypothesis per session, or do not rent

### 4.1 The arithmetic that forces a longer gate

Worst-case cold-cache budget for one *scored kernel* hypothesis — the identity slot plus
one kernel slot, two columns:

| Phase | Cold worst case | Basis |
|---|---:|---|
| provision → sshd → container up | 5-10 min | rentals 18-22, **observed** |
| torch + sync + 9.32 GB checkpoint + GPU suite | 10-15 min | estimate, **never measured** |
| reference `compiled` compile | ~40 min | rental 22, **observed** |
| slot 0 identity: candidate compile (same graph, in-process cache hits) | 5-15 min | **inferred** |
| slot 0 correctness + bench | 4-8 min | estimate |
| slot 1 kernel: candidate compile (one module class differs) | 10-25 min | **inferred** |
| slot 1 correctness + bench | 4-8 min | estimate |
| teardown: results pull, cache pull, destroy | 5-8 min | the cache pull is new |
| **Total** | **83-129 min** | |

A 120-minute gate minus the 12-minute reserve leaves **108 usable minutes**. The upper half
of that range does not fit — and the upper half is exactly the cold-cache case, which is
what the next rental will be, because the cache starts empty.

**So: session gate 120 → 180 minutes, watchdog 150 → 210.**

This costs nothing when unused. Vast bills by the minute and the run destroys as soon as
the batch finishes, so a warm session still ends in ~40 minutes and pays for ~40. **The gate
is a ceiling, not a spend commitment**; the real budget control is the $45 month-to-date
gate, which is untouched. Three hours at $0.356/hr is $1.07.

### 4.2 The guarantee is a mechanism, not a number

Every number in §4.1 will drift, so nothing below depends on them being right.

* **Measured timings are carried forward.** The persisted cache directory also holds
  `phases.json` — last rental's measured phase timings for that GPU/torch key. Budget
  arithmetic uses those when they exist and the cold worst case only when they do not.
* **Pre-flight starvation check.** Before provisioning, `run_remote.sh` compares the
  session's remaining budget against `setup + reference_compile + slot0 + slot1` and
  **refuses to rent** with its own exit code when one hypothesis cannot fit, saying how
  short the gate is. Not renting beats renting to produce nothing, which is what nine
  rentals did.
* **`starved` is a recorded outcome.** If the budget runs out before any scoring slot
  completes, the remaining slots record `starved` rather than `not_run`, and the summary
  says the clock ended the session. That is the evidence a future session needs in order to
  raise the gate again instead of re-deriving this table.
* **Slots 0 and 1 are never capped below the remaining budget.** The §3.5 cap exists to
  stop slot 7 eating slot 8. Applying it to the first two slots would defeat the guarantee
  it is there to protect.

## 5. Batch 002: the calibration rental

`002-compile-cost`: the identity champion plus **two kernels chosen for cheap compilation
rather than for their byte share** — the smallest graph change among those already written
and gated on CPU — two columns, on a cold cache. Ranking by ceiling resumes once the cost
of a slot is known; this rental is measuring the cost, so it buys the cheapest slots that
still exercise a real kernel.

This deliberately breaks the "7 is the floor" rule in `docs/BATCHES.md`. That floor exists
to amortise fixed cost across measurements; here the cost measurement *is* the product.
The rule gains an explicit exemption for a batch that declares itself a calibration batch,
rather than being left for the next session to find violated.

Success for this rental is: the identity slot returns 1.00 ± noise (the project's first
calibrated harness), one kernel hypothesis is scored, and `phases.json` comes home with
real compile numbers.

## 6. Testing — all on CPU, before any money

| Test | Asserts |
|---|---|
| `scripts_test.py` | cache env vars reach the remote command line; the cache pull is ordered after the results pull and before destroy, and cannot block it; the pre-flight check refuses with its exit code; the watchdog default stays above the gate |
| `batch_test.py` | budget arithmetic with and without `phases.json`; `starved` distinguished from `not_run`; slots 0-1 exempt from the per-slot cap |
| `batch_run_test.py` | a fake slow slot hits the cap, records `error`, and the batch continues; the cudagraph reset is skipped without raising when the symbol is absent |
| `batches_test.py` | the 7-12 rule holds unless a batch declares itself a calibration batch |

Anything needing a GPU keeps the `gpu` marker and skips on a real condition.

## 7. Docs this changes

Per `AGENT.md` §6.1: `AGENT.md` §4 and §5 (gate, watchdog, the guarantee), `README.md`
cost control, `docs/BATCHES.md` (cost table, the calibration exemption, and the "After the
rental" step gaining `phases.json`).

## 8. What remains unproven after this lands

Stated plainly, because the failure mode this project guards against is a doc that quietly
carries a wrong number:

* **Every "inferred" row in §4.1.** The candidate compile costs are reasoning about cache
  behaviour, not measurements. Batch 002 exists to replace them.
* **That the graph-pool release is the OOM.** It fits the evidence and it is the only
  allocation in the loop that survives `empty_cache()`, but rental 21's memory was never
  broken down per slot. If the OOM survives §3.3 and §3.4, the next suspect is warmup
  itself, not construction — rental 22 measured construction at 0.11 GiB.
* **That a warm cache actually helps across rentals.** Inductor keys entries by hardware
  and shapes; a rental that lands on a different GPU model gets no benefit by design. The
  first rental to reuse a cache is the one that proves it.
