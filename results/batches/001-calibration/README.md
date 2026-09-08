# Batch 001 — calibration. Void again, and further along.

**No hypothesis in this batch has been measured.** Two rentals on 2026-09-08 (21 and 22)
reached the batch loop; neither produced a ratio. The identity champion never returned a
number, so the batch is **void by its own rule** — `calibrated: false`,
`counts: {"error": 9}`, `prediction_record: {"correct": 0, "scored": 0}`.

Not one of the nine predictions was scored. A slot that errored never tested its
prediction, and counting those as wrong would understate the record exactly as counting
them right would flatter it.

The JSON records here are **rental 21's**. Rental 22 completed no slot, so it overwrote
nothing.

---

## What is now settled

**Blocker 6 is fixed and proven.** Rental 16 lost all nine slots inside
`BatchRunner._build_candidate` to a double allocation of the weights. On rental 21 every
slot built its candidate and cleared the correctness gates. Construction is no longer where
this dies, and the meta-device fix in `1d5f2a9` needs no further testing.

**Blocker 7 was found and fixed** — the readiness poll waited on the rental contract rather
than the container. See `docs/GPU-ACCESS.md`; it had been costing rentals since 11 and
masquerading as flaky hosts.

**A kernel bug that had never run.** `007-gated-delta-fused-step` errored in 0s with
`ImportError: cannot import name 'STATE_DTYPE' from 'deltaforge.config'` — it lives in
`reference`. Kernels import from the package inside function bodies to keep Triton off the
CPU import path, so nothing checked that name until the function ran on a rented GPU.
`registry_test.py` now resolves every deferred relative import statically, and reverting
the fix reproduces the rental's exact message on CPU.

## Where it dies now: the benchmark, not the build

Rental 21 lost eight slots to a CUDA OOM at **30.71 GiB of 31.36**, every one of them at
the *benchmarking* step, 16-18s in — past construction, past correctness. The batch
recorded peak memory only for slots that **succeeded**, which is precisely the empty set
when memory is the problem, so nine identical failures said 30.71 GiB and nothing about
where it went. That instrumentation now runs on the error path too.

Rental 22 ran with only the two scoring columns and reported the breakdown:

| After | Allocated |
|---|---:|
| loading weights | 7.83 GiB |
| reference column `compiled` | 7.95 GiB |
| candidate build | 7.96 GiB |
| candidate column `candidate_compiled` | 8.07 GiB |

**Construction is nearly free.** Weights are 7.83 GiB and the candidate shares them, adding
0.11 GiB. Four columns cannot cost 30 GiB at construction either — so the ~22 GiB in rental
21 was accumulated *during warmup*, by the two eager columns that the default includes and
the scoring pair does not need. `run_remote.sh --columns` now exists to say so; batch mode
had always accepted the flag and nothing passed it.

**This is a hypothesis, not a result.** Rental 22 was stopped by the session gate before
warmup completed, so the two-column configuration has never been observed to survive the
benchmark. It is the next thing to test, and it is cheap to test.

## The wall in front of the next session

Rental 22 spent **~40 minutes inside slot 0 without finishing it.** `nvidia-smi` showed the
GPU at 0% with python at 129% CPU and an inductor worker alongside: a cold
`max-autotune` compile, not a hang.

That breaks the arithmetic the whole batch design rests on. `AGENT.md` and
`docs/BATCHES.md` cost a rental at ~15 minutes fixed plus 2-4 per slot, which is what makes
7-12 hypotheses per rental worth doing. A 40-minute first compile means a nine-slot batch
does not fit a 90-minute session at all, and the session gate — not the science — decided
when this run ended.

**Do not fill another batch until that number is known.** Time one `max-autotune`
compilation, then either bring it down (`mode="reduce-overhead"`, a warm inductor cache
carried between rentals, `torch.compiler` caching to disk) or re-cut the batch size around
what a compile actually costs. Either is a better use of the next rental than nine more
slots that will not run.

## Cost

Five rentals, 91.39 billed minutes, $0.567 — session `batch001-20260908T211825Z`, which
ended at its 90-minute gate. Zero leaked instances. Three of the five died to blocker 7
before it was understood.
