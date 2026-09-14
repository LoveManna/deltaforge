# Batch 001 — calibration. The harness is calibrated; the batch is not yet measured.

> **Current state (2026-09-14, rental 35).** All nine slots ran to completion for the
> first time. `calibrated: true` — `000-identity` measured **1.0018** against an IQR of
> **0.0018**, correctness exact. Six of eight kernels failed the correctness gate, which is
> a real result. **No hypothesis has an admissible ratio**, because dynamo hit
> `recompile_limit` inside slot 2 and slots 3-8 timed an eager candidate. Full account in
> "Rentals 33-35" below; everything above that heading is older and kept for the record.

---

**No hypothesis in this batch had been measured as of 2026-09-08.** Two rentals (21 and 22)
reached the batch loop; neither produced a ratio. The identity champion never returned a
number, so the batch was **void by its own rule** — `calibrated: false`,
`counts: {"error": 9}`, `prediction_record: {"correct": 0, "scored": 0}`.

Not one of the nine predictions was scored. A slot that errored never tested its
prediction, and counting those as wrong would understate the record exactly as counting
them right would flatter it.

The JSON records here were **rental 21's**. Rental 22 completed no slot, so it overwrote
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

## Rental 28, 2026-09-12 — the transformers hypothesis is dead

**No slot ran. The batch loop was not reached.** The GPU suite refused it, which is what
the suite is for. Session `batch001-20260912T021420Z`: instance 50682531, RTX 4090, machine
5629, $0.4030/hr, **7.12 billed minutes, $0.0478**, destroyed cleanly, nothing leaked.

**Two things worked for the first time.**

*The authenticated Docker Hub pull.* Credentials went into `.env` and the create request
carried `image_login`. The default `vastai/base-image` pulled and sshd answered in about a
minute. Eight rentals never got one layer to "Pull complete" anonymously and rental 23 hit
the same wall on 2026-09-10, so blocker 1 is now fixed *and proven*, not fixed and hoped.

*The offer walk.* No RTX 5090 met the filters, the search fell through to the 4090, found
one candidate, and created on the first try. Blocker 10's fix was exercised on a real
market; it was not stressed, because nothing refused.

**And the thing that was supposed to work did not.** With `transformers==5.16.1` installed
and verified in the log (`accelerate 1.15.0 transformers 5.16.1`):

```
FAILED src/deltaforge/oracle_test.py::test_reference_greedy_decode_matches_the_oracle_token_for_token
At index 2 diff: 11540 != 1528
```

**The same index. The same two token ids. As rental 27.** Rental 27 was an RTX 5090 running
transformers 5.17.0; rental 28 was an RTX 4090 running 5.16.1. Neither the card nor the
library version moved the result by a single token.

### The hypothesis is refuted, and refuted cleanly

The floor `>=5.16,<6` resolved to **5.16.1 on 2026-09-07 as well** — 5.17.0 did not exist
until 2026-09-09. So the rental that passed and the rental that failed ran *the same
transformers*. The release-date argument that made 5.17.0 look guilty was real but
irrelevant: it established that the input *could* have moved, not that it *did* move on the
day that mattered. Pinning was still correct — an unpinned oracle is a defect whatever the
outcome — but it was not the cause, and blocker 13 is not fixed.

### What is left, after everything checkable was checked

| input | 2026-09-06 (passed) | 2026-09-12 (failed) |
|---|---|---|
| `reference.py`, `model.py`, `config.py`, `weights.py` | unchanged since 2026-09-06 | identical |
| `oracle_test.py` and its prompt | unchanged since 2026-09-06 | identical |
| `transformers` | 5.16.1 | 5.16.1 |
| checkpoint `Qwen/Qwen3.5-4B` | last modified 2026-03-02 | identical |
| torch / triton / python | 2.11.0+cu128 / 3.6.0 / 3.12.3 | identical |
| GPU | RTX 5090 | RTX 4090 (and rental 27's 5090 diverged identically) |

Every input anyone has named is constant. That reframes the question: not *what moved*, but
**whether exact token equality was ever the right gate.**

### The reading that now fits the evidence

The test is an exact-equality check over 32 sequential argmaxes in bf16. The prompt is
"The chunked delta-rule recurrence is a sequential scan with matrix-valued state." Both
models emit `\nThe`; ours then continues by echoing the prompt (`chunked`), HuggingFace by
picking `state` — token 14 of the same prompt. **Both are ordinary continuations**, and
they are exactly the kind of pair a sub-ULP difference reorders.

Note what commit `5722aaa` did on 2026-09-06. It found three fp32-era bounds in this very
file that no correct bf16 implementation could meet, fixed all three to score relative to
the logit scale, and left *this* test alone — deliberately, "until the oracle could
adjudicate". The oracle adjudicated once, in its favour, and has now contradicted itself
twice without a single input changing. A gate that flips on an unchanged system is
measuring something other than what it claims to.

**That is a reading, not a result, and it must not be promoted without the number.** A
correct reference and a fragile gate look identical from here; so does a real bug in the
reference that the logits test is too coarse to see.

### The number that decides it, and what it costs

`test_report_the_first_greedy_divergence` is now in `oracle_test.py`. It asserts nothing,
fails nothing, and changes no gate. It replays both models to the diverging step and writes
to `results/diagnostics/oracle-divergence.json` — which teardown pulls home before
destroying the instance, as it did on this rental:

* how decisively each model preferred its own token, in logits;
* each model's own top-2 gap at that step;
* the largest disagreement between the two models' logits anywhere in that row, and one
  bf16 ULP at that scale.

If the gap that separates `11540` from `1528` is smaller than the disagreement the two
models show anyway — measured at 1-2 ULP (0.28125) on 2026-09-07 — they are splitting a
tie and the gate is wrong. If it is much larger, the reference has a real bug and this
gate has been right all along and rental 50123509 was the fluke.

Rental 28 cost $0.0478 to reach the GPU suite. **The number costs about the same**, because
the suite runs before the batch regardless and this test runs inside it.

## What stood between this batch and a number — 2026-09-11

No rental happened on this date; this is a bench session that cleared what it could clear
without one. Four things were in the way, and two of them are now gone.

**Cleared: the pull.** Docker Hub credentials are in `.env`, so the create request carries
`image_login` and the default `vastai/*` image is pulled authenticated. Blocker 1 recurred
as recently as rental 23, on the default image, so this was live and not historical.

**Cleared: the trap in the offer search.** Blocker 10 had no fix, only `--exclude-machines`
applied by a human who knew to reach for it. The search now emits the five cheapest offers
and the create step walks them, so a listed-but-unrentable ask costs one warning instead of
a run. Proven by tests against a stub API, not by a GPU.

**Restored, not proven: the reference.** `transformers` was installed from a floor, which
let the oracle move between rentals; it is now pinned to 5.16.1, the version that passed on
2026-09-07. The GPU suite gate refuses to run a batch on an unvalidated reference, so until
that test passes on a card, batch 001 cannot legally produce a number. **This is the one
remaining hard blocker, and it is settled by the first rental that runs.**

> **Rental 28 settled it, against the pin.** The floor had already resolved to 5.16.1 on
> 2026-09-07, so the pin changed nothing about that day's stack and the test failed
> identically. See the section above; this paragraph is kept as written because the
> prediction it made is what the rental tested.

**Untouched: the arithmetic.** A cold cache still needs ~131 minutes to reach the end of
the *second* slot — 1500s setup, 2400s reference compile, two slots at 1980s — against a
180-minute gate. Batch 001 has nine slots. **It will not finish in one rental and should
not be expected to.**

That is not a reason to re-cut it. The batch stops itself before a slot it cannot finish,
records each slot as it completes, and the compile cache now comes home, so the honest
shape of this batch is:

* **Rental A (cold).** Oracle result settles blocker 13. Slot 0 calibrates. Slot 1 is the
  first measured hypothesis this project has ever produced. The batch stops itself, and
  `phases.env` comes home carrying the first *measured* compile cost — which replaces the
  pessimistic cold estimates in `batch.COLD_PHASE_ESTIMATES` for every rental after it.
* **Rental B (warm, same card).** The pre-flight gate now reasons from measured numbers
  instead of guesses, and the batch resumes into the slots A could not reach.

Two rentals to finish batch 001 is the plan, not a failure of one. The alternative —
shrinking the batch to fit a cold cache — throws away the only thing that makes the
remaining slots cheap.

## Rentals 30-32, 2026-09-13 — the calibration slot finally says why

Three rentals, $2.740, and still no ratio. What changed is that the reason is no longer
unknown.

### The oracle gate passes, on three independent hosts

`results/diagnostics/oracle-divergence.json` records `status: "agree"`,
`first_divergence_index: null`, ours ≡ theirs across all 32 tokens, and
`oracle-gate.json` records `tie_break_count: 0`. Not a coin-flip that fell the right way —
there were no ties to break. The criterion was registered at `c5ce9d7` **before** any data
existed, which is what makes the pass mean something.

The 2026-09-10 divergence is therefore resolved, and its cause remains unexplained. Rental
28 had already refuted the `transformers`-version story. `AGENT.md` §8's architecture
facts were never in doubt and still stand.

### One missing `no_grad` was three failures

`harness/bench.py` timed the workload with autograd live; `harness/correctness.py:120`
did not. Hence gates passing and only benchmarks failing. Rental 31, same model, same card,
same rental:

| phase | time | autograd |
|---|---|---|
| `correctness` (passes) | **17.1 s** | `no_grad` |
| benchmark (crashes) | **3162 s** | none |

Slot 0 is `000-identity`: it installs nothing, so the candidate *is* the reference and the
failure could never have been a kernel bug. Slot 1 failed identically (3206 s) with a
different kernel installed — a fault in the shared timing path. Inductor was compiling the
backward graph as well as the forward, which also killed the one-core theory for rental
22's ~40 minutes: 64 cores on rental 30, **256** on rental 31, no improvement.

Fixed in `f461025`, three tests watched failing first.

### The fix works, and is not enough

Rental 32, same card and same 256 cores as rental 31, changing only the fix: **no
`BackendCompilerFailed`**, correctness gates passed, memory flat at 8.07 GiB of 31.36.
Slot 0 then ran its **full 6980.9 s cap without completing the compile**.

The honest correction is to what rental 31 measured: 3162 s was *time-until-crash*, not a
completed compile. **No run has ever finished a cold `max-autotune` on this model.** The
crash was hiding the cost, not causing it.

### Scorecard

Nine hypotheses, **nine unscored**. Slot 0 errored in all three rentals, so
`CALIBRATION FAILED` fired every time and correctly voided everything else: a batch whose
identity champion never measures 1.00 has established nothing about what its harness
measures. Slot 1 produced a number on rental 31 and it is recorded for diagnosis, not as a
finding.

### What it cost

| Rental | Billed | Cost | Outcome |
|---|---:|---:|---|
| 30 | 69.72 min | $0.5448 | oracle gate passes; host dropped ssh mid-slot-0 |
| 31 | 131.03 min | $0.8069 | 2 slots, `BackendCompilerFailed` — root cause found |
| 32 | 176.80 min | $1.3878 | crash fixed; slot 0 hit its cap |
| | | **$2.740** | month-to-date $5.102 of $45 |

### What the next session inherits

**47 MB / 1800 inductor and triton entries** compiled *without* autograd, against rental
31's 312 KB compiled with it. Inductor caches per kernel, so rental 32's timed-out compile
banked real progress and the next rental on this card starts genuinely warm.

The one thing to do with it: **finish a single cold `max-autotune` compile and time it**,
off that warm cache, before filling another batch. If it still does not complete, the lever
is `mode="reduce-overhead"` or fewer columns, not more cores and not a bigger batch.

## Cost

Five rentals, 91.39 billed minutes, $0.567 — session `batch001-20260908T211825Z`, which
ended at its 90-minute gate. Zero leaked instances. Three of the five died to blocker 7
before it was understood.

## Rentals 33-35, 2026-09-14 — nine slots ran, and the harness is calibrated

Three rentals, **$0.662**, and the first measurements this project has taken. Two of the
three bought a layer that the one before it had been hiding.

### The compile finished

Rental 32 spent its entire 6980.9s cap inside slot 0 without completing a cold
`max-autotune`. No run ever had. The cause was not cores, not autograd, and not the cache:

`gated_delta_rule` scans the sequence with a Python `for t in range(seq_len)`, and dynamo
**unrolls that into the graph**. Measured on a CPU against the tiny config, the captured
graph grows linearly with prefill length at ~22 FX nodes per token per linear-attention
layer — 466 nodes at prefill 8, 5922 at 256. Qwen3.5-4B has **24** linear-attention layers
of 32 and the workload prefills **2048** tokens, so `setup()` handed inductor roughly
**1.08 million FX nodes** under `max-autotune`, once per column.

And `run_interleaved` excludes every `setup` from the timed region *by construction*. The
compile that consumed four rentals was of a graph nothing measures.

The prefill now runs on the eager module; the decode steps, which are the measurement, still
go through the compiled wrapper.

| | slot-0 reference compile |
|---|---|
| rental 32 (cold, autograd already fixed) | **6980.9s, did not finish** |
| rental 34 (47 MB cache from 32) | **267.5s** |
| rental 35 (298 MB cache from 34) | **57.4s** |

Every timed round after the compile is ~900ms with an IQR under 0.005.

### Calibration passes

| rental | identity ratio | IQR | correctness |
|---|---:|---:|---|
| 34 | **1.0009** | 0.0018 | exact — `max_abs_err` 0.0, 32/32 tokens on 5 prompts |
| 35 | **1.0018** | 0.0043 | exact |

`summary.json` records `calibrated: true`. The identity champion is the strongest claim in
the batch and it has never returned a number before. **Every other number in this batch is
now admissible in principle** — which is what makes the rest of this section a statement
about the numbers rather than about the harness.

### Rental 34: the reclaim that emptied the batch

Slot 0 calibrated, and then **all eight scoring slots died in 23s each**:

```
AssertionError: Running CUDAGraph after shutdown
```

`release_compiled_state` called `reset_cudagraph_trees` to give back the graph pool a
finished slot was holding, on the premise — written in its own docstring — that the
reference would "re-record on the next slot's first warmup call". It does not. The shutdown
is permanent for an already-recorded callable, and inductor's trees are per **device**, not
per model, so releasing the candidate tore down the reference columns too.

This could not appear until a slot completed, which is why 33 rentals never saw it.

The pools now stay, and rental 34 measured what that costs: slot 0 recorded CUDA graphs on
both columns and sat at **8.07 GiB allocated / 8.08 reserved of 31.36 before and after the
release alike**. At two columns with autograd off, the reclaim returned less than the logged
number resolves. (It also settles blocker 8: nine slots never exceeded 8.07 GiB.)

### Rental 35: nine slots, and six of them are not measurements

| slot | outcome | ratio | IQR | ref ms | cand ms | correctness |
|---|---|---:|---:|---:|---:|---|
| 000-identity | inconclusive | 1.0018 | 0.0043 | 893.2 | 892.4 | pass |
| 001-fused-rmsnorm-residual | incorrect | 0.5536 | 0.0067 | 911.8 | 1637.6 | **fail** |
| 002-rmsnorm-only | incorrect | 0.5787 | 0.0118 | 924.9 | 1589.9 | **fail** |
| 003-qk-norm-triton | loss | 0.1546 | 0.0022 | 920.7 | 5950.4 | pass |
| 004-fused-swiglu | loss | 0.1460 | 0.0014 | 902.2 | 6201.3 | pass |
| 005-fused-qkv-rope | incorrect | 0.1560 | 0.0024 | 898.0 | 5793.7 | **fail** |
| 006-gqa-no-expand | incorrect | 0.1552 | 0.0017 | 912.9 | 5919.2 | **fail** |
| 007-gated-delta-fused-step | incorrect | 0.1568 | 0.0036 | 919.3 | 5872.1 | **fail** |
| 008-flash-decode-splitkv | incorrect | 0.1527 | 0.0030 | 898.4 | 5944.0 | **fail** |

Six kernels attacking six different operations — QK norm, SwiGLU, RoPE, GQA, the delta-rule
scan, split-K attention — returning ratios between 0.146 and 0.157. That is not six
coincidences. It is one systematic effect, and the log names it:

```
[0/8] torch._dynamo hit config.recompile_limit (8)
```

Dynamo caches compiled code per **code object** with guards, and a fresh candidate module is
a fresh guard. Every slot compiles one against the same `ReferenceModel.forward`, so slot N
is cache entry N. At the limit, dynamo stops compiling that code object and **runs it
eagerly for the rest of the process**, with a warning and no error.

The per-round progress log added this session shows it happening. The candidate's *first*
round, which contains its compile:

| slot | candidate round 0 |
|---|---:|
| 0 | 906.9 ms |
| 1 | 235 848.1 ms — compiling |
| 2 | 171 750.6 ms — compiling; the limit trips here |
| 3 | 6 020.0 ms — **not compiling** |
| 4 | 6 162.5 ms |
| 5 | 5 965.9 ms |
| 6 | 5 945.9 ms |
| 7 | 5 928.7 ms |
| 8 | 5 991.9 ms |

**So slots 3-8 timed an eager candidate against a compiled reference.** Their ratios measure
`torch.compile` itself — worth about 6.5x on batch-1 decode, which is its own datum — and say
nothing about any kernel. `003-qk-norm-triton` and `004-fused-swiglu` are recorded as `loss`
and **are not losses**; they are void, and they are the two that passed correctness.

Slots 1 and 2 *did* compile, so their ratios are admissible. Both failed correctness, so
neither is a finding about speed.

### What is trustworthy from rental 35

**The correctness gate**, for all eight kernels — it runs eagerly, before the benchmark, and
does not depend on compilation at all. Six of eight kernels failed it. That is a real
result, and this project's first.

**Why they failed is not settled**, and the detail matters. For 001 and 002 the *layer-1
kernel checks passed* — max relative error 0.0076 and 0.0074 against the tensor's own
scale, a couple of bf16 ULP. It is **layer 2**, exact token equality, that failed: on 1 of 5
prompts, at token 9, **identically for two different kernels**.

Two independent kernels diverging at the same token of the same prompt, at relative errors
inside the bf16 bound, is the signature of a borderline argmax — not of two coincident bugs.
`AGENT.md` §7a records this project getting that call wrong in one direction (four fp32-era
bounds that looked like model bugs) and the oracle saga getting it wrong in the other. **Do
not promote either reading without the number**: the top-2 logit gap at the diverging step,
against the disagreement the two implementations show anyway. That is exactly what
`test_report_the_first_greedy_divergence` already computes for the oracle, and the same
treatment applies here.

### Scorecard

`prediction_record: {"correct": 1, "scored": 9}`. Read it with care: the one correct
prediction is the identity slot. Six of the eight kernel predictions were scored against
ratios that measure a fallback rather than a kernel, so the scorecard is not yet meaningful
for them either.

### What it cost

| Rental | GPU | Billed | Cost | Outcome |
|---|---|---:|---:|---|
| 33 | RTX 5090 (machine 44927) | 8.90 min | $0.0602 | CUDA `Error 804` — blocker 11 recurred |
| 34 | RTX 5090 | 30.15 min | $0.2192 | compile finished; calibrated; 8 slots lost to the cudagraph reset |
| 35 | RTX 5090 | 52.65 min | $0.3829 | **all nine slots ran**; 6 lost to the recompile limit |
| | | 91.70 min | **$0.6623** | month-to-date $5.764 of $45 |

Zero leaked instances. The session ends at 91.70 of its 180 billed minutes, and the
pre-flight gate refuses the next rental on its own arithmetic.

### What the next session should do first

**One thing, and it is written and tested on CPU already:** `recompile_limit_for` in
`batch_run.py` raises dynamo's limit to cover every slot, and each slot record now carries
`graphs_compiled`, so a candidate that never compiled says so in the record instead of
publishing a plausible ratio. Neither has run on a GPU.

Run batch 001 again. Slot 0 is now ~194s including a warm compile, a slot averages ~200s,
and a full nine-slot batch took 44 minutes of GPU time — so this is one rental, and it
should produce eight admissible ratios and the project's first real leaderboard.

Then, separately and cheaply, settle the layer-2 correctness question above with the logit
gap rather than by loosening the gate.
