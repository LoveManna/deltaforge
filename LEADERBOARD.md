# Leaderboard

## Champion

**None — no baseline has been recorded yet.**

| | |
|---|---|
| Kernel | — |
| Replaces | — |
| Median ratio vs `torch.compile(max-autotune)` | — |
| IQR of the scoring rounds | — |
| Last verified on | — |
| Result record | — |

**Still no champion — but for the first time the reason is a measurement rather than a
defect.** On 2026-09-16 (rental 37) batch 003 returned **seven admissible ratios**:
`calibrated: true`, `graphs_compiled` non-zero on every slot, no slot voided. Every one of
them lost. Nothing here is estimated, projected, or placeheld.

**The baseline is now characterised, which it never was.** The compiled reference runs at
**6.84 ms/token — 1308 GB/s, 73% of an RTX 5090's 1790 GB/s vendor peak** — against a
5.11 ms/token roofline. That number is the denominator for every hypothesis in the backlog
and it says the compiler is already most of the way to the wall.

**The reference's reading of the weight *values* is no longer settled** — it matched
HuggingFace token-for-token on 2026-09-07 and diverged on 2026-09-10 and again on
2026-09-12; see below. That was the outstanding precondition, so it is now the outstanding
*question*. The oracle's version is pinned rather than floored since 2026-09-11, which was
right on its own terms and ruled itself out as the cause on the next rental. **The calibrated harness now exists**: on rentals 34 and 35 the identity champion
measured 1.0009 and 1.0018 against IQRs of 0.0018 and 0.0043, with correctness exact. That
was the other outstanding precondition and it is met. There is still no baseline and no
champion, because no candidate has produced an admissible ratio — see the hypotheses table.

### The reference was validated on 2026-09-07, and disagreed on 2026-09-10.

**2026-09-07 — the weight-value oracle passed for the first time.**

```
test_reference_greedy_decode_matches_the_oracle_token_for_token  PASSED
```

Our from-scratch `reference.py` greedy-decoded 32 tokens **identically to HuggingFace's own
Qwen3.5-4B**, on the real checkpoint, on an RTX 5090. `AGENT.md` §4 calls this the
precondition for every number downstream of it, and it had never run in nine previous
rentals. It means `head_dim` 256 (not 160), the `1 + weight` RMSNorm convention, the
sigmoid output gate, partial mRoPE, the fp32 recurrent state and the GatedDeltaNet
projection layout were all corroborated.

**2026-09-10 — the same test failed on rental 27.**

```
assert ours == theirs
At index 2 diff: 11540 != 1528
```

**This claim is therefore no longer settled, and nothing should be built on it until it
is.** What did *not* change: in the same run the logits oracle passed (`relative < 1e-2`
against the tensor's own scale) and so did the mRoPE reduction test, so the architecture
facts above are still corroborated. The disagreement is one argmax at token 2 of 32, with
logits inside the bf16 bound — the exact failure the test's docstring anticipates, since
"small drifts change argmax".

The leading suspect is that **`transformers` moved under us**: the install is
`>=5.16,<6`, a floor rather than a pin, and rental 27 resolved it to 5.17.0 where the
2026-09-07 run predates that release. The assertion compares our tokens to *theirs*, so a
change on their side fails it with nothing in this repo changing. **That is a hypothesis,
not a finding** — one GPU minute settles it. Full account in
`results/batches/002-compile-cost/README.md`.

**No benchmark ratio exists yet.** Batch 001 has now been attempted on two more rentals and
is still void: the identity champion has never returned a number, so nothing else it reports
would mean anything. Full account: `results/batches/001-calibration/README.md`.

What moved on 2026-09-08: candidate *construction* is fixed and proven, the rental path is
fixed and proven, and the failure has relocated to the benchmark's own warmup — a CUDA OOM
at 30.71 GiB of 31.36 with the default four columns. A cold `max-autotune` compile also
turns out to cost ~40 minutes rather than the 3-4 the batch cost model assumes, which is now
the binding constraint on how many hypotheses fit a rental.

Twenty-seven rentals have now been billed across the project, $2.314 lifetime, **zero
leaked**. Five of those were 2026-09-10, which measured no hypothesis and instead found
five defects between renting a box and running one — including a stall guard that destroyed
a healthy rental at the moment its container pull succeeded. See
`results/batches/002-compile-cost/README.md`.

### The chain of blockers, and where it stands

Each rental that got further than its predecessor did so by exposing the next problem:

| | Blocker | Status |
|---|---|---|
| 1 | Anonymous Docker Hub pulls stall from vast egress ranges | fixed |
| 2 | The image refuses the account's ssh key | fixed, **proven** on a live box |
| 3 | The image has `python3` but no `python` | fixed, proven |
| 4 | `accelerate` absent, so the oracle cannot even be constructed | fixed, proven |
| 5 | Four `oracle_test.py` bounds were fp32-era absolutes applied to bf16 | fixed, proven |
| 6 | Candidate construction double-allocates the 8.4 GB of weights | fixed, **proven** — every slot on rental 21 built and passed correctness |
| 7 | Readiness waited on `cur_state` (the rental contract), not `actual_status` (the container) | fixed, proven |
| 8 | The benchmark OOMs at warmup with the default four columns | fixed, **proven** — rentals 34-35 held 8.07 GiB of 31.36 across nine slots |
| 9 | **The stall guard destroyed a healthy rental** the moment its pull finished | fixed, **proven** — three instances have since passed through that state |
| 10 | A phantom ask traps the deterministic, price-ordered offer search | manual `--exclude-machines` only, **no real fix** |
| 11 | Host driver older than our torch build → CUDA `Error 804` | **recurred on rental 33** — `DF_MIN_CUDA` filters an advertised `cuda_max_good`, not a driver |
| 12 | The ghcr image ships no Python headers, so Triton's JIT shim will not build | fixed, **proven** — the GPU suite has passed on rentals 34 and 35 |
| 13 | The oracle's greedy decode no longer matches HuggingFace | **closed** — agrees token-for-token on rentals 30-32 and 34-35, zero tie-breaks |
| 14 | A cold `max-autotune` compile never finishes inside a session | fixed, **proven** — the unrolled prefill scan; 6980.9s (unfinished) → 267.5s → 57.4s warm |
| 15 | `reset_cudagraph_trees` between slots tears down the reference columns | fixed, **proven** — it emptied rental 34, and rental 35 ran all nine slots |
| 16 | Dynamo's `recompile_limit` (8) makes a batch silently time **eager** candidates | fixed, **proven** — batch 003 ran seven slots with `graphs_compiled: 3` on every one |

Blocker 7 is worth reading even though it is closed: it had been costing rentals since 11
while wearing a convincing disguise as flaky hosts, and the "2 in 17 rentals go to hosts
that never answer sshd" line this file used to carry has been withdrawn.

**Blocker 16 is closed.** `recompile_limit_for(7)` raised dynamo's limit to 22 on rental 37
and every one of batch 003's seven slots reported `graphs_compiled: 3`. This project now has
admissible ratios; what it does not have is a kernel that beats the compiler, which is a
scientific problem rather than an infrastructural one and is the first time that has been
true.

Note what blockers 14, 15 and 16 have in common: none of them could be seen until the one
before it was fixed. 14 stopped any slot finishing, so 15 (which only fires *between* two
slots) was invisible; 15 emptied the batch after slot 0, so 16 (which only fires once
several candidates have compiled) was invisible in turn. Three rentals in one day, each
buying exactly one layer.

Blocker 11 is the one regression: its fix was recorded as proven on the strength of rental
27 and rental 33 disproved it. `DF_MIN_CUDA` filters the offer's advertised `cuda_max_good`,
which is a claim rather than a driver, and a host sitting exactly on the floor turned out to
be the risky case. See `docs/GPU-ACCESS.md`,
`results/batches/002-compile-cost/README.md`, and
`results/batches/001-calibration/README.md`.

## Baseline

| | |
|---|---|
| Status | **not recorded** |
| Definition | `src/deltaforge/reference.py` under `torch.compile(mode="max-autotune")` |
| Headline workload | batch 1, context 2048, 128 decoded tokens |
| Secondary workload | batch 32, context 2048, 128 decoded tokens |
| Model | `Qwen/Qwen3.5-4B` (see `docs/ARCHITECTURE.md` on why not a newer one) |
| Roofline at headline | 5.11 ms/token, 196 tok/s on an RTX 5090 — `docs/roofline.py` |
| Record | `results/baseline/` (empty) |

## Hypotheses

Nine written and shipped. **Nine ran on rental 35 (2026-09-14), and none has an admissible
ratio.** The batch is *calibrated* for the first time — `000-identity` measured 1.0018 with
an IQR of 0.0018 — so the harness is trustworthy. What is not trustworthy is six of the
eight numbers below, and the reason is recorded rather than guessed: dynamo hit
`recompile_limit` (8) inside slot 2 and stopped compiling candidates, so slots 3-8 timed an
**eager** candidate against a compiled reference. Six unrelated kernels returning ratios
between 0.146 and 0.157 is that fallback, not six coincidences.

`ratio*` marks a number measured against a candidate that never compiled. It is kept
because deleting evidence is worse than labelling it, and it is not a result.

| ID | Hypothesis | Replaces | Median ratio | IQR | GPU | Correctness | Outcome | Record |
|---|---|---|---:|---:|---|---|---|---|
| 001 | Fused residual add + RMSNorm | `rms_norm_residual` | 0.5536 | 0.0067 | RTX 5090 | **fail** — layer 2, 1 of 5 prompts, token 9 | `incorrect` | [dir](results/batches/001-calibration/) |
| 002 | Standalone Triton RMSNorm, hidden-size sites | `rms_norm` | 0.5787 | 0.0118 | RTX 5090 | **fail** — layer 2, 1 of 5 prompts, token 9 | `incorrect` | [dir](results/batches/001-calibration/) |
| 003 | The same kernel on `q_norm`/`k_norm` | `rms_norm` | 0.1546* | 0.0022 | RTX 5090 | pass | **void** — candidate ran eager | [dir](results/batches/001-calibration/) |
| 004 | Fused SwiGLU activation | `swiglu_mlp` | 0.1460* | 0.0014 | RTX 5090 | pass | **void** — candidate ran eager | [dir](results/batches/001-calibration/) |
| 005 | Fused partial mRoPE | `qkv_projection_rope` | 0.1560* | 0.0024 | RTX 5090 | **fail** | `incorrect`, ratio void | [dir](results/batches/001-calibration/) |
| 006 | GQA decode without the head expansion | `gqa_attention` | 0.1552* | 0.0017 | RTX 5090 | **fail** | `incorrect`, ratio void | [dir](results/batches/001-calibration/) |
| 007 | Fused gated delta-rule step | `gated_delta_rule` | 0.1568* | 0.0036 | RTX 5090 | **fail** | `incorrect`, ratio void | [dir](results/batches/001-calibration/) |
| 008 | Split-KV flash decode | `gqa_attention` | 0.1527* | 0.0030 | RTX 5090 | **fail** | `incorrect`, ratio void | [dir](results/batches/001-calibration/) |

**2026-09-13 (rentals 30-32): still no ratio, but the reason is now known.** The oracle
gate passes on three independent hosts with zero tie-breaks, and correctness gates pass.
Slot 0 errored in all three rentals, so `CALIBRATION FAILED` fired each time and correctly
voided everything else. The cause of two of those errors was one missing `no_grad` in
`harness/bench.py` (fixed in `f461025`); the third is that a cold `max-autotune` compile
does not finish inside a session — which no run has ever managed. See
`results/batches/001-calibration/README.md`.

**2026-09-14 (rentals 33-35): the harness is calibrated, and the batch is not
trustworthy past slot 2.** `summary.json` records `calibrated: true`, `counts:
{inconclusive: 1, incorrect: 6, loss: 2}` and `1 correct of 9 scored`. Read that scorecard
with care: the one correct prediction is the identity slot, and the two `loss` outcomes are
the artefact above rather than measurements of a kernel.

What *is* trustworthy from this rental, because it does not depend on compilation at all:
**the correctness gate ran eagerly for all eight kernels and six of them failed.** That is a
real result about the kernels, and it is the first one this project has.

What is not settled is *why* they failed. For 001 and 002 the layer-1 kernel checks
**passed** (max relative error 0.0076 and 0.0074, a couple of bf16 ULP) and it is layer 2 —
exact token equality — that failed, on the same 1 of 5 prompts at the same token 9, for two
different kernels. An identical divergence from two independent kernels, at relative errors
inside the bf16 bound, is the signature of a borderline argmax, not of two coincident bugs.
This project has twice mistaken that signature for a model bug (`AGENT.md` §7a) and once for
a fragile gate. **Do not promote either reading without the top-2 logit gap at the diverging
step**, which is the same number `test_report_the_first_greedy_divergence` already produces
for the oracle.

**001 is no longer graveyarded on mechanism.** It was closed on the argument that a 0.018%
ceiling is not worth a rental — an argument about *cost*, which batching dissolves. Its
arithmetic still stands as the *prediction*; what changed is that the prediction is now
falsifiable in practice. Same for 004 and 005. **None of them is graveyarded on this
rental either**: a kernel whose candidate never compiled has not been shown to be slow, and
a kernel that fails an exact-token gate at 2 ULP has not been shown to be wrong.

### Batch 003 — weight-only quantisation, 2026-09-16 (rental 37)

**Seven admissible ratios, seven losses, and no kernel bugs.** The first batch in this
project where every slot compiled, every slot was scored, and nothing was voided. Predictions
scored **1 correct of 7**, and the one that was right is the identity slot.

| ID | Hypothesis | Replaces | Median ratio | IQR | GPU | Correctness | Outcome | Record |
|---|---|---|---:|---:|---|---|---|---|
| 000 | Identity champion | — | **1.0024** | 0.0190 | RTX 5090 | 5/5 exact | calibrated | [dir](results/batches/003-int8-weight-only/) |
| 009 | Hand-written bf16 GEMV (control) | `decode_step` | 0.2801 | 0.0106 | RTX 5090 | **fail** — exact gate, 1 of 5 prompts | `incorrect` | [dir](results/batches/003-int8-weight-only/) |
| 010 | Weight-only int8 via PyTorch dequant (control) | `decode_step` | 0.9893 | 0.0212 | RTX 5090 | **fail** — layer 1, see below | `incorrect` | [dir](results/batches/003-int8-weight-only/) |
| 011 | int8 fused dequant-GEMV, MLP only | `swiglu_mlp` | 0.4669 | 0.0254 | RTX 5090 | 256/264, KL 0.00082 | `incorrect` | [dir](results/batches/003-int8-weight-only/) |
| 012 | int8 fused dequant-GEMV, all layer projections | `decode_step` | 0.1962 | 0.0042 | RTX 5090 | 255/264, KL 0.00110 | `incorrect` | [dir](results/batches/003-int8-weight-only/) |
| 013 | 012 plus the tied LM head | `decode_step` | 0.1878 | 0.0026 | RTX 5090 | 256/264, KL 0.00122 | `incorrect` | [dir](results/batches/003-int8-weight-only/) |
| 014 | int4 group-128 fused dequant-GEMV, all sites | `decode_step` | 0.1797 | 0.0040 | RTX 5090 | 226/264, KL 0.09185 | **`loss`** | [dir](results/batches/003-int8-weight-only/) |

**Read the `incorrect` column with care: not one of them is a kernel bug.** Every Triton
kernel passed layer 1 at every probe, worst relative error 7.8e-3 — about one bf16 ULP. The
six failures are three defects in the *gates*:

* **009** was gated `exact` on the reasoning that it computes the same function as the
  reference. It does; it does not compute the same *bits*. fp32 accumulation in a different
  order from cuBLAS lands one bf16 ULP away, and this model's top-2 logit gaps flip an argmax
  on that. `AGENT.md` §7a already records this trap and rental 35 already paid for it.
* **010-013** missed a top-1 agreement bar chosen from priors rather than from measurement,
  and set finer than the statistic can resolve. At n=264 positions agreement quantises to
  1/264 = 0.0038; **013 missed its bar by 0.000303 — eight hundredths of one token.**
* **010**'s layer 1 failed against a reference it never claimed to match: `Int8DequantLinear`
  rounds the dequantised weight to bf16 as any real PyTorch implementation would, while the
  shared layer-1 reference dequantises in fp32.

**Why the kernels lost is a single fact.** They were never bandwidth-bound. As they removed
bytes they got *slower* — 26.90 → 38.68 → 42.49 ms/token while traffic fell 9158 → 5588 →
2850 MB/token — because a cross-lane `tl.sum` reduction ran once per K-iteration and the
dequantisation landed on an already-saturated issue port. Achieved bandwidth: **332 GB/s at
bf16, 141 at int8, 65 at int4, against the compiled baseline's 1308.**

**And a two-week-old claim in the backlog is refuted.** `docs/HYPOTHESES.md` asserted that
inductor materialises the dequantised weight and is therefore *slower* than bf16. Measured:
0.9893 ± 0.0212 — the same. The materialisation story needs 2525 GB/s on a 1790 GB/s card,
so it is impossible; inductor either fuses the dequant or keeps the transient in L2.

Full account, including the bandwidth table and the ranked plan that follows from it:
[`results/batches/003-int8-weight-only/README.md`](results/batches/003-int8-weight-only/README.md).

### Column definitions

- **ID** — `NNN`, matching the branch `hyp/NNN-slug` and `results/hypotheses/NNN-slug.json`.
- **Hypothesis** — one line, stating the mechanism expected to produce the win.
- **Replaces** — the reference operation the kernel substitutes, from `REPLACEABLE_OPS`.
- **Median ratio** — `median(t_compiled / t_candidate)` over the scoring rounds. Greater
  than 1 means faster than the compiler. This is the score; absolute milliseconds are
  provenance and live in the result record.
- **IQR** — interquartile spread of those per-round ratios: the noise band. A margin
  smaller than this is not a win.
- **GPU** — the card it was measured on. Every session rents a different one, so a ratio
  is only comparable within its own run.
- **Correctness** — `pass` only when both gates pass. Otherwise the worst error magnitude,
  or the prompt and token index where the end-to-end match diverged.
- **Outcome** — one of:
  - `win` — beat the incumbent by more than the IQR. Promoted to champion.
  - `loss` — slower. Goes to the graveyard with the mechanism that failed.
  - `inconclusive` — inside the noise band. Neither promoted nor buried; the hypothesis
    stays open for a cleaner measurement.
  - `graveyarded on mechanism` — closed by arithmetic before or instead of measurement,
    because its ceiling is below the noise band. Cheaper than measuring, and a legitimate
    result: `docs/HYPOTHESES.md` records the numbers that closed it.
  - `incorrect` — failed a correctness gate. Recorded with its error magnitudes, because a
    candidate that was fast but wrong is among the most useful things to read.
