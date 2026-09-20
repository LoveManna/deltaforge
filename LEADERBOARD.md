# Leaderboard

## Champion

**`022-int4-head` — group-128 int4 on the tied LM head.**

| | |
|---|---|
| Kernel | `tiled_int4_head` (`src/deltaforge/kernels/tiled_gemv.py`) |
| Replaces | `decode_step` — `ReferenceModel.project_logits`, and nothing else |
| Median ratio vs `torch.compile(max-autotune)` | **1.0791** |
| IQR of the scoring rounds | **0.00034** |
| Correctness | layer 1 one bf16 ULP (7.8e-3); layer 2 top-1 0.9318, mean KL 0.01674 nats |
| Last verified on | RTX 5090, rental 40, 2026-09-19. **Not re-benchmarked on rental 42** — see below. |
| Result record | [`results/batches/005-launch-and-head/`](results/batches/005-launch-and-head/) |

**A hand-written Triton kernel has beaten what `torch.compile(mode="max-autotune")`
generates on the decode path of Qwen3.5-4B** — by 7.91%, at an interquartile spread of
0.03%, on a harness that calibrated at 1.0008 in the same run. Two other candidates have
won since, on mechanisms that share nothing with it: `025-fused-causal-conv` at **1.0144**
and `034-static-cache-cudagraphs` at **1.0196**, both bit-identical to the reference.

**The champion's number is one rental old and was not re-verified on rental 42.** That
batch replaced `022` with `028-int4-head-tuned` — the same kernel on the same site with a
searched tile — which measured **0.9920**. The tile is the only difference and it is worth
2.3x (§ batch 006), so 1.0791 stands as `022`'s own measurement and re-measuring it on a
second card is the first slot of the next rental. Recording it any other way would be
carrying a number across sessions, which this file forbids.

**That slot is registered**: `035-int4-head`, the champion unchanged at the heuristic tile,
opening batch `007-compose-and-retile` (`src/deltaforge/batches.py`; the run plan is
[`docs/superpowers/plans/2026-09-20-batch-007-compose-and-retile.md`](docs/superpowers/plans/2026-09-20-batch-007-compose-and-retile.md)).
Nothing in this file moves until it has run.

**The mechanism, stated before the measurement and confirmed by it.** The tied LM head is
248320 x 2560 — **1271.40 MB/token, 14.80% of everything the compiled column moves, in one
matmul.** Storing it at 4 bits removes **953.55 MB/token as the byte model counts it** —
1271.40 becomes 317.85 of packed nibbles — for a **1.1249x** ceiling, and the kernel
collected 70% of it. The kernel also reads 19.87 MB/token of fp32 group scales that
`decode_bytes_per_token` does not count, so the true saving is 933.68 and the true ceiling
**1.1220**; the difference is 0.26% of per-token bytes and it makes every achieved-bandwidth
figure for a quantised candidate conservative rather than flattering. (The 943.7 MB/token
this line used to carry assumed 2-byte scales and matched neither.)

**Why here and not on the 248 projections two rentals lost on.** At BLOCK_N=64 the head
launches **3880 programs** on a 170-SM card; `in_proj_a` is 32 channels wide and launches
four. The same kernel family achieved 319 and 228 GB/s averaged over the layer projections
and **656 GB/s on the head**, with its flop padding and split-K unchanged. The GEMV was
grid-starved, not structurally slow — and that could not be seen while every slot installed
on all 248 sites at once and reported one aggregate number.

**What the batch cost to learn it: 32.32 billed minutes, $0.2350.**

**How the project got here.** Batch 003 (rental 37) returned seven admissible ratios and
every one lost; batch 004 (rental 38) rewrote the kernel that lost hardest around the cause
batch 003 named — and it got **worse**, 0.2801 → 0.1934. Batch 005 stopped rewriting the
kernel and changed where it was installed. Nothing here is estimated, projected, or
placeheld.

**The baseline is now characterised properly, and the earlier figure was wrong twice.**
Rental 38's `TORCH_LOGS=output_code` dump — the diagnostic `docs/HYPOTHESES.md` had called
mandatory and four rentals had skipped — shows that inductor **folds the GQA head expansion
into index arithmetic**, so the compiled column never moves the 570.43 MB/token the roofline
attributes to it. Against the **8587.80 MB/token it actually moves**, the compiled reference
runs at **7.30 ms/token — 1177 GB/s, 65.7% of an RTX 5090's 1792 GB/s vendor peak.** (The
"1308 GB/s, 73%" recorded after batch 003 was high on both counts: it counted the folded
expansion, and it mixed SI megabytes with binary gigabytes per second, which is worth 2.4%.)

**The baseline is generated Triton, not cuBLAS, and it is fused.** `extern_kernels` is
called for convolution and nothing else — no `mm`, no `addmm` — and inductor's matmul
kernels carry the residual add and the RMSNorm inside them. It also runs with **no CUDA
graphs**, because `_causal_conv` mutates its cache in place. See
[`results/batches/004-bandwidth-bound-gemv/README.md`](results/batches/004-bandwidth-bound-gemv/README.md).

**The reference's reading of the weight values is settled.** It matched HuggingFace
token-for-token on 2026-09-07, diverged on 2026-09-10 and 2026-09-12, and has agreed on
every rental since — rentals 30-32, 34, 35, 37, 38 and 40, on six physical hosts, with zero
tie-breaks. That was the outstanding precondition for every number downstream of it and it
is met; blocker 13 in the table below is closed. The historical account of the divergence
is kept in `results/batches/002-compile-cost/README.md` because it cost five rentals to
resolve and the shape of it is worth reading, not because the question is open.

**The harness is calibrated, repeatedly and tightly.** The identity champion has measured
1.0009, 1.0018, 1.0024, 0.9913, 1.0008 and — on rental 42 — **1.0053 at an IQR of
0.00086**. A batch whose identity slot misses 1.00 voids every other number in it; none
since rental 35 has. Note the sign: rental 42 carried a **+0.53% offset**, so every ratio
in batch 006 is that much flattering and `034`'s 1.0196 is ~1.4% net.

**Forty-one instances have now been created and forty-one destroyed, $7.177 lifetime,
zero leaked.** Every one was destroyed cleanly by the trap, including two cancelled
mid-flight with SIGTERM. The cheapest informative rental in the set remains rental 28 at
$0.0478; the most expensive mistake remains rental 32 at $1.3878, which spent its entire
cap inside one compile. (The count is the ledger's, and it is one lower than the "forty
rentals" this file carried before rental 42 created two instances: the prose had drifted
one ahead of `ledger/spend.jsonl`, which is the authority.)

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
| 11 | Host driver older than our torch build → CUDA `Error 804` | **recurred on rentals 33 and 42** — three for three, a host advertising `cuda_max_good` of *exactly* 12.8 fails and 13.0 does not. `DF_MIN_CUDA=12.9` is the filter; it empties the 4090 pool at $0.45, so the rate ceiling has to move with it |
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
| Status | **characterised, and beaten** — 6.70 ms/token, 1282 GB/s, 71.5% of peak (rental 40); 7.30 ms/token and 1177 GB/s on rental 38's card |
| Definition | `src/deltaforge/reference.py` under `torch.compile(mode="max-autotune")` |
| Headline workload | batch 1, context 2048, 128 decoded tokens |
| Secondary workload | batch 32, context 2048, 128 decoded tokens |
| Model | `Qwen/Qwen3.5-4B` (see `docs/ARCHITECTURE.md` on why not a newer one) |
| Roofline at headline | 5.11 ms/token, 196 tok/s on an RTX 5090 — `docs/roofline.py` |
| Bytes the compiled column actually moves | **8587.80 MB/token** — the roofline's 9158.23 less the GQA expansion inductor folds away. `harness.bytes_model` defaults to this since rental 40; it had been dividing by the eager total. |
| CUDA graphs | **none, measured** — 0 recorded and 128 skipped on every slot of rental 40, because the decode cache is mutated in place |
| Record | `results/baseline/` (empty); the numbers are in `results/batches/005-launch-and-head/` and `004-bandwidth-bound-gemv/` |

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

### Batch 004 — a GEMV that is actually bandwidth-bound, 2026-09-17 (rental 38)

**The rewrite made it slower, and the free diagnostic was worth more than the batch.** Two
slots ran, five declined on a precondition. Predictions scored **1 of 2**; the five untested
ones are recorded as untested, not as wrong.

| ID | Hypothesis | Replaces | Median ratio | IQR | GPU | Correctness | Outcome | Record |
|---|---|---|---:|---:|---|---|---|---|
| 000 | Identity champion | — | **0.9913** | 0.0159 | RTX 5090 | 5/5 exact | calibrated | [dir](results/batches/004-bandwidth-bound-gemv/) |
| 015 | Tiled bf16 GEMV: `tl.dot`, K-major, split-K (control) | `decode_step` | **0.1934** | 0.0028 | RTX 5090 | pass — 261/264, KL 0.00060 | **`loss`** | [dir](results/batches/004-bandwidth-bound-gemv/) |
| 016 | fp8 e4m3, all layer projections | `decode_step` | — | — | — | — | `precondition_failed` | [dir](results/batches/004-bandwidth-bound-gemv/) |
| 017 | 016 plus the tied LM head | `decode_step` | — | — | — | — | `precondition_failed` | [dir](results/batches/004-bandwidth-bound-gemv/) |
| 018 | fp8 e4m3, MLP only | `swiglu_mlp` | — | — | — | — | `precondition_failed` | [dir](results/batches/004-bandwidth-bound-gemv/) |
| 019 | int8, all layer projections | `decode_step` | — | — | — | — | `precondition_failed` | [dir](results/batches/004-bandwidth-bound-gemv/) |
| 020 | int4 group-128, all sites plus head | `decode_step` | — | — | — | — | `precondition_failed` | [dir](results/batches/004-bandwidth-bound-gemv/) |

**015 is 009 with a different inside.** Same sites, same bytes, same hypothesis — a
hand-written bf16 GEMV moving exactly what the compiler moves — with the cross-lane `tl.sum`
that batch 003 blamed replaced by a `tl.dot` accumulator, a K-major layout and split-K:

| | ratio | ms/token | achieved |
|---|---:|---:|---:|
| compiled baseline | 1.0000 | 7.30 | **1177 GB/s** |
| `009` naive GEMV (rental 37) | 0.2801 | 26.90 | 319 GB/s |
| `015` tiled GEMV (rental 38) | **0.1934** | **37.73** | **228 GB/s** |

**1.40× slower.** Batch 003's conclusion that its kernel was not bandwidth-bound survives;
its diagnosis that the per-iteration reduction was the cause does not.

**And this time the kernel is demonstrably correct**, which 009 never was: layer 1 worst
relative error 7.75e-3 (one bf16 ULP), top-1 agreement 261/264 with a 95% Wilson interval of
[0.967, 0.996], mean KL 0.00060 nats. Under batch 003's gates it would have been reported
`incorrect` and the 0.1934 discarded.

**Backlog entry 2 is closed by the dump, not by a kernel.** Inductor already folds the GQA
expansion — 6.23% of per-token bytes, the second-largest share in the model — into index
arithmetic. Full account:
[`results/batches/004-bandwidth-bound-gemv/README.md`](results/batches/004-bandwidth-bound-gemv/README.md).

### Batch 005 — the launch, and the head, 2026-09-19 (rental 40)

**The first two wins this project has recorded.** Six slots ran, two declined. Predictions
scored **3 of 5**, and the two that were wrong were wrong in ways the record can name.

| ID | Hypothesis | Replaces | Median ratio | IQR | GPU | Correctness | Outcome | Record |
|---|---|---|---:|---:|---|---|---|---|
| 000 | Identity champion | — | **1.0008** | 0.00017 | RTX 5090 | exact | calibrated | [dir](results/batches/005-launch-and-head/) |
| 021 | Static decode cache, for CUDA graphs | `decode_cache` | 0.9986 | 0.00058 | RTX 5090 | 264/264, 0.00000 nats | **`loss` — untested, see below** | [dir](results/batches/005-launch-and-head/) |
| 022 | **int4 group-128 on the tied LM head alone** | `decode_step` | **1.0791** | 0.00034 | RTX 5090 | 0.9318, 0.01674 nats | **`win` — CHAMPION** | [dir](results/batches/005-launch-and-head/) |
| 023 | int8 per-channel on the head alone | `decode_step` | 1.0373 | 0.00011 | RTX 5090 | **fail — layer 1, rel 4511** | `incorrect` | [dir](results/batches/005-launch-and-head/) |
| 024 | e4m3 on the head alone | `decode_step` | — | — | — | — | `error` — did not compile | [dir](results/batches/005-launch-and-head/) |
| 025 | **Fused four-tap causal conv step** | `causal_conv` | **1.0144** | 0.00107 | RTX 5090 | 264/264, **0.0 abs** | **`win`** | [dir](results/batches/005-launch-and-head/) |
| 026 | 022 + 021 | `decode_step`, `decode_cache` | — | — | — | — | `precondition_failed` | [dir](results/batches/005-launch-and-head/) |
| 027 | 022 + 021 + 025 | three | — | — | — | — | `precondition_failed` | [dir](results/batches/005-launch-and-head/) |

**Why 022 won where two rentals of GEMV work had lost: parallelism, not the kernel.**
Batches 003 and 004 installed on all 248 layer projections at once and reported one
aggregate byte rate. Measured per site, the same kernel family gives:

| | achieved |
|---|---:|
| `009` naive GEMV, 248 layer projections (rental 37) | 319 GB/s |
| `015` tiled GEMV, 248 layer projections (rental 38) | 228 GB/s |
| `022` int4, **the LM head alone** | **656 GB/s** |
| `023` int8, **the LM head alone** | **847 GB/s** |

The head launches 3880 programs at BLOCK_N=64; `in_proj_a` is 32 channels wide and launches
four. The flop padding and split-K that batch 004 suspected are unchanged here.

**int8 beats int4 on byte rate and loses on time**, which settles a question two rentals
could not: at this site the kernel is substantially bandwidth-bound, with a nibble unpack
costing 29% of the achieved bandwidth. Batch 003's issue-bound regime was a property of the
sites it measured, not of the kernel.

**`021` did not test its hypothesis, and the record says so rather than implying a
refutation.** `cudagraph_nodes: 0`, `cudagraph_skips: 127` — the candidate was refused for
mutated inputs exactly as the reference is, so 0.9986 says nothing about what CUDA graphs
are worth. Layer 1 passed, so the marking itself worked; the break is between
`mark_static_address` and `func.static_input_idxs`, and `TORCH_LOGS=cudagraph_static_inputs`
brackets it for free next time. Slot 0's own counters — 0 nodes, 128 skips — are the first
*measurement* that the baseline has never been CUDA-graphed on any rental.

**Two kernel defects, both in code that had never executed.** `023` failed layer 1 at a
relative error of **4511**: `_reduce_partials_kernel` declared `SCALE` and `HAS_SCALE` and
read neither, so the int8 and fp8 paths returned unscaled integer dot products. `024` did
not compile at all — `other=0` is an int32 literal and will not cast to e4m3, in a kernel
body shared by both dtypes. Batch 004 built five slots on that kernel and its preconditions
declined every one, so it shipped, passed CI, and was never run. Both are fixed, and
`kernels/kernel_contract_test.py` now catches the class of each on a CPU.

Full account: [`results/batches/005-launch-and-head/README.md`](results/batches/005-launch-and-head/README.md).

### Batch 006 — the tile, and the promise that pays without the graph, 2026-09-20 (rental 42)

**The batch's premise was refuted by its own first kernel slot, and the one win came from
a mechanism that never fired.** Five slots ran, three declined. Predictions scored
**1 of 4** — every slot predicted `win`.

| ID | Hypothesis | Replaces | Median ratio | IQR | GPU | Correctness | Outcome | Record |
|---|---|---|---:|---:|---|---|---|---|
| 000 | Identity champion | — | **1.0053** | 0.00086 | RTX 5090 | exact | calibrated | [dir](results/batches/006-tile-and-sites/) |
| 028 | int4 head, tile chosen by search | `decode_step` | 0.9920 | 0.00481 | RTX 5090 | 0.9318, 0.01674 nats | **`loss`** | [dir](results/batches/006-tile-and-sites/) |
| 029 | 028 + the fused causal conv | `decode_step`, `causal_conv` | **0.7937** | 0.01706 | RTX 5090 | 0.9318, 0.01674 nats | **`loss`** | [dir](results/batches/006-tile-and-sites/) |
| 030 | int4 on the 96 MLP projections | `swiglu_mlp` | **0.3545** | 0.00063 | RTX 5090 | 0.8977, 0.04868 nats | **`loss`** | [dir](results/batches/006-tile-and-sites/) |
| 031 | 030 + the head | `swiglu_mlp`, `decode_step` | — | — | — | — | `precondition_failed` | [dir](results/batches/006-tile-and-sites/) |
| 032 | int4 on the 200 wide sites + head | `decode_step` | — | — | — | — | `precondition_failed` | [dir](results/batches/006-tile-and-sites/) |
| 033 | 032 + the fused conv | `decode_step`, `causal_conv` | — | — | — | — | `precondition_failed` | [dir](results/batches/006-tile-and-sites/) |
| 034 | Static decode cache, for CUDA graphs | `decode_cache` | **1.0196** | 0.00149 | RTX 5090 | 264/264, **0.0 nats** | **`win`** | [dir](results/batches/006-tile-and-sites/) |

**The tile is not what separates the head from the layer projections.** `028` is the
champion's kernel on the champion's site with the tile chosen by an install-time search
instead of a heuristic, and the search picked BLOCK_N=256 where the heuristic picks 64:

| the head at group-128 int4 | tile | programs | in-situ achieved |
|---|---|---:|---:|
| `022`, rental 40 | BLOCK_N 64, SPLIT_K 1 | 3880 | **656 GB/s** |
| `028`, rental 42 | BLOCK_N **256**, SPLIT_K 1 | 970 | **282 GB/s** |

**Per-site micro-benchmarking does not select tiles for this workload.** The tuner timed
the head at 0.2 ms for 327 MB — **1639 GB/s, faster than the whole compiled model** — and
the tile it endorsed ran at 282 in the decode step. `DF_TILE_TUNE` now defaults to off.

**And the layer projections are 65-67 GB/s whatever is done to them.** `030` put int4 on
the MLP — 52.75% of per-token bytes, a 1.6545x ceiling — at a searched tile, and the MLP
ran at **67 GB/s** against batch 003's **65** at a different tile in a different kernel
structure. Three rentals, two structures, two tile-selection methods, the same number.
The correctness bar was derived in advance from `014` minus `022` at ~0.048 nats and the
measurement was **0.04868**, so the gate method is working and the kernel is not.

**`034` won at 1.0196 with `cudagraph_nodes: 0`.** All 64 decode-cache tensors were marked
static, the candidate is bit-identical (264/264, 0.0 nats, 0.0 absolute error), and no
CUDA graph was ever recorded — so entry 8's hypothesis is **still untested at 0 for 2**
while the slot is a real win. The 2.0% is the *other* thing `mark_static_address` does:
64 tensors leave inductor's per-call alignment check on every one of 128 decode steps, and
achieved bandwidth rose 1198 → **1222 GB/s on identical bytes**.

**Batch 005's two ranked suspects for `021` are dead, killed on a CPU for nothing.**
`TORCH_LOGS=cudagraph_static_inputs` on the tiny config prints `Adding static input pos 5
for source L['cache'].layers[0].conv`: the mark does reach `static_input_indices` through
a plain Python object, and `_extract_tensor_dict` does stamp it. The skip message captured
on the box still says `mutated inputs (64 instances)` with all 64 marked, so the leading
suspect is now the torch version — 2.11 on the box against 2.14 on the laptop.

**The largest unexplained number this project holds is `029`.** The tiled head cost
+0.064 ms/token and the fused causal conv, which *saved* 0.093 ms/token alone on rental
40, added ~1.85 ms/token composed with it. Same kernel, unmodified. `027` was built to
test that pair on rental 40 and its precondition declined it, so this is the first time
the two have ever run together.

Full account, including what the three refusals cost before a card was obtained:
[`results/batches/006-tile-and-sites/README.md`](results/batches/006-tile-and-sites/README.md).

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
