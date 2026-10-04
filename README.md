# DeltaForge

Hand-written Triton kernels for the **Qwen3.5-4B** decode path, measured against
`torch.compile(mode="max-autotune")` running an identical pure-PyTorch reference.

Every attempt is recorded — the wins and the losses — so that independent working
sessions compound instead of rediscovering the same dead ends.

**Start with the write-up:** [**RESULTS.md**](RESULTS.md) — the result first, then each
finding mechanism-first, the method, and the loose ends. The same write-up is served as a
page at **<https://lovemanna.github.io/deltaforge/>** (source in [`site/`](site/)).
[`LEADERBOARD.md`](LEADERBOARD.md) has every hypothesis attempted, with its ratio, noise
band, correctness and the rental it belongs to.

---

## Headline result

**Both of this project's champions were obtained by deleting a hand-written kernel, and the
second of them beat the kernel it replaced by 3.2%.**

On 2026-09-23 (rental 46) three registrations of *one program* — the group-128 int4
dequantise-GEMV on the tied LM head — ran in a single process on a single card. They
compute the same function, and the record says so: layer 2 returned **0.9318 agreement and
0.01674 nats** for every one of them, as it has on four rentals.

| | how the program reaches `torch.compile` | ratio | IQR |
|---|---|---:|---:|
| `054-int4-head` | a hand-written Triton kernel behind `torch.library.custom_op` | **0.9851** | 0.0099 |
| `055-int4-head-triton-op` | the same kernel behind `torch.library.triton_op` | **errored** | — |
| `056-int4-head-torch-dequant` | torch operations, no kernel of ours at all | **1.0171** | 0.0130 |

**The generated code explains it, and refutes what we predicted.** The slot with no kernel
was registered in advance as a predicted **loss**, on the argument that inductor would
materialise the head's 1271.40 MB/token bf16 weight. It does not. Its final kernel is one
reduction carrying `__rshift__`, `bitwise_and`, the `mm`, the final RMSNorm and the residual
add *together* — the entire grouped nibble unpack fused into the matmul prologue — and the
graph holds **no weight-sized buffer anywhere**, allocating 48 buffers where the reference
allocates 59.

So the hand-written kernel lost twice over: a custom op cannot keep the RMSNorm fusion the
reference welds into the lm_head matmul, **and** the register-level unpack that was its
whole claim to necessity is something the compiler emits inline. `tiled_int4_head` is
retired. See [`results/batches/009-visible-kernels/`](results/batches/009-visible-kernels/).

**The other champion has the same shape.** On 2026-09-23 (rental 45) `045-inline-causal-conv`
— the four-tap causal convolution written as torch operations rather than as one opaque
call — returned **1.0765**, against **0.7854** for the identical arithmetic wrapped in a
custom op. Both were bit-identical to the reference (264/264, 0.00000 nats). It reproduced
at **1.0650** on rental 46, with the candidate column reaching **1254 GB/s, 70% of an RTX
5090's 1792 GB/s vendor peak.** One variable, 37%.

**The finding is the mechanism, not the number: a custom op costs its own kernel plus
everything the compiler can no longer fuse around it, and that bill is invisible at the call
site.** It is also what settled a 20% regression two rentals had blamed on composing two
kernels — there was no composition effect; the conv lost 21% on its own.
See [`results/batches/008-ingredients-and-barriers/`](results/batches/008-ingredients-and-barriers/).

**Where that leaves the thesis.** The premise of this project is that a hand-written Triton
kernel can beat what `torch.compile(mode="max-autotune")` generates, and the useful result
was always going to be *where the compiler wins and where it cannot*. Nine batches in, the
answer on this model's decode path is that the compiler wins wherever it can see the
program — and that the way to beat it is to stop hiding things from it.

**The retired kernel, and the four rentals it took to settle it.** `022-int4-head` — the
**tied LM head**, 248320 x 2560, *14.80% of every byte the compiled column moves, in a
single matmul*, stored at 4 bits with group-128 scales — returned **1.0791 at an
interquartile spread of 0.00034** on 2026-09-19 (rental 40), then **1.0161** (rental 43),
**1.0105, inconclusive** (rental 45) and finally **0.9851, a loss** (rental 46), the last
at an IQR of 0.0099 once the batch ran 15 scoring rounds instead of 5. The kernel never
changed and never returned a correctness result other than 0.9318 / 0.01674. What changed
is that the measurement got sharp enough to resolve it. **The site's 1.1249x byte ceiling
is arithmetic and stands — it is collected by inductor now.**

**The card is an uncontrolled variable and this file names the rental for every number.**
Two RTX 5090s reporting the same memory clock, driver and torch differed by **1.61x** on
the baseline, and one card lost 17% of its SM clock partway through a rental. Ratios
survive that — every slot times its own reference in the same interleaved rounds — but
absolute predictions and cross-rental comparisons do not.

`034-static-cache-cudagraphs` won on a third, unrelated mechanism — **1.0196** (rental 42)
and **1.0150** (rental 46), bit-identical — and it won *without* its stated mechanism
firing, which the record says out loud because the slot carries a CUDA-graph node count
beside its ratio. It is also worth **nothing** once the fused causal conv is installed:
two dispatch savings compete for the same microseconds, so this project's launch accounting
is not additive.

**Rental 42 also refuted the obvious next step, which is the more useful half.** If the
GEMV won on the head because the head has parallelism, a better *tile* should have unlocked
the other 83% of the bytes. The tile was then searched on the card — BLOCK_N, split-K,
BLOCK_K, warps, pipeline depth — and the search made the champion's own site **2.3x
slower** (656 → 282 GB/s, 1.0791 → 0.9920) while the MLP came back at **67 GB/s**, within
noise of the 65 measured two rentals earlier at a different tile in a different kernel.
The layer projections are not a tiling problem. **Rental 43 then pinned two more tiles on
the head in advance — BLOCK_N 128 at 8 warps, and `num_stages` 5 — and both lost (0.9343
and 0.9502); rental 45 pinned the last untried direction, BLOCK_N 32, and it lost too
(0.9617).** Five measured points now, four deliberate attempts, and the heuristic tile
nobody chose on purpose is the best of them. **The tile question is closed.** See
[`results/batches/006-tile-and-sites/`](results/batches/006-tile-and-sites/) and
[`results/batches/007-compose-and-retile/`](results/batches/007-compose-and-retile/).

**And a win plus a win is not a win.** `022` (int4 on the head) and `025` (a fused causal
convolution) each beat the compiler on mechanisms that share nothing. Composed, they
return **0.79-0.81** — 19% *slower* than the baseline — while launching **24 fewer kernels
than it**. The `output_code` dump of that exact pair says why: a hand-written kernel
installed as an opaque custom op is a **fusion barrier**, so inductor must materialise its
inputs and outputs (59 → 190 buffer allocations per decode step) and it splits a producer
chain it had been fusing, then recomputes the shared prologue rather than reading the
buffer it just wrote — the linear-attention state reduction runs **twice per layer, 24
times per token**. **A custom op costs its own kernel plus everything the compiler can no
longer fuse across it, and that second term is invisible at the call site.**

**The finding is not "our kernel is fast". It is where the compiler can be beaten and why
two earlier attempts could not find it.** Batches 003 and 004 installed a hand-written GEMV
on all 248 layer projections at once and lost by 5x — 0.2801, then 0.1934 after a rewrite.
Measured per site, the same kernel family gives **228-319 GB/s over the projections and
656 GB/s on the head**. The head launches 3880 programs on a 170-SM card; `in_proj_a` is 32
channels wide and launches four. The kernel was never structurally slow: **it was
grid-starved, and an aggregate number over 248 sites could not show that.**

The most useful number the project has produced is not a ratio. It is the baseline:

> **`torch.compile(mode="max-autotune")` runs this model's decode at 6.70-7.30 ms/token —
> 1177-1282 GB/s, 65.7-71.5% of an RTX 5090's 1792 GB/s vendor peak** (two different
> physical cards), against a 4.79 ms/token roofline on the bytes it actually moves.

And the most useful thing rental 38 produced was not a ratio either. It was **reading the
code we are trying to beat**, which had never been done:

* **Inductor already eliminates the GQA head expansion** — `x1 // 4` on the unexpanded KV
  cache, no materialisation — which closes the backlog's second-largest hypothesis (6.23% of
  per-token bytes) without a kernel, and corrects the baseline's traffic from the roofline's
  9158.23 MB/token to the **8587.80** it actually moves.
* **There is no cuBLAS in the decode path.** `extern_kernels` is called for convolution and
  nothing else: inductor generates Triton for every matmul, and it welds the residual add and
  the RMSNorm *into* them. A hand-written GEMV gives all of that fusion up, at 248 projection
  sites per token.
* **The compiled baseline runs with no CUDA graphs**, because the causal conv mutates its
  cache in place.

That reframes the whole exercise. The compiler is two thirds of the way to the memory wall
*and* fusing everything around the matmuls it generates, so the only large win left is to
move fewer bytes — and taking a matmul away from inductor has a measured price attached.

**Rental 40 is where that price became payable.** At the LM head it is paid once, on a
matmul so large that a hand-written kernel finally has the card to itself. At the 248 layer
projections it is paid 248 times, on matmuls as narrow as 32 channels. Same kernel, same
arithmetic, opposite result — which is the mechanistic account this project exists to
produce, and it was registered before the measurement rather than after it.

Batch 003 tried exactly that — weight-only int8 and int4 with the dequantisation fused into
the GEMV's K-loop, attacking the 91.85% of per-token bytes that are weights — and produced
a clean negative result with its cause attached. **The kernels were never bandwidth-bound:
as they removed bytes they got slower**, 26.90 → 38.68 → 42.49 ms/token while traffic fell
9158 → 5588 → 2850 MB/token. A cross-lane reduction ran once per K-iteration where it should
run once per output, and the dequantisation landed on an already-saturated issue port. The
1.85× ceiling is real arithmetic and unreachable by an implementation that is not spending
its time on memory in the first place.

Two things that only a control slot could have established, and both change the backlog:

* **Weight-only int8 written in plain PyTorch costs the same as bf16** — 0.9893, IQR 0.0212.
  This repository had asserted for two weeks that inductor materialises the dequantised
  weight and is therefore *slower*. That story needs 2525 GB/s on a 1790 GB/s card, so it is
  refuted by arithmetic alone.
* **None of the six `incorrect` verdicts was a kernel bug.** Every Triton kernel passed its
  numerics gate at every probe, worst relative error 7.8e-3 — one bf16 ULP. The failures were
  three defects in the gates themselves, including a threshold set finer than the statistic
  could resolve: one slot missed its bar by eight hundredths of a single token.

There are no estimated, placeholder or illustrative numbers anywhere in this repo. Every
number here came off a real card, and the ones that do not mean what they appear to mean say
so. `LEADERBOARD.md` and `results/batches/003-int8-weight-only/README.md` carry the detail.

## What is being claimed

That a hand-written Triton kernel can beat what `torch.compile(mode="max-autotune")`
generates on an LLM decode path — and, more importantly, that we can say **where**, **by
how much**, and **why**, *in advance*.

The second half is the actual contribution. "Hand-written beats the compiler" is the
premise the entire inference-serving industry is built on; re-proving it is not a finding.
A correct, mechanistic account of where the compiler wins and where it structurally cannot
— registered as a prediction before the measurement, then confirmed by it — is.

That account starts from arithmetic, not intuition. `docs/roofline.py` prints where every
byte goes in one decode step:

| What moves | share of per-token bytes |
|---|---:|
| Weights, streamed once | **91.85%** |
| GQA `repeat_interleave` materialisation | **6.23%** |
| Recurrent state | 1.10% |
| KV cache read | 0.78% |
| SwiGLU intermediates | 0.026% |
| Norm + residual | 0.018% |

**A hypothesis cannot beat the share of bytes it touches.** Batch-1 decode is a
weight-streaming problem, and inductor already reaches the roofline on simple memory-bound
work — it emits Triton, so hand-writing a fused RMSNorm means hand-writing the kernel it
already generates, for a ceiling of 0.018%. Five hypotheses are in the graveyard because of
this table, closed by arithmetic rather than by renting a GPU. `docs/HYPOTHESES.md` ranks
what is left.

The win condition is a **median ratio** `t_compiled / t_candidate` over interleaved rounds,
not an absolute millisecond count. Every session rents a different physical GPU, so absolute
times are provenance, never the score.

Interleaving reference and candidate *within* each round is what cancels thermal drift and
clock changes, so both models must be resident at once. They are separate module trees —
installing a kernel swaps a class on the candidate's modules, and a shared tree would alter
the reference — but they **share one set of parameter tensors**, since nothing writes to a
weight under `no_grad`. That is 8.4 GB on the card rather than 16.8, and one read of the
checkpoint rather than two. `cli._assert_parameters_are_shared` fails the run if that ever
silently stops being true.

Four columns are reported by default; `--columns all` adds a fifth:

| Column | What it is | Role |
|---|---|---|
| `eager` | `reference.py` in eager PyTorch | Context: shows how much is Python overhead |
| `compiled` | `reference.py` under `torch.compile(mode="max-autotune")` | **The win condition** |
| `compiled_nocudagraphs` | the same, `mode="max-autotune-no-cudagraphs"` | **Opt-in.** A debugging column: both scored columns already have CUDA graphs, so this only earns its compile when a result is confusing or the hypothesis is about launch overhead |
| `candidate` | `reference.py` with Triton kernels substituted, eager | Diagnostic: the gap to `candidate_compiled` is the compiler's contribution |
| `candidate_compiled` | the candidate under `max-autotune` | **The scoring column.** Same treatment as `compiled`, so the only difference is who wrote the kernel |

## What is *not* being claimed

These are non-goals, stated up front because a knowledgeable reader will ask:

- **Beating cuBLAS on dense bf16 GEMM.** We will not win there and this README says so.
  At batch 1 the linear layers are already at the bandwidth roofline; you cannot beat a
  roofline with a better kernel, only by moving fewer bytes (quantisation) or running a
  different algorithm (a chunked recurrent scan). That is what the backlog targets.
- **Beating a compiler at elementwise fusion.** Inductor's home turf, and it emits Triton.
  Those hypotheses are in the graveyard with the arithmetic that closed them.
- **Beating vLLM or SGLang.** Those are already hand-tuned Triton and CUDA. They are
  out of scope as a win condition and may later appear only as an unscored reference point.
- **Training kernels.** Inference decode path only.
- **Multi-GPU.** Single card throughout.
- **Building a serving engine.** No batching scheduler, no API server, no continuous batching.

## What is and is not benchmarked

Qwen3.5-4B (`Qwen3_5ForConditionalGeneration`) ships three things in one checkpoint.
DeltaForge benchmarks exactly one of them:

| Component | Params | Status |
|---|---|---|
| Text decoder (`model.language_model.*`) | ~4.21 B | **Benchmarked.** This is the subject. |
| Vision tower (`model.visual.*`) | ~0.33 B | **Excluded.** Text-only decode; never instantiated. |
| Multi-token-prediction head (`mtp.*`) | ~0.12 B | **Excluded.** See below. |

The **vision tower** is cleanly separable: it is configured entirely by the top-level
`vision_config` and shares no tensor with `text_config`. Text-only decode builds from
`text_config` alone.

The **MTP head** is excluded for a methodological reason, not a convenience one. It is a
speculative-decoding head: including it would change how many tokens come out per forward
pass, which makes a decode-latency number incomparable with one produced without it.
Benchmarking it would quietly turn "our kernels are faster" into "we enabled speculative
decoding". Both exclusions are enforced in `src/deltaforge/weights.py`, which reports every
tensor it skipped.

## The baseline is ours, on purpose

The baseline is `src/deltaforge/reference.py` — a pure-PyTorch implementation of the
decode forward pass containing **no custom kernels of any kind**.

HuggingFace's own Qwen3.5 modeling code dispatches the Gated DeltaNet layers to
hand-written Triton via `flash-linear-attention`, and its attention to FlashAttention.
Benchmarking against that would compare hand-tuned Triton to hand-tuned Triton while
claiming to beat a compiler. The claim would be false and a knowledgeable reader would
catch it in a minute. HuggingFace `transformers` is used only to load weights, tokenize,
and act as a **correctness oracle**. It is never the baseline.

For the same reason `reference.py` does not call `torch.nn.functional.scaled_dot_product_attention`:
SDPA *is* the fused attention kernel we are trying to hand-write. See the module docstring
for the exact line drawn between "plain PyTorch primitive" and "the fused algorithm under test".

## Correctness

A fast kernel that is wrong is worse than no kernel. Two gates, both required:

1. **Per kernel** — `allclose` against the reference operation at bf16 tolerance
   (`rtol=1e-2`, `atol=1e-2`) on real model shapes. Max absolute and max relative error
   are **recorded as numbers**, never reduced to a bare pass/fail: the magnitude is
   informative even when the gate passes.
2. **End to end** — greedy-decode 128 tokens from five fixed prompts (checked into the
   repo, at `src/deltaforge/harness/prompts.py`). The candidate's token sequence must
   match eager exactly. This catches the case where every kernel passes in isolation but
   the assembled pipeline accumulates drift.

Failures are recorded, not discarded. A candidate that was fast but wrong goes into
`results/` and the graveyard with its error magnitudes.

## Cost control

Budget is **$50/month**, enforced by code rather than by discipline:

- Month-to-date gate: provisioning refuses at ≥ $45.
- Session GPU-time soft gate: 180 cumulative billed minutes, checked *before* a run starts
  and never during one. Three hours because one hypothesis on a cold compile cache does not
  fit in two: a single `max-autotune` compile has been observed to take ~40 minutes.
- **A pre-flight refusal**: provisioning is refused outright when the remaining budget
  cannot fit the identity slot plus one kernel slot. Not renting beats renting to produce
  nothing, which is how the first nine rentals were spent.
- Per-slot wall-clock cap, so a slot that runs away inside a compile costs a slot rather
  than the rental. The first two slots are exempt — capping them would defeat the guarantee.
- Hard watchdog: a local-side process destroys the instance at 210 minutes regardless of
  what the remote is doing — always above the gate, so the routine control is the gate and
  a watchdog firing stays a fault. A remote that hangs cannot defeat its own kill switch.
- A batch stops *itself* before a hypothesis it cannot finish before the session deadline,
  so the watchdog never has to. A watchdog firing is treated as a reportable fault.
- Teardown pulls results back **before** destroying the instance, so a failure late in a
  long batch does not take the slots that already succeeded with it.
- Trap-based teardown: the whole remote run sits inside a shell trap on `EXIT`/`INT`/`TERM`,
  so a crash still destroys the instance.
- The spend row is written to `ledger/spend.jsonl` *before* the instance is used, then
  reconciled with actuals at destroy.
- The compile cache is pulled off the box before it is destroyed and pushed back to the
  next rental on the same card, so a cold `max-autotune` compile is paid once
  rather than every session. It changes how long compilation takes, not what it emits.
  **Now demonstrated.** Rental 34 completed the first cold `max-autotune` compile this
  project has ever finished, in **267.5 s**, off the 47 MB rental 32 banked; rental 35 then
  did it in **57.4 s** off the 298 MB rental 34 brought home. Teardown reports warmth from
  what actually landed on disk, because a log that overstates it would corrupt the very
  number the cache is meant to improve.

Every script in `remote/` supports `--dry-run`, which exercises the full logic path —
including all the gates and the teardown trap — without contacting the create or destroy
endpoints. That is how the cost machinery is verified with no money at risk.

## One rental, many hypotheses

A rental's fixed cost — container image, torch, a 9.32 GB checkpoint, the GPU test suite,
and one `max-autotune` compile of the reference — is about nineteen minutes, measured.
Each additional hypothesis costs three to six. Testing one hypothesis per rental therefore
pays nineteen minutes of setup to buy three of measurement.

So a rental measures a **batch of 7–12 hypotheses**. The reference is compiled once and
kept; each candidate is built, gated, benchmarked and freed in turn. Only the *compilation*
is amortised — the reference column is re-timed inside every hypothesis's own interleaved
rounds, so per-round drift still divides out exactly as it does for a single measurement.

Every slot runs in its own try/except and writes its record the moment it finishes, so a
wrong kernel costs a slot rather than the rental. Each batch opens with an **identity
champion** that installs nothing: it must measure 1.00 ± noise, and if it does not, every
other number in that batch is void and the writeup says so.

Each hypothesis registers its **predicted outcome and reasoning in the manifest, committed
before the rental**, and the batch summary scores every prediction against what was
measured. That scorecard is the point: this project's claim is that a mechanistic account
stated in advance survives contact with the measurement, and until batches arrived nothing
here recorded a prediction anywhere it could be checked.

Details in `docs/BATCHES.md`.

## Reproduction

```sh
# CPU-only checkout: everything except the GPU steps.
uv sync --extra dev
uv run pytest

# Where the bytes go, and therefore what is worth optimising. No GPU, no checkpoint.
uv run python docs/roofline.py

# Verify the cost machinery without spending anything.
remote/run_remote.sh --dry-run --session-id smoke --batch 001-calibration

# Read what the compiler does with a candidate, before renting a card to ask.
uv run python -m deltaforge.cli fusion --batch 009-visible-kernels     # which slots install a barrier
uv run python -m deltaforge.cli fusion --install inline_causal_conv    # compile here, diff the code
```

The full run — provision, sync, correctness gates, benchmark, pull results, destroy —
is one command, and requires a funded Vast.ai account and `VAST_API_KEY` in a gitignored
`.env` at the repo root:

```sh
remote/run_remote.sh --session-id "$(date -u +%Y%m%dT%H%M%SZ)" --batch 001-calibration
```

## Repository map

| Path | What it is |
|---|---|
| `src/deltaforge/reference.py` | The baseline. Pure PyTorch, no custom kernels, ever. |
| `src/deltaforge/weights.py` | Safetensors → reference model, with an explicit name map. |
| `src/deltaforge/kernels/` | The Triton kernels and the champion registry. |
| `src/deltaforge/harness/` | Interleaved timing, correctness gates, results records. |
| `remote/` | Instance lifecycle, in POSIX shell so it works before the env exists. |
| `docs/roofline.py` | Where the bytes go. Run before picking a hypothesis; needs no GPU. |
| `docs/ARCHITECTURE.md` | The resolved Qwen3.5-4B facts every kernel must honour. |
| `docs/HYPOTHESES.md` | The idea backlog, and the graveyard of what failed and why. |
| `docs/BATCHES.md` | How a batch works, and what filling one requires. |
| `src/deltaforge/fusion.py` | What inductor generated, parsed; which registrations are opaque. No GPU. |
| `src/deltaforge/batches.py` | The batch manifests, predictions registered in advance. |
| `AGENT.md` | **Start here.** The single entry point for a working session. |
| `LEADERBOARD.md` | The current champion, and every hypothesis attempted. |
| `RESULTS.md` | The write-up: the result, the findings and the method, for a reader who was not here. |
| `site/index.html` | The same write-up as one standalone page, served at [lovemanna.github.io/deltaforge](https://lovemanna.github.io/deltaforge/). |

Licensed Apache-2.0, matching the target model.
