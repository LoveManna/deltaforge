# DeltaForge

Hand-written Triton kernels for the **Qwen3.5-4B** decode path, measured against
`torch.compile(mode="max-autotune")` running an identical pure-PyTorch reference.

Every attempt is recorded — the wins and the losses — so that independent working
sessions compound instead of rediscovering the same dead ends.

---

## Headline result

**No champion. Fifteen kernels have run on a GPU and the compiler has beaten all of them.**

On 2026-09-16 (rental 37) batch 003 returned **seven admissible ratios** — `calibrated:
true`, every slot compiled, nothing voided, for the first time in this project. The identity
champion measured **1.0024 with an IQR of 0.0190**. Every other slot lost.

The most useful number the project has produced is not a ratio. It is the baseline:

> **`torch.compile(mode="max-autotune")` runs this model's decode at 6.84 ms/token —
> 1308 GB/s, 73% of an RTX 5090's 1790 GB/s vendor peak**, against a 5.11 ms/token roofline.

That reframes the whole exercise. The compiler is already three quarters of the way to the
memory wall, so the headroom for *any* kernel is 1.37× from efficiency alone, and the only
large win left is to move fewer bytes.

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
| `src/deltaforge/batches.py` | The batch manifests, predictions registered in advance. |
| `AGENT.md` | **Start here.** The single entry point for a working session. |
| `LEADERBOARD.md` | The current champion, and every hypothesis attempted. |

Licensed Apache-2.0, matching the target model.
