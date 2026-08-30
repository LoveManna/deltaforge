# DeltaForge — Design Spec

**Date:** 2026-08-29
**Status:** Approved, pending implementation plan
**Repo:** `~/github/deltaforge` (private)

---

## 1. Goal

Produce hand-written Triton kernels for the **Qwen3.5-4B** decode path that measurably outperform
`torch.compile(mode="max-autotune")` running an identical pure-PyTorch reference, on a rented modern
NVIDIA GPU.

Every attempt — successful or not — is recorded so that independent working sessions compound on each
other's results instead of rediscovering the same dead ends.

The output is two things at once: a genuine performance result, and a legible artifact that a hiring
manager can read in ten minutes and believe.

## 2. Success criteria

1. A reproducible end-to-end decode benchmark showing hand-written Triton beating
   `torch.compile(mode="max-autotune")` by a margin large enough to survive measurement noise
   (target: ≥1.25× median ratio on the headline configuration).
2. Every kernel that ships passes a two-layer correctness gate against the reference implementation.
3. A leaderboard that states the current champion unambiguously, plus a graveyard of failed
   hypotheses with the reason each failed.
4. Total GPU spend stays under $50/month, enforced by code rather than by discipline.
5. The repo can be flipped from private to public without a rewrite.

## 3. Non-goals

- **Beating cuBLAS on dense GEMM.** We will not win there and the README will say so. Wins live in
  memory-bound operations, fusion, and the chunked recurrent scan.
- **Beating vLLM or SGLang.** Those are already hand-tuned Triton and CUDA. Out of scope as a win
  condition; may appear as an unscored reference point later.
- **Training kernels.** Inference decode path only.
- **Multi-GPU.** Single card throughout.
- **The vision tower.** Qwen3.5-4B appears to be multimodal; we benchmark text-only decode.
- **Building a serving engine.** No batching scheduler, no API server, no continuous batching.

## 4. Target model: Qwen3.5-4B

Apache 2.0, released March 2026. Chosen over the safer Qwen3-8B and the harder gpt-oss-20b for
specific reasons:

**Architecture.** 32 layers, hidden dim 2560, arranged as 8 repetitions of
`3 × (Gated DeltaNet → FFN)` followed by `1 × (Gated Attention → FFN)`. So roughly 75% of layers use
linear attention and 25% use full softmax attention with GQA (16 query heads, 4 KV heads).
RMSNorm, RoPE, SwiGLU. Native context 262,144 tokens.

**Why this model.** The chunked delta-rule recurrence inside Gated DeltaNet is a sequential scan with
matrix-valued state. A compiler cannot restructure that into an efficient chunked parallel form on its
own — that restructuring is the hand-tuning. This gives a large, honest win margin rather than a 5%
squeak, and it sits on current-generation architecture rather than re-treading fused RMSNorm on a
2024-era dense model, which Liger Kernel and unsloth already own.

At bf16 the weights are roughly 8–9 GB, leaving substantial headroom on a 32 GB card for batch and
context sweeps.

**Unknowns to resolve in the bootstrap session:** whether the 4B variant's FFN is dense or sparse-MoE
(sources conflict), and confirming the vision tower can be cleanly excluded from the decode path.

## 5. Baseline definition

This is the single most important decision in the project, because it determines whether the headline
claim survives scrutiny.

**The baseline is `src/deltaforge/reference.py`** — a single-file, pure-PyTorch implementation of the
Qwen3.5-4B decode forward pass, containing no custom kernels of any kind. HuggingFace `transformers`
is used only to load weights, tokenize, and act as a correctness oracle. It is never the baseline.

**Rationale.** HF's own Qwen3.5 modeling code very likely dispatches to hand-written Triton kernels for
the Gated DeltaNet layers via the `flash-linear-attention` library. Benchmarking against that would
compare hand-tuned Triton to hand-tuned Triton while claiming to beat a compiler. The claim would be
false and a knowledgeable reader would catch it immediately.

**Three columns are reported on every run:**

| Column | What it is | Role |
|---|---|---|
| `eager` | `reference.py` in eager PyTorch | Context; shows how much is Python overhead |
| `compiled` | `reference.py` under `torch.compile(mode="max-autotune")` | **The win condition** |
| `candidate` | `reference.py` with Triton kernels substituted | The submission |

A fourth column, `compiled_nocudagraphs` (`mode="max-autotune-no-cudagraphs"`), is also recorded so a
reader can verify the candidate is not merely winning on kernel-launch overhead.

## 6. Benchmark methodology

**The core constraint:** every session provisions a *different physical GPU*. Absolute milliseconds are
therefore not comparable across sessions and must never be the win condition. Clock locking
(`nvidia-smi -lgc`) generally requires privileged container access we will not reliably have, so the
methodology cannot depend on it.

**Interleaved A/B/A measurement.** Within a single process on a single card:

```
for round in 1..R:
    t_eager[round]        = time(eager)
    t_compiled[round]     = time(compiled)                # max-autotune
    t_compiled_ncg[round] = time(compiled_nocudagraphs)   # max-autotune-no-cudagraphs
    t_candidate[round]    = time(candidate)
```

Rounds 1–2 are discarded as warmup. The reported score is
`median(t_compiled[r] / t_candidate[r])` over the remaining rounds. Interleaving cancels slow thermal
drift and noisy-neighbour effects that would corrupt a sequential "measure A ten times, then B ten
times" design.

**Timing.** CUDA events with explicit `torch.cuda.synchronize()` around each measured region.
`R = 7` by default.

**Configurations.** Headline is single-stream latency, batch 1 / context 2048 / 128 decoded tokens —
the memory-bound regime where fusion wins are real. A secondary batch-32 configuration is recorded to
show behaviour as the workload becomes compute-bound.

**Absolute numbers are recorded** alongside the ratio, tagged with GPU model, driver version, observed
clocks, and torch/Triton versions. They are evidence and provenance, never the score.

**The reigning champion is re-benchmarked every session.** A stored number from a previous session's
hardware is never trusted or carried forward as a comparison point.

## 7. Correctness gates

A fast kernel that is wrong is worse than no kernel. Two layers, both required.

**Layer 1 — per kernel.** `torch.allclose` against the corresponding reference operation at bf16
tolerance (`rtol=1e-2`, `atol=1e-2`), on shapes drawn from the real model configuration. Max absolute
error and max relative error are recorded numerically, not reduced to pass/fail — the magnitude is
informative even when the gate passes.

**Layer 2 — end to end.** Greedy-decode 128 tokens from five fixed prompts. The candidate's token
sequence must match the eager reference exactly. This catches the case where each kernel passes in
isolation but the assembled pipeline accumulates drift.

**Failures are recorded, not discarded.** A candidate that was fast but incorrect is written to the
results directory and the graveyard with its error magnitudes. That is among the most valuable things
a future session can read.

## 8. Cost control

Budget is **$50/month**, sessions provision autonomously, and an agent that forgets to destroy an
instance burns roughly $13/day. Guardrails are therefore structural.

1. **Month-to-date gate.** `remote/provision.sh` reads `ledger/spend.jsonl` and refuses to provision if
   month-to-date spend is ≥ $45, leaving headroom under the ceiling.
2. **Session GPU-time gate (soft, 60 minutes).** Before starting any new hypothesis run — whether that
   means provisioning a fresh instance or launching another run on a live one — the session sums the
   billed instance minutes it has already consumed, read from `ledger/spend.jsonl`. At 60 cumulative
   minutes the session **may not start another hypothesis**. It destroys any live instance, then
   finishes recording, writing up, and merging the work it already completed — none of which needs a
   GPU — and ends.

   The check happens *before* a run starts, never during one. A benchmark already executing at minute
   59 finishes normally rather than being killed halfway, which would waste the money already spent on
   it.

   This is the gate expected to fire in ordinary operation. It exists to kill the "just one more
   attempt" pattern that otherwise runs a session into the hard watchdog with work in flight.
3. **Per-instance hard watchdog (90 minutes).** A watchdog process runs *on the local machine*, not the
   rented box, and destroys the instance after 90 minutes regardless of what the remote is doing. A
   remote that hangs cannot defeat its own kill switch. A remote-side `shutdown` timer is set as a
   second line of defence. This is a backstop, not a routine control: if it ever fires, something went
   wrong and the session must report that rather than treat it as a normal ending.
4. **Per-session dollar cap.** ~$2 per session, independent of the clock, as protection against an
   unexpectedly expensive fallback card. At the $0.60/hr ceiling the 60-minute gate normally binds
   first, at roughly $0.60 per session.
5. **Trap-based teardown.** The whole remote run executes inside a shell trap on `EXIT`/`INT`/`TERM`, so
   a crash, a failed benchmark, or an interrupted session still destroys the instance.
6. **Ledger written before use.** The spend row is appended at provision time with the hourly rate and
   an estimated ceiling, then reconciled with actuals at destroy time. A session that dies mid-run
   still leaves a record of what it started.
7. **Instance filters.** On-demand only (never interruptible — a benchmark that dies mid-sweep costs
   more than it saves), host reliability > 0.98, single GPU, hourly rate ceiling.

## 9. Remote GPU: provider and instance selection

**Provider: Vast.ai.** Cheapest verified marketplace with root access inside the container, which we
need for profiling and for any clock manipulation that turns out to be available.

**Primary target: RTX 5090** — 32 GB, ~1.8 TB/s memory bandwidth, Blackwell `sm_120`. Roughly
$0.45–0.60/hr against a cross-provider median of $0.54. The bandwidth matters: memory-bound kernel
wins are most visible on the fastest memory.

**Fallback: RTX 4090** — 24 GB, Ada `sm_89`, roughly $0.35–0.50/hr (median $0.43). Used when no 5090
meets the filters.

**Explicitly rejected:**

- **Salad at $0.18/hr** — a network of idle consumer PCs. Cheapest per hour, useless for reproducible
  timing.
- **Interruptible instances at $0.29–0.31/hr** — a run that dies mid-sweep wastes more than it saves.
- **H100 at ~$2/hr** for routine work. Worth a small number of hours late in the project to produce a
  second results column on Hopper, not for iteration.

**Software.** A Vast template with PyTorch 2.11 + CUDA 12.8 preinstalled, to avoid paying for a long
dependency install on every provision. Model weights (~9 GB) download from HuggingFace per instance;
Qwen3.5-4B is ungated so no token is required.

**Credential required from the captain:** a Vast.ai API key. Nothing real runs without it.

## 10. Repository layout

```
deltaforge/
  README.md                   public-facing: headline number, chart, reproduction steps
  AGENT.md                    instructions each new working session reads first
  LEADERBOARD.md              current champion + every hypothesis attempted
  pyproject.toml              uv-managed dependencies
  src/deltaforge/
    reference.py              pure-PyTorch Qwen3.5-4B decode path (the baseline)
    weights.py                HF weight loading into the reference module
    model.py                  assembles reference + registered champion kernels
    kernels/
      __init__.py             registry: name -> (impl, replaces, status)
      <kernel>.py             one module per hand-tuned kernel
    harness/
      bench.py                interleaved A/B/A timing, median ratios
      correctness.py          per-kernel allclose + end-to-end token match
      report.py               emits results JSON + markdown fragment
  remote/
    provision.sh              vast.ai search + create, with hard filters and budget gate
    watchdog.sh               local-side wall-clock auto-destroy
    sync.sh                   rsync repo up, results down
    run_remote.sh             one-shot: provision -> sync -> validate -> bench -> pull -> destroy
  results/
    baseline/<gpu>-<date>.json
    hypotheses/<NNN>-<slug>.json
  ledger/
    spend.jsonl               append-only cost record
  docs/
    HYPOTHESES.md             idea backlog + graveyard of what failed and why
    superpowers/specs/        design specs (this file)
```

## 11. Component boundaries

Each unit has one purpose, a defined interface, and explicit dependencies.

**`reference.py`** — *What:* the Qwen3.5-4B decode forward pass in plain PyTorch. *Interface:*
`ReferenceModel(config).forward(input_ids, cache) -> logits, cache`. *Depends on:* torch only.
*Invariant:* contains no custom kernels, ever. This file is the definition of the baseline and changes
to it invalidate stored results.

**`kernels/`** — *What:* one Triton kernel per module, each a drop-in replacement for a named reference
operation. *Interface:* every module exports a callable with the reference operation's exact signature,
plus metadata declaring which reference operation it replaces. *Depends on:* triton, torch.
*Registry:* `kernels/__init__.py` maps name → implementation, and marks exactly one champion per
replaceable operation.

**`harness/bench.py`** — *What:* interleaved timing. *Interface:* takes a list of named callables and a
configuration, returns per-round timings and median ratios. *Depends on:* torch. Knows nothing about
Qwen, Triton, or the registry — it times opaque callables.

**`harness/correctness.py`** — *What:* both gate layers. *Interface:* takes reference and candidate,
returns a structured pass/fail with error magnitudes. *Depends on:* torch, reference.

**`remote/`** — *What:* instance lifecycle and file transport. Pure shell, no Python dependency, so it
works before the environment is built. *Interface:* `run_remote.sh` is the only entry point a session
calls; everything else is internal to it.

## 12. Session workflow

Codified in `AGENT.md`. Each fresh working session:

1. Read `LEADERBOARD.md` — current champion, and the graveyard of what has already failed.
2. Pick one unexplored hypothesis from `docs/HYPOTHESES.md`, or form a new one. **State it in one
   sentence before writing any code**, including what mechanism is expected to produce the win.
3. Branch `hyp/NNN-slug`.
4. Write the Triton kernel and its correctness test locally. No GPU needed for this step.
5. **Check the session GPU-time gate.** If this session has already consumed 60 cumulative billed
   minutes, do not start another hypothesis: destroy any live instance, finish recording and merging
   whatever is already complete, and end the session cleanly.
6. Run `remote/run_remote.sh`, which provisions, syncs, runs correctness gates, benchmarks, pulls
   results back, and destroys the instance.
7. Record the outcome in `results/hypotheses/` and `LEADERBOARD.md` — **win or lose**. A loss updates
   the graveyard with the mechanism that failed and why.
8. Promote to champion in the registry only if the candidate's median ratio exceeds the incumbent's
   by more than the interquartile spread of the scoring rounds. A margin inside the noise band is
   recorded as *inconclusive*, not as a win — it neither promotes nor enters the graveyard, and the
   hypothesis stays open for a cleaner measurement.
9. Open a PR whose body is the writeup; self-merge once gates are green and the ledger is updated.
10. Confirm the spend ledger reflects the run.

**Session 1 is a bootstrap and ships no hypothesis.** It builds the harness, the reference
implementation, the provisioning and cost machinery, and establishes the recorded baseline. Attempting
a kernel in the same session as the harness that measures it produces a broken harness and a
meaningless number.

## 13. Record formats

**`LEADERBOARD.md`** — a champion block stating the current best ratio, the kernel that achieved it,
and the GPU it was last verified on; followed by a table of every hypothesis with its ID, one-line
description, ratio, and outcome.

**`results/hypotheses/<NNN>-<slug>.json`** — hypothesis ID and statement; git SHA; GPU model, driver,
torch and Triton versions; per-round raw timings for all four columns; median ratios; correctness
results with error magnitudes; instance cost and duration; outcome.

**`ledger/spend.jsonl`** — one append-only row per instance: timestamp, instance ID, GPU model, hourly
rate, estimated ceiling at provision, actual minutes and cost at destroy, and the session or hypothesis
that spent it.

## 14. Hypothesis backlog (seed)

Ordered roughly by expected yield per GPU hour:

1. Fused RMSNorm + residual add.
2. Fused SwiGLU (gate/up projection with activation and multiply fused into the epilogue).
3. Fused QKV projection + RoPE.
4. Chunked delta-rule scan: block size, chunk length, and state layout tuning.
5. Flash decode for the GQA attention layers (the ~25% of layers using softmax attention).
6. KV-cache layout and gather strategy.
7. Fused MoE routing and grouped GEMM, if the 4B's FFN turns out to be sparse.
8. Persistent-kernel decode step, collapsing per-layer launches.

## 15. Risks and mitigations

| Risk | Mitigation |
|---|---|
| Dense GEMM is unbeatable vs cuBLAS | Stated as a non-goal in the README; hypotheses target memory-bound and fusible work |
| Cross-session hardware variance corrupts comparisons | Median ratio from interleaved same-process measurement; champion re-benchmarked every session |
| Runaway GPU spend | Local watchdog, exit traps, month-to-date gate, ledger written before use |
| Clock locking unavailable in container | Methodology never depends on it; interleaving absorbs drift |
| Nsight Compute counters blocked | Profiling leans on CUDA events and `torch.profiler`; `ncu` is a bonus, not a dependency |
| Reference implementation is subtly wrong | Validated against HF transformers as an oracle before any kernel work begins |
| Qwen3.5-4B architecture unknowns | Resolved in the bootstrap session before any kernel is written |
| $50/mo is only ~90 GPU-hours | The 60-minute soft gate caps a session near $0.60, giving roughly 75 sessions per month; kernels are written locally, so GPU time buys validation and measurement only |

## 16. Open items

- Vast.ai API key needed from the captain before the first real run.
- Qwen3.5-4B FFN: dense or sparse-MoE — resolve in bootstrap.
- Vision tower exclusion — confirm in bootstrap.
- No deadline set; roadmap is open-ended and can be reshaped backwards from one if the captain names it.

## 17. Decisions log

| Decision | Choice | Why |
|---|---|---|
| Model | Qwen3.5-4B | Current-gen hybrid architecture; chunked recurrence gives a large, honest win margin; small enough to iterate on one card |
| Baseline | Pure-PyTorch reference we write | HF's own code ships Triton kernels; using it would compare hand-tuned to hand-tuned |
| Win condition | Beat `torch.compile(mode="max-autotune")` | The only apples-to-apples reading of "beat what the compiler generates" |
| Metric | End-to-end decode headline + per-kernel diagnostics | Strong claim, with wins still attributable to a specific kernel |
| Score | Median ratio, not absolute ms | Every session runs on different hardware |
| Provider | Vast.ai, RTX 5090 on-demand | Cheapest verified provider with root; best bandwidth per dollar; on-demand avoids mid-sweep death |
| Tooling scope | Triton only | Matches the goal and keeps every hypothesis comparable |
| Correctness | Per-kernel allclose + end-to-end token match | Catches both a wrong kernel and a drifting pipeline |
| Git flow | PR per hypothesis, always merged | Visible failures are what make the log credible |
| Merge authority | Session self-merges on green gates | Keeps the loop unblocked |
| Spend autonomy | Autonomous within code-enforced caps | Prose guardrails do not stop a forgotten instance |
| Session GPU time | 60-min soft gate, 90-min hard watchdog | The soft gate fires routinely and preserves work in flight; the watchdog is a backstop for hangs and its firing is itself a reportable fault |
| Visibility | Private now, written for public | Flipping it public should be one click, not a rewrite |
