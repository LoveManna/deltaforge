# DeltaForge

Hand-written Triton kernels for the **Qwen3.5-4B** decode path, measured against
`torch.compile(mode="max-autotune")` running an identical pure-PyTorch reference.

Every attempt is recorded — the wins and the losses — so that independent working
sessions compound instead of rediscovering the same dead ends.

---

## Headline result

**None yet.**

This repository is at the end of its bootstrap session. The harness, the reference
implementation, the correctness gates, the provisioning machinery and the cost controls
are built and tested. **No baseline has been recorded and no kernel has been written.**

The bootstrap session had no GPU available, so every number-producing step is wired,
marked deferred, and skipped rather than faked. `LEADERBOARD.md` and `results/` say the
same thing. There are no estimated, placeholder, or illustrative numbers anywhere in
this repo; when a number appears here it will have come off a real card.

## What is being claimed

That a human writing Triton by hand can beat what an optimising compiler generates,
on the operations where fusion and memory movement dominate — measured end to end on
a decode workload, not on a microbenchmark chosen to flatter the result.

The win condition is a **median ratio** `t_compiled / t_candidate` over interleaved
rounds, not an absolute millisecond count. Every session rents a different physical
GPU, so absolute times are provenance, never the score.

Four columns are reported on every run:

| Column | What it is | Role |
|---|---|---|
| `eager` | `reference.py` in eager PyTorch | Context: shows how much is Python overhead |
| `compiled` | `reference.py` under `torch.compile(mode="max-autotune")` | **The win condition** |
| `compiled_nocudagraphs` | the same, `mode="max-autotune-no-cudagraphs"` | Shows the win is not just launch overhead |
| `candidate` | `reference.py` with Triton kernels substituted | The submission |

## What is *not* being claimed

These are non-goals, stated up front because a knowledgeable reader will ask:

- **Beating cuBLAS on dense GEMM.** We will not win there and this README says so.
  Wins are expected in memory-bound operations, in fusion across operation boundaries,
  and in the chunked recurrent scan — not in matrix multiply.
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
- Session GPU-time soft gate: 60 cumulative billed minutes, checked *before* a run starts
  and never during one.
- Hard watchdog: a local-side process destroys the instance at 90 minutes regardless of
  what the remote is doing. A remote that hangs cannot defeat its own kill switch.
- Trap-based teardown: the whole remote run sits inside a shell trap on `EXIT`/`INT`/`TERM`,
  so a crash still destroys the instance.
- The spend row is written to `ledger/spend.jsonl` *before* the instance is used, then
  reconciled with actuals at destroy.

Every script in `remote/` supports `--dry-run`, which exercises the full logic path —
including all the gates and the teardown trap — without contacting the create or destroy
endpoints. That is how the cost machinery is verified with no money at risk.

## Reproduction

```sh
# CPU-only checkout: everything except the GPU steps.
uv venv
uv pip install --index-url https://download.pytorch.org/whl/cpu torch
uv pip install -e ".[dev]"
uv run pytest

# Verify the cost machinery without spending anything.
remote/run_remote.sh --dry-run
```

The full run — provision, sync, correctness gates, benchmark, pull results, destroy —
is one command, and requires a funded Vast.ai account and `VAST_API_KEY` in a gitignored
`.env` at the repo root:

```sh
remote/run_remote.sh --session-id "$(date -u +%Y%m%dT%H%M%SZ)"
```

## Repository map

| Path | What it is |
|---|---|
| `src/deltaforge/reference.py` | The baseline. Pure PyTorch, no custom kernels, ever. |
| `src/deltaforge/weights.py` | Safetensors → reference model, with an explicit name map. |
| `src/deltaforge/kernels/` | The Triton kernels and the champion registry. Currently empty of kernels. |
| `src/deltaforge/harness/` | Interleaved timing, correctness gates, results records. |
| `remote/` | Instance lifecycle, in POSIX shell so it works before the env exists. |
| `docs/ARCHITECTURE.md` | The resolved Qwen3.5-4B facts every kernel must honour. |
| `docs/HYPOTHESES.md` | The idea backlog, and the graveyard of what failed and why. |
| `AGENT.md` | What each new working session reads first. |
| `LEADERBOARD.md` | The current champion, and every hypothesis attempted. |

Licensed Apache-2.0, matching the target model.
