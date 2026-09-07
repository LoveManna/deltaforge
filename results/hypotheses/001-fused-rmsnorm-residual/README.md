# Hypothesis 001 — fused residual-add + RMSNorm

**Status: GRAVEYARDED ON MECHANISM — written, gated, never measured, and now closed.**

No benchmark number exists for this kernel and none ever will. It has never executed on a
GPU. Nothing in this folder is a result, an estimate, or a projection.

> **Closed 2026-09-04, by arithmetic rather than by measurement.** The residual add and the
> two hidden-size norms per layer move **1.64 MB/token against 9158 MB total — 0.018%** of
> per-token bytes at batch-1 decode. An infinitely fast kernel here buys 0.018%, which is
> far below the harness's own noise band, so **no measurement could have shown a win.** The
> eight failed rentals below are incidental to the outcome. Reproduce the arithmetic with
> `uv run python docs/roofline.py`; the full reasoning is in `docs/HYPOTHESES.md`.
>
> The kernel and its tests stay in the tree, registered as `RETIRED`, so a later session
> finds the dead end already explored. The registry now has **no champion**, which is the
> identity-champion state a first GPU session should use to calibrate the harness.

| | |
|---|---|
| Hypothesis id | `001-fused-rmsnorm-residual` |
| Branch | `hyp/001-fused-rmsnorm-residual` |
| Replaces | `rms_norm`, `rms_norm_residual` |
| Outcome | **graveyarded on mechanism** — ceiling 0.018%, below the noise band |
| GPU time bought | 94.2 billed minutes across 8 rentals |
| Spend | $0.4783 |
| Numbers produced | none |

## The hypothesis, in one sentence

Fusing the residual add with the RMSNorm that immediately follows it should beat what
`torch.compile(mode="max-autotune")` generates for the same work, because at batch 1 the
decode path is entirely memory-bound and the fused kernel makes one pass over the hidden
state where the compiler's elementwise chain makes several.

## Mechanism

A reference decoder layer runs:

```
residual = h;  h = input_layernorm(h);           h = attn(h);  h = residual + h    # (A)
residual = h;  h = post_attention_layernorm(h);  h = mlp(h);   return residual + h # (B)
```

The add at (A) and the `post_attention_layernorm` that immediately follows it are the
fusable pair: the add's output is the norm's only consumer, and both are pure elementwise
passes over a `hidden_size`-wide row. Separately they read and write the hidden state
three times; fused they read it once and write it twice — the updated residual stream,
which later layers need, and the normalised activation.

Nothing here is arithmetic-limited. At batch 1 the row is 2560 elements — 5 KB at bf16 —
so the cost is traffic and launch overhead, not flops. That is the whole basis for
expecting a win, and it is also why this hypothesis is first: it is the cheapest to write
and the one that calibrates the harness.

## What was replaced, and what deliberately was not

Replaced: the two `hidden_size`-wide norms in every decoder layer, and the model's final
norm. `q_norm` and `k_norm` inside the attention layers are also RMSNorm, but they run on
a 256-wide head dimension in a different shape regime — leaving them to the reference
keeps any measured win attributable to one change.

Also not fused: a layer's final add (B) with the *next* layer's `input_layernorm`. That
pair straddles the layer boundary, and capturing it means threading a residual through
`ReferenceModel.forward`, which restructures the model. Per layer this replaces four
passes with three; capturing the boundary pair would make it two, and that is a separate
hypothesis rather than a detail of this one.

## Implementation notes worth carrying forward

* **Two custom ops, not raw kernels.** `deltaforge::add_rms_norm` and
  `deltaforge::rms_norm` are registered through `torch.library.custom_op` with fake
  implementations. That lets inductor schedule and CUDA-graph everything around them while
  forbidding it from decomposing them back into the ops this hypothesis exists to replace.
* **The add is rounded before the norm sees it.** `residual + hidden_states` in the
  reference produces a bf16 tensor which `RMSNorm.forward` then upcasts, so the kernel
  rounds the sum to the tensor dtype and reloads it. Normalising the unrounded fp32 sum
  would be *more* accurate than the reference and therefore a different function — the
  gate compares against the reference, not against infinite precision.
* **The scale is `1 + weight`.** Qwen3.5 stores norm weights centred on zero. A kernel
  reading them as a plain scale produces an all-zero activation on a fresh module and a
  subtly wrong one on real weights. There is a test for exactly this.
* **No PyTorch fallback.** Both ops raise on a non-CUDA tensor. Falling back would run the
  reference while the harness recorded the result as the candidate — the worst failure
  this harness could have.
* **Installation swaps `__class__` on the instance**, rather than binding a method onto
  `forward`. Dynamo then sees an ordinary module with an ordinary forward, which is what
  keeps the compiled candidate column free of graph breaks. A test asserts a second model
  built from the same classes is untouched.

## A harness change this hypothesis forced

The `candidate` column was built eager while `compiled` got `max-autotune` with CUDA
graphs. At batch-1 decode that compares Python dispatch across 32 layers, not kernels: the
candidate loses several-fold no matter how good the Triton is, and the resulting number
says nothing about the hypothesis.

A `candidate_compiled` column now gives the candidate the *same* max-autotune treatment,
leaving exactly one difference between it and `compiled` — who wrote the kernel. That is
what "beat what the compiler generates" has to mean. `candidate` stays as a diagnostic:
the gap between it and `candidate_compiled` is the compiler's contribution, and the gap
between `candidate_compiled` and `compiled` is the kernel's.

## What happened: eight rentals, no measurement

Every rental was destroyed on exit, every one reconciled in the ledger, and no instance
leaked. The money bought infrastructure findings rather than a number, and those findings
are now fixes with tests rather than folklore.

| # | Instance | Billed | Cost | What it bought |
|---|---|---|---|---|
| 1 | 49700454 | 10.1 min | $0.0357 | `GET /api/v0/instances/<id>/` returns `{"instances": null}` for a live instance instead of failing, so the readiness poll reported "unknown" 60 times and timed out. Listing moved to `/api/v1/`. |
| 2 | 49701623 | 10.0 min | $0.0355 | Lost to a monitoring wrapper of mine whose timeout killed the run. The trap still destroyed the instance and reconciled the ledger. |
| 3 | 49702340 | 2.6 min | $0.0092 | Diagnostic probe: `actual_status` is null on this host while `cur_state` reads "running" from the moment of creation — the rental state, not the container's. |
| 4 | 49702514 | 10.1 min | $0.0357 | ssh proxy unreachable; "running" does not mean sshd is accepting. Added an ssh probe before the first real command. |
| 5 | 49703204 | 1.3 min | $0.0045 | Diagnostic probe of the proxy vs direct path. |
| 6 | 49704554 | 20.1 min | $0.1192 | First *verified* host: reports `actual_status: loading` with `status_msg` carrying pull progress. Spent the whole rental pulling the 14.1 GB `pytorch/pytorch:...-devel` image without finishing. |
| 7 | 49706056 | 20.1 min | $0.1192 | Same with the 4.26 GB `-runtime` tag. Still "Pulling from pytorch/pytorch" at 20 minutes. |
| 8 | 49707685 | 20.1 min | $0.1192 | Different machine, different registry: `vastai/base-image` 2.5 GB. Layers showed "Pulling fs layer" and never advanced. |

## Why it is blocked, and the cheapest way to find out

Four machines, three images, and every instance created, billed, and never finished
pulling.

> **Corrected 2026-09-05.** This paragraph used to say "two registries". That was wrong:
> `pytorch/pytorch` and `vastai/base-image` are *both* Docker Hub, so all eight failures
> were anonymous Docker Hub pulls and no other registry had ever been tried. Rental 9 tried
> `ghcr.io` and the image pulled in three minutes. See `docs/GPU-ACCESS.md`; the
> account-restriction theory below is ruled out.

The account state is the strongest candidate:

```
balance          = 0
credit           = 7.94      (signup credit, and it IS being drawn down)
paid_verified    = 0.0
has_billing      = False
```

The account has never made a payment. Instances are created and billed against signup
credit, but image pulls do not progress — consistent with a new-account restriction on
unpaid accounts rather than anything in this repo.

**The test is cheap:** add a payment method or make a small deposit, then re-run

```sh
remote/run_remote.sh --session-id "$(date -u +%Y%m%dT%H%M%SZ)" \
                     --hypothesis 001-fused-rmsnorm-residual
```

unchanged. If a pull completes, everything downstream is already wired: the run installs
torch, executes `pytest -m gpu` (which is where the deferred weight-value oracle finally
runs), then both correctness gates, then the five-column benchmark, and pulls the records
back here.

If pulls still stall on a funded account, the next thing to try is a host in a different
region with the `--image` flag pointing at a tag that host already caches.

## What a future session must not do

Do not record a ratio for this hypothesis from anything in this folder. There is no
measurement, and re-running it would not produce a useful one.

**Do not resurrect this mechanism without new arithmetic.** If a future workload changes
the byte shares enough to lift the ceiling above the noise band — a much smaller model, or
a regime where the hidden state stops being rounding error next to the weights — say so
with numbers from `docs/roofline.py` first.

**The lesson that matters more than the kernel.** This hypothesis was ranked first because
it was the cheapest to *write*, and "calibrate the harness on a cheap kernel" was used to
justify that order. Both were wrong, and the whole GPU budget went to the least promising
entry in the backlog. A harness is calibrated with an **identity champion** — an installer
that changes nothing, whose columns must return 1.00 ± noise — at zero kernel-writing risk.
`docs/HYPOTHESES.md` now leads with the byte table so the ordering mistake is hard to
repeat.

## Files here

| File | What it is |
|---|---|
| `kernel-as-written.py` | Frozen copy of the Triton kernel at this attempt. Source of truth is `src/deltaforge/kernels/fused_rmsnorm_residual.py`. |
| `kernel-tests.py` | Frozen copy of its tests: CPU structural gates that pass, GPU numerical gates that have never run. |
| `correctness-review.md` | Line-by-line correctness review of the kernel, done by reading rather than by running. |
