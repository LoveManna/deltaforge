# Batch 004 — the rewrite made it worse, and the dump says why the premise was wrong

> **Rental 38, 2026-09-17.** RTX 5090, instance 51361206, machine 148413, $0.4363/hr,
> **39.65 billed minutes, $0.2883**, destroyed cleanly. Lifetime $6.4451 across 37 billed
> rentals.
>
> `calibrated: true`. `graphs_compiled: 3` on both slots that ran. **Two slots ran and five
> declined**: the bf16 control returned 0.1934 against a 0.56 floor, the preconditions fired,
> and the batch stopped rather than re-measuring a settled question five more times.
>
> **Predictions scored 1 of 2.** The identity slot was right. The control was not: it
> predicted 0.75-0.95 and measured 0.1934 — **worse than the 0.2801 it was rewritten to
> improve on.**

---

## The scorecard

| Slot | Outcome | Ratio | IQR | Layer 1 | Layer 2 | Predicted |
|---|---|---:|---:|---|---|---|
| `000-identity` | calibrated | **0.9913** | 0.0159 | — | exact | `identity` ✅ |
| `015-tiled-gemv-bf16` | `loss` | **0.1934** | 0.0028 | pass (7.8e-3) | 261/264, 0.0006 nats | `inconclusive` ❌ |
| `016-fp8-all-linear` | `precondition_failed` | — | — | — | — | `win` (untested) |
| `017-fp8-full` | `precondition_failed` | — | — | — | — | `win` (untested) |
| `018-fp8-mlp` | `precondition_failed` | — | — | — | — | `win` (untested) |
| `019-int8-all-linear` | `precondition_failed` | — | — | — | — | `inconclusive` (untested) |
| `020-int4-full` | `precondition_failed` | — | — | — | — | `win` (untested) |

**No kernel in this batch was wrong, and the one kernel that ran was slower than the one it
replaced.** Those are both facts about the same slot and the rest of this file separates
them.

---

## 1. The measurement, against the thing it was built to beat

Batch 003's `009-gemv-bf16-control` and batch 004's `015-tiled-gemv-bf16` are the same
hypothesis — a hand-written bf16 GEMV on every layer projection, moving exactly the bytes
the compiler moves — with only the kernel's internal structure changed.

| | ratio | ms/token | achieved |
|---|---:|---:|---:|
| compiled baseline (rental 38) | 1.0000 | 7.30 | **1177 GB/s** |
| `009` naive GEMV (rental 37) | 0.2801 | 26.90 | 319 GB/s |
| `015` tiled GEMV (rental 38) | **0.1934** | **37.73** | **228 GB/s** |

**The rewrite is 1.40× slower.** It removed exactly what batch 003's writeup said was the
cause — the cross-lane `tl.sum` that ran once per K-iteration, 20 times for K=2560 and 72
for K=9216 — and the kernel got worse. The ratio fell by 1.45×.

This is the first falsification condition the plan registered in advance:

> **Slot 1 lands below 0.56 after the rewrite.** Then the problem is not the reduction
> structure, and the remaining suspects are the `tl.dot` row-0 extraction, occupancy, or a
> split-K reduction that costs more than it buys.

It is worth being exact about what is now known. Batch 003 established that its kernel was
**not bandwidth-bound**; that survives — 228 GB/s against a baseline at 1177 is not a kernel
near the memory wall. What batch 003 also claimed, and what this rental refutes, is that the
**per-iteration cross-lane reduction was the cause**. It was not, or not mainly: removing it
cost 40% more time.

### Correctness is not the issue, and this is the first time that has been demonstrable

`009` was gated `exact` and matched 1 of 5 prompts, so batch 003 could never separate "this
kernel is wrong" from "this kernel is slow". `015` is the same class of kernel under the
gates this batch fixed first, and it passes cleanly:

* **Layer 1**: worst relative error **7.75e-3** across all six probes — about one bf16 ULP,
  which is what reordering a K-sum against cuBLAS costs and nothing more.
* **Layer 2**: top-1 agreement **261/264** with a 95% Wilson interval of [0.967, 0.996], mean
  KL **0.00060 nats** against a 0.01 bar.

The record also carries `top1_resolved: false` — the agreement bar was inside the interval its
own sample supports, so the slot passed on KL, which is the policy this batch introduced. Under
batch 003's gates this slot would have been reported `incorrect` and its 0.1934 discarded.

---

## 2. What the compiler actually emits — the diagnostic that had never been run

`docs/HYPOTHESES.md` has said in bold since it was written that dumping
`TORCH_LOGS=output_code` "is not optional". Four rentals went by without it. This one ran it
as a step, for 4.7 MB of generated Triton in
[`results/diagnostics/inductor-output-code.txt`](../../diagnostics/inductor-output-code.txt),
and it answers two of the three open questions outright — and reframes the batch.

### 2a. The baseline is not cuBLAS. It is fused Triton, and the fusion is the moat.

Across the whole compiled decode graph, `extern_kernels` is called for exactly one thing:

```
     72 extern_kernels.convolution
```

**Not one `mm`, not one `addmm`.** Under `max-autotune` inductor generated its own Triton
kernel for every matmul in the decode path. So this project's stated contest — "a hand-written
Triton kernel versus what `torch.compile` generates" — is literally Triton against Triton, with
no vendor library in between.

And the generated matmuls are not bare matmuls. This is inductor's kernel for `in_proj_a` and
`in_proj_b` (`xnumel=32`, `r0_numel=2560`), lightly trimmed:

```python
def triton_red_fused__to_copy__unsafe_view_add_mean_mm_mul_pow_rsqrt_t_view_23(
        in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr0, out_ptr1, ...):
    for r0_offset in tl.range(0, r0_numel, R0_BLOCK):
        tmp0  = tl.load(in_ptr0 + r0_1)                  # hidden state
        tmp1  = tl.load(in_ptr1 + r0_1)                  # residual
        tmp3  = tmp0 + tmp1                              # residual add
        tmp11 = libdevice.rsqrt(tmp8 + 1e-06)            # RMSNorm
        tmp16 = tmp14 + 1.0                              #   ... scaled by 1 + weight
        tmp17 = tmp12 * tmp16
        tmp20 = tl.load(in_ptr4 + (r0_1 + 2560*x0))      # in_proj_a
        tmp26 = tl.load(in_ptr5 + (r0_1 + 2560*x0))      # in_proj_b
        _tmp24 = _tmp24 + tmp17 * tmp20                  # two matmuls,
        _tmp30 = _tmp30 + tmp19 * tmp26                  #   one pass over the hidden state
```

**One kernel does the residual add, the RMSNorm, and two projections**, reading the hidden
state once and writing two outputs.

A hand-written GEMV cannot enter that competition. It replaces a matmul, so:

* the residual add and the RMSNorm become separate kernels again;
* two projections that shared one pass become two launches;
* and this kernel's split-K adds a third — an fp32 partial buffer written and read back.

There are **248 two-dimensional projection weights** inside the decoder layers. The candidate
launches at least two kernels at each of them, per token, where the baseline launches
substantially fewer and does more per launch.

**This reframes the hypothesis.** The backlog's entry 1 says weight-only quantisation wins by
moving fewer bytes and that the compiler cannot change the representation. That is still true.
What this dump shows is the *price* of taking the matmul away from inductor: you also take away
every fusion it had built around it, and on a batch-1 decode step — where each matmul is tiny
and there are 248 of them — that price is large.

### 2b. Inductor folds the GQA expansion. Backlog entry 2 is dead, and the compiler killed it.

`docs/HYPOTHESES.md` entry 2 — eliminating the GQA head expansion, **6.23% of per-token bytes,
the second-largest share in the model** — has been explicitly conditional on this check since it
was written, and the check had never been done.

`repeat_interleave` appears **zero** times in the generated code. Here is the attention `bmm`:

```python
tmp2 = tl.load(in_ptr1 + (r0_2 + 256*x0 + 557056*(x1 // 4)), ...)
```

`x1` is the query head (0-15), and `557056 = 2176 × 256` is one KV head's stride through
(context × head_dim). **`x1 // 4` is the GQA head mapping, folded into index arithmetic.**
Sixteen query heads read four KV heads directly out of the unexpanded cache; nothing is
materialised and nothing is written back.

So the compiler already has this optimisation, completely. Entry 2 is closed — not by a
measurement of our kernel, but by reading the code we were proposing to beat, which cost a
step rather than a rental.

**It also corrects the roofline for the compiled baseline.** The 570.43 MB/token the roofline
attributes to the expansion is traffic the reference's eager code performs and the compiled
column does not. Against the 8587.80 MB/token the compiled baseline actually moves, it runs at
**1177 GB/s, 65.7% of an RTX 5090's 1792 GB/s** — and the remaining 34% is now the open
question, not 27%.

### 2c. The baseline runs with no CUDA graphs at all

128 occurrences of:

```
skipping cudagraphs due to mutated inputs (64 instances). Found from :
  File ".../reference.py", line 431, in _causal_conv
    cache.conv.copy_(x[..., -history:].to(cache.conv.dtype))
```

Inductor refuses to CUDA-graph a region with mutated inputs, and the depthwise causal conv
updates its cache in place. **The `compiled` column has therefore never been CUDA-graphed**, on
this rental or any previous one, which means:

* `compiled` and `compiled_nocudagraphs` are the same measurement on this model, and the
  `--columns all` option buys nothing here;
* per-step launch dispatch is fully exposed in the baseline — and *also* in the candidate,
  where there is much more of it.

Nobody knew this. It is a plausible part of the baseline's missing 34%, and it is a cheap
experiment for the next rental: making `_causal_conv` write out-of-place would let inductor
CUDA-graph the whole step, and the difference is a measurement of what launch overhead costs.

### 2d. What the dump does *not* answer

**Why `010-int8-dequant-torch` moved 39% fewer bytes for 0% less time is still unexplained.**
The dump captures the reference and a candidate with no kernels installed, because nothing is
registered as champion — so there is no compiled quantised linear in it. Naming this a step
before the batch was right; pointing it at a candidate that installs nothing was not. The next
rental should dump the batch's own slots.

---

## 3. Why the rewrite lost — what is measured and what is inferred

**Measured:** ratio 0.1934 ± 0.0028, 37.73 ms/token against the baseline's 7.30, correct to one
bf16 ULP, three graphs compiled. And the three structural facts in §2.

**Inferred, and labelled as such — this rental did not profile:**

1. **The 16× flop padding is not free after all.** The plan argued arithmetic intensity is ~2
   flop/byte against a machine balance near 150, so wasting 15 of 16 MMA rows costs nothing.
   That argument holds only for a kernel whose time is set by memory. At 228 GB/s this one's is
   not, so the padding multiplies the work on whatever resource *is* binding.
2. **Split-K costs a second kernel and an fp32 round trip at every one of 248 sites, every
   token.** It was added to give `in_proj_a`/`in_proj_b` — 32 output channels, 8 programs on a
   170-SM card — something to fill the card with. It buys occupancy on 48 of 248 sites and
   charges a launch and a buffer at all of them.
3. **Losing inductor's fusion is a real cost, not a rounding error** (§2a).

These are three separable causes and this batch cannot separate them. The cheap discriminator
is an ablation, not another quantisation batch: the same kernel with `SPLIT_K=1`, and the same
kernel with a plain FMA accumulator instead of `tl.dot`, are two slots that would say which of
(1) and (2) is paying.

---

## 4. What the preconditions bought

Five slots declined. Batch 003, in the identical situation, ran all five and re-measured the
same settled fact at five bit widths for five slots' worth of rental.

| | batch 003 | batch 004 |
|---|---|---|
| Control's ratio | 0.2801 | 0.1934 |
| Slots run after it | 5 | **0** |
| Billed minutes | 55.55 | **39.65** |
| Cost | $0.3786 | **$0.2883** |

The batch ended having answered the only question that mattered, and the five untested
predictions are recorded as untested — `precondition_failed` scores `None`, not wrong. That is
the machinery working. Setting the floor at 0.56 was also correct in hindsight: at 0.1934 no
byte saving was collectable, so every skipped slot would have measured the kernel's slowness
and not its own hypothesis.

---

## 5. Defects this rental found in its own harness

**The achieved-bandwidth feature did not ship.** Task 1's whole purpose was to stop computing
GB/s by hand after the rental, and every slot logged `no byte model ... bandwidth not
reported`. `_repo_for` matched `ModelConfig.name` against the transcriptions in `MODELS`; a
rental's config comes from the checkpoint's `config.json`, where `from_hf_config` sets `name`
from `model_type` — `'qwen3_5'`, never `'qwen3.5-4b'`. The lookup resolved exactly the configs
that never need resolving and failed on every one that does. Fixed to match on shape, with the
regression test named after this rental. Every number in this file was still recoverable from
the recorded per-column medians — which is precisely the by-hand reconstruction the feature was
meant to end.

**The layer-1 probe labels name the wrong tile.** The probes come from
`quantised_linear._probe_weights`, which groups sites by *that module's* `_launch_shape`, so the
labels read `BLOCK_N=8` / `16` / `32` while `tiled_gemv` actually launched 32 or 64. The probes
still covered three distinct shapes and the checks are valid; the labels are misleading and the
grouping is not derived from the kernel under test.

**A precondition-skipped slot writes no individual record.** `run_batch` appends the result but
does not call `on_slot`, so `results/batches/004-.../` holds two files for seven slots. The
skips are all in `summary.json` with their reasons. This mirrors how `not_run` already behaves
and is probably right, but it is worth knowing before looking for a file that is not there.

---

## 6. What is settled, and what the next rental should do

**Settled by this rental:**

* The compiled baseline is generated Triton, matmuls fused with their surrounding norm and
  residual, no cuBLAS and no CUDA graphs, running at **1177 GB/s — 65.7% of peak** against the
  **8587.80 MB/token it actually moves**.
* **Inductor already eliminates the GQA expansion.** Backlog entry 2 is closed.
* The cross-lane reduction was **not** the reason batch 003's GEMV was slow. Removing it cost
  40% more time.
* The fixed gates work: an approximate kernel that computes the same function as the reference
  now passes correctness on KL with its agreement interval recorded, where batch 003 called the
  same thing `incorrect`.

**Not settled:** why `010` was free; where the baseline's remaining 34% goes; which of the three
inferred causes in §3 dominates.

**The next rental is an ablation batch, not a quantisation batch.** Weight-only quantisation is
not refuted — its arithmetic is untouched and its ceiling is still 1.85× at 8 bits — but it is
unreachable until a hand-written GEMV can hold a substantial fraction of 1177 GB/s, and two
attempts with opposite kernel structures have now landed at 319 and 228. The open questions are
cheap and mechanical: `SPLIT_K=1`, FMA instead of `tl.dot`, a fused norm+GEMV that competes with
inductor on its own terms, an out-of-place `_causal_conv` to see what CUDA graphs are worth, and
a dump pointed at a slot that actually installs a kernel.

**And the comparison `docs/HYPOTHESES.md` has been asking for should come first.** A published
int4 kernel — Marlin, machete — as an unscored column would say whether 1177 GB/s is beatable by
anyone on this shape, or whether batch-1 decode on a 2560-wide model is simply a regime where a
fusing compiler wins. Two rentals have now been spent finding out that our kernel is slow. One
rental would find out whether *any* hand-written kernel is fast here, and that is the more
valuable question.
