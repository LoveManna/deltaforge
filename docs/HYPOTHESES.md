# Hypothesis backlog

Ordered by **share of per-token bytes attacked**, not by how easy the kernel is to write.
Pick one, state it in a sentence before writing any code, and record the outcome whether
it wins or loses.

---

## Read this before picking anything

> **Batching changed what "worth trying" means.** A rental now measures 7-12 hypotheses,
> not one, so a slot costs about three minutes instead of a whole rental. The ceiling
> arithmetic below still decides what can *win*; it no longer decides what is worth
> *measuring*. Several entries in the graveyard were closed as "unmeasurable — not worth a
> rental", which was an argument about cost rather than about truth. They are being
> measured now, and each will be amended from a predicted null to a measured one. See
> `docs/BATCHES.md`.

> **2026-09-16 (rental 37): the first fully admissible batch, and it changes this file.**
> Batch 003 ran seven slots with `calibrated: true` and `graphs_compiled` non-zero on every
> one. Hypothesis 1 below is **amended, not closed**: its ceiling arithmetic survives intact
> and one of its stated mechanisms is **refuted by measurement**. The batch also produced
> the number this file never had — **the compiled baseline reaches 1308 GB/s, 73% of a
> 5090's vendor peak** — which is the denominator every entry here is implicitly divided by.
> Three new entries (5, 6, 7) come out of it. See
> `results/batches/003-int8-weight-only/README.md`.

> **2026-09-14: eight of these ran on a GPU, and nothing here changes status yet.** Rental
> 35 executed all nine slots of batch 001. The batch is calibrated, so the harness is sound
> — but six of the eight kernels failed the correctness gate, and the two that passed were
> timed after dynamo stopped compiling candidates, so their ratios compare eager against
> compiled. **No entry below is promoted, graveyarded or amended on this rental.** A kernel
> whose candidate never compiled has not been shown to be slow, and a kernel that fails an
> exact-token gate by ~2 bf16 ULP has not been shown to be wrong. See
> `results/batches/001-calibration/README.md`.

At batch 1, decode is a weight-streaming problem. Here is where every byte goes for
`Qwen/Qwen3.5-4B` at batch 1, context 2048 — reproduce it with `docs/roofline.py`:

| What moves | MB/token | share |
|---|---:|---:|
| Weights, streamed once | 8411.51 | **91.85%** |
| GQA `repeat_interleave` materialisation | 570.43 | **6.23%** |
| Recurrent state, read + write | 100.66 | 1.10% |
| KV cache read | 71.30 | 0.78% |
| SwiGLU intermediates | 2.36 | 0.026% |
| Norm + residual | 1.64 | 0.018% |
| QKV / RoPE intermediates | 0.33 | 0.004% |
| **Total** | **9158.23** | |

Roofline on an RTX 5090 (1.79 TB/s): **5.11 ms/token, 196 tok/s.**

Two consequences, and they decide the whole backlog:

**1. A kernel that already streams its data once cannot be beaten.** Once you are at the
roofline the only thing left is the hardware. Inductor is *good* at getting simple
memory-bound elementwise and reduction kernels to roofline — that is its home turf, and it
emits Triton, so "hand-written Triton beats compiler-generated Triton" on a fused RMSNorm
is a claim about two nearly identical kernels.

**2. The ceiling of a hypothesis is the share of bytes it touches.** A perfect, infinitely
fast fused RMSNorm buys 0.018%. That is far below the noise band the harness itself
declares, so the hypothesis is not merely unlikely — it is **unmeasurable**. Do this
arithmetic before writing a kernel, not after renting a GPU.

### The three ways a hand-written kernel can actually win

Every entry below names which one it uses. An entry that names none of them does not
belong in this file.

| | How it wins | Why the compiler cannot |
|---|---|---|
| **A** | The compiler's kernel is far from roofline | Rare for elementwise and reductions; real for irregular access and hardware features inductor does not emit |
| **B** | It moves fewer bytes | Quantisation, layout changes, and eliminating a materialised intermediate are choices about representation, not scheduling |
| **C** | It runs a different algorithm | Chunked parallel scans, online-softmax attention, split-K/split-KV: mathematical reassociations a scheduler will not derive |

### Before you write a kernel, read what you are trying to beat

```sh
TORCH_LOGS=output_code python -m deltaforge.cli bench --weights … 2>&1 | tee inductor.txt
```

That prints the exact Triton inductor generated. **This is not optional.** You cannot claim
to beat code you have not read, you cannot explain *why* you won without it, and it costs
nothing. Several entries below are graveyarded precisely because inductor already emits the
kernel someone was about to hand-write.

Also confirm the baseline actually compiled: a silent graph break makes `compiled` fall
back toward eager and inflates every ratio in your favour. That is the failure mode a
sceptical reader looks for first.

---

## Open

### 1. Weight-only quantisation with a fused dequantise-GEMV

**Share of bytes: 91.85%. Ceiling: 1.85× at fp8/int8, 3.21× at int4.** Category **B**.

**Mechanism.** Decode at batch 1 reads every weight once per token and does almost no
arithmetic with them, so the model runs at the bandwidth roofline. You cannot beat a
roofline with a better kernel; you beat it by moving fewer bytes. Storing weights at 4 or 8
bits and dequantising them *inside* the GEMV's K-loop cuts weight traffic by 2–4×.

~~The compiler cannot do this, and it fails in a specific, checkable way.~~ **Refuted on
rental 37.** This entry asserted that given `dequant(W_int4, scales) @ x`, inductor
materialises the full bf16 weight into global memory and calls cuBLAS, *adding* an 8.4 GB
write and making the quantised version **slower** than bf16.
`010-int8-dequant-torch` measured exactly that program under `max-autotune`: **0.9893, IQR
0.0212** — the same speed as bf16, inside the noise band.

The materialisation story is refuted outright by a bandwidth budget, with no profiler
needed. It would require 19868 MB/token at the measured 7.68 ms/token, which is **2525 GB/s
on a card whose vendor peak is 1790** — physically impossible. Inductor therefore either
fuses the dequantisation into the matmul prologue or keeps the transient in the 96 MB L2.

**What is still unexplained is why a 39% cut in DRAM traffic bought 0% of time**, and the
`TORCH_LOGS=output_code` dump this file already calls mandatory is what would answer it. It
was skipped before rental 37; entry 5 exists so that stops being possible.

So the compiler is not *worse* at weight-only quantisation than a hand-written kernel. It is
**exactly as good as doing nothing** — a different claim, and a more useful one, because it
moves the question from "can the compiler express this" to "why does halving the weight
stream not show up on the clock".

**Replaces.** `swiglu_mlp`, `qkv_projection_rope`, and the linear-attention input
projections — 91.8% of weight bytes sit behind those three.

**Measured on rental 37: the ceiling is intact and out of reach of a naive kernel.** The
1.85× is real arithmetic. It is collectable only by an implementation that is *itself*
bandwidth-bound, and batch 003's was not — as it removed bytes it got **slower** (26.90 →
38.68 → 42.49 ms/token for 9158 → 5588 → 2850 MB/token), because a cross-lane `tl.sum` ran
once per K-iteration and the int8→fp32 conversion landed on an already-saturated issue port.
An int8 GEMV needs **715 GB/s just to tie** the compiled baseline and ~1170 GB/s to win;
batch 003's reached **141**. Entry 6 is the prerequisite for retrying this one.

**Watch for.**
* **The `output_code` dump is still mandatory and has still never been done.** See entry 5.
* **Your real competition is Marlin / machete / AWQ kernels, not torch.compile.** Beating
  the compiler here is easy and proves little. Record the comparison against a published
  int4 kernel as an unscored column. If you are at 0.6× Marlin, say so.
* Correctness changes shape: a quantised candidate is not bit-comparable to a bf16
  reference, so the layer-2 exact-token gate will fail by construction. Decide *before*
  measuring what the correctness claim is — the usual answer is perplexity or KL against
  the bf16 reference on a fixed prompt set, plus exact-match on the dequantise kernel
  itself against a PyTorch dequantise.

### 2. Eliminating the GQA head expansion

**Share of bytes: 6.23%. Ceiling: 6.2%.** Category **B**.

**Mechanism.** The reference expands 4 KV heads to 16 query heads with a real
`repeat_interleave` (`reference.py:579`), which materialises 4× the KV cache — written,
then read back by the attention matmul. At context 2048 that is 570 MB/token, 300× more
than every elementwise fusion in this file combined. A kernel that indexes the unexpanded
cache directly never pays it.

**Replaces.** `kv_cache_update`, `gqa_attention`.

**Watch for.** **This one may already be won by the compiler — check first.** Dump the
generated code and see whether inductor keeps the expansion materialised or folds the index
arithmetic into the consumer. If it folds it, `compiled` already has this and there is
nothing to take; record that as the finding and move on, because it is a genuinely
interesting fact about inductor. Note also that this number is partly a property of *our*
baseline's choice to copy rather than alias, so the honest framing is "against
`max-autotune`", never "against eager".

**The share grows with context.** The expansion scales with the KV cache, so at 32k it is
already 32.5% of per-token bytes on the 27B config. Run `docs/roofline.py --context N` for
the context you actually intend to measure before deciding this is a 6% hypothesis.

### 3. Chunked delta-rule scan — **at prefill and long context, not at batch-1 decode**

**Share of bytes at batch-1 decode: 1.10%.** Category **C**.

**Mechanism.** The delta rule is a sequential scan with matrix-valued state. Restructuring
it into a chunked parallel form converts a long dependency chain into blocked matrix work.
No compiler will derive that: it is a reassociation of the recurrence, not a fusion or
tiling decision. 24 of 32 layers are affected.

**Watch for — this is why the entry moved.** A chunked scan needs *a sequence to chunk*. At
batch-1 single-token decode there is no sequence: it is one rank-1 update to a 128×128
state per head, and the state is 1.1% of traffic. **Measured in the `headline` workload this
hypothesis cannot express itself.** Running it there and recording a null would be a
measurement error, not a result.

To attempt it, add a prefill or long-context workload to `DEFAULT_WORKLOADS` and score it
there, stating in the writeup that the workload differs from the headline. Also: the
recurrent state must stay **fp32** (`mamba_ssm_dtype`) whatever the surrounding weights
are, and `reference_test.py` already pins the contract any chunked implementation must
satisfy — step-by-step, whole-sequence and chunk-by-chunk must all agree. Compare against
`flash-linear-attention` as an unscored reference point; beating a compiler at a recurrence
is trivial, and matching a tuned kernel is the result worth publishing.

### 4. Quantised KV cache and long-context attention decode

**Share of bytes at ctx 2048: 0.78%, and it grows linearly with context.** Category **B**
for the cache, **C** for the split-KV decode kernel.

**Mechanism.** Two things that only matter once the context is long. Storing K/V at fp8
halves cache traffic. And at batch 1 there is no batch or head parallelism to fill the GPU,
so a flash-decode kernel that splits the reduction across KV blocks and combines partial
softmax results manufactures parallelism a scheduler will not invent.

**Watch for.** Both are worthless at 2048 tokens. Pair this with a 32k or 128k workload or
do not run it. At 128k the KV cache alone exceeds the weights, which inverts the whole
table above — recompute it for the context you intend to measure.

### 5. Read what the compiler actually emits — the free diagnostic this file keeps demanding

**Share of bytes: none. It is not a kernel.** Category: prerequisite.

**Mechanism.** `TORCH_LOGS=output_code` on both scoring columns, plus one profile of the
decode step, as a *step inside* the next rental rather than a rental of its own. It answers
three open questions, each worth more than a slot:

1. **Why did `010` move 39% fewer bytes for 0% less time?** A fused prologue, or an
   L2-resident transient? The answer decides whether weight-only quantisation is reachable
   *through the compiler* — which would be worth more than reaching it through a kernel.
2. **Where does the baseline's remaining 27% go?** It runs at 1308 GB/s of 1790. That gap is
   a 1.37× hypothesis in its own right and nobody knows what is in it.
3. **Does inductor fold the GQA `repeat_interleave`?** Entry 2 has been explicitly
   conditional on this since it was written, and the condition has never been checked.

**Why this is an entry and not a chore.** This file says in bold that reading the generated
code "is not optional". Batch 003 skipped it, went straight to a rental, and came back with
a number whose mechanism it cannot name. Making it numbered makes it schedulable.

### 6. A GEMV that is actually bandwidth-bound

**Share of bytes: 0%. Ceiling: 1.0 by construction.** Category **A**.

**Mechanism.** The same bf16 GEMV as `009`, rewritten so that its time is set by memory
rather than by instruction issue:

* **Accumulate a `(BLOCK_N, BLOCK_K)` tile and reduce once**, instead of a cross-lane
  `tl.sum` per K-iteration — 20 reductions for K=2560 and 72 for K=9216 collapse to one.
* **`tl.dot` with M padded to 16**, so the MMA pipeline schedules the loads. The 16× flop
  waste is free: arithmetic intensity at batch-1 decode is ~2 flop/byte against a machine
  balance near 150.
* **Split-K with a second reduction pass**, for projections that cannot fill the card by
  output channel — `in_proj_a`/`in_proj_b` are 32 wide and get 4 programs at any tile size.

**It cannot win, and it is still the most valuable slot in the next batch**, because it is
the precondition for entries 1, 4 and 7. Batch 003 spent five slots on quantisation variants
whose outcome was already determined when this control returned 0.2801.

**Gate: the bf16 control must reach ≥ 0.56 before any quantised slot is worth running** —
and ≥ 0.75 to expect a comfortable win. **The ≥ 0.90 first written here was wrong**, and
wrong in the expensive direction: it would have cancelled a batch that could have won.
Quantisation halves the bytes, so int8 *ties* the baseline when the kernel reaches half the
baseline's byte rate (`f` = 0.50), and that shows up on the bf16 control — which moves the
full bf16 bytes — as only **0.562**. The full table is in
`docs/superpowers/plans/2026-09-17-bandwidth-bound-gemv.md`.

### 7. fp8 rather than int8, on Blackwell's conversion path

**Share of bytes: 91.85%. Ceiling: 1.85×.** Category **B**.

**Mechanism.** The same byte saving as int8 without the tax that made int8 *slower* than
bf16 in batch 003. On an RTX 5090 (sm_120) e4m3 feeds the MMA path directly, so the
conversion happens inside the tensor-core pipeline rather than as ALU instructions on the
critical path. Batch 003 measured that tax precisely: **int8 cost 1.438× the time of bf16**
in the same kernel on the same sites, and int4 a further **1.046×** for its nibble unpack.

**Strictly after entry 6.** fp8 removes a conversion cost from the inner loop and does
nothing about a loop that is issue-bound for other reasons. Running it first would reproduce
batch 003 with a different dtype.

**Watch for.** e4m3 carries 3 mantissa bits against int8's effective 7 at per-channel scale,
so expect accuracy between batch 003's int8 (0.0011 nats, 8/264 flips) and its int4 (0.0919
nats, 38/264). **Derive the gate from those two measured points, not from priors** — batch
003's priors were wrong in exactly this way, and its thresholds failed four slots that were
working correctly.

---

## Graveyard

Hypotheses that were measured and lost, or that were ruled out before measurement. Each
entry records the **mechanism that failed and why**, so a later session does not pay to
learn it twice.

The four entries below were ruled out by arithmetic, not by measurement. That was a
legitimate and much cheaper way to close a hypothesis **under a workflow where measuring
one cost a whole rental.** It is no longer the cheaper option: batch 001 puts all four back
on a card as slots 1-5, at about three minutes each, so the repo will hold measured nulls
instead of predicted ones. Each entry keeps its arithmetic — that arithmetic is the
prediction being tested — and gains its measurement when the batch reports.

### Fused RMSNorm + residual add — ruled out 2026-09-04

**Ceiling 0.018% of per-token bytes.** The residual add and the two hidden-size norms per
layer move 1.64 MB/token against 9158 MB total. An infinitely fast kernel is unmeasurable
against the harness's own noise band. Scored instead as pure launch overhead — 64 fused
pairs at 1–2 µs — the ceiling is 1–3%, and CUDA graphs plus inductor's fusion already
collect most of that. Inductor emits a single persistent reduction kernel here that loads
the row once, computes the sum of squares and rescales: the same kernel, generated.

**Attempted before the arithmetic was done.** Branch `hyp/001-fused-rmsnorm-residual`,
2026-09-03: the kernel was written and passed its CPU gates but was **never measured** —
eight Vast.ai rentals ($0.4783, all destroyed cleanly) produced no number because no
instance finished pulling its container image. The kernel, its gates and a full account are
in `results/hypotheses/001-fused-rmsnorm-residual/`. It is graveyarded on the mechanism,
not on the failed rentals: even a perfect measurement could not have shown a win.

**Lesson, and the reason this file now leads with a byte table:** the hypothesis was ranked
first because it was the cheapest to *write*, and "calibrate the harness on a cheap kernel"
was used to justify the order. Both were wrong. A harness is calibrated with an **identity
champion** — an installer that changes nothing — whose columns must come back at 1.00 ±
noise. That measures the harness with zero kernel-writing risk.

### Fused SwiGLU — ruled out 2026-09-04

**Ceiling 0.026%.** The MLP is 53.9% of weight bytes, but the *fusion target* is the
activation, not the weights: at batch 1 the two 9216-wide intermediates are 18 KB each
against 141 MB of weights per layer. The GEMMs are cuBLAS territory and the epilogue is
rounding error. Concatenating `gate_proj` and `up_proj` into one GEMM reduces launches but
reads exactly the same bytes.

### Fused QKV projection + RoPE — ruled out 2026-09-04

**Ceiling 0.004%**, the smallest in the file. Same argument: the projections' cost is
streaming their weights, and the Q/K/V intermediates being written and re-read are 0.33
MB/token across all 8 full-attention layers.

### Persistent-kernel decode step — ruled out 2026-09-04

**No byte-share to attack.** The premise was that batch-1 decode is launch-bound, but at
91.8% weight streaming it is bandwidth-bound: the kernels are not tiny-and-idle, they are
each waiting on memory. CUDA graphs already remove the launch overhead that remains, which
is what the `compiled_nocudagraphs` column exists to show. Reviving this needs a
*measurement* first — a profile showing gaps between kernels — not an assumption.

---

## Adding an entry

State the **mechanism**, the **category** (A, B or C above), and the **share of per-token
bytes** it attacks. "Fuse it and see" is not a hypothesis. An entry whose ceiling is below
the noise band goes straight to the graveyard with its arithmetic, and that is a result
worth recording.
