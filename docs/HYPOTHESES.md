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

The compiler cannot do this, and it fails in a specific, checkable way: given
`dequant(W_int4, scales) @ x`, inductor materialises the full bf16 weight tensor into
global memory and then calls into cuBLAS. That *adds* an 8.4 GB write on top of the read,
making the quantised version **slower** than bf16. The hand-written kernel never
materialises anything.

**Replaces.** `swiglu_mlp`, `qkv_projection_rope`, and the linear-attention input
projections — 91.8% of weight bytes sit behind those three.

**Watch for.**
* **Verify the claim above before building on it.** Dump inductor's code for a small
  quantised linear first. Recent inductor has prologue fusion into its mm templates and may
  fuse *some* of the dequant; at M = 1 it likely is not using a template at all. Either way,
  measure, do not assume.
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
