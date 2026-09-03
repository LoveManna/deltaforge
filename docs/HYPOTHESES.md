# Hypothesis backlog

Ordered by expected yield per GPU hour. Pick one, state it in a sentence before writing
any code, and record the outcome whether it wins or loses.

**Nothing here has been attempted.** The bootstrap session built the harness and shipped
no kernel; benchmarking a kernel with the harness written alongside it produces a broken
harness and a meaningless number.

## Format

Each entry states the **mechanism** — the specific reason the win is expected — because
"fuse it and see" is not a hypothesis and produces results nobody can learn from. An entry
moves to the graveyard when it is measured and loses, and it stays here when a measurement
comes back inside the noise band.

---

## Open

### 1. Fused RMSNorm + residual add

**Mechanism.** The residual add, the norm's reduction and its rescale each read and write
the full hidden state. At batch 1 this is entirely memory-bound: three passes over
`2560 × 2` bytes per layer, 64 norms per forward. Fusing them into one pass should cut
traffic by roughly two thirds for this operation.

**Replaces.** `rms_norm`, `rms_norm_residual`.
**Watch for.** `torch.compile` already fuses some of this — the honest comparison is
against `max-autotune`, not against eager. This may be the smallest margin in the list,
which is why it is first: it is also the cheapest to write and calibrates the harness.

**Attempted 2026-09-03 on branch `hyp/001-fused-rmsnorm-residual`: kernel written and
gated, NOT MEASURED.** Eight rentals ($0.4783, all destroyed cleanly) produced no number —
every instance was billed but never finished pulling its container image. This stays
**open**, not graveyarded: a graveyard entry means a mechanism was tried and failed, and
this mechanism has not been tried. The kernel, its gates and a full account are in
`results/hypotheses/001-fused-rmsnorm-residual/`. A funded session should re-run it before
picking anything else from this list.

### 2. Fused SwiGLU

**Mechanism.** `silu(gate_proj(x)) * up_proj(x)` materialises two `9216`-wide
intermediates before the elementwise combine. Fusing activation and multiply into the GEMM
epilogue avoids writing and re-reading both.

**Replaces.** `swiglu_mlp`.
**Watch for.** The GEMMs themselves are cuBLAS territory and we will not beat them; the
win must come from the epilogue and the avoided round trip, not from the matmul.

### 3. Fused QKV projection + RoPE

**Mechanism.** One projection pass feeding RoPE directly, instead of writing Q/K/V out and
reading them back to rotate.

**Replaces.** `qkv_projection_rope`.
**Watch for.** RoPE here is **partial** — only the first 64 of each 256-wide head is
rotated and the other 192 pass through — and the query projection is **doubled** for the
output gate. Both change the kernel's shape. See `docs/ARCHITECTURE.md`.

### 4. Chunked delta-rule scan

**Mechanism.** This is the project's reason for existing. The baseline runs the delta rule
as a sequential scan with matrix-valued state, one token at a time. A compiler cannot
restructure a sequential scan into a chunked parallel form on its own — that restructuring
is the hand-tuning, and it converts a long dependency chain into blocked matrix work.
24 of 32 layers are affected.

**Replaces.** `gated_delta_rule`.
**Watch for.** The recurrent state must stay **fp32** (`mamba_ssm_dtype`), regardless of
what the surrounding weights are. `reference_test.py` already pins the contract any
chunked implementation has to satisfy: step-by-step and whole-sequence must agree, and so
must chunk-by-chunk. Tune block size, chunk length and state layout separately — this is
one hypothesis by mechanism but several by parameter, so budget for more than one session.

*Expected to be the largest win in the project. It is number 4 rather than number 1
because it is also the hardest, and the earlier entries calibrate the harness first.*

### 5. Flash decode for the GQA attention layers

**Mechanism.** The 8 full-attention layers currently compute attention as explicit
matmul + softmax, materialising the score matrix. A flash-style decode kernel keeps it in
registers.

**Replaces.** `gqa_attention`.
**Watch for.** This is the one hypothesis where the reference is *deliberately*
unoptimised: the baseline does not call `F.scaled_dot_product_attention`, because SDPA is
itself the fused attention kernel we are trying to write. Beating our own explicit
softmax attention is therefore not the interesting claim — **the honest comparison is
against `torch.compile(max-autotune)`, which is free to select a fused attention kernel
itself.** State that plainly in the writeup, and record how the result compares to SDPA as
an unscored reference point. Getting this wrong would be the easiest way to publish a
misleading number.

### 6. KV-cache layout and gather strategy

**Mechanism.** GQA expands 4 KV heads to 16 query heads. The baseline does a real
`repeat_interleave` copy. A layout that avoids materialising the expansion, or that stores
K/V in an access-friendlier order, removes both the copy and its bandwidth.

**Replaces.** `kv_cache_update`, `gqa_attention`.

### 7. Persistent-kernel decode step

**Mechanism.** At batch 1 the decode step is launch-bound: 32 layers × several kernels
each, all tiny. Collapsing per-layer launches into one persistent kernel removes launch
overhead that dominates when there is almost no work per kernel.

**Replaces.** `decode_step`.
**Watch for.** CUDA graphs already attack this problem, and the `compiled` column has them
on. That is exactly what the `compiled_nocudagraphs` column is for: if this wins against
`compiled` it is a real win, and if it only wins against `compiled_nocudagraphs` it is
launch overhead that CUDA graphs already remove. Report both.

---

## Graveyard

Hypotheses that were measured and lost, or that were ruled out before measurement. Each
entry records the **mechanism that failed and why**, so a later session does not pay to
rediscover it.

Format:

```
### NNN. <title>  —  retired YYYY-MM-DD
**Expected mechanism.** What was supposed to produce the win.
**What happened.** The measurement, with the median ratio and its IQR, or the reason no
measurement was needed.
**Why it failed.** The actual cause, not a restatement of the result.
**Result record.** results/hypotheses/NNN-slug.json
```

A hypothesis whose measurement landed **inside the noise band** does not belong here. That
is recorded as *inconclusive*: it neither promotes nor enters the graveyard, and the
hypothesis stays open for a cleaner measurement.

### 7'. Fused MoE routing and grouped GEMM — retired 2026-08-30, before any GPU time

*(Numbered 7 in the design spec's original backlog; the open list above has been
renumbered.)*

**Expected mechanism.** If the 4B variant's FFN were sparse, fusing expert routing with a
grouped GEMM would avoid a scatter/gather round trip per token.

**What happened.** No measurement was needed. The published `config.json` and the
checkpoint's own tensor list settle it: `intermediate_size` is 9216, `mlp_only_layers` is
empty, there is no expert, router or MoE key anywhere in `text_config`, and every one of
the 32 layers carries exactly three dense SwiGLU projections and no expert tensors.

**Why it failed.** The premise was false. Sources conflicted about whether the 4B variant
was sparse; it is dense. There is no routing to fuse.

**Result record.** None — retired on architecture, not on measurement. The evidence is in
`docs/ARCHITECTURE.md` and is asserted by `config_test.py::test_ffn_is_dense`, which reads
the committed checkpoint manifest rather than trusting this document.
