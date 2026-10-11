# DeltaForge — Results

Hand-written Triton kernels for the **Qwen3.5-4B** decode path, measured against
`torch.compile(mode="max-autotune")` on an identical pure-PyTorch reference.

Twelve batches, 56 GPU rentals, $11.49 of billed compute, 2026-10-02.

---

## The result, in short

**The two shipped champions, composed, run this model's decode 1.2179x faster than
`torch.compile(mode="max-autotune")`** — 21.8% off the step, ~19% after correcting for that
card's calibration offset. The margin comes off a *strong* baseline, which is the whole
difficulty: the compiled reference already moves bytes at **65.7–71.5% of an RTX 5090's
vendor peak bandwidth**, so the headroom above it is thin, and finding any of it meant
knowing exactly where the compiler's remaining slack was and proving it before renting a
card.

The champions are a four-tap causal convolution at **1.0765–1.0778**, bit-identical to the
reference at 264/264 tokens and 0.00000 nats, and a group-128 int4 tied LM head at **1.0251**
(IQR 0.0104). Both follow from this project's central finding, which is a fact about the
compiler rather than about any one kernel: **an opaque `torch.library.custom_op` is a fusion
barrier, and it costs its own kernel plus everything inductor can no longer fuse across it —
a bill that is invisible at the call site.** Keeping the same arithmetic visible to inductor,
as fusible torch operations, wins at both sites.

The convolution is bit-identical either way and measured **0.7854 behind the custom op
against 1.0765 as fusible ops — one variable, 37%**, because the barrier drove buffer
allocations from 59 to 190 and made the linear-attention state reduction run twice per layer,
24 times per token. At the LM head the fusible form beat a hand-written Triton GEMV **by 3.2%
in the same process on the same card**: a custom op forfeits the RMSNorm fusion inductor
welds into the matmul, *and* the register-level 4-bit unpack that was the kernel's reason to
exist is something the compiler emits inline. That the two are the same program is not an
assumption — layer 2 returned 0.9318 agreement and 0.01674 nats for both, digit for digit, on
six rentals.

**Every batch number here came off a rented card, with its outcome and magnitude registered in a
manifest committed before the GPU existed** — 12 batches, 56 rentals, $11.49 of compute, an
identity control that can void a batch, two correctness gates, and 56 of 56 instances
destroyed.

---

## 1. How it works, in plain terms

To produce *one* token, the GPU reads every one of the model's 4.21 billion weights, and then
reads them all again for the next token. The arithmetic is nearly free; the fetching is
**91.85%** of the work. So there are only two ways to go faster: **move fewer bytes**, or
**stop interrupting the flow**. There is one champion for each.

`torch.compile` turns the model into GPU kernels, and its most valuable trick is **fusion**:
when several steps in a row touch the same data, it welds them into one pass over memory
instead of several. It is good enough at this to reach 65.7–71.5% of the card's peak bandwidth
before anyone hand-writes anything.

But it can only weld across code it can *see into*. An opaque call — a vendor library kernel, a
`torch.library.custom_op`, a hand-written Triton kernel — is a wall: inductor has to write that
call's inputs out to memory, hand over, and read the results back, and the steps on either side
can no longer be welded to each other. **That wall is what "fusion barrier" means in this
document, and both champions are the removal of one.**

```
WITH A WALL — 0.7854, slower than changing nothing
  [ qkv projection ]  ╳  [ F.conv1d — opaque ]  ╳  [ state reduction ]  ╳  [ state reduction, again ]
        └──────────────────┴─ GPU memory ─┴──────────────────┘
                      three round trips, and the reduction recomputed

NO WALL — 1.0765, and bit-identical output
  [ qkv projection + four multiplies + state reduction — one kernel ]
        └────────────────────── GPU memory ──────────────────────┘
                              one pass
```

The wall does not only cost its own kernel. It splits a producer chain inductor had been fusing,
and then inductor *recomputes* the shared prologue rather than reading the buffer it just wrote —
so the `(1, 32, 128, 128)` state reduction runs **twice per layer, 24 times per token**. Both arms
are bit-identical to the reference, 264/264 tokens at 0.00000 nats.

### A. Stop interrupting the flow — four multiplies instead of a library call

Twenty-four times per token, the model blends the last four timesteps together:
`a·w1 + b·w2 + c·w3 + x·w4`, per channel. It is four multiplies and three adds. The reference
expressed it as `F.conv1d`, which inductor hands to `extern_kernels.convolution` — **the only
vendor call left anywhere in this decode path** — with a `cat` stranded in front of it and a
`copy_` stranded behind.

The champion writes those four multiplies out as torch operations instead.
`inline_causal_conv.py` contains **no Triton and no custom op**; its own docstring calls it "a
deletion rather than a kernel". The arithmetic does not change and neither does the output. What
changes is that inductor can now weld the step into the kernels on either side of it.

And the control is what makes it a finding rather than a tweak: **the identical arithmetic,
hand-written as a fast Triton kernel behind a `custom_op`** — a wall in a different place —
measured **0.7854** against this one's **1.0765**, same card, same process, minutes apart.
Writing the kernel was worse than writing nothing, and the reason is the wall, not the kernel.

### B. Move fewer bytes — the biggest table in the model, stored four times smaller

The tied LM head scores all **248,320** possible next tokens from a 248,320 x 2,560 table. It is
the largest single operand in the model and **14.80% of every byte the compiled baseline moves**.
Storing it at 4 bits instead of 16, with one scale per group of 128 weights, cuts that nearly
fourfold.

The objection is real, and it is why this slot was **registered in advance as a predicted loss**:
multiplying needs full-width numbers. If inductor unpacks that table into memory before
multiplying, the candidate moves the 317.85 MB of packed nibbles *and* the 1271.40 MB bf16 weight
it built from them, and no matmul however good can recover that.

**It does not unpack it into memory.** The unpack happens in registers, inside the matmul's own
prologue: one reduction kernel carries the shift, the mask, the `mm`, the final RMSNorm and the
residual add together, and the graph holds **no weight-sized buffer anywhere**. The same unpack
hand-written as a Triton kernel behind a custom op measured **0.9851**, a loss, because a wall at
that site forfeits the RMSNorm fusion the reference welds into the `lm_head` matmul.

> Neither champion computes anything faster than the compiler does. Both win by making more of
> the program *visible* to it — one by deleting a library call, one by deleting a hand-written
> kernel. That is the result, and it is why the most valuable entries in the leaderboard are
> losses.

---

## 2. What was measured, and against what

| | |
|---|---|
| Model | `Qwen/Qwen3.5-4B`, text decoder only (~4.21 B params; vision tower and MTP head excluded in `weights.py`) |
| Workload | batch 1, context 2048, 128 decoded tokens |
| Baseline | `src/deltaforge/reference.py` — pure PyTorch, **no custom kernels of any kind** — under `torch.compile(mode="max-autotune")` |
| Score | `median(t_compiled / t_candidate)` over interleaved rounds; > 1 beats the compiler |
| Noise band | interquartile spread of the per-round ratios. **A margin smaller than its IQR is not a win.** |
| Hardware | 47 RTX 5090 rentals, 9 RTX 4090; torch 2.11.0+cu128, Triton 3.6.0 |

**The opponent is the compiler.** The question is where a hand-written kernel beats
`torch.compile(mode="max-autotune")`, so the baseline has to be the compiler's own output and
nothing else: `reference.py` is pure PyTorch with no custom kernels anywhere in it, and
`max-autotune` compiles it.

HuggingFace's Qwen3.5 modeling code answers a different question. It dispatches Gated DeltaNet
to hand-written Triton via `flash-linear-attention` and attention to FlashAttention, so a
candidate measured against it would be scored on whether my Triton beats theirs — a worthwhile
question, and not this one. `transformers` loads weights, tokenizes, and serves as a correctness
oracle; it is never the baseline. For the same reason `reference.py` never calls
`scaled_dot_product_attention`: SDPA *is* the fused kernel under test.
`reference_purity_test.py` enforces this in CI.

**The baseline is the project's most reusable number:**

> `torch.compile(mode="max-autotune")` runs this decode path at **6.70–7.30 ms/token —
> 1177–1282 GB/s, 65.7–71.5% of an RTX 5090's 1792 GB/s vendor peak**, against a 4.79
> ms/token roofline on the bytes it actually moves.

---

## 3. Arithmetic before kernels

`docs/roofline.py` prints where every byte of one decode step goes. It needs no GPU and no
checkpoint, and it decides what *can* win:

| What moves | share of per-token bytes |
|---|---:|
| Weights, streamed once | **91.85%** |
| GQA `repeat_interleave` materialisation | **6.23%** |
| Recurrent state, read + write | 1.10% |
| KV cache read | 0.78% |
| SwiGLU intermediates | 0.026% |
| Norm + residual | 0.018% |

A hypothesis cannot beat the share of bytes it touches. Batch-1 decode is a weight-streaming
problem; hand-writing a fused RMSNorm has a ceiling of 0.018%, which is below the harness's
own noise band. **Five hypotheses are in the graveyard with the arithmetic that closed
them**, at a cost of zero GPU minutes.

**Then I read the generated code**, which three rentals had skipped and which cost nothing:

- **Inductor already eliminates the GQA head expansion** — `x1 // 4` indexing on the
  unexpanded KV cache, never materialised. That closes the backlog's second-largest
  hypothesis (6.23% of per-token bytes) **without a kernel**, and corrects the baseline's
  true traffic from 9158.23 to **8587.80 MB/token**.
- **There is no cuBLAS in this decode path.** `extern_kernels` is called for convolution and
  nothing else: inductor generates Triton for every matmul and welds the residual add and
  the RMSNorm *into* them. A hand-written GEMV gives all of that fusion up — 248 times per
  token.
- **The compiled baseline runs with no CUDA graphs at all**, because the causal conv mutates
  its cache in place (0 recorded, 128 skipped, measured on every slot of rental 40).

That reframed the exercise: the compiler is two thirds of the way to the memory wall *and*
fusing everything around the matmuls it emits, so the only large win left is to move fewer
bytes — and taking a matmul away from inductor has a price that is measurable.

---

## 4. Headline results

**Two champions, and neither contains a line of Triton.**

| | `045-inline-causal-conv` | `056-int4-head-torch-dequant` |
|---|---|---|
| Replaces | `causal_conv` | `decode_step` |
| Median ratio | **1.0765** (r45), **1.0650** (r46), **1.0778** (r56) | **1.0171** (r46), 0.9971 (r56), **1.0251** (r57) |
| IQR | 0.0552 / 0.0213 / 0.0322 | 0.0130 / 0.0230 / **0.0104** |
| Correctness | **bit-identical** — 264/264 tokens, 0.00000 nats | 0.9318 top-1 agreement, 0.01674 nats |
| Best candidate bandwidth | **1254 GB/s — 70% of vendor peak** | 1068 GB/s |

**Composed, they are the largest margin this project has recorded: `068-champion-pair` =
1.2179** (rental 56), on a card whose identity slot carried +2.36%, so read it as ~1.19
corrected. Notably it is **16% above the product of its two ingredients measured alone in
the same process** (1.0778 × 0.9971 = 1.0747) — a cross-slot comparison on a drifting card,
so the leaderboard declines to promote on it and says why.

---

## 5. Findings

### 5.1 An opaque custom op is a fusion barrier, and the bill is invisible at the call site

The cleanest controlled pair in the project. `044-fused-causal-conv` and
`045-inline-causal-conv` compute **the same function** — both bit-identical to the reference,
layer-1 relative error `0.0`, layer-2 264/264 at 0.00000 nats — and differ only in whether
the four taps reach inductor as a `torch.library.custom_op` or as torch operations it may
fuse. Measured in the same process, on the same card, minutes apart:

| | ratio | IQR | Δ ms/token |
|---|---:|---:|---:|
| `044` — the taps behind `custom_op` | **0.7854** | 0.0446 | +1.919 |
| `045` — the same taps as fusible torch ops | **1.0765** | 0.0552 | **−0.727** |

**One variable, 37%.** The `output_code` dump gives the mechanism, not a story: a barrier
forces inductor to materialise the op's inputs and outputs (**59 → 190 buffer allocations**
per decode step) and it splits a producer chain it had been fusing, then *recomputes* the
shared prologue rather than reading the buffer it just wrote — the linear-attention state
reduction runs **twice per layer, 24 times per token** (298 reductions in the reference, 297
in the inline candidate, **321** with the custom op).

Two corollaries worth as much as the number:

- **It settled a 20% regression that two earlier rentals had attributed to composing two
  kernels.** There was no composition effect: the conv loses 21% *alone*, and composed with
  the int4 head it loses 20% (`050` − `044` = +0.011, inside both IQRs). Nobody had measured
  the ingredient before the composition.
- **The allocation count was a red herring.** Rental 43 had ranked 59 → 190 allocations as a
  suspect. The winning candidate allocates **232 and wins**. The duplicated reduction is the
  mechanism; the allocation count is a symptom.

### 5.2 At the project's best site, the compiler beat the hand-written kernel

Three registrations of **one program** — the group-128 int4 dequantise-GEMV on the tied LM
head — ran in a single process on a single card. They compute the same function, and the
record proves it: layer 2 returned **0.9318 agreement and 0.01674 nats** for every one of
them, as it has on six rentals.

| | how the program reaches `torch.compile` | ratio | IQR |
|---|---|---:|---:|
| `054-int4-head` | a hand-written Triton kernel behind `torch.library.custom_op` | **0.9851** | 0.0099 |
| `055-int4-head-triton-op` | the same kernel behind `torch.library.triton_op` | **errored** | — |
| `056-int4-head-torch-dequant` | torch operations, no kernel of mine at all | **1.0171** | 0.0130 |

**The slot that won was registered in advance as a predicted loss**, on the argument that
inductor would materialise the head's 1271.40 MB/token bf16 weight. It does not. Its final
kernel is one reduction carrying `__rshift__`, `bitwise_and`, the `mm`, the final RMSNorm
and the residual add **together** — the whole grouped nibble unpack fused into the matmul
prologue — and the graph holds **no weight-sized buffer anywhere**, allocating 48 buffers
where the reference allocates 59.

So the hand-written kernel lost twice over: a custom op cannot keep the RMSNorm fusion the
reference welds into the `lm_head` matmul, **and** the register-level unpack that was its
whole claim to necessity is something the compiler emits inline. `tiled_int4_head` is
retired.

`055` is a recorded negative with engineering value: `torch.library.triton_op` does not
trace this GEMV on torch 2.11 — dynamo ran the body under `FakeTensorMode` and the launch
reached `.data_ptr()` instead of being intercepted by `wrap_triton`. It cost 34 s of GPU
time. The CPU suite could assert the op was *registered* and structurally could not assert
it *traces*, because tracing needs Triton.

### 5.3 At the MLP, inductor writes the matmul's product to memory

The obvious generalisation of 5.2 — "inductor fuses grouped dequantisation into a GEMV
prologue" — is **false at the MLP**, and the dump showed it before a kernel was written. At
the MLP projections a *pointwise* kernel carries the activation's producers, the whole
unpack and a multiply, and writes a full `(1, K, N)` **fp32** tensor — 94.37 MB a site —
which a **separate** split-K reduction reads back. Zero `extern_kernels.mm` anywhere.

That buffer is **not the dequantised weight**, which is what this section first said. It is
the product of inductor's M=1 `mm` decomposition, `(x.unsqueeze(2) * W.unsqueeze(0)).sum(1)`
in fp32, which `max-autotune` enables. The kernel stores `x[k] · W[k, n]`, and the shape
matches `W`'s, which is how it was misread. Inductor writes it out when the op counts of the
fused producers of `x` and `W` together cross `realize_opcount_threshold`. §7 has the
mechanism, the standalone repro and the upstream fix.

Counted per decode step from the candidate's graph, by kernel **launch** rather than by
allocation:

| product | sites | written per graph | of how many sites |
|---|---|---:|---:|
| `(1, 2560, 9216)` fp32 | `gate_proj` / `up_proj`, K=2560 N=9216 | **32** | 64 |
| `(1, 9216, 2560)` fp32 | `down_proj`, K=9216 N=2560 | **32** | 32 |

**64 of 96 sites.** An earlier census counted `empty_strided_cuda` allocations and got 17,
because inductor recycles the buffers through `# reuse` lines. The size-based reading — that
inductor "materialises when it can afford to" — is withdrawn: the rule that writes these
buffers counts ops and never looks at size.

And the prize this hides — int4 on 52.75% of per-token bytes — had been registered as a
predicted win **twice** and declined unexecuted both times on a precondition gate. Run
ungated, it is a **measured loss at 0.6390, IQR 0.0060**, the tightest band in its batch.
Taking the same construction to 200 sites and 97.85% of the bytes returned **0.6150**. Both
were measured with the compiler bug in place, and neither has been re-run without it.

### 5.4 The GEMV was grid-starved, not structurally slow — and an aggregate could not show it

Batches 003 and 004 installed a hand-written GEMV on all 248 layer projections at once and
lost by 5x (0.2801, then **0.1934 after a rewrite aimed at the diagnosed cause**). Measured
per site, the same kernel family gives:

| | achieved bandwidth |
|---|---:|
| naive GEMV, 248 layer projections | 319 GB/s |
| tiled GEMV (`tl.dot`, K-major, split-K), 248 layer projections | 228 GB/s |
| int4, **the LM head alone** | **656 GB/s** |
| int8, **the LM head alone** | **847 GB/s** |

At BLOCK_N=64 the head launches **3880 programs** on a 170-SM card; `in_proj_a` is 32
channels wide and launches **four**. Same kernel, same flop padding, same split-K, opposite
result. **An aggregate number over 248 sites could not distinguish a slow kernel from a
starved grid** — which is why the first two quantisation batches drew the wrong conclusion
from a correct measurement.

Relatedly, batch 003's kernels were never bandwidth-bound in the first place: **as they
removed bytes they got slower** — 26.90 → 38.68 → 42.49 ms/token while traffic fell 9158 →
5588 → 2850 MB/token — because a cross-lane `tl.sum` ran once per K-iteration instead of
once per output, and the dequantisation landed on an already-saturated issue port. The 1.85x
int8 ceiling is real arithmetic and unreachable by an implementation that is not spending
its time on memory.

### 5.5 The matmul dtype is a no-op, because the decomposition computes in fp32 either way

One operator apart, two arms per site width:

| pair | fp32 arm | bf16 arm | margin | arms' IQRs |
|---|---:|---:|---:|---|
| tied head, 248320 wide | **1.0251** | **1.0218** | −0.0033 | 0.0104 / 0.0081 |
| 96 MLP projections, 9216 wide | **0.6390** | **0.6452** | +0.0062 | 0.0060 / 0.0259 |

Both margins sit inside their own interquartile spreads and point in opposite directions;
the candidate columns agree to the digit on achieved bandwidth (**458 against 458 GB/s** at
the MLP). The dump says why: the bf16 candidate writes the same fp32 products as the fp32
one and no bf16 buffer of either shape. The M=1 decomposition runs under
`pw_cast_for_opmath`, which upcasts bf16 operands to fp32 before the multiply, so the
product is fp32 whatever dtype the source asks for. **Changing the matmul's dtype cannot
change the dtype of a product the compiler computes in fp32 regardless.**

### 5.6 A speculative verify is a step, not a slope

Nobody had put a number on what a `k+1`-token verify costs at batch 1, so I built
instruments rather than candidates: drafters that are **wrong on purpose**, so acceptance is
zero by construction (confirmed by histogram across 2159 cycles) and the measured ratio is
exactly `1/γ`, the inflation factor.

- 1 → 3 tokens costs **31.6%**; 3 → 5 costs **6.7%**. A one-token verify *is* ordinary
  decode at γ = 1.000, so **~23% of the cost is a discontinuity at the `seq=1 → seq>1`
  boundary** — dispatch and kernel-selection, not traffic. The registered cost model had no
  term for it.
- What γ really moves is **break-even**: from 0.06 to **0.317** mean accepted tokens per
  cycle for a free drafter.
- **Both registered kill criteria were crossed and both were withdrawn as underived** — the
  spec's own formula with the measured γ still gives 1.34x at k=2 and 1.45x at k=4 for an
  int4 self-draft at p = 0.9, with k=4 winning by more than the k=2 its criterion spared.
- The n-gram drafter measurably works and is **20x too weak**: 0.016 accepted against a
  0.317 bar. Its histogram is the reason — `{0: 2108, 1: 0, 2: 17}`: never one token, and
  0.8% of the time the whole block. That is prompt-lookup finding a literal repeat or
  nothing, which is a property of the workload, not a tuning problem.

### 5.7 The measurement is the instrument, and it had to be sharpened before three conclusions flipped

The retired int4 head is the clearest case. **The kernel never changed** and never returned a
correctness result other than 0.9318 / 0.01674. What changed is resolution:

| | rental 40 | rental 43 | rental 45 | rental 46 |
|---|---:|---:|---:|---:|
| reference achieved | **1282 GB/s** | 845 GB/s | 1197 GB/s | 1167 GB/s |
| `000-identity` calibration | 1.0008 | 1.0101 | 0.9972 | 1.0044 |
| the int4 head | **1.0791** | 1.0161 | 1.0105 (inconclusive) | **0.9851 (loss)** |
| IQR | 0.00034 | 0.00407 | 0.0263 | 0.0099 |

Going from 5 scoring rounds to **15** took the worst IQR in a batch from 0.1511 to 0.0213
and `inconclusive` slots from six of eleven to one of ten, **on the same physical host
drifting the same way** — and it is what resolved the int4 head's loss, the static cache's
+1.5% win, and the fact that dispatch savings are **not additive** (the cache is +1.5% alone
and +0.0% on top of the fused conv; `058` and `060` return the same 7.251 ms/token from
genuinely separate runs).

**The card is an uncontrolled variable the size of the effects under test**, and the
write-ups name the rental for every number. Two RTX 5090s reporting the same memory clock,
driver and torch differed by **1.61x** on the baseline; one card lost 17% of its SM clock
mid-rental (2910 → 2400 MHz). Seven identity slots separate on exactly one field — the
**host driver**: every driver seen twice sits in 1197–1283 GB/s, and the only outliers
(441 GB/s, the worst in the project) are the only drivers seen once. Interleaving the
reference and candidate *within* each round divides drift out of every ratio, which is why
no slot was voided; absolute predictions and cross-rental comparisons do not survive it.

**And the obvious next step was refuted rather than assumed.** If the GEMV won at the head
because the head has parallelism, a better *tile* should unlock the other 83% of the bytes.
Searched on the card — BLOCK_N, split-K, BLOCK_K, warps, pipeline depth — the search made the
champion's own site **2.3x slower** (656 → 282 GB/s) while the MLP came back at 67 GB/s,
within noise of the 65 measured two rentals earlier at a different tile in a different
kernel. Two more tiles were then pinned in advance (0.9343, 0.9502) and the last untried
direction after that (0.9617). **Five measured points, four deliberate attempts, and the
heuristic tile nobody chose on purpose is the best of them. The tile question is closed.**

---

## 6. Method

A kernel is specific to one model on one card; a measurement you can trust is not. This is how
every number above was produced, and it is the part of the project that transfers.

**Interleaved scoring.** Both models are resident at once and timed in alternating rounds,
so thermal drift and clock changes cancel. They are separate module trees — installing a
kernel swaps a class on the candidate's modules — but they **share one set of parameter
tensors**, since nothing writes to a weight under `no_grad`: 8.4 GB on the card instead of
16.8, and one read of the checkpoint instead of two. `cli._assert_parameters_are_shared`
fails the run if that silently stops being true.

**A calibration slot that can void the batch.** Every batch opens with an identity champion
that installs nothing and must measure 1.00 ± noise. It has measured 1.0009, 1.0018, 1.0024,
0.9913, 1.0008, 1.0053 (IQR 0.00086) and 1.0002 (IQR **0.0001**). When it carried +2.36% on
rental 56, that batch's write-up reports every slot in a *corrected* column and promotes
nothing. **A batch whose identity slot misses 1.00 voids every number in it, in writing.**

**Two correctness gates, with magnitudes recorded as numbers.** Per kernel, `allclose`
against the reference operation at bf16 tolerance on real model shapes, with max absolute
and max relative error recorded rather than reduced to pass/fail. End to end, greedy-decode
128 tokens from five fixed prompts checked into the repo; the candidate's token sequence must
match eager exactly. Failures are kept in `results/` with their error magnitudes, because a
candidate that was fast but wrong is among the most useful things to read.

That discipline paid directly: **none of batch 003's six `incorrect` verdicts was a kernel
bug.** Every Triton kernel passed its numerics gate at every probe, worst relative error
7.8e-3 — about one bf16 ULP. All six were defects in the *gates*: an `exact` gate on a kernel
that computes the same function but not the same bits (fp32 accumulation in a different order
from cuBLAS lands one ULP away, and this model's top-2 logit gaps flip an argmax on that); a
top-1 agreement bar set finer than the statistic can resolve (at n=264, agreement quantises
to 1/264 = 0.0038 — **one slot missed its bar by 0.000303, eight hundredths of a single
token**); and a slot gated against a reference it never claimed to match. Five more gate
defects have been found and recorded since, including a zero-floor margin gate that passes
on +0.0062 against an IQR of 0.0259.

**Predictions registered before the rental.** Each hypothesis commits its predicted outcome,
magnitude range and reasoning to `src/deltaforge/batches.py` **before** the GPU is created,
and every batch write-up scores them:

| batch | predictions scored |
|---|---|
| 003 int8 weight-only | 1 of 7 |
| 004 bandwidth-bound GEMV | 1 of 2 (five slots declined, recorded as untested) |
| 005 launch and head | 3 of 5 |
| 006 tile and sites | 1 of 4 |
| 007 compose and retile | 2 of 6 |
| 008 ingredients and barriers | 5 of 11 |
| 009 visible kernels | 5 of 10 |
| 010 speculative verify | 3 of 5 |
| 011 bytes not kernels | 4 of 5 scored slots (identity did not calibrate) |
| 012 the matmul dtype | 5 of 6 verdicts, **3 of 6 magnitudes** |

Low scores are the point: an unfalsifiable account scores 100%. The three misses that
mattered most each corrected a mechanism — the predicted-loss slot that won (5.2), the
generalisation the dump refuted for free (5.3), and a cost model with no dispatch term
(5.6). Batch 012 scores *magnitudes* separately from verdicts, because a registered
prediction that gets the verdict right for the wrong reason is worth less than a verdict
column makes it look.

**An explicit outcome taxonomy**, so that "we didn't measure it" never reads as "it
doesn't work": `win`, `loss`, `inconclusive` (inside the noise band — neither promoted nor
buried), `graveyarded on mechanism` (closed by arithmetic, with the numbers that closed it),
`incorrect` (failed a gate, with error magnitudes), `precondition_failed` and `error`. A
precondition-skipped slot is recorded as **unexecuted, not refuted** — two defects survived
a rental that way, and the 52.75% prize in 5.3 was declined twice before it was finally
measured.

**Batching, because fixed cost dominates.** A rental's fixed cost — image, torch, a 9.32 GB
checkpoint, the GPU test suite and one `max-autotune` compile of the reference — is ~19
minutes measured; each additional hypothesis costs 3–6. So a rental measures **7–12
hypotheses**, compiling the reference once, with the reference column re-timed inside every
hypothesis's own interleaved rounds so per-round drift still divides out exactly. Each slot
runs in its own try/except and writes its record the moment it finishes, so a wrong kernel
costs a slot rather than a rental.

**Cost control in code, not discipline.** $50/month: a month-to-date gate that refuses at
$45, a session soft gate at 180 billed minutes checked before a run and never during one, a
**pre-flight refusal when the remaining budget cannot fit the identity slot plus one kernel
slot**, per-slot wall-clock caps, a local-side watchdog that destroys the instance at 210
minutes regardless of what the remote is doing, a batch that stops *itself* before a
hypothesis it cannot finish, teardown that pulls results back **before** destroying, and a
shell trap on `EXIT`/`INT`/`TERM` so a crash still destroys the instance. Every script in
`remote/` supports `--dry-run`, which exercises every gate and the teardown trap without
contacting the create or destroy endpoints.

**Ledger:** 61 instances provisioned, **61 destroyed, zero leaked**, 1714 billed GPU
minutes, **$12.1196 lifetime**, including two cancelled mid-flight with SIGTERM. Five of
those instances ($0.63) ran the upstream report in §7 rather than a batch. The
cheapest informative rental cost $0.0478; the most expensive mistake cost $1.3878 and spent
its entire cap inside one compile.

**The compile cache is a measured artifact, not a hope.** A cold `max-autotune` compile had
never finished inside a session — 6980.9 s unfinished — until an unrolled prefill scan was
found and fixed; it then completed in **267.5 s**, and **57.4 s** warm off the cache banked
from the previous rental on the same card. Teardown reports warmth from bytes that actually
landed on disk, because a log that overstates it would corrupt the number the cache exists
to improve. A later rental spent ~33 minutes pushing a 2.2 GB cache to save ~3.5 minutes of
compiling, so pushes above `DF_CACHE_MAX_PUSH_MB` are now refused — fixed cost on the same
host fell from ~72 minutes to ~22.

**Twenty-two blockers, and three that hid each other.** They are tracked with status and
proof, because each rental that got further than its predecessor did so by exposing the next
one. The three could not even be *seen* until the one before was fixed: a cold compile
that never finished hid a teardown bug that only fires *between* slots, which in turn hid
dynamo's `recompile_limit` (8) silently timing **eager** candidates against a compiled
reference — the cause of six unrelated kernels all returning ratios between 0.146 and 0.157.
Those six numbers are kept in the leaderboard marked `void — candidate ran eager`, because
deleting evidence is worse than labelling it.

**Scale:** ~37k lines of Python and POSIX shell and **1160 tests — 1142 of which pass on a
laptop with no GPU and no checkpoint** (18 skip without a card), plus `cli fusion` — which compiles a
candidate locally and diffs the generated code, so "does this registration install a fusion
barrier?" is answered **before** renting a card. That tool is what refuted 5.3's
generalisation for free.

---

## 7. Upstream: the 5.3 buffers are a compiler bug, now reported with a fix

Reading rental 57's generated code more closely changed what 5.3 and 5.5 are about. **The
`(1, K, N)` fp32 buffers are not the dequantised weight.** They are the product of inductor's
M=1 `mm` decomposition, `(x.unsqueeze(2) * w.unsqueeze(0)).sum(1)` in fp32, which
`coordinate_descent_tuning` (and so `max-autotune`) turns on. The producers of `x` and `w` are
fused into that product, and when their op counts together cross `realize_opcount_threshold`
(30 on GPU) while neither crosses it alone, inductor writes the product to memory and reads it
back: 4·K·N bytes each way, against 2·K·N for a bf16 weight and K·N/2 for int4. Counted
through the `# reuse` lines that the 17-buffer census missed, rental 57 writes it at **64 of
96 MLP sites per decode step**. Sections 5.3 and 5.5 carry the corrected account; this
section is how it was established.

It reproduces outside this project, on torch nightly and the 2.14.1 release. On an RTX 5090,
nightly, with the threshold raised to 1000 as the comparison (a workaround that changes
realisation for the whole graph):

| standalone case | K × N | as compiled | threshold 1000 | slower by |
|---|---|---:|---:|---:|
| `(silu(a)*b) @ dequant_int4(W)` | 9216 × 2560 | **92.8 µs** | 18.3 µs | 5.1x |
| same | 2560 × 9216 | **95.4 µs** | 17.0 µs | 5.6x |

It does not need int4. A bf16 `x @ W` (9216 × 2560) with 24 cheap pointwise ops on `x` takes
**121.9 µs**, against 37.1–37.8 µs with 22 ops or fewer and 39.4–40.2 µs with 28 or more,
where `x` alone crosses the threshold and inductor realises it instead.

The fix realises the small broadcast input (`x`, K elements) instead of the broadcast result,
the choice inductor already makes when one input crosses the threshold by itself. On one RTX
4090, main against the fix: 221.3 → 74.3 µs for the bf16 case and **179.5 → 27.1 µs (6.6x)**
for the int4 one, with unchanged numerics and no change wherever the rule does not fire.

- **Issue:** [pytorch/pytorch#200539](https://github.com/pytorch/pytorch/issues/200539), filed
  2026-10-10.
- **Fix:** [pytorch/pytorch#200540](https://github.com/pytorch/pytorch/pull/200540), a draft PR
  with two regression tests that fail on main. It awaits maintainer triage and is **not
  merged**.
- **Cost:** five more rentals, $0.63, in the same ledger as the rest. They ran outside the
  batch harness: no manifest registered predictions for them, and the timings are Triton
  `do_bench` times of one compiled function, not interleaved ratios against a reference.

**What it means here is untested.** The int4 slots `077` (MLP, 0.6390) and `081` (wide, 0.6150)
most likely lost to this bug: 64 product round trips are enough bytes to cover the whole gap
§9 describes. A slot with the threshold raised, or with the fix, is the next experiment;
until it runs, those losses stand as measured.

---

## 8. What is *not* claimed

Stated so that the win condition is unambiguous:

- **Not beating cuBLAS on dense bf16 GEMM.** At batch 1 the linear layers are at the
  bandwidth roofline; you do not beat a roofline with a better kernel, only by moving fewer
  bytes or running a different algorithm.
- **Not beating a compiler at elementwise fusion.** Inductor's home turf, and it emits
  Triton. Those hypotheses are in the graveyard with the arithmetic that closed them.
- **Not beating vLLM or SGLang.** Those are already hand-tuned Triton and CUDA; out of scope
  as a win condition.
- **Not a serving engine, not training, not multi-GPU.** Single-card inference decode only.
- **Every number is measured.** Nothing in the repository is estimated, illustrative or
  placeholder; each figure came off a rented card, and the ones that do not mean what they
  appear to mean say so in the line that reports them.

---

## 9. Open, and recorded as open

- **How much of the 5.3 loss the product buffers explain is not measured.** The
  candidate's traffic, inferred from its achieved bandwidth, is 13.44 GB/token against a
  5.19 GB/token byte model: a gap of 8.25 GB. The 17-buffer census priced its buffers at
  3.21 GB and left ~5 GB unattributed. The 64 products actually written, priced at full DRAM
  cost, come to **12.08 GB/token**, which is more than the gap, so the census no longer
  under-explains it. A 94 MB buffer read back right after it is written, on a card with a 96 MB
  L2, probably does not cost its full DRAM price, but nothing has measured that split. The
  direct test is the one §7 names: run `077` without the product being written.
- **One rental-40 number has never reproduced.** The int4 head's 1.0791 belongs to that
  rental; later rentals on healthy cards gave 1.0161, 1.0105 and 0.9851. The leaderboard
  does not pretend otherwise.
- **One infrastructure blocker has no real fix**, only a manual `--exclude-machines`
  workaround: a phantom ask traps the deterministic, price-ordered offer search.

---

## 10. Where to look

| Path | What it is |
|---|---|
| `README.md` | The project's own front page: claims, non-goals, reproduction |
| `LEADERBOARD.md` | Every hypothesis attempted, with ratio, IQR, correctness, outcome and the rental it belongs to |
| `docs/roofline.py` | Where the bytes go. No GPU, no checkpoint — run it first |
| `docs/HYPOTHESES.md` | The backlog, and the graveyard with the arithmetic that closed each entry |
| `docs/BATCHES.md` | How a batch works and what filling one requires |
| `src/deltaforge/reference.py` | The baseline. Pure PyTorch, no custom kernels, ever |
| `src/deltaforge/fusion.py` | What inductor generated, parsed; which registrations are opaque. No GPU |
| `src/deltaforge/batches.py` | Batch manifests — the predictions, committed before each rental |
| `src/deltaforge/harness/` | Interleaved timing, correctness gates, result records |
| `remote/` | Instance lifecycle in POSIX shell, so it works before the environment exists |
| `results/batches/0NN-*/README.md` | One write-up per rental, scored against its registered predictions |

Reproduce the parts that need no GPU:

```sh
uv sync --extra dev && uv run pytest          # 1142 pass, 18 skip, no GPU needed
uv run python docs/roofline.py                # where the bytes go
remote/run_remote.sh --dry-run --session-id smoke --batch 001-calibration
uv run python -m deltaforge.cli fusion --install inline_causal_conv
```

Licensed Apache-2.0, matching the target model.
