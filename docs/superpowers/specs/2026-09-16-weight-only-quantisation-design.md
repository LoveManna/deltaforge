# Weight-only quantisation — batch 003

**Date:** 2026-09-16
**Branch:** `batch/003-int8-weight-only`
**Status:** written before the rental. Every prediction below is registered in
`src/deltaforge/batches.py` and committed before any GPU is billed.

---

## 1. Why this hypothesis and not another

Batch 001 measured eight kernels. Their byte shares were 0.004%, 0.018%, 0.026%, 0.78%,
1.10% and 6.23%. **Not one of them could have won by a margin worth reporting**, because a
hypothesis cannot beat the share of per-token bytes it touches, and the largest share in
that batch was 6.23%.

`docs/roofline.py`:

| What moves | MB/token | share |
|---|---:|---:|
| **Weights, streamed once** | **8411.51** | **91.85%** |
| GQA `repeat_interleave` materialisation | 570.43 | 6.23% |
| everything else | 176.29 | 1.92% |

At batch-1 decode **the model is the weight stream**. Every other hypothesis in the backlog
is a rounding error against it, and the only way to attack it is to make the weights
smaller. That is hypothesis 1 in `docs/HYPOTHESES.md`, it is the only open entry whose
ceiling is above 1.0, and it has never been attempted.

```
weight-only quantisation (hypothesis 1's ceiling)
  fp8/int8  total   4.95 GB  ->   1.85x speedup
  int4      total   2.85 GB  ->   3.21x speedup
```

## 2. The mechanism, and why the compiler cannot take it

A GEMV at batch 1 reads `W` once and does two flops per weight. It is pinned to the
bandwidth roofline, and **a roofline is not beaten by a better kernel**. It is beaten by
moving fewer bytes — Category **B** in `docs/HYPOTHESES.md`, a choice about representation
rather than about scheduling.

Store `W` at 8 or 4 bits and dequantise it **inside the GEMV's K-loop**: the int8 value is
loaded, converted in a register, multiplied into the accumulator, and never written
anywhere. Weight traffic halves or quarters.

The compiler cannot express that. Given `dequant(W_q, s) @ x` it has no way to keep the
dequantised weight off the memory bus: it materialises a full bf16 `W` into global memory
and calls cuBLAS, **adding** an 8.4 GB write on top of the read.

**That claim is an assertion in this repository, not a measurement**, and its own entry
says so: *"Verify the claim above before building on it… Either way, measure, do not
assume."* Recent inductor has prologue fusion into its mm templates and might fuse some of
the dequant — though at M=1 it is probably not using a template at all. So the batch
measures it. `010-int8-dequant-torch` is exactly that program, written in PyTorch and
handed to `max-autotune` like any other candidate, and it is predicted to **lose**.

## 3. The batch

Seven slots. Two of them are controls, and they are the difference between a ratio and an
explanation.

| # | Slot | Sites | Share of per-token bytes | Ceiling | Predicted |
|---|---|---|---:|---:|---|
| 0 | `000-identity` | none | — | 1.00 | `identity` |
| 1 | `009-gemv-bf16-control` | every layer projection, bf16 | 0.0 | 1.00 | `inconclusive` |
| 2 | `010-int8-dequant-torch` | every layer projection, torch dequant | 77.9% | <1.0 | **`loss`** |
| 3 | `011-int8-mlp` | MLP only | 49.5% | 1.33× | `win` |
| 4 | `012-int8-all-linear` | every layer projection | 77.9% | 1.64× | `win` |
| 5 | `013-int8-full` | + the tied LM head | 91.8% | 1.85× | `win` |
| 6 | `014-int4-full` | the same at 4 bits, group 128 | 91.8% | 3.21× | `win` |

**Slots 3-5 are a dose-response ladder, not three results.** They are the same kernel over
a strictly increasing set of sites. If a larger share of the weight stream does not buy a
larger win, then whatever is happening is not the mechanism claimed here, and the batch
says so without needing a second rental. `batches_test.py` asserts the ladder is a real
subset chain, because two slots accidentally covering the same sites would erase it and
each would still return a plausible number.

**Slot 1 is the divisor.** A hand-written GEMV competing with cuBLAS is an open question
independent of quantisation: if `009` returns 1.0 the int8 wins are bytes, if it returns
0.6 the kernel is simply slow and int8's true margin is larger than it looks, and if it
returns 1.3 then part of every int8 win belongs to the GEMV rather than to the bits. No
other slot can separate those.

**Slot 2 is the claim under test.** The outcome that would matter most is the one that
refutes the prediction: if the torch dequant *wins*, the compiler can express weight-only
quantisation, the kernels in slots 3-6 are worth far less than claimed, and the backlog's
top entry needs rewriting.

## 4. Correctness — the part that needed new harness code

**A quantised candidate cannot pass the layer-2 exact-token gate.** Not because it is
broken, but because it computes a deliberately different function. Running it through the
exact gate would record `incorrect` for every weight-only hypothesis this project will ever
run, and a gate that cannot tell *wrong* from *different* reports nothing.

`docs/HYPOTHESES.md` already specifies the answer. It is now implemented:

**Layer 1 — unchanged, and sharper here than usual.** Each Triton op is compared against a
PyTorch expression computing the identical quantised arithmetic in the identical order —
dequantise in fp32, matmul in fp32. The comparison is deliberately *not* against the bf16
reference: that difference is the quantisation error, which is the hypothesis. Anything but
near-ULP agreement at layer 1 is a kernel bug.

**The probes are derived from the model, not hardcoded.** `_launch_shape` picks 8, 16 or 32
rows per program by width, and each branch is a different tiling with a different tail mask.
A fixed pair of probes — `gate_proj` and `down_proj` — covers the 32 and 16 branches and
leaves the 8 branch (`k_proj`, `v_proj`, `in_proj_a`, `in_proj_b`) entirely untested, on a
kernel that installs there in three of the seven slots. `_probe_weights` therefore groups
the sites a hypothesis actually installs on by branch and takes the narrowest `N` in each,
because the narrowest is the one whose last block is masked.

**Layer 2 — `check_distribution`.** Teacher-force both models over the reference's own
greedy continuation (prompt + 128 tokens, five prompts, ~765 positions) and score:

* **top-1 agreement** — the fraction of positions where the argmaxes match. Teacher-forced
  rather than free-running, because once two decoders disagree once they are reading
  different text: a free-running agreement rate measures the first divergence and then
  nothing.
* **mean KL(ref ‖ cand)** in nats. Agreement alone can pass a distribution that has been
  shredded everywhere the argmax happens to be safe. A monotone rescaling of the logits
  keeps agreement at exactly 1.0 and is a different model; the KL bar sees it, and
  `correctness_test.py` demonstrates precisely that case.

**The gate stays on the decode path.** The context goes through in 64-token chunks carried
by the model's own cache, not in one 150-row pass. That is not a memory decision: above
`GEMV_MAX_ROWS` the op takes its dense fallback, so a single-pass gate would score the
fallback, the kernel would never run there, and the only thing between a wrong kernel and a
reported win would be layer 1's probes. Chunked and whole-sequence are equivalent —
`reference_test.py` pins that — and both models are chunked identically, so any residual
rounding difference cancels in the comparison rather than entering it.

**Both bars are registered per hypothesis, in `batches.py`, before the rental**, for the
same reason the prediction is. `Hypothesis.__post_init__` refuses an approximate slot that
carries no bars, refuses an exact slot that carries bars nothing would read, and refuses an
identity slot gated approximately — a calibration slot that could not match tokens would
hide a harness fault behind a tolerance.

| Slot | top-1 floor | KL ceiling (nats) | Why |
|---|---:|---:|---|
| `010`, `011`, `012` | 0.98 | 0.01 | per-channel int8 is the mildest weight-only scheme there is |
| `013` | 0.97 | 0.02 | the head is the last projection before the argmax; nothing downstream attenuates its error |
| `014` | 0.85 | 0.15 | 16 levels, and `in_proj_a`/`in_proj_b` feed an exponential |

The measured numbers go into the record whether they clear the bar or not. A fast candidate
that fails the gate is a result and is recorded as one.

## 5. Implementation notes worth keeping

**Installation is a `__class__` swap on existing `nn.Linear` instances**, and the quantised
copies are **buffers**, not parameters. `cli._assert_parameters_are_shared` walks
`named_parameters` and would reject a quantised weight registered as one — it has no
counterpart in the reference. The bf16 `weight` Parameter therefore stays registered and
shared, and the candidate still costs the reference's 8.4 GB plus the quantised copies
(3.6 GB at int8 over layer projections, 4.2 GB including the head, 2.1 GB at int4).

**`reference.py` gained one method and no arithmetic.** The tied LM head is 15.1% of weight
bytes and had no `nn.Linear` to swap: with `tie_word_embeddings` it was
`F.linear(h, self.lm_head_weight.to(h.dtype))` written inline in `forward`. That expression
is now `ReferenceModel.project_logits`, moved verbatim. Extracting it is not the
accommodation `AGENT.md` §8 forbids — no kernel enters the file, nothing computes
differently, and `reference_purity_test.py` still holds. What invalidates stored results is
a change to what the baseline *computes*.

**`GEMV_MAX_ROWS = 64`.** A GEMV re-reads the whole weight per row of `x`, so at the
2048-row prefill it would read 7 GB per layer and the rental would end inside slot 0. Above
the threshold the op dequantises and calls `F.linear` — the same quantised numerics, a
different implementation, and never inside a timed region, because `run_interleaved`
excludes every `setup` and the prefill is setup. Both `DEFAULT_WORKLOADS` batch sizes are
1 and 32; `quantised_linear_test.py` pins that they stay under the threshold, since a
workload climbing past it would quietly stop measuring the kernel — blocker 16's shape
exactly.

**The launch geometry is the one number no CPU can validate.** A GEMV has no M to spread
across the card at batch 1, so the grid *is* `ceil(N / BLOCK_N)`. At 64 rows per program
`down_proj` (N=2560) would put **40 programs on a 170-SM card** and lose three quarters of
the GPU on 24% of the weight bytes — correct, and slow for a reason that has nothing to do
with quantisation. `_launch_shape` therefore uses 8/16/32 rows by width, keeping every
projection at ≥128 programs, except `in_proj_a` and `in_proj_b`, which are 32 output
channels wide and cannot be spread by any block size. Those need split-K, which is a
different kernel; they are 0.03% of per-token bytes and `012`'s rationale says so.

## 6. What would make this batch a failure rather than a result

* **The identity slot misses 1.00.** Everything else is void, as always.
* **`graphs_compiled` is 0 anywhere.** Blocker 16's fix has never run on a GPU. Seven slots
  raises dynamo's limit to 22; a 0 in any record means that slot's ratio is not a
  comparison, and the batch must say so rather than report it.
* **Layer 1 fails.** Then the kernel is wrong and the ratios measure a wrong kernel. This is
  the gate that matters most for this batch and it is exact.

### The near miss this design already had

The first launch of this batch was killed two minutes in, at $0.014, before the checkpoint
download. The gate as written fed the whole ~150-token context through in one pass, which is
above `GEMV_MAX_ROWS`, so **layer 2 would have scored the dense fallback and never run the
kernel at all** — and layer 1's two hardcoded probes missed the 8-row launch branch, which
three slots install on. A wrong kernel on `k_proj`, `v_proj`, `in_proj_a` and `in_proj_b`
would have passed both gates and reported a ratio. Both holes are closed above; the tests
that pin them are `test_layer_one_probes_every_launch_shape_branch_the_kernel_will_take` and
`test_the_approximate_gate_stays_on_the_decode_path`.

The general shape is the one `AGENT.md` §8 keeps finding: **a gate whose subject was never
reached.** Blocker 7 was a guard firing before its subject existed, blocker 9 one firing
after its subject had succeeded, blocker 15 a cleanup whose justification nothing had
exercised. This was a gate that would have passed without the thing it gates ever running.

## 7. What is deliberately not here

* **A comparison against Marlin / machete / AWQ.** `docs/HYPOTHESES.md` is right that they,
  not `torch.compile`, are the real competition for a weight-only kernel, and that beating
  the compiler here proves less than it looks. But the project's win condition is stated
  against `torch.compile(mode="max-autotune")` and changing it mid-batch would make this
  result incomparable with every other row on the leaderboard. The honest framing belongs
  in the writeup, and a published-kernel column belongs in a later batch as an unscored
  reference point.
* **Activation quantisation.** Weight-only is the whole hypothesis: activations at batch 1
  are kilobytes against gigabytes of weights, so quantising them buys nothing and costs
  accuracy.
* **A long-context workload.** The KV cache and the GQA expansion grow with context and
  would change the denominator this batch's shares are computed against. Batch 003 measures
  the headline workload the leaderboard is defined on.
