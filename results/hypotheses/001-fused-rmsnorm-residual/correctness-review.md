# Correctness review — hypothesis 001

Reviewed by reading, against `src/deltaforge/reference.py`. **The kernel has never
executed**, so nothing below is confirmed by measurement. That distinction is the most
important thing on this page: the analysis says the arithmetic should match the reference,
and only the GPU gates can turn "should" into "does".

## What the review can and cannot establish

| Claim | Status |
|---|---|
| The fused op computes the same function as the reference, on paper | Reviewed, argued below |
| Installation swaps the right modules and leaves other models alone | **Verified** — CPU tests, passing |
| Weights survive installation by identity, not reconstruction | **Verified** — CPU test on `data_ptr` |
| Ops refuse to run without CUDA rather than falling back | **Verified** — CPU tests |
| Shape, dtype and layout contracts | **Verified** — CPU tests |
| Launch geometry covers the whole row for every width used | **Verified** — CPU tests |
| bf16 output matches the reference within `rtol=atol=1e-2` | **Not verified** — GPU test written, never run |
| The residual stream matches the reference bit for bit | **Not verified** — GPU test written, never run |
| The installed model's logits match the reference's | **Not verified** — GPU test written, never run |
| The custom ops survive `torch.compile(max-autotune)` | **Not verified** — GPU test written, never run |
| The kernel is *faster* | **Not measured at all** |

## The arithmetic, step by step

Reference (`reference.py:77-92`, and the layer at `reference.py:617-638`):

```python
hidden = residual + hidden  # bf16 add -> bf16 result
out = hidden.float()
out = out * torch.rsqrt(out.pow(2).mean(-1, keepdim=True) + eps)
out = out * (1.0 + weight.float())
return out.type_as(hidden)  # back to bf16
```

Kernel (`_add_rms_norm_kernel`):

1. `summed = (x.f32 + r.f32).to(RESID.dtype)` — the add is performed in fp32 and **rounded
   to bf16 before anything else reads it**. This matters: `residual + hidden_states` in the
   reference produces a bf16 tensor, and the norm upcasts *that rounded value*. Keeping the
   unrounded fp32 sum would be more accurate than the reference and therefore a different
   function. Addition is commutative here, so `attn_out + hidden` matching the reference's
   `residual + attn_out` is not an issue.
2. `s = summed.f32`, `var = tl.sum(s*s)/N`, `rstd = 1/sqrt(var + EPS)` — same reduction, in
   fp32, dividing by the true row width `N` rather than by `BLOCK`. Masked lanes load `0.0`
   and contribute exactly zero to the sum, so padding to a power of two does not perturb the
   mean. This is the single most likely place for a silent error and it is handled.
3. `y = s * rstd * (1.0 + w)` — the `1 + weight` convention. Qwen3.5 stores norm weights
   centred on zero; reading them as a plain scale produces an all-zero activation on a
   freshly initialised module. There is a GPU test aimed squarely at this.
4. Both stores are masked, so no lane writes past the row.

The plain `rms_norm` path is the same minus step 1.

**Conclusion:** on paper the kernel computes the reference function, with the same
intermediate rounding. The residual output should be bit-identical (a single bf16 add); the
normalised output should differ only in the last bits from a different fp32 reduction order,
well inside the `1e-2` bf16 gate.

## Dataflow around the kernel

`FusedDecoderLayer.forward` against `DecoderLayer.forward`:

| Reference | Fused | Same? |
|---|---|---|
| `h1 = input_layernorm(h)` | `rms_norm(h, w_in, eps)` | yes |
| `a = attn(h1)` | unchanged | yes |
| `r = h + a` | first output of `add_rms_norm(a, h, ...)` | yes (commutative) |
| `h2 = post_attention_layernorm(r)` | second output of the same call | yes |
| `return r + mlp(h2)` | `return residual + self.mlp(normed)` | yes |

The layer's final add is left to PyTorch. Fusing it would require pairing it with the *next*
layer's `input_layernorm`, which straddles the layer boundary — deliberately out of scope
here and recorded as such.

## Findings from the review, and what was done

**1. Non-contiguous input was refused, then coerced anyway.** `_check` raised on a
non-contiguous activation while the very next line called `.contiguous()`, making the
coercion unreachable. Worse, refusing is the wrong behaviour: the reference norm accepts any
layout, so the candidate would have failed where the baseline works — a harness bug wearing
a kernel's clothes. **Fixed:** layout is coerced, the row stride is read after the coercion,
and a GPU test feeds it a genuinely strided view.

**2. `add_rms_norm` never checked that `residual` matched `x`.** A broadcastable mismatch —
`[1, 8]` against `[2, 8]` — would have launched cleanly and computed something else, because
the kernel indexes both tensors with one row stride taken from `x`. **Fixed:** shape and
dtype must agree exactly; two tests cover it.

**3. The weight was never checked for being a contiguous vector.** The kernel indexes it by
column with unit stride. **Fixed**, with a test.

**4. Two of my own guard tests passed for the wrong reason.** With the device check first,
the CPU tests for findings 2 and 3 raised on "requires a CUDA tensor" and never reached the
check they claimed to exercise — they would have passed with the fix absent. **Fixed:** the
structural checks now run before the device check, and the tests assert on the specific
message. A guard test that cannot fail is worse than no test, because it reports safety
that is not there.

## What remains unverified, and why that is the whole story

The numerics are argued, not observed. Specifically unproven:

* that Triton's `tl.sum` reduction over 4096 lanes lands within the bf16 tolerance on the
  real weight distribution;
* that `.to(RESID.dtype.element_ty)` rounds identically to PyTorch's bf16 add;
* that inductor keeps the custom ops opaque under `max-autotune` and still CUDA-graphs the
  rest, rather than graph-breaking around them — which would silently make the scoring
  column measure something other than what it claims;
* that the assembled model's logits track the reference over 128 greedy tokens.

Every one of these has a written test that skips on a real condition and will run
unattended the moment a GPU session succeeds. None of them has run.

**No promotion.** The registry lists this kernel as champion of `rms_norm_residual` only
because being champion is what puts it in the candidate column and gets it measured at all.
It has not beaten anything, and `LEADERBOARD.md` records no ratio for it.
