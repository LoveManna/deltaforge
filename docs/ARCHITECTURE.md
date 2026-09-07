# Qwen3.5-4B: the resolved architecture

Every fact here was read from the published checkpoint on **2026-08-30** — from
`config.json` and from the safetensors headers of both shards — not from a blog post, a
sibling model, or inference from the model card. The design spec left two items open
(section 16); both are closed below.

> **Why this model and not a newer one.** Qwen3.8 ships no small checkpoint: the family is
> 27B (55.6 GB), a 2.4T MoE, and a 360 GB Flash-Next. Qwen3.6 is 27B/35B-A3B only. The
> newest *small* Qwen is therefore 3.5, and 4B is the largest of them that leaves room for
> a bf16 baseline and a candidate in one process on one rented card — which the interleaved
> A/B/A protocol requires. `Qwen/Qwen3.8-27B` is the same architecture (`model_type:
> qwen3_5`); its config is transcribed and verified in `config.py` and selectable with
> `--model`, for a future session with an 80 GB card. All 851 of its decode parameters were
> checked against the published safetensors headers on 2026-09-04 and match.

The tensor names, dtypes and shapes are checked into the repo at
`src/deltaforge/data/qwen3_5_4b_manifest.json` (metadata only, ~80 KB, no weight data).
`weights_test.py` uses it to prove on CPU, with no download, that every one of the 738
real tensors is mapped or deliberately skipped and that every reference parameter has
exactly the right shape.

## The two open items, closed

**The FFN is dense, not sparse-MoE.** `intermediate_size` is 9216, `mlp_only_layers` is
empty, and there is no expert, router or MoE key anywhere in `text_config`. The checkpoint
agrees: every layer carries exactly three dense SwiGLU projections
(`mlp.{gate,up,down}_proj`) and there are no expert tensors. **A fused-MoE-routing
hypothesis is therefore retired before it started**, and never entered the backlog.

**The vision tower is cleanly separable.** The checkpoint is
`Qwen3_5ForConditionalGeneration` with a top-level `vision_config` (depth 24, hidden 1024,
patch 16, `out_hidden_size` 2560) fully disjoint from `text_config`. Every vision tensor
sits under `model.visual.`, and text-only decode builds from `text_config` alone.

## The decode path

| | |
|---|---|
| Layers | 32, as 8 repetitions of `3 × linear_attention` then `1 × full_attention` |
| Full-attention layers | 3, 7, 11, 15, 19, 23, 27, 31 (`full_attention_interval` 4) |
| `hidden_size` | 2560 |
| `intermediate_size` | 9216, SwiGLU (`hidden_act` silu) |
| `rms_norm_eps` | 1e-6 |
| `vocab_size` | 248320, `tie_word_embeddings` true (no `lm_head.weight` in the checkpoint) |
| Text params | ~4.21 B |

### Gated attention (the 8 full-attention layers)

| | |
|---|---|
| Query heads | 16 |
| KV heads | 4 (GQA 4:1) |
| `head_dim` | **256** |
| `attn_output_gate` | true |
| `partial_rotary_factor` | **0.25** → 64 of 256 dims rotated |

**`head_dim` is not `hidden_size / num_heads`.** 2560/16 is 160, but `head_dim` is 256, so
none of the projections are square. This is the single most likely early bug.

**The output is gated.** `q_proj` emits `16 × 256 × 2 = 8192` rows: viewed as
`(…, 16 heads, 512)` and split in half per head into query and gate. The attention output
is multiplied by the gate before `o_proj`. Splitting the flat 8192 in half instead of
per-head is wrong and will look almost right.

**The gate's nonlinearity is config-driven, not fixed.** Qwen3.5 omits `output_gate_type`
and means `sigmoid`; Qwen3.8 declares `swish` (SiLU). It is therefore a `ModelConfig` field
read by `reference.apply_output_gate`, not a constant. Hardcoding either one produces a
model that runs and emits plausible logits on the other checkpoint — the failure mode this
project is least able to detect.

**RoPE is partial.** Only the first 64 dimensions of each 256-wide head are rotated; the
remaining 192 pass through untouched. This changes the shape of any kernel that fuses the
projection with the rotation.

**`q_norm` and `k_norm` are applied over the head dimension, before RoPE.**

### mRoPE

`mrope_interleaved: true`, `mrope_section: [11, 11, 10]`, `rope_theta: 1e7`. The sections
sum to 32 = `rotary_dim / 2`, confirming `rotary_dim` is 64.

Positions are 3-dimensional `(t, h, w)` and the frequency bands are **interleaved**
(`T H W T H W … T T`) rather than laid out in contiguous chunks.

**For text-only decode this reduces exactly to standard RoPE.** HuggingFace expands one
row of text positions across all three sections, so every band — whichever section it is
drawn from — reads the same position index.

This reduction was **verified, not assumed**, in two places:

* `reference_test.py::test_mrope_reduces_exactly_to_standard_rope_for_text_only` asserts
  bit-exact equality (`rtol=0, atol=0`) between our interleaved-mRoPE output and a
  standard RoPE computed independently from first principles. A companion test feeds
  genuinely different section positions and requires the result to *change*, so the
  equality is a property of text-only input rather than of the interleaving being a no-op.
* `oracle_test.py::test_mrope_reduction_holds_against_the_oracle` checks the same against
  HuggingFace's own rotary module. That one needs a GPU and the full checkpoint, and is
  **deferred to the first funded session**.

### Gated DeltaNet (the 24 linear-attention layers)

| | |
|---|---|
| Key heads | 16, `linear_key_head_dim` 128 → key dim 2048 |
| Value heads | 32, `linear_value_head_dim` 128 → value dim 4096 |
| Value groups | 2 value heads per key head (q and k are `repeat_interleave`d by 2) |
| Conv | `linear_conv_kernel_dim` 4, depthwise, causal, over all 8192 qkv channels |
| Recurrent state | **fp32** (`mamba_ssm_dtype: float32`) even though weights are bf16 |

**The projection layout differs from Qwen3-Next**, and this matters for anyone porting
code across. Qwen3-Next fuses `in_proj_qkvz` and `in_proj_ba` and needs a
`fix_query_key_value_ordering` step to undo per-key-head interleaving. Qwen3.5 instead has
**four separate projections**:

| Tensor | Shape | Meaning |
|---|---|---|
| `in_proj_qkv.weight` | `[8192, 2560]` | q (2048), k (2048), v (4096), contiguous |
| `in_proj_z.weight` | `[4096, 2560]` | the output gate |
| `in_proj_b.weight` | `[32, 2560]` | β, one per value head |
| `in_proj_a.weight` | `[32, 2560]` | a, one per value head |

So the qkv split is a plain contiguous three-way split with **no head interleaving to
undo**. Assuming the Qwen3-Next layout here produces a model that runs and is wrong.

The recurrence itself, per token: decay the state by `exp(g)`, recall `k·S`, correct
toward `v` at rate β, then read out with `q`. Where
`g = -exp(A_log) · softplus(a + dt_bias)` and `β = sigmoid(b)`, both computed in fp32.
Queries and keys are L2-normalised before use, and queries are scaled by `1/√d`.

**Any future scan kernel must keep the state in fp32.** This is the binding constraint on
the chunked delta-rule scan in `docs/HYPOTHESES.md`.

### The two RMSNorm conventions

The model uses **two different** normalisation conventions, and mixing them up is silent:

| Module | Scale | Order |
|---|---|---|
| `RMSNorm` (all layer norms, `q_norm`, `k_norm`, final norm) | **`1 + weight`** | — |
| `GatedRMSNorm` (inside Gated DeltaNet only) | plain `weight` | normalise **then** gate |

The checkpoint stores the ordinary norm weights **centred on zero**, so the effective
scale is `1 + w`. Reading them as a plain `w` yields an all-zero activation — a model that
runs, emits finite logits, and is completely wrong.

The gated norm inside Gated DeltaNet is the opposite: its weight is a plain multiplier
(stored as fp32), and normalisation happens *before* the SiLU gate is applied, not after.

## What is excluded, and why

| Component | Tensors | Params | Why excluded |
|---|---|---|---|
| Vision tower `model.visual.*` | 297 | ~0.33 B | Text-only decode. Never instantiated. |
| MTP head `mtp.*` | 15 | ~0.12 B | Speculative-decoding head — see below. |

The **multi-token-prediction head** (`mtp_num_hidden_layers: 1`) is one extra full-attention
decoder layer plus `mtp.fc` `[2560, 5120]`, which combines an embedding with a hidden
state. It is excluded from the decode path *and* from the benchmark for a methodological
reason: it changes how many tokens come out per forward pass. Including it would quietly
turn "our kernels are faster" into "we enabled speculative decoding", and make the headline
number incomparable with one produced without it.

`weights.py` skips both families deliberately and reports how many tensors it skipped, so
the exclusion is visible in every load rather than assumed.

## Checkpoint facts

- Two safetensors shards plus an index, ~9 GB bf16 total, 738 tensors.
- Ungated: no HuggingFace token required.
- `A_log` and `linear_attn.norm.weight` are stored **fp32**; everything else is bf16.
- No `lm_head.weight`: the LM head reuses the embedding matrix.

## Where the baseline draws the line

`reference.py` contains no custom kernels of any kind, which needs a precise meaning
because `F.linear` also lands in hand-written cuBLAS. The rule:

- **Allowed** — primitives the compiler can see through or dispatch normally: `F.linear`,
  `F.conv1d`, `matmul`, `softmax`, elementwise math. Beating cuBLAS on dense GEMM is an
  explicit non-goal, so using it costs nothing we were going to claim.
- **Excluded** — anything that *is itself the fused algorithm we intend to hand-write*.
  That means `F.scaled_dot_product_attention` (it dispatches to FlashAttention, which is
  itself a hypothesis) and `flash-linear-attention` (whose chunked delta-rule scan is
  another). Attention is therefore written out as matmul + softmax.

`reference_purity_test.py` enforces this by inspecting the module's AST, and includes a
test proving the detector actually detects.
