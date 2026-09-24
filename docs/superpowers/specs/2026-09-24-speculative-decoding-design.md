# Speculative decoding: the one hypothesis that is not bounded by the roofline

**Status:** design, registered before any measurement. 2026-09-24, after rental 46.

Every open entry in `docs/HYPOTHESES.md` is bounded by the same number. The compiled column
moves **8587.80 MB/token** and the card does **1790 GB/s**, so 4.80 ms/token is the floor and
the measured column is at **7.05**: about **1.47x** exists, and only for an implementation
that reaches vendor peak. Weight-only quantisation moves the floor rather than approaching
it, which is why entry 1 is the largest prize in the file — and after nine batches the best
composed candidate this project ships is **1.034x**.

Speculative decoding changes a term no other entry touches: **how many tokens come out per
weight-stream.** The weights are read once per *forward pass*, not once per *token*, so a
forward pass that verifies four drafted tokens costs almost exactly what a forward pass that
produces one costs. That is not a scheduling or fusion choice, so it is category **C**: a
compiler will not derive it, because it is a statement about greedy decoding being
verifiable, not about the graph.

It is also the only hypothesis here that can put the model **below 4.80 ms/token**.

---

## 1. The arithmetic

One cycle drafts `k` tokens, verifies them in a single forward pass over `k+1` positions,
and emits every draft token the verifier agrees with plus one token of its own:

```
                 k·d  +  γ(k)·v₁
ms/token  =  ─────────────────────────
              1  +  Σ(i=1..k) pⁱ
```

| term | what it is | what we know |
|---|---|---|
| `v₁` | one reference decode step | **7.05 ms**, measured, rental 46, 15 rounds |
| `γ(k)` | cost of a `k+1`-token verify, relative to `v₁` | **predicted 1.11 at k=4**; unmeasured |
| `d` | one draft token | **unmeasured**; ~2.5 ms if an int4 model streams 2804 MB at ~1150 GB/s |
| `p` | per-token acceptance | **already measurable for free** — see §2 |

`γ` is traffic plus dispatch. The weights do not move: 8411.51 MB either way. What scales
with `k+1` is the KV read (71.30 MB), the recurrent state (100.66), and the small
intermediates — 176.29 MB/token, so **+8.2%** of bytes at `k=4` — plus the state versioning
§3 needs, another **+2.9%**. Dispatch is the open half: the linear-attention scan runs
`k+1` steps per layer instead of one, and this project has 0.9–2.5 ms of its step in launch
overhead. **γ is the first thing to measure and it can be measured without a drafter at
all** (§6, slot `a`).

### What the model says

At `v₁ = 7.05`, `γ(4) = 1.11`, `γ(2) = 1.06`:

| draft | k | p = 0.9 | p = 0.8 | p = 0.7 | p = 0.6 |
|---|---:|---:|---:|---:|---:|
| int4 self-draft, `d = 2.5` | 4 | **1.62x** | 1.33x | 1.10x | 0.91x |
| int4 self-draft, `d = 2.5` | 2 | **1.53x** | 1.38x | 1.24x | 1.11x |

| draft | k | p = 0.5 | p = 0.3 | p = 0.1 |
|---|---:|---:|---:|---:|
| zero-cost draft, `d = 0` | 4 | **1.74x** | 1.28x | 1.00x |

Three things fall out of the table and they shape the plan:

1. **`k = 2` is the robust setting, not `k = 4`.** A short block gives up the top of the
   range and moves break-even acceptance from ~0.65 to ~0.51. Start there.
2. **A zero-cost draft breaks even at p ≈ 0.10.** Whatever the drafter is, `γ−1` is the
   whole downside, so the cheapest drafter is the one to try first if any acceptance exists.
3. **The int4 self-draft is only worth building if the int4 model is fast.** `d` is a full
   forward of the quantised model, so this hypothesis inherits `061-int4-mlp-torch-dequant`
   as a **hard prerequisite**. If int4 in torch does not clear ~1.0 on the whole model,
   `d ≈ v₁` and self-drafting is dead by arithmetic.

**`k·d` is only `k·d` if the drafter's catch-up is merged into its first draft pass.**
A drafter carries its own recurrent state, and at the start of each cycle that state is
wrong in two ways: it ran ahead through drafts that were rejected, and it never saw the
verifier's own token. It must therefore consume the newly committed tokens before it can
draft again. Done as a separate pass that is `(k+1)·d` and the `k=4`, `p=0.9` cell falls
from **1.62x to 1.42x**; done as the *first* draft pass of the next cycle — a multi-token
input whose last position is the first new draft — it is free, because the weights stream
once either way. **The implementation must merge it, and the plan tests that it did.**

---

## 2. What already works, unchanged

**The verify pass needs no new model code.** `ReferenceModel.forward` already takes a
multi-token input with a cache — that is what the 2048-token prefill is — and the
full-attention path already builds a position-aware causal mask when `seq_len > 1`
(`reference.py`, the `if seq_len > 1` branch), while the linear-attention path already runs
its scan over a sequence. A verify is `model(draft_tokens, cache)`.

**The rollback primitive exists as of today.** `DecodeCache.snapshot` / `restore` were added
for the benchmark's rounds (copy the state out once, copy it back per round, in place so no
address moves). That is exactly the operation a rejected draft needs.

**The acceptance rate is a statistic this repo already computes.** The layer-2 correctness
gate's teacher-forced top-1 agreement between a candidate and the reference **is** `p`. The
int4 head alone measured **0.9318** over 264 positions on rentals 40, 43, 45 and 46.

---

## 3. What makes this model hard: the state is not indexed by position

24 of 32 layers are gated delta-rule layers carrying a `(1, 32, 128, 128)` fp32 recurrent
state updated in place, once per token. A KV cache rolls back by moving `seq_len`; a
recurrent state does not roll back at all.

So a cycle that drafts `k` tokens, verifies them, and accepts `j < k` must recover the state
as of position `j`. Three options, and the cheap one is not the obvious one:

| | cost | verdict |
|---|---|---|
| Re-run the accepted prefix through the model | a second weight stream, ~7 ms | **kills the hypothesis** |
| Snapshot before the verify, re-run only the *scan* for `j` steps from the already-computed `(k, v, β, g)` | ~50 MB × j, no weights | correct, needs the layer to hand back its projections |
| Keep one state copy per step during the verify, then select index `j` | `(k+1) × 50.33 MB` of writes, ~0.2 ms at `k=4` | **v1**: simplest thing that is exactly right |

v1 takes the third. It costs the 2.9% already in `γ`, it touches one module
(`GatedDeltaNet`, through the ordinary installer mechanism), and it has no correctness
argument to get wrong. The second option is the optimisation to reach for **only if `γ`
comes back above ~1.15 and the versioning is why**.

The causal conv history rolls back the same way and is negligible: 24 layers × 4096 × 3 ×
2 bytes = **590 KB** per version.

---

## 4. Correctness: neither existing gate is the right instrument

**The claim is that the emitted token sequence is the reference's.** A draft token is
accepted only when it equals `argmax` of the *verifier's* logits at that position, and at
the first disagreement the verifier's own token is emitted and the rest of the block is
discarded. Every emitted token is therefore an argmax of logits this model computed.

**It is not bit-exact, and the reason is this checkpoint's own sensitivity.** The verifier
computes `k+1` positions in one matmul; the reference computes them one at a time. A
different reduction order lands one bf16 ULP away, and `AGENT.md` §7a records that one ULP
flips an argmax on this model — `009-gemv-bf16-control` matched 1 prompt of 5 on exactly
that.

**And this is where the first draft of this spec was wrong.** It registered the slot as
`approximate`, which is the policy every quantised slot uses. That policy runs
`check_distribution`, which **teacher-forces the two models and never calls the decode
loop** — and a speculative candidate has the reference's own weights, so it would return
264/264 and 0.00000 nats by construction, including with a rollback that silently corrupts
the recurrent state. It is not a weak gate here; it is a gate that cannot fail.

The other policy, `exact`, does exercise the loop — `check_end_to_end` free-runs
`greedy_decode` on both models and compares token ids — but `Hypothesis.__post_init__`
refuses `exact` for anything that installs a kernel, for the reason that produced this
rule three times: computing the same function is not producing the same bits.

So this hypothesis needs a **third policy, `sequence`**, and the gate that goes with it:

* free-run both models, compare token ids, and report **the index of the first
  divergence** rather than a bare pass/fail;
* at that index, report the **reference's own top-2 logit gap**, which
  `oracle_test.py::test_report_the_first_greedy_divergence` already computes;
* **pass** when there is no divergence, or when the first divergence sits at a position
  whose top-2 gap is below the ULP scale — that is a flip the reduction order explains.
  **Fail** when a divergence lands on a position the reference was confident about, which
  is what a rollback bug looks like and what no distribution statistic would have caught.

Registered bars: `first_divergence_gap_ceiling = 0.02` (bf16 has ~3 decimal digits at
logit magnitudes here, so a gap below this is not a decision the model made), and **no
divergence at all in the `k = 0` diagnostic**, which must be bit-identical because it
degenerates to ordinary decoding. The `k = 0` case runs on CPU with `tiny_config`, so it
is a unit test rather than a rental line.

## 5. What has to change in the harness

Ordered by who blocks whom.

1. **`greedy_decode` dispatches to a decode loop.** Today the bench times
   `greedy_decode(runnable, …)`, which is a fixed loop over `model(next_token, cache)`. A
   speculative candidate replaces the *loop*, not a module, so `greedy_decode` reads
   `model.decode_loop` when one is installed and keeps its current body as the default.
2. **A `decode_loop` replaceable op**, with the installer keyed by kernel name like every
   other, so a slot names it in `Hypothesis.kernels` the way it names anything else.
3. **`batch_run._build_candidate`'s no-op guard must see it.** It currently refuses a
   candidate whose module classes all match the reference — which a loop installer's would.
   Extend the guard to accept a changed `decode_loop`, and keep it refusing everything else:
   that guard is what stops a no-op candidate being recorded as a well-behaved 1.00.
4. **The slot records acceptance, not just a ratio.** Mean accepted tokens per cycle, the
   distribution over `0..k`, and cycles per timed call. Without them a ratio cannot
   distinguish "the drafter is bad" from "the verify is expensive", which are opposite next
   steps — the mistake `034` made about CUDA graphs, where a win was recorded for a
   mechanism that never fired.
5. **No byte model for the candidate column.** Bytes per token are now variable, so the
   candidate declares none and reports no GB/s rather than a rate it did not achieve.
   `decode_bytes_per_token` keeps describing the reference.
6. **A text workload.** `DEFAULT_WORKLOADS["headline"]` prompts with `torch.randint` token
   ids. Acceptance on random ids is not acceptance on text — the model's continuation of
   noise is its own distribution, and a drafter can look far better or far worse there than
   it ever will in use. Add `headline_text` (the same 2048/128 shape, prompt tokenized from
   `harness/prompts.py`) and **report both**. Every earlier hypothesis was indifferent to
   the prompt; this one is not, and that is a property of the hypothesis, not a flaw in the
   harness.

7. **`DecodeCache` needs `rewind`, and the bench needs `k` positions of headroom.**
   `advance` is one-way and `ensure_capacity` refuses to write past `max_seq_len`. A cycle
   writes `k+1` positions and may keep `j+1`, so the cache is allocated at
   `context + decode_tokens + k` and rewound by `k − j` after every verify.
8. **The drafter cannot live in the candidate's module tree.**
   `cli._assert_parameters_are_shared` requires every candidate parameter to share storage
   with a reference parameter of the same name, and int4 draft weights have no counterpart.
   That check is load-bearing — it is what catches a candidate that silently loaded a
   second 8.4 GB copy — so the drafter is held by the loop object rather than registered as
   a submodule, and the check stays exactly as strict as it is.

Two smaller ones: the drafter's catch-up pass has a **data-dependent length** (`j+1`, which
varies per cycle), so its sequence dimension is marked dynamic — otherwise dynamo compiles a
graph per length and the slot is voided by recompilation. And `graphs_compiled` must be read
as usual to confirm neither column fell back to eager.

---

## 6. The plan, as slots

Each is one variable from its neighbour, and each names it — the design that produced
`044`/`045` and `054`/`056`, now required by `batch.unpaired_slots`.

| slot | what it installs | the one variable | prediction |
|---|---|---|---|
| `a` **verify inflation, k=4** | the loop, drafting deliberate garbage (p = 0) | — | **`loss`, and the ratio is `1/γ(4)`: 0.87–0.95** |
| `b` **verify inflation, k=2** | same, k = 2 | `k` | `loss`, 0.93–0.97 |
| `c` **int4 self-draft, k=2** | loop + int4 draft weights | the drafter | `win` if `061` cleared and p ≥ 0.75 |
| `d` **int4 self-draft, k=4** | same, k = 4 | `k` | `win`, above or below `c` depending on p |
| `e` **n-gram draft, k=4** | loop + prompt-lookup drafter, `d = 0` | the drafter | `inconclusive` on random-id prompts, `win` on text |

**Slot `a` is the important one and it is worth a rental on its own terms.** It measures
`γ` — the whole downside of the hypothesis — with no drafter, no acceptance and no
quantisation in it. If `γ(4)` comes back above ~1.25, blocks longer than 2 are dead and the
table in §1 collapses to its second row. Registering it as a predicted **loss** is the point:
the number it produces is the input to everything else.

**Stage 0 costs nothing and runs first.** `p` for any drafter is the layer-2 agreement the
correctness gate already computes, so the rental measures acceptance during the correctness
phase, before a single timed round. A drafter whose agreement is below break-even should
decline its own slots — with a `Precondition` on the comparison, not on an absolute ratio.

---

## 7. Kill criteria, registered in advance

* **`γ(4) > 1.25`** → `k = 4` is dead; only `k = 2` survives.
* **`γ(2) > 1.15`** → the whole hypothesis is dead at this model's shape, and the reason is
  dispatch rather than traffic, which is worth writing up as a result about the decode graph.
* **full-int4 agreement `< 0.65`** → the int4 self-draft is dead at `k = 4`, before any
  timing.
* **`061-int4-mlp-torch-dequant < 1.0`** → `d ≈ v₁`, and the int4 self-draft is dead
  outright. The n-gram drafter is unaffected, because it has no `d`.

---

## 8. What this is not

It is **not** a way to make a kernel faster, and it does not compose with the launch-count
hypotheses the way a byte saving would: entry 8's dispatch savings and this one are both
spending the same microseconds, and rental 46 showed those do not add (`+1.5%` alone, `+0.0%`
composed). It does compose with quantisation, but only through `d`.

It is also **not** a general throughput result. At batch 1 the card is idle enough that
verifying four tokens is nearly free; at batch 32 it is not, and the table in §1 does not
apply there. The claim is about single-stream latency, which is what this project measures.


---

## 9. What the review changed, 2026-09-24

The spec was checked against the code the same day it was written, before any
implementation. Four things changed, and they are recorded here rather than silently
edited, because a design that was wrong in a way the code could have told you is worth the
same note as a prediction that missed:

1. **§4 was wrong about the gate.** `approximate` teacher-forces the models and never calls
   the decode loop; on a candidate with the reference's own weights it cannot fail. A third
   policy and a divergence-attribution gate replace it.
2. **§1 understated the drafter's bookkeeping.** A drafter has its own recurrent state and
   must consume the newly committed tokens each cycle. Merged into the first draft pass that
   is free; done separately it is `(k+1)·d` and costs the headline cell 0.2x.
3. **§5 was missing the cache rewind and the headroom** it implies, and the fact that
   `_assert_parameters_are_shared` refuses a drafter inside the module tree.
4. **The variable-length catch-up is a recompilation source**, which would void a slot the
   way `recompile_limit` voided six of nine in batch 008.

Verified against the code and unchanged: the multi-token verify needs no new model code
(`ReferenceModel.forward` takes `num_logits_to_keep=k+1`; `_causal_conv` prepends the cache
history; `recurrent_gated_delta_rule` takes `initial_state` and scans; gated attention takes
`cache_offset`), and the per-step conv windows are free because the layer has already
concatenated history and input into one tensor. The per-step *recurrent* states are not
free: `recurrent_gated_delta_rule` returns only the final state, so the versioning is a
module swap, which is what makes it an installer like any other.
