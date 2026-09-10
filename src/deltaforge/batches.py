"""The batch manifests.

**Every prediction in this file is committed before the rental it is tested on.** That is
the whole point of the file existing separately from the kernels: `AGENT.md` §1 says the
finding is a mechanistic account registered in advance and then confirmed, and a
prediction written after seeing the number is not a prediction.

Order within a batch is deliberate and is never sorted:

* the **calibration slot first**, so a broken harness is discovered in three minutes
  rather than at the end of the rental;
* then cheapest and most diagnostic;
* the **riskiest kernels last**, so everything already measured is on disk before one of
  them fails.

Byte shares are fractions, from `docs/roofline.py` at the headline workload — batch 1,
context 2048. Run it for the context you actually mean to measure before treating any of
them as fixed; several grow with the KV cache.
"""

from __future__ import annotations

from .batch import Batch, Hypothesis

__all__ = ["BATCHES", "BATCH_001", "BATCH_002", "get_batch"]


BATCH_001 = Batch(
    batch_id="001-calibration",
    description=(
        "The first batch, and the first measurement this project has ever taken: nine "
        "rentals have been billed and none produced a number. Slot 0 calibrates the "
        "harness; slots 1-5 collect real measurements for hypotheses the backlog "
        "graveyarded on *cost* rather than on truth, now that a slot is three minutes "
        "instead of a rental; slots 6-8 attack the only byte shares in the model large "
        "enough to matter at all."
    ),
    hypotheses=(
        Hypothesis(
            slug="000-identity",
            kernels=(),
            category="calibration",
            byte_share=0.0,
            replaces=(),
            mechanism=(
                "Install nothing. The candidate is then bit-identical to the reference, so "
                "every column must measure the same thing."
            ),
            prediction="identity",
            rationale=(
                "Not a hedge but the strongest claim in the batch: the candidate IS the "
                "reference, so a ratio away from 1.00 means the harness is measuring "
                "something other than the kernel under test. If this misses, every other "
                "number below is void and the writeup says so rather than reporting them."
            ),
            notes=(
                "AGENT.md section 4 asks the first working session to calibrate with an "
                "identity champion before writing any kernel. Batching means it no longer "
                "costs a whole rental to obey that."
            ),
        ),
        Hypothesis(
            slug="001-fused-rmsnorm-residual",
            kernels=("fused_rmsnorm_residual",),
            category="A",
            byte_share=0.00018,
            replaces=("rms_norm_residual", "rms_norm"),
            mechanism=(
                "Fuse the residual add with the RMSNorm that follows it, reading the hidden "
                "state once instead of across three separate elementwise kernels."
            ),
            prediction="inconclusive",
            rationale=(
                "Ceiling 0.018% of per-token bytes — below the harness's own noise band, so "
                "even a perfect kernel is unmeasurable. Inductor also emits a single "
                "persistent reduction here that loads the row once, which is the same "
                "kernel generated. Written in September 2026 and never measured because "
                "eight rentals died on a container pull; this is its first measurement."
            ),
        ),
        Hypothesis(
            slug="002-rmsnorm-only",
            kernels=("rmsnorm_hidden",),
            category="A",
            byte_share=0.00018,
            replaces=("rms_norm",),
            mechanism=(
                "The same Triton RMSNorm as 001 on the same three hidden-size sites, but "
                "without fusing the residual add."
            ),
            prediction="inconclusive",
            rationale=(
                "Same 0.018% ceiling as 001, so the same argument applies. Its real job is "
                "to be subtracted: 001 changes the kernel AND the fusion, 002 changes only "
                "the kernel, so 001 minus 002 is the fusion's contribution. Neither number "
                "alone can separate them."
            ),
        ),
        Hypothesis(
            slug="003-qk-norm-triton",
            kernels=("rmsnorm_qk",),
            category="A",
            byte_share=0.00001,
            replaces=("rms_norm",),
            mechanism=(
                "The same kernel again on q_norm/k_norm: 256-wide rows instead of 2560-wide, "
                "ten times as many of them, in the 8 full-attention layers only."
            ),
            prediction="inconclusive",
            rationale=(
                "A tenth of the reduction width and ten times the row count is a genuinely "
                "different regime for a persistent reduction — narrow rows are where a "
                "hand-written kernel's fixed launch cost is worst amortised, and where "
                "inductor's tiling heuristics have the most room to differ. The byte share "
                "is ~0.001%, far below anything measurable, so the prediction is a null; "
                "the reason to spend a slot is that the *shape regime* is untested and 001 "
                "explicitly declined to enter it."
            ),
        ),
        Hypothesis(
            slug="004-fused-swiglu",
            kernels=("fused_swiglu",),
            category="A",
            byte_share=0.00026,
            replaces=("swiglu_mlp",),
            mechanism=(
                "Fuse SiLU and the gate multiply into one pass over the two 9216-wide MLP "
                "intermediates, replacing three elementwise kernels with one."
            ),
            prediction="inconclusive",
            rationale=(
                "Ceiling 0.026%. The MLP is 53.9% of weight bytes, but the fusion target is "
                "the activation, not the weights: at batch 1 the two intermediates are 18 KB "
                "each against 141 MB of weights per layer. The GEMMs are cuBLAS territory "
                "and this epilogue is rounding error."
            ),
        ),
        Hypothesis(
            slug="005-fused-qkv-rope",
            kernels=("fused_rope",),
            category="A",
            byte_share=0.00004,
            replaces=("qkv_projection_rope",),
            mechanism=(
                "Partial mRoPE in one pass: rotate 64 of 256 head dims in registers and pass "
                "the other 192 through, replacing a slice/negate/cat/cat chain that copies "
                "three quarters of every element for no reason but to rebuild contiguity."
            ),
            prediction="inconclusive",
            rationale=(
                "Ceiling 0.004%, the smallest in the backlog. The projections' cost is "
                "streaming their weights; the Q/K/V intermediates are 0.33 MB/token across "
                "all 8 full-attention layers. The cat-elimination is real but it is real "
                "against a denominator of nothing."
            ),
        ),
        Hypothesis(
            slug="006-gqa-no-expand",
            kernels=("gqa_decode",),
            category="B",
            byte_share=0.0623,
            replaces=("gqa_attention", "kv_cache_update"),
            mechanism=(
                "Index the unexpanded KV cache directly instead of materialising "
                "repeat_interleave's 4x copy: each program maps its query head to hq // "
                "groups and reads the one KV head that actually exists."
            ),
            prediction="win",
            rationale=(
                "The only slot here with a byte share worth anything: 570 MB/token, 6.23% "
                "of all per-token traffic, 300x every elementwise fusion in this batch "
                "combined. Category B — moving fewer bytes is a choice about "
                "representation, which a scheduler is not obliged to make. **Conditional:** "
                "if inductor already folds the expansion's index arithmetic into its "
                "consumer rather than materialising it, `compiled` has this win already and "
                "there is nothing to take. Nobody has read the output code to find out, so "
                "6.23% is an upper bound, and the honest first move next session is "
                "TORCH_LOGS=output_code, which costs nothing."
            ),
            notes=(
                "The share grows with context — 32.5% at 32k on the 27B config — so a null "
                "here is a null at 2048, not a null for the hypothesis."
            ),
        ),
        Hypothesis(
            slug="007-gated-delta-fused-step",
            kernels=("gated_delta_step",),
            category="A",
            byte_share=0.0110,
            replaces=("gated_delta_rule",),
            mechanism=(
                "One pass over the (128, 128) fp32 recurrent state instead of five: decay, "
                "recall, rank-1 correction and read-out all held in registers. 24 of 32 "
                "layers are linear-attention layers."
            ),
            prediction="inconclusive",
            rationale=(
                "1.10% of per-token bytes, and the ceiling is only the fraction of that the "
                "extra passes waste — the state must be read and written once regardless. "
                "That lands near the noise band before discounting. A memory-bound "
                "elementwise chain is also inductor's home turf, so it may already fuse most "
                "of it. NOT the chunked scan from the backlog, which is a larger claim that "
                "cannot express itself where there is no sequence to chunk."
            ),
        ),
        Hypothesis(
            slug="008-flash-decode-splitkv",
            kernels=("flash_decode_splitkv",),
            category="C",
            byte_share=0.0078,
            replaces=("gqa_attention",),
            mechanism=(
                "Split the KV scan across 8 programs and merge their partial softmax results "
                "with the online-softmax rescaling identity, turning 16 programs at batch 1 "
                "into 128 and manufacturing parallelism the workload does not have."
            ),
            prediction="inconclusive",
            rationale=(
                "A prediction about the experiment, not about the kernel. The KV read is "
                "0.78% of per-token bytes at context 2048 while weights are 91.85%: "
                "parallelism in attention cannot show up in an end-to-end decode ratio when "
                "attention is not the bottleneck. docs/HYPOTHESES.md is explicit that this "
                "needs a 32k or 128k workload, and batch 001 has none. It is here as the "
                "**control for 006** — the two differ only in how the scan is parallelised, "
                "so 008 matching 006 is evidence that 006's win, if any, comes from not "
                "materialising the expansion — and to put a gated, measured kernel on disk "
                "for the session that adds a long-context workload."
            ),
        ),
    ),
)


BATCH_002 = Batch(
    batch_id="002-compile-cost",
    is_calibration=True,
    description=(
        "Three slots, and the product is the clock rather than the ratios. Rental 22 spent "
        "~40 minutes inside one cold `max-autotune` compile, which breaks the arithmetic "
        "every batch size in this repo rests on: `docs/BATCHES.md` costs a rental at ~15 "
        "minutes fixed plus 2-4 per slot, and that has never been measured. This batch "
        "measures it -- with the compile cache carried home afterwards, so the next rental "
        "can also say what a WARM compile costs, which is the number that decides whether "
        "7-12 slots is a batch or a fantasy."
    ),
    hypotheses=(
        Hypothesis(
            slug="000-identity",
            kernels=(),
            category="calibration",
            byte_share=0.0,
            replaces=(),
            mechanism=(
                "Install nothing. The candidate is then bit-identical to the reference, so "
                "every column must measure the same thing."
            ),
            prediction="identity",
            rationale=(
                "Still the strongest claim in any batch, and this project still does not "
                "have it: two rentals reached the batch loop and neither returned a ratio "
                "from this slot. Until it measures 1.00 within the noise band there is no "
                "calibrated harness, and every other number ever recorded here is void."
            ),
            notes=(
                "Also the cleanest possible compile measurement. The candidate is the same "
                "graph as the reference, so this slot's compile time is the cost of "
                "compiling a candidate with a FULLY warm in-process cache -- the floor that "
                "every other slot is measured against."
            ),
        ),
        Hypothesis(
            slug="003-qk-norm-triton",
            kernels=("rmsnorm_qk",),
            category="A",
            byte_share=0.00001,
            replaces=("rms_norm",),
            mechanism=(
                "The same Triton RMSNorm on q_norm/k_norm: 256-wide rows instead of "
                "2560-wide, ten times as many of them, in the 8 full-attention layers only."
            ),
            prediction="inconclusive",
            rationale=(
                "Chosen for its compile cost, not its ceiling. It swaps one module class in "
                "8 of 32 layers and restructures nothing around it, which makes it the "
                "smallest graph change available -- and therefore the cheapest possible "
                "measurement of what a candidate recompile costs on top of a warm reference. "
                "Its byte share is ~0.001%, far below the noise band, so the ratio itself is "
                "predicted null and would be worth almost nothing on its own. The clock is "
                "what this slot is for."
            ),
        ),
        Hypothesis(
            slug="002-rmsnorm-only",
            kernels=("rmsnorm_hidden",),
            category="A",
            byte_share=0.00018,
            replaces=("rms_norm",),
            mechanism=(
                "The same Triton RMSNorm on the three hidden-size sites, 2560-wide rows in "
                "all 32 layers, without fusing the residual add."
            ),
            prediction="inconclusive",
            rationale=(
                "The same kernel as the slot before it, applied to four times as many sites "
                "in four times as many layers. Two slots differing mainly in how much of the "
                "graph they touch is what turns a single compile timing into a slope, which "
                "is what `docs/BATCHES.md` needs in order to cost a batch it has not run "
                "yet. The ratio is again predicted null on a 0.018% ceiling."
            ),
        ),
    ),
)


#: Every batch, by id.
BATCHES: dict[str, Batch] = {BATCH_001.batch_id: BATCH_001, BATCH_002.batch_id: BATCH_002}


def get_batch(batch_id: str) -> Batch:
    try:
        return BATCHES[batch_id]
    except KeyError:
        raise SystemExit(f"unknown batch {batch_id!r}; known batches: {sorted(BATCHES)}") from None
