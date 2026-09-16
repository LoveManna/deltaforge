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

__all__ = ["BATCHES", "BATCH_001", "BATCH_002", "BATCH_003", "get_batch"]


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


BATCH_003 = Batch(
    batch_id="003-int8-weight-only",
    description=(
        "Weight-only quantisation with a fused dequantise-GEMV -- hypothesis 1 in "
        "docs/HYPOTHESES.md, and the only open entry whose ceiling is above 1.0 rather "
        "than below the noise band. Everything batch 001 measured attacks at most 6.23% "
        "of per-token bytes; this attacks 91.85%, because at batch-1 decode the model IS "
        "the weight stream. Two of the seven slots are controls rather than hypotheses, "
        "and they are what make a win mean something: 009 asks whether a hand-written "
        "GEMV is competitive with cuBLAS at all, and 010 asks whether the compiler could "
        "have done this itself. A ratio without those two is a number; with them it is a "
        "mechanism."
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
                "It measured 1.0018 with an IQR of 0.0018 on rental 35, so unlike every "
                "batch before this one there is a reason to expect it. That is exactly why "
                "it still runs first: a calibration slot is worth nothing as a belief and "
                "everything as a check, and the six slots behind it are the first numbers "
                "this project would ever promote. If it misses, they are all void."
            ),
            notes=(
                "It also tests the blocker 16 fix on a GPU for the first time. "
                "`recompile_limit_for(7)` raises dynamo's limit to 22; every slot record "
                "carries `graphs_compiled`, and a 0 anywhere in this batch means that "
                "slot's ratio is not a comparison."
            ),
        ),
        Hypothesis(
            slug="009-gemv-bf16-control",
            kernels=("gemv_bf16",),
            category="A",
            byte_share=0.0,
            replaces=("decode_step",),
            mechanism=(
                "The hand-written Triton GEMV reading the reference's own bf16 weights, on "
                "every projection in every decoder layer. Identical bytes to cuBLAS; only "
                "the author changes."
            ),
            prediction="inconclusive",
            rationale=(
                "Byte share 0.0 -- it moves not one byte fewer than the baseline, which is "
                "the point. At batch 1 a GEMV is pure weight streaming, so both "
                "implementations are pinned to the same roofline and should tie. This slot "
                "is the divisor for the three int8 slots behind it: if 012 returns 1.6 and "
                "this returns 1.0, the win is the bytes. If this returns 0.6, my kernel is "
                "simply slow and int8's real margin is larger than it looks. No other slot "
                "can separate those, and a quantisation result reported without this one is "
                "a number whose cause is unknown."
            ),
        ),
        Hypothesis(
            slug="010-int8-dequant-torch",
            kernels=("int8_dequant_torch",),
            category="B",
            byte_share=0.779,
            replaces=("decode_step",),
            mechanism=(
                "The same int8 weights, dequantised the only way PyTorch can express it -- "
                "materialise the bf16 weight, then call F.linear -- and handed to "
                "torch.compile(max-autotune) like every other candidate."
            ),
            prediction="loss",
            rationale=(
                "docs/HYPOTHESES.md ASSERTS that inductor cannot fuse this: that it "
                "materialises the full bf16 weight into global memory and calls cuBLAS, "
                "ADDING an 8.4 GB write on top of the read and making the quantised version "
                "slower than the bf16 baseline. The entry then says, in its own Watch-for: "
                "'Verify the claim above before building on it... Either way, measure, do "
                "not assume.' Nobody has. Recent inductor has prologue fusion into its mm "
                "templates and might fuse some of the dequant -- though at M=1 it likely is "
                "not using a template at all. So the prediction is a loss around 0.6-0.8x, "
                "and the outcome that would matter most is the one that refutes it: if this "
                "slot WINS, the compiler can express weight-only quantisation, the "
                "hand-written kernels behind it are worth much less than claimed, and the "
                "backlog's top entry needs rewriting."
            ),
            notes=(
                "The correctness thresholds match 012 exactly, and so should the measured "
                "numbers: it is the same arithmetic, so a divergence between the two would "
                "mean one of the implementations is wrong rather than merely slower."
            ),
            correctness="approximate",
            top1_threshold=0.98,
            kl_threshold=0.01,
        ),
        Hypothesis(
            slug="011-int8-mlp",
            kernels=("int8_mlp",),
            category="B",
            byte_share=0.495,
            replaces=("swiglu_mlp",),
            mechanism=(
                "int8 weight-only with a fused dequantise-GEMV on the three MLP "
                "projections: the int8 weight is loaded and rescaled inside the K-loop, so "
                "the dequantised bf16 weight is never written to memory at all."
            ),
            prediction="win",
            rationale=(
                "The MLP is 4.530 GB of the 8.411 GB of weights read per token -- 49.5% of "
                "ALL per-token traffic, eight times the largest share batch 001 attacked. "
                "Halving it saves 2.265 GB of 9.158, for a ceiling of 1.33x, which is two "
                "orders of magnitude outside the harness's measured noise band of 0.002-0.004. "
                "Category B: moving fewer bytes is a choice about representation, and a "
                "scheduler is not entitled to make it. Predicted 1.15-1.30x -- short of the "
                "ceiling because a GEMV at N=9216 puts only 144 programs on 170 SMs and the "
                "int8 loads are not perfectly efficient at BLOCK_N=64."
            ),
            notes=(
                "The cheap diagnostic before the expensive claim: 012 minus this is exactly "
                "what the attention and linear-attention projections are worth, and if this "
                "slot fails there is no point reading 012 or 013 at all."
            ),
            correctness="approximate",
            top1_threshold=0.98,
            kl_threshold=0.01,
        ),
        Hypothesis(
            slug="012-int8-all-linear",
            kernels=("int8_all_linear",),
            category="B",
            byte_share=0.779,
            replaces=("decode_step",),
            mechanism=(
                "The same fused dequantise-GEMV on every projection inside a decoder layer: "
                "MLP, gated attention and gated delta-net, 7.140 GB of the 8.411 GB of "
                "weights."
            ),
            prediction="win",
            rationale=(
                "77.9% of per-token bytes at 8 bits saves 3.570 GB of 9.158, for a ceiling "
                "of 1.64x. The mechanism is identical to 011 and the only thing that changes "
                "is how much of the weight stream it covers, which makes the pair a dose-"
                "response test rather than two separate experiments: if 011 wins and this "
                "does not win by MORE, the win is not coming from bytes and the account is "
                "wrong. Predicted 1.35-1.55x. Note the small-N sites here -- in_proj_a and "
                "in_proj_b are 32 output channels against K=2560, so one BLOCK_N program "
                "does the whole projection and the kernel is latency-bound rather than "
                "bandwidth-bound on them. They are 0.03% of the bytes, so it does not "
                "matter much, but it is the reason this may land below the dose-response "
                "line rather than above it."
            ),
            correctness="approximate",
            top1_threshold=0.98,
            kl_threshold=0.01,
        ),
        Hypothesis(
            slug="013-int8-full",
            kernels=("int8_full",),
            category="B",
            byte_share=0.918,
            replaces=("decode_step",),
            mechanism=(
                "012 plus the tied LM head, through `ReferenceModel.project_logits`. The "
                "head is one 248320 x 2560 GEMV per token -- the single largest weight read "
                "in the model -- and with tie_word_embeddings it had no nn.Linear to swap."
            ),
            prediction="win",
            rationale=(
                "The head is 1.271 GB, 15.1% of weight bytes, and quantising it takes the "
                "batch to 100% of the weight stream at 8 bits: ceiling 1.85x, which is what "
                "docs/roofline.py prints for fp8/int8 and the largest number in this "
                "repository's backlog. Predicted 1.5-1.7x. It is placed after 012 rather "
                "than folded into it because the head is the one site where quantisation is "
                "genuinely risky for accuracy -- it is the last projection before the "
                "argmax, so its error is not attenuated by anything downstream, and the "
                "logit gaps this model produces are small enough that 001 and 002 flipped a "
                "token on 2 bf16 ULP. If exactly one of 012 and 013 fails the distribution "
                "gate, that is the finding."
            ),
            notes=(
                "The embedding LOOKUP stays bf16. It reads one row of the table per token, "
                "not the table, so it is not on the bandwidth path and quantising it would "
                "add error for nothing."
            ),
            correctness="approximate",
            top1_threshold=0.97,
            kl_threshold=0.02,
        ),
        Hypothesis(
            slug="014-int4-full",
            kernels=("int4_full",),
            category="B",
            byte_share=0.918,
            replaces=("decode_step",),
            mechanism=(
                "The same sites as 013 at 4 bits with per-group-of-128 scales, two values "
                "packed per byte and unpacked in registers inside the K-loop."
            ),
            prediction="win",
            rationale=(
                "Ceiling 3.21x -- the largest in the repository. Riskiest and therefore "
                "last, on two counts. Numerically: 4 bits is 16 levels, and in_proj_a and "
                "in_proj_b feed an exponential through A_log, so a quantisation error there "
                "is amplified rather than averaged; its thresholds are set loosest in the "
                "batch for that reason and the honest outcome may be a fast candidate that "
                "fails the gate, which is a result and is recorded as one. Mechanically: "
                "the unpack costs shifts and masks in the inner loop, and at 4 bits the "
                "kernel may stop being bandwidth-bound and start being ALU-bound, at which "
                "point the extra halving buys nothing. Predicted 1.8-2.4x, well short of "
                "3.21x, and predicted to have the widest gap between ceiling and measurement "
                "of any slot here."
            ),
            correctness="approximate",
            top1_threshold=0.85,
            kl_threshold=0.15,
        ),
    ),
)


#: Every batch, by id.
BATCHES: dict[str, Batch] = {
    BATCH_001.batch_id: BATCH_001,
    BATCH_002.batch_id: BATCH_002,
    BATCH_003.batch_id: BATCH_003,
}


def get_batch(batch_id: str) -> Batch:
    try:
        return BATCHES[batch_id]
    except KeyError:
        raise SystemExit(f"unknown batch {batch_id!r}; known batches: {sorted(BATCHES)}") from None
