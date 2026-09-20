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

from .batch import Batch, Hypothesis, Precondition

__all__ = [
    "BATCHES",
    "BATCH_001",
    "BATCH_002",
    "BATCH_003",
    "BATCH_004",
    "BATCH_005",
    "BATCH_006",
    "BATCH_007",
    "get_batch",
]


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
            historical_exact_gate=True,
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
            historical_exact_gate=True,
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
            historical_exact_gate=True,
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
            historical_exact_gate=True,
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
            historical_exact_gate=True,
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
            historical_exact_gate=True,
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
            historical_exact_gate=True,
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
            historical_exact_gate=True,
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
            historical_exact_gate=True,
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
            historical_exact_gate=True,
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
            historical_exact_gate=True,
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
BATCH_004 = Batch(
    batch_id="004-bandwidth-bound-gemv",
    description=(
        "Batch 003 produced seven admissible ratios and every one was a loss -- and it "
        "produced the cause too: the hand-written GEMV ran at 332 GB/s against a compiled "
        "baseline near 1200, and got *slower* as it removed bytes, which is the signature "
        "of a kernel bound by instruction issue rather than bandwidth. This batch fixes "
        "that kernel and then quantises on top of it. Slot 1 is the whole result: it moves "
        "exactly cuBLAS's bytes, so its ratio is the divisor every quantised slot behind it "
        "is read against, and every one of them is gated on it clearing 0.56 -- the point "
        "below which halving the weight stream cannot even tie."
    ),
    hypotheses=(
        Hypothesis(
            slug="000-identity",
            kernels=(),
            category="calibration",
            byte_share=0.0,
            mechanism=(
                "Install nothing. The candidate is the reference, so the measured ratio is "
                "the harness's own noise floor rather than a property of any kernel."
            ),
            prediction="identity",
            rationale=(
                "Must return 1.00 within the noise band. If it does not, the harness is "
                "measuring something other than the kernel under test and every other "
                "number in this batch is void -- which is a statement about the rental, not "
                "about any hypothesis, and the writeup has to say so rather than reporting "
                "the rest as findings."
            ),
        ),
        Hypothesis(
            slug="015-tiled-gemv-bf16",
            kernels=("tiled_gemv_bf16",),
            category="A",
            byte_share=0.0,
            replaces=("decode_step",),
            correctness="approximate",
            correctness_positions=264,
            top1_threshold=246 / 264,
            kl_threshold=0.01,
            weight_bits={},
            mechanism=(
                "009's GEMV rewritten around a tl.dot accumulator with a K-major weight "
                "layout and split-K, so the cross-lane reduction that ran once per "
                "K-iteration -- 20 times for K=2560, 72 for K=9216 -- runs once in total."
            ),
            prediction="inconclusive",
            rationale=(
                "Byte share 0.0: it moves not one byte fewer than the baseline, which is "
                "the point. Tying is the honest expectation and its job is to be the "
                "divisor, not to win. 009 returned 0.2801 on identical bytes and that one "
                "number settled five slots behind it; the required improvement is 1.8x, not "
                "4x, because quantisation ties at f=0.50 and that shows up here as 0.562. "
                "Predicted 0.75-0.95: removing 20-72 reductions per output from an inner "
                "loop is comfortably worth 1.8x, and the remaining gap to 1.0 is the "
                "split-K reduction pass and whatever cuBLAS does that this does not. "
                "Gated approximately rather than exactly: it computes the same function as "
                "the reference but sums K in a different order from cuBLAS, which lands one "
                "bf16 ULP away, and one ULP flips an argmax on this model -- 009 was gated "
                "exact on that reasoning and matched 1 of 5 prompts. Its bars are int8's, "
                "because one ULP of reordering is far inside them."
            ),
        ),
        Hypothesis(
            slug="016-fp8-all-linear",
            kernels=("tiled_fp8_all_linear",),
            category="B",
            byte_share=0.7796,
            replaces=("decode_step",),
            correctness="approximate",
            correctness_positions=264,
            top1_threshold=228 / 264,
            kl_threshold=0.03,
            weight_bits={"layers": 8},
            requires=Precondition(
                slug="015-tiled-gemv-bf16",
                floor=0.56,
                reason=(
                    "quantisation halves the bytes, so it cannot even tie unless the kernel "
                    "reaches half the baseline's byte rate; below 0.56 this slot would "
                    "re-measure batch 003's 0.1962 in a new dtype"
                ),
            ),
            mechanism=(
                "e4m3 on every projection inside a decoder layer, converted to bf16 inside "
                "the K-loop and rescaled once per output channel at the end. 77.9% of "
                "per-token bytes at 8 bits: 5588 MB/token against 9158, a 1.64x ceiling."
            ),
            prediction="win",
            rationale=(
                "The direct fp8 counterpart of batch 003's 012, which returned 0.1962 on a "
                "kernel that could not collect the saving. Predicted 1.15-1.30 against a "
                "1.64x ceiling: the shortfall is the split-K reduction and the 22% of bytes "
                "left in bf16. fp8 rather than int8 because batch 003 measured int8 at "
                "1.438x the time of bf16 in the same kernel -- int8->fp32 is an ALU "
                "instruction on the critical path, where sm_120 converts e4m3 inside the "
                "MMA pipeline, and every finite e4m3 value is exactly a bf16 value so the "
                "conversion is lossless. The KL bar is interpolated between two measured "
                "points rather than taken from priors, which is how batch 003's bars went "
                "wrong: int8 measured 0.0011 nats and int4 0.0919. e4m3's per-element "
                "absolute error is about 3x int8's per-channel error on a Gaussian row and "
                "about a quarter of group-128 int4's, and KL grows as the square of the "
                "perturbation, so expect ~0.01 nats. The bar is 0.03, three times that. "
                "The top-1 bar allows 36 flips of 264 against the ~24 that flip rate "
                "implies, and it is written as a count because agreement quantises to 1/264 "
                "and a bar finer than one sample is not a decision procedure."
            ),
        ),
        Hypothesis(
            slug="017-fp8-full",
            kernels=("tiled_fp8_full",),
            category="B",
            byte_share=0.9184,
            replaces=("decode_step",),
            correctness="approximate",
            correctness_positions=264,
            top1_threshold=228 / 264,
            kl_threshold=0.03,
            weight_bits={"layers": 8, "head": 8},
            requires=Precondition(
                slug="015-tiled-gemv-bf16",
                floor=0.56,
                reason="the same floor as 016: below it no byte saving can be collected",
            ),
            mechanism=(
                "016 plus the tied LM head, which is 15.1% of weight bytes and the largest "
                "single GEMV in the model. 91.8% of per-token traffic at 8 bits: 4953 "
                "MB/token, the full 1.85x roofline ceiling. The embedding *lookup* stays "
                "bf16 -- it reads one row, not the table."
            ),
            prediction="win",
            rationale=(
                "The top rung of the ladder and the batch's best chance at a champion. "
                "Predicted 1.25-1.45 against a 1.85x ceiling. 017 minus 016 is exactly what "
                "the head is worth, which batch 003 tried to measure as 013 minus 012 and "
                "got -0.0084 -- a difference that says nothing, because both kernels were "
                "issue-bound and the head's extra bytes were not what set their time. Same "
                "bars as 016: same dtype, same quantiser, one more site."
            ),
        ),
        Hypothesis(
            slug="018-fp8-mlp",
            kernels=("tiled_fp8_mlp",),
            category="B",
            byte_share=0.4946,
            replaces=("swiglu_mlp",),
            correctness="approximate",
            correctness_positions=264,
            top1_threshold=228 / 264,
            kl_threshold=0.03,
            weight_bits={"mlp": 8},
            requires=Precondition(
                slug="015-tiled-gemv-bf16",
                floor=0.56,
                reason="the same floor as 016: below it no byte saving can be collected",
            ),
            mechanism=(
                "e4m3 on the three MLP projections only: 53.9% of the model's weight bytes "
                "and 49.5% of per-token traffic, 6893 MB/token, a 1.33x ceiling."
            ),
            prediction="win",
            rationale=(
                "The low rung of the dose-response ladder, at 49.5% against 016's 77.9% and "
                "017's 91.8%. Predicted 1.08-1.18 against a 1.33x ceiling. Its value is not "
                "its own ratio: three slots on one kernel over three increasing shares is "
                "what separates 'the mechanism works' from 'something else moved', and a "
                "win here that is *smaller* than 016's is the evidence, where three "
                "unrelated wins of similar size would be a reason to distrust all of them."
            ),
        ),
        Hypothesis(
            slug="019-int8-all-linear",
            kernels=("tiled_int8_all_linear",),
            category="B",
            byte_share=0.7796,
            replaces=("decode_step",),
            correctness="approximate",
            correctness_positions=264,
            top1_threshold=246 / 264,
            kl_threshold=0.01,
            weight_bits={"layers": 8},
            requires=Precondition(
                slug="015-tiled-gemv-bf16",
                floor=0.56,
                reason="the same floor as 016: below it no byte saving can be collected",
            ),
            mechanism=(
                "The same sites and the same bit width as 016, stored int8 instead of e4m3. "
                "Identical byte traffic; the only difference is which unit converts the "
                "weight, and on sm_120 that is the ALU rather than the MMA pipeline."
            ),
            prediction="inconclusive",
            rationale=(
                "Two measurements in one slot. Against 016 it isolates the conversion tax "
                "and nothing else -- same sites, same bytes, same kernel structure -- which "
                "batch 003 could only infer at 1.438x from kernels that were issue-bound "
                "anyway. Against batch 003's 012 it isolates the value of the kernel "
                "rewrite: 0.1962 against whatever this returns, same sites and same "
                "quantisation, with only the kernel structure changed. Predicted "
                "inconclusive at 1.05-1.25 rather than a win, because if the conversion tax "
                "survives the rewrite it lands here and nowhere else, and predicting a win "
                "for both dtypes would make the pair unable to say anything. Bars are batch "
                "003's own measurements at these exact sites: 012 scored 0.001096 nats and "
                "9 flips of 264, so 0.01 nats and 18 flips leave an order of magnitude and "
                "a factor of two."
            ),
        ),
        Hypothesis(
            slug="020-int4-full",
            kernels=("tiled_int4_full",),
            category="B",
            byte_share=0.9184,
            replaces=("decode_step",),
            correctness="approximate",
            correctness_positions=264,
            top1_threshold=207 / 264,
            kl_threshold=0.15,
            weight_bits={"layers": 4, "head": 4},
            requires=Precondition(
                slug="016-fp8-all-linear",
                floor=1.05,
                reason=(
                    "int4's nibble unpack and its in-loop group scale are only worth trying "
                    "once 8 bits has actually won something; if fp8 cannot clear 1.05 the "
                    "extra work cannot be paid for by the extra bytes saved"
                ),
            ),
            mechanism=(
                "017's sites at 4 bits with per-group (128) scales, two values packed per "
                "byte and unpacked in registers. 2850 MB/token: a 3.21x ceiling, the "
                "largest in the backlog. The group scale varies along K so it cannot leave "
                "the loop; it is applied to the weight tile before the dot rather than to a "
                "partial sum, which keeps the accumulator's job unchanged."
            ),
            prediction="win",
            rationale=(
                "Highest ceiling and highest risk, so it runs last and behind the tightest "
                "precondition in the batch. Predicted 1.4-1.9 against 3.21x: the shortfall "
                "is the unpack, the in-loop scale multiply, and the bf16 rounding of the "
                "scaled weight that the fp8 path does not pay. Bars are batch 003's own "
                "int4 measurements at these exact sites -- 014 scored 0.091854 nats and 38 "
                "flips of 264 -- so 0.15 nats and 57 flips leave roughly 1.5x on each. "
                "in_proj_a and in_proj_b feed an exponential and are 32 output channels "
                "wide; that is where this breaks if it breaks, and the layer-1 probes cover "
                "that launch branch specifically."
            ),
        ),
    ),
)

# ======================================================================================
# Batch 005 — the dispatch path, and the one site with parallelism to spare
# ======================================================================================
#
# Two rentals have now established that a hand-written GEMV installed on all 248 layer
# projections is slow: 0.2801 with one kernel structure, 0.1934 with the opposite one.
# Neither says *why*, and neither attacked the number rental 38's dump actually turned up.
#
# **The denominator this batch is written against is 8587.80 MB/token**, what the compiled
# column moves once the GQA expansion inductor folds away is taken out — not the 9158.23
# the roofline prints for eager. `harness.bytes_model` now defaults to it too, so every
# byte share below and every bandwidth the rental reports use the same number.

BATCH_005 = Batch(
    batch_id="005-launch-and-head",
    description=(
        "Rental 38's `TORCH_LOGS=output_code` dump is a 4.7 MB file this repository already "
        "has, and counting it answers a question no kernel had asked: the decode graph "
        "issues **508 kernel launches per token** (483 generated Triton plus 25 extern), "
        "and **not one of them is CUDA-graphed**, because the decode cache is mutated in "
        "place and inductor counts 64 mutated inputs. The compiled baseline spends 7.30 "
        "ms/token moving 8587.80 MB -- 4.79 ms of that at vendor peak, 5.3-6.4 at an "
        "achievable one -- so 0.9-2.5 ms is spread over those 508 dispatches. This batch "
        "attacks that, and it attacks the only matmul in the model big enough for a "
        "hand-written GEMV not to be grid-starved: the tied LM head, 248320 x 2560, "
        "14.80% of everything the compiled column moves, in one kernel. Eight slots, no "
        "control that cannot win, and two compositions gated on the first result."
    ),
    hypotheses=(
        Hypothesis(
            slug="000-identity",
            kernels=(),
            category="calibration",
            byte_share=0.0,
            mechanism=(
                "Install nothing. The candidate is the reference, so the measured ratio is "
                "the harness's own noise floor rather than a property of any kernel."
            ),
            prediction="identity",
            rationale=(
                "Must return 1.00 within the noise band. If it does not, the harness is "
                "measuring something other than the kernel under test and every other "
                "number in this batch is void -- which is a statement about the rental, not "
                "about any hypothesis, and the writeup has to say so rather than reporting "
                "the rest as findings. It runs first for the same reason it always has: a "
                "broken harness then costs three minutes rather than a rental."
            ),
        ),
        Hypothesis(
            slug="021-static-cache-cudagraphs",
            kernels=("static_decode_cache",),
            category="A",
            byte_share=0.0,
            replaces=("decode_cache",),
            correctness="approximate",
            correctness_positions=264,
            top1_threshold=264 / 264,
            kl_threshold=1e-06,
            weight_bits={},
            mechanism=(
                "Allocate the decode cache with `torch._dynamo.mark_static_address`, which "
                "is the promise inductor's cudagraph mutation check requires and cannot "
                "infer from a tensor handed in as an argument. The candidate then runs the "
                "same 508 kernels in the same order from one graph replay instead of 508 "
                "Python dispatches. No Triton, no arithmetic change, no byte saved."
            ),
            prediction="win",
            rationale=(
                "Byte share 0.0 and the largest ceiling in the batch, which is a "
                "combination this repository has not seen before -- every previous "
                "hypothesis was priced in bytes. The arithmetic is dispatch, not traffic: "
                "8587.80 MB/token is 4.79 ms at an RTX 5090's 1792 GB/s vendor peak and "
                "5.3-6.4 ms at the 75-90% a real kernel reaches, against a measured 7.30. "
                "That residue over 508 launches is 1.8-4.9 us each, which is what "
                "inductor's Python launch path costs uncaptured. **Predicted 1.15-1.30 "
                "against a ~1.26x ceiling**, and it is a win or an explained null rather "
                "than a coin toss because the mechanism is verified in torch's own source "
                "rather than assumed: `cudagraph_utils.check_for_mutation` exempts an index "
                "in `static_input_idxs`, and `_dynamo_static_input_type` is what puts it "
                "there. The two ways it can fail are both informative and both visible in "
                "the record. `cudagraph_nodes == 0` means it did not engage -- most likely "
                "because `cache_offset` reaches the graph as a symint and cudagraph trees "
                "key a recording per distinct int, so 128 decode steps want 128 recordings. "
                "`cudagraph_nodes > 0` with a ratio near 1.00 means launch dispatch was "
                "never the gap, which retires the whole line and is worth knowing. Gated "
                "approximately at 264/264 and 1e-6 nats because the policy forbids `exact` "
                "for a non-identity slot; unlike every other candidate here this one really "
                "is bit-identical, and a bar it can only miss by being broken is the right "
                "shape for that claim."
            ),
        ),
        Hypothesis(
            slug="022-int4-head",
            kernels=("tiled_int4_head",),
            category="B",
            byte_share=0.1480,
            replaces=("decode_step",),
            correctness="approximate",
            correctness_positions=264,
            top1_threshold=240 / 264,
            kl_threshold=0.06,
            weight_bits={"head": 4},
            mechanism=(
                "Group-128 int4 on the tied LM head and nothing else. 1271.40 MB/token "
                "becomes 327.7 including its bf16 group scales: a 1.123x ceiling from one "
                "site, with two kernel launches replacing one rather than 496 replacing "
                "248."
            ),
            prediction="win",
            rationale=(
                "**The head has never been measured on its own, and it is the only site in "
                "this model where a hand-written GEMV is not grid-starved by "
                "construction.** N=248320 launches 3880 programs at BLOCK_N=64 on a 170-SM "
                "card; `in_proj_a` is 32 wide and launches four. Batches 003 and 004 "
                "installed on all 248 layer projections and reported one aggregate rate -- "
                "319 GB/s, then 228 -- which cannot distinguish a kernel that is slow "
                "everywhere from one that is slow where there is no parallelism to have. "
                "What breaking even takes is arithmetic, not hope: the baseline spends "
                "1271.40 MB / 1177 GB/s = 1.08 ms/token in that matmul, so int4 ties at "
                "**303 GB/s** -- 1.33x the aggregate already measured, on the friendliest "
                "shape in the model -- and collects the full 1.123x at 550. Predicted "
                "1.05-1.12. It runs before the 8-bit slots because its bar is the lowest "
                "and its ceiling the highest, and its error is the batch's largest: the "
                "head is the one weight whose perturbation reaches the argmax with no "
                "further layer to attenuate it. The bars come from measurement rather than "
                "priors, which is how batch 003 failed four working kernels. Batch 003 ran "
                "int8 both without the head (012: 0.00110 nats, 255/264) and with it (013: "
                "0.00122, 256/264), so the head at 8 bits is worth about 0.0001 nats; "
                "group-128 int4 perturbs roughly 6x harder and KL goes as the square, "
                "giving ~0.004 and at worst ~0.03 if the grouping helps less than expected. "
                "The bar is 0.06, and 240/264 allows 24 flips against batch 003's 38 for "
                "int4 on *every* site."
            ),
        ),
        Hypothesis(
            slug="023-int8-head",
            kernels=("tiled_int8_head",),
            category="B",
            byte_share=0.1480,
            replaces=("decode_step",),
            correctness="approximate",
            correctness_positions=264,
            top1_threshold=256 / 264,
            kl_threshold=0.003,
            weight_bits={"head": 8},
            mechanism=(
                "The same site at 8 bits with a per-channel scale. 1271.40 MB/token becomes "
                "635.7: a 1.080x ceiling, twice int4's bytes and none of its nibble unpack."
            ),
            prediction="win",
            rationale=(
                "This slot exists to be read *against* 022, and the pair is a cleaner "
                "instrument than either alone. int8 needs **588 GB/s** to tie where int4 "
                "needs 303, so if the kernel is bandwidth-bound at this site 022 beats 023 "
                "and the ordering follows the bytes. If 023 beats 022 the kernel is still "
                "issue-bound and the unpack is on the critical path -- which is exactly "
                "what batch 003 measured when int4 cost 1.046x int8 *while moving half the "
                "bytes*, and what two rentals have failed to explain. Predicted 1.02-1.08: "
                "the bar is higher than 022's but not out of reach on a site with this much "
                "parallelism, and the ceiling is 1.080 so there is no room to be wrong by "
                "much in either direction. Bars are batch 003's own measurements at this "
                "exact site rather than an extrapolation: 013 minus 012 puts the int8 head "
                "at ~0.0001 nats and about one flip, so 0.003 nats and 8 flips of 264 leave "
                "an order of magnitude on the first and 8x on the second."
            ),
        ),
        Hypothesis(
            slug="024-fp8-head",
            kernels=("tiled_fp8_head",),
            category="B",
            byte_share=0.1480,
            replaces=("decode_step",),
            correctness="approximate",
            correctness_positions=264,
            top1_threshold=252 / 264,
            kl_threshold=0.012,
            weight_bits={"head": 8},
            mechanism=(
                "The same site and the same 8 bits as 023, stored e4m3 instead of int8, so "
                "the conversion happens inside sm_120's MMA pipeline rather than as an ALU "
                "instruction on the critical path."
            ),
            prediction="win",
            rationale=(
                "Adjacent to 023 so the only difference between the two slots is the "
                "conversion tax, which batch 003 measured at **1.438x** -- int8 cost that "
                "much more time than bf16 in the same kernel on the same sites, and nothing "
                "since has isolated it on a site where the kernel was not already bound by "
                "something else. Identical ceiling to 023 (1.080x, identical bytes), so "
                "023 minus 024 is the tax and nothing else. Predicted 1.03-1.08, a little "
                "above 023 for the same reason batch 004 ordered fp8 ahead of int8. The "
                "accuracy goes the other way and the bar says so: e4m3 carries 3 mantissa "
                "bits against int8's effective 7 at per-channel scale, so batch 004 "
                "interpolated fp8's error at ~3x int8's in amplitude and ~9x in KL. Applied "
                "to the ~0.0001 nats batch 003 measured for the int8 head that is ~0.001; "
                "the bar is 0.012 and 252/264 allows 12 flips. Both are derived from the "
                "two measured points this repository owns, not from what fp8 is generally "
                "said to cost."
            ),
        ),
        Hypothesis(
            slug="025-fused-causal-conv",
            kernels=("fused_causal_conv",),
            category="A",
            byte_share=0.00055,
            replaces=("causal_conv",),
            correctness="approximate",
            correctness_positions=264,
            top1_threshold=260 / 264,
            kl_threshold=0.001,
            weight_bits={},
            mechanism=(
                "One Triton kernel per linear-attention layer for the four-tap depthwise "
                "causal convolution: the `cat` that builds its input, the cuDNN "
                "`extern_kernels.convolution` itself, the `silu`, and the `copy_` that "
                "advances the history, in one launch. 72 of the step's 508 launches become "
                "24."
            ),
            prediction="win",
            rationale=(
                "**The one place in this model where launch count and byte count are wildly "
                "out of proportion: 14.2% of the dispatch for 0.055% of the traffic.** At "
                "batch-1 decode that convolution is 8192 channels x 4 taps -- 32768 "
                "multiplies, roughly a microsecond of arithmetic -- wrapped in three "
                "launches, one of which is the only `extern_kernels` call left in the whole "
                "decode graph. Inductor cannot close this itself, and that is the "
                "mechanistic claim: a scheduler cannot fuse a producer and a consumer "
                "across an opaque external call, so the `cat` and the `copy_` are stranded "
                "either side of it by construction rather than by oversight. Worth 48 of "
                "508 launches, 9.4% of the dispatch, plus whatever a cuDNN convolution "
                "dispatch costs above a Triton launch -- against 0.9-2.5 ms of launch "
                "overhead that is **0.08-0.24 ms of 7.30, a ratio of 1.011 to 1.033**. The "
                "low end is inside the noise band and the high end is several times outside "
                "it, so this is a real coin with a weighted edge rather than a certainty, "
                "and it is predicted `win` because the launch saving is structural and the "
                "arithmetic it replaces is trivially small. It also pairs with 021: if "
                "launches cost nothing once the step is CUDA-graphed, this kernel is worth "
                "nothing inside 027, and that is a falsifiable pair rather than two "
                "independent hopes. Gated approximately because the policy forbids `exact`, "
                "but the kernel rounds exactly as the reference does -- fp32 accumulate, "
                "round to bf16, *then* silu -- so the bars are tight on purpose: a real "
                "disagreement here is a bug, not a dtype."
            ),
        ),
        Hypothesis(
            slug="026-int4-head-static-cache",
            kernels=("tiled_int4_head", "static_decode_cache"),
            category="B",
            byte_share=0.1480,
            replaces=("decode_step", "decode_cache"),
            correctness="approximate",
            correctness_positions=264,
            top1_threshold=240 / 264,
            kl_threshold=0.06,
            weight_bits={"head": 4},
            requires=Precondition(
                slug="021-static-cache-cudagraphs",
                floor=1.02,
                reason=(
                    "this slot's only novel ingredient is the CUDA graph; if marking the "
                    "cache static did not clear the noise band on its own, this is 022 "
                    "measured a second time and the batch already has that number"
                ),
            ),
            mechanism=(
                "022 and 021 composed: fewer bytes at the largest matmul, and the whole "
                "step dispatched as one graph replay."
            ),
            prediction="win",
            rationale=(
                "The two mechanisms are disjoint -- one removes bytes from a kernel, the "
                "other removes dispatch from every kernel -- so the naive expectation is "
                "the product, **~1.26 x 1.123 = 1.41x, the batch's only realistic shot at a "
                "champion.** Predicted 1.20-1.40. But the composition is a measurement "
                "rather than an inference, and it can come out *super*-additive in a way "
                "worth catching: a hand-written GEMV replaces one fused inductor kernel "
                "with two launches, and that penalty is paid in dispatch, which is exactly "
                "what 021 removes. If 026 exceeds 021 x 022 then launch dispatch was part "
                "of what has been making every hand-written kernel here look slow, and that "
                "reframes batches 003 and 004 rather than merely adding to them. The bars "
                "are 022's unchanged: same quantiser, same site, and 021 changes no "
                "arithmetic at all."
            ),
        ),
        Hypothesis(
            slug="027-conv-head-static-cache",
            kernels=("tiled_int4_head", "static_decode_cache", "fused_causal_conv"),
            category="B",
            byte_share=0.1485,
            replaces=("decode_step", "decode_cache", "causal_conv"),
            correctness="approximate",
            correctness_positions=264,
            top1_threshold=240 / 264,
            kl_threshold=0.06,
            weight_bits={"head": 4},
            requires=Precondition(
                slug="021-static-cache-cudagraphs",
                floor=1.02,
                reason=(
                    "the same floor as 026: with no CUDA graph this is 022 and 025 "
                    "composed, and both are measured alone in this batch with disjoint "
                    "costs, so the composition would carry no new information"
                ),
            ),
            mechanism=(
                "Everything this batch has that works, on one candidate: the int4 head, the "
                "fused causal conv, and a decode cache the compiler can CUDA-graph."
            ),
            prediction="win",
            rationale=(
                "Last because it is the riskiest -- three installers, the largest surface "
                "for one of them to interact badly with another -- and everything cheap is "
                "on disk by the time it runs. Predicted 1.20-1.45. It is also the slot that "
                "tests 025's stated pair: once the step is a single graph replay, saving 48 "
                "launches should be worth **nothing**, so 027 materially above 026 would "
                "mean the fused conv is buying something other than dispatch (most likely "
                "cuDNN's own algorithm selection on an 8192x4 problem), and 027 equal to "
                "026 confirms the launch account. Either way the comparison is the finding, "
                "and it costs one slot because the two compositions share a precondition. "
                "Bars are 022's: the conv kernel is numerically the reference and 021 "
                "changes no arithmetic, so the head is still the only source of error."
            ),
        ),
    ),
)


BATCH_006 = Batch(
    batch_id="006-tile-and-sites",
    description=(
        "Batch 005 found that the hand-written GEMV was **grid-starved, not structurally "
        "slow**: the same kernel family achieved 228-319 GB/s averaged over all 248 layer "
        "projections and **656 GB/s on the tied LM head**, which launches 3880 programs "
        "where `in_proj_a` launches four. That reframed the site. It left the *tile* "
        "un-examined, and the tile is a heuristic no rental has ever timed: "
        "`_launch_shape` targets 256 programs -- one wave on a 170-SM card -- and it "
        "*narrows* BLOCK_N to 32 for every site with N <= 4096, which halves the "
        "contiguous run each program reads from 128 bytes to 64. Both are the opposite of "
        "what the head won on. So this batch measures the tile instead of deriving it, on "
        "the card in hand, exactly as `max-autotune` does for the code we are trying to "
        "beat -- and then runs the same int4 kernel on the 52.75% of per-token bytes "
        "behind the MLP and the 83.03% behind the 200 layer projections wide enough for a "
        "tile to matter. Nine slots. Every one of them can beat the champion's 1.0791: "
        "there is no control here whose ceiling is 1.0, because after batch 005 the "
        "cheapest control is a slot that also has a ceiling."
    ),
    hypotheses=(
        Hypothesis(
            slug="000-identity",
            kernels=(),
            category="calibration",
            byte_share=0.0,
            mechanism=(
                "Install nothing. The candidate is the reference, so the measured ratio is "
                "the harness's own noise floor rather than a property of any kernel."
            ),
            prediction="identity",
            rationale=(
                "Must return 1.00 within the noise band. If it does not, the harness is "
                "measuring something other than the kernel under test and every other "
                "number in this batch is void -- a statement about the rental rather than "
                "about any hypothesis, and the writeup has to say so rather than reporting "
                "the rest as findings. It has measured 1.0009, 1.0018, 1.0024, 0.9913 and "
                "1.0008 on the five rentals that got this far, so a miss here is news."
            ),
        ),
        Hypothesis(
            slug="028-int4-head-tuned",
            kernels=("tiled_int4_head_tuned",),
            category="B",
            byte_share=0.1480,
            replaces=("decode_step",),
            correctness="approximate",
            correctness_positions=264,
            top1_threshold=240 / 264,
            kl_threshold=0.06,
            weight_bits={"head": 4},
            mechanism=(
                "The champion's site and the champion's arithmetic, with "
                "`tune_launch_shape` timing every candidate tile on the card instead of "
                "`_launch_shape` deriving one from a comment about SM counts. Same bytes, "
                "same kernel, same 1.1249x ceiling; only the launch geometry moves."
            ),
            prediction="win",
            rationale=(
                "**The cheapest slot in the batch and the only one directly comparable to "
                "a number already on the leaderboard.** `022-int4-head` measured 1.0791 "
                "against a 1.1249x ceiling -- it collected **70%**, at 656 GB/s where the "
                "baseline runs at 1282 -- with BLOCK_N=64, BLOCK_K=64, SPLIT_K=1, "
                "num_warps=4 and Triton's default 3 stages. Not one of those five numbers "
                "was ever measured. The search space includes that exact configuration "
                "first, so **the floor of this slot is the champion** and the only "
                "question is how much of the remaining 30% a measured tile collects. "
                "Predicted **1.08-1.12**. It runs before every other kernel because it is "
                "the tuner's own control: if the tuner cannot improve on the heuristic at "
                "the one site where the heuristic was accidentally right, the slots behind "
                "it are far less likely to -- and if it comes back *below* 1.0791 the "
                "tuner itself is broken and the batch says so in three minutes rather than "
                "at the end. Bars are 022's, unchanged and already met at 0.9318 and "
                "0.01674 nats: the arithmetic is identical, so a different correctness "
                "result here would be a tiling bug rather than a quantisation effect."
            ),
        ),
        Hypothesis(
            slug="029-head-and-conv",
            kernels=("tiled_int4_head_tuned", "fused_causal_conv"),
            category="B",
            byte_share=0.1486,
            replaces=("decode_step", "causal_conv"),
            correctness="approximate",
            correctness_positions=264,
            top1_threshold=240 / 264,
            kl_threshold=0.06,
            weight_bits={"head": 4},
            mechanism=(
                "Batch 005's two winners on one candidate: int4 on the tied LM head, and "
                "the four-tap causal convolution's cat + cuDNN call + silu + cache copy "
                "collapsed into one Triton kernel per linear-attention layer. One removes "
                "943.7 MB/token of weight traffic; the other removes 48 of 508 launches "
                "and no bytes at all."
            ),
            prediction="win",
            rationale=(
                "**The one slot here that is near-certain, and it is in the batch for "
                "exactly that reason.** Both halves won on rental 40 -- 1.0791 and 1.0144 "
                "-- on mechanisms that share nothing: the head slot changes what one "
                "matmul reads, the conv slot changes how 24 layers dispatch. Their "
                "installers patch disjoint attributes (`project_logits` via the root "
                "class, and the conv step inside each linear-attn module), so neither can "
                "discard the other -- the failure `AGENT.md` records for two root-class "
                "installers. Naive expectation is the product of the measured savings, "
                "0.493 ms + 0.093 ms of 6.70, which is **1.0957**; predicted "
                "**1.09-1.13** because 028's tile can only add to the first term. If this "
                "slot does not clear 1.0791 then something about composing two installs is "
                "wrong, which is worth knowing before five slots depend on it. It also "
                "banks a champion early: after this the batch can spend its remaining "
                "slots on hypotheses that might fail without risking the session's result."
            ),
        ),
        Hypothesis(
            slug="030-int4-mlp",
            kernels=("tiled_int4_mlp",),
            category="B",
            byte_share=0.5275,
            replaces=("swiglu_mlp",),
            correctness="approximate",
            correctness_positions=264,
            top1_threshold=220 / 264,
            kl_threshold=0.10,
            weight_bits={"mlp": 4},
            mechanism=(
                "Group-128 int4 on the 96 MLP projections and nothing else: 4529.85 "
                "MB/token to 1167.8, a **1.6545x ceiling** from the largest homogeneous "
                "block of bytes in the model. Every site is wide -- 9216 for gate_proj and "
                "up_proj, 2560 for down_proj -- and each gets a tile chosen by measurement."
            ),
            prediction="win",
            rationale=(
                "**The hypothesis the batch exists to test, and the number it turns on is "
                "331 GB/s.** At the baseline's 1282 GB/s the MLP's 4529.85 MB/token costs "
                "3.53 ms of 6.70; int4 moves 1167.8 MB there, so the slot ties at "
                "**1167.8 / 3.53 ms = 331 GB/s** and collects the full 1.6545x at 1282. "
                "That tie point sits **between the two rates this project has measured** "
                "-- 196-280 GB/s for the untuned kernel over the layer projections "
                "(rentals 37 and 38), 656 for the tuned-by-accident head (rental 40) -- "
                "which is what makes it a coin worth flipping rather than an argument. "
                "What moves it is the tile, and both defects are arithmetic rather than "
                "suspicion: `down_proj` at N=2560 gets BLOCK_N=32 and SPLIT_K=4, which is "
                "320 program instances on 170 SMs *and* a 64-byte contiguous read where "
                "the hardware transacts 128; `up_proj` at N=9216 gets 288. The head got "
                "3880 and a full 128-byte line. Predicted **1.05-1.40**: the low end is a "
                "tile that helps a little, the high end is one that reaches the head's "
                "byte rate, and below 1.0 means the layer projections are slow for a "
                "reason that is not the launch geometry -- which would be the first "
                "evidence for that, because nobody has varied the geometry before. Bars "
                "come from two measured points: batch 003's `014` put int4 on all 248 "
                "sites plus the head at 0.09185 nats and 226/264, and `022` put the head "
                "alone at 0.01674, so the layer projections carry ~0.075 nats and the MLP "
                "is 63.5% of their weight bytes -- call it ~0.048. The bar is 0.10 nats "
                "and 220/264, which is looser than `014` measured while quantising twice "
                "as much."
            ),
        ),
        Hypothesis(
            slug="031-int4-mlp-and-head",
            kernels=("tiled_int4_mlp", "tiled_int4_head_tuned"),
            category="B",
            byte_share=0.6755,
            replaces=("swiglu_mlp", "decode_step"),
            correctness="approximate",
            correctness_positions=264,
            top1_threshold=215 / 264,
            kl_threshold=0.12,
            weight_bits={"mlp": 4, "head": 4},
            requires=Precondition(
                slug="030-int4-mlp",
                floor=1.00,
                reason=(
                    "this slot is 030 plus a site the batch has already measured alone in "
                    "028; if int4 on the MLP cannot reach parity on its own then the MLP "
                    "half contributes a known loss and the composition re-measures 028 "
                    "with a handicap, which is a number the batch already has"
                ),
            ),
            mechanism=(
                "030 and 028 composed: the two largest blocks of weight bytes in the model "
                "at 4 bits, 67.55% of what the compiled column moves, for a 2.0269x "
                "ceiling. Disjoint installs -- the MLP swaps `nn.Linear` subclasses, the "
                "head swaps the root class's `project_logits`."
            ),
            prediction="win",
            rationale=(
                "The mechanisms are the same mechanism at two sets of sites, so unlike "
                "029 this is not a product of two unrelated effects but a single question "
                "asked of more bytes: **does the tuned kernel hold its byte rate as the "
                "share it carries grows?** That is not guaranteed and the failure mode is "
                "specific -- 96 extra pairs of launches where inductor had fused the "
                "projection into the norm and the residual, against a dispatch path that "
                "rental 38's dump measured at 508 launches per token with no CUDA graph "
                "anywhere. Predicted **1.15-1.55**: the midpoint is 030 and 028 collecting "
                "the same fraction of their ceilings together as they did apart. "
                "**031 materially below 030 x 028 is the interesting outcome**, because it "
                "prices the fusion loss per site for the first time -- batch 004 named it "
                "as a suspect and nothing has ever isolated it. Bars add the two measured "
                "contributions: ~0.048 nats for the MLP and 0.01674 for the head, so 0.12 "
                "and 215/264 leave roughly a factor of two on each."
            ),
        ),
        Hypothesis(
            slug="032-int4-wide-and-head",
            kernels=("tiled_int4_wide",),
            category="B",
            byte_share=0.9783,
            replaces=("decode_step",),
            correctness="approximate",
            correctness_positions=264,
            top1_threshold=210 / 264,
            kl_threshold=0.18,
            weight_bits={"mlp": 4, "linear_attn": 4, "full_attn": 4, "head": 4},
            requires=Precondition(
                slug="030-int4-mlp",
                floor=1.00,
                reason=(
                    "the same floor and the same reason as 031: this slot adds 104 more "
                    "sites of the mechanism 030 measures, so a 030 below parity makes this "
                    "a larger measurement of a loss the batch has already recorded, and "
                    "batch 003 spent five slots doing exactly that"
                ),
            ),
            mechanism=(
                "int4 on every site a tile can help: the 200 layer projections with "
                "N >= 1024, plus the tied head, in **one** kernel -- the registry allows "
                "one champion per replaceable operation and a head installed beside this "
                "would claim `decode_step` twice. **97.83% of what the compiled column "
                "moves, a 3.7578x ceiling.** The 48 excluded projections are `in_proj_a` "
                "and `in_proj_b`, 32 channels wide and 7.86 MB/token between them."
            ),
            prediction="win",
            rationale=(
                "**The full ceiling, minus the only sites no tile can reach.** Batches 003 "
                "and 004 installed on all 248 and reported one aggregate byte rate, which "
                "batch 005 showed cannot separate a slow kernel from a starved grid -- so "
                "this slot removes the 48 that are starved by construction and costs "
                "0.09% of per-token bytes to do it. Ties at the same 331 GB/s as 030, "
                "because the tie point is a property of int4 against a 1282 GB/s baseline "
                "and not of how many sites carry it. Predicted **1.10-2.20**, a wide range "
                "on purpose: this is 030's mechanism at 1.85x the byte share plus 028's, "
                "and if the tuned tile holds its rate the arithmetic says 1.9, while every "
                "per-site cost that does not scale with bytes -- the forfeited fusion, the "
                "second launch per site, split-K's reduction pass -- is multiplied by 200 "
                "here against 96 in 030. Reading 032 against 031 against 030 is a "
                "per-site-cost curve, which is the thing two rentals of aggregate numbers "
                "could not produce. Bars: batch 003's `014` measured this configuration "
                "plus the 48 gates at **0.09185 nats and 226/264**, so 0.18 nats and "
                "210/264 are twice that measurement's error and sixteen more flips."
            ),
        ),
        Hypothesis(
            slug="033-int4-wide-head-and-conv",
            kernels=("tiled_int4_wide", "fused_causal_conv"),
            category="B",
            byte_share=0.9789,
            replaces=("decode_step", "causal_conv"),
            correctness="approximate",
            correctness_positions=264,
            top1_threshold=210 / 264,
            kl_threshold=0.18,
            weight_bits={"mlp": 4, "linear_attn": 4, "full_attn": 4, "head": 4},
            requires=Precondition(
                slug="032-int4-wide-and-head",
                floor=1.05,
                reason=(
                    "the conv is worth 0.093 ms of 6.70 and this slot's only novel "
                    "ingredient is that 1.4%; if 032 has not cleared the noise band by "
                    "more than the conv can contribute, this re-measures 032 and the "
                    "batch already has that number with a tighter interval"
                ),
            ),
            mechanism=(
                "Everything in this batch that can compose: 032's int4 on 97.83% of the "
                "weight bytes, and the fused causal convolution on 48 of the step's 508 "
                "launches. Two installers patching disjoint attributes."
            ),
            prediction="win",
            rationale=(
                "The batch's best shot at a champion and its riskiest slot, so it runs "
                "last among the kernels with everything cheaper already on disk: three "
                "installers, 200 quantised sites and the largest resident memory in the "
                "batch. Predicted **1.12-2.25**, which is 032 plus the conv's measured "
                "0.093 ms. It also asks a question the arithmetic cannot answer: **the "
                "conv's win was a dispatch win, and 032 adds ~200 launches to the step.** "
                "If a saved launch is worth less when the step dispatches more of them, "
                "033 minus 032 comes out below rental 40's 1.0144 and the launch account "
                "is not linear; if it comes out at 1.0144 the two effects are independent "
                "and the account holds. Either reading is a finding, and it costs one slot "
                "because the ingredients are measured separately above."
            ),
        ),
        Hypothesis(
            slug="034-static-cache-cudagraphs",
            kernels=("static_decode_cache",),
            category="A",
            byte_share=0.0,
            replaces=("decode_cache",),
            correctness="approximate",
            correctness_positions=264,
            top1_threshold=264 / 264,
            kl_threshold=1e-06,
            weight_bits={},
            mechanism=(
                "`021` again, with the diagnostic it was missing. The decode cache is "
                "allocated under `mark_static_address`, which is the promise inductor's "
                "cudagraph mutation check requires and cannot infer from a tensor handed "
                "in as an argument. Same 508 kernels in the same order, from one graph "
                "replay instead of 508 Python dispatches."
            ),
            prediction="win",
            rationale=(
                "**Still the largest unattacked number in the project, and rental 40 did "
                "not test it**: `cudagraph_nodes: 0, cudagraph_skips: 127` -- the candidate "
                "was refused exactly as the reference is, so 0.9986 said nothing. Its "
                "writeup ranked two suspects, and **this session refuted both on a CPU for "
                "nothing.** Running the tiny config under "
                "`TORCH_LOGS=cudagraph_static_inputs` prints `Adding static input pos 5 "
                "for source L['cache'].layers[0].conv` and the same for every recurrent "
                "state and KV slice: the mark does reach `static_input_indices`, through a "
                "plain Python object, and `_extract_tensor_dict` does stamp it. So the "
                "break is downstream of both, and the two remaining candidates are the int "
                "graph input -- `cache.seq_len` reaches the graph as a symint, "
                "`cudagraphify_impl` keys `fn_cache` on every int input, and a 128-token "
                "decode therefore wants 128 recordings against a "
                "`cudagraph_unexpected_rerecord_limit` of exactly 128 -- and a skip for "
                "some reason entirely unrelated to mutation. **This slot can now tell "
                "those apart**, because `cudagraphs_during` captures the skip *message* "
                "and not only the counter; batch 005 had the count and spent a section of "
                "its writeup ranking suspects the sentence beside it would have named. "
                "Ceiling ~1.26x from rental 38's dump: 8587.80 MB/token is 4.79 ms at an "
                "RTX 5090's vendor peak and 5.3-6.4 at an achievable one, against 6.70 "
                "measured, so 0.3-1.9 ms sits in 508 dispatches. Predicted **1.05-1.26 if "
                "it engages**, and it runs last because capturing 128 graphs of a "
                "508-kernel step is the most likely thing in this batch to be slow or to "
                "exhaust memory, and everything else is on disk by then. Gated at 264/264 "
                "and 1e-6 nats because the policy forbids `exact` for a non-identity slot "
                "and this candidate really is bit-identical: a bar it can only miss by "
                "being broken is the right shape for that claim."
            ),
        ),
    ),
)


BATCH_007 = Batch(
    batch_id="007-compose-and-retile",
    description=(
        "Rental 42 left the leaderboard in a state no rental should: **the champion's "
        "1.0791 was not re-measured**, because the slot that was meant to improve on it "
        "replaced its tile and returned 0.9920. So this batch opens by re-measuring "
        "`022-int4-head` unchanged, on a second card, and everything behind that slot is "
        "read against it rather than against a number from another rental. "
        "The rest is the two things rental 42 showed are worth a slot each and nothing "
        "else is. **Compositions, because they are not predictable**: the project holds "
        "three measured wins on disjoint mechanisms -- int4 on the head (1.0791), the "
        "fused causal conv (1.0144) and the static decode cache (1.0196) -- and the one "
        "pair ever composed returned 0.7937, which is the largest unexplained number here. "
        "**Tiles, because on this one site the tile is worth 2.3x** and the only two ever "
        "timed in the decode step were a heuristic and a micro-benchmark's pick. Both "
        "pinned tiles are registered here, before the rental, and ranked by the ratio the "
        "whole step returns -- the one selection method rental 42 did not discredit. "
        "Nine slots. Every one of them can beat 1.0791: every kernel slot carries the "
        "head's 1.1249x ceiling, and the two that add the conv and the cache carry their "
        "measured savings on top of it."
    ),
    hypotheses=(
        Hypothesis(
            slug="000-identity",
            kernels=(),
            category="calibration",
            byte_share=0.0,
            mechanism=(
                "Install nothing. The candidate is the reference, so the measured ratio is "
                "the harness's own noise floor rather than a property of any kernel."
            ),
            prediction="identity",
            rationale=(
                "Must return 1.00 within the noise band or every other number in this "
                "batch is void. It has measured 1.0009, 1.0018, 1.0024, 0.9913, 1.0008 and "
                "1.0053 on the six rentals that got this far. **Read its sign before "
                "reading anything else**: rental 42 carried +0.53%, which is a third of "
                "what `034` appeared to win, and this batch compares slots against a "
                "champion measured on a rental that carried +0.08%."
            ),
        ),
        Hypothesis(
            slug="035-int4-head",
            kernels=("tiled_int4_head",),
            category="B",
            byte_share=0.1480,
            replaces=("decode_step",),
            correctness="approximate",
            correctness_positions=264,
            top1_threshold=240 / 264,
            kl_threshold=0.06,
            weight_bits={"head": 4},
            mechanism=(
                "The champion exactly as rental 40 ran it: group-128 int4 on the tied LM "
                "head alone, at the heuristic tile -- BLOCK_N 64, BLOCK_K 64, SPLIT_K 1, 4 "
                "warps, 3 stages -- with the install-time tuner off. 1271.40 MB/token of "
                "bf16 weight becomes 317.85 MB of packed nibbles plus its group scales, in "
                "one matmul that launches 3880 programs."
            ),
            prediction="win",
            rationale=(
                "**Not a new hypothesis: the incumbent, re-measured on hardware that has "
                "never run it.** `AGENT.md` section 6 forbids carrying a number across "
                "sessions and rental 42 carried one anyway, because `028` replaced this "
                "slot with a searched tile and the leaderboard has stood on a single "
                "rental's measurement ever since. Everything else in this batch is read "
                "against this slot, so it runs first among the kernels and nothing is "
                "gated on it -- a batch that made its whole tail conditional on one slot "
                "would have no result if that slot errored. Predicted **1.06-1.10**: the "
                "mechanism is measured and the arithmetic is fixed at a 1.1249x ceiling, "
                "and the width of the range is the card, not the kernel. Rental 40's 5090 "
                "ran the reference at 1282 GB/s and rental 42's at 1198, and a slower card "
                "moves the head's share of the step, not just its clock. **Below 1.05 is "
                "the interesting outcome** and it would mean the champion is "
                "card-dependent in a way nothing has recorded. Bars are 022's own, already "
                "met at 0.9318 and 0.01674 nats; this slot re-measures those too, and it "
                "is the only slot here whose correctness result is a re-test rather than a "
                "prediction."
            ),
        ),
        Hypothesis(
            slug="036-int4-head-static-cache",
            kernels=("tiled_int4_head", "static_decode_cache"),
            category="B",
            byte_share=0.1480,
            replaces=("decode_step", "decode_cache"),
            correctness="approximate",
            correctness_positions=264,
            top1_threshold=240 / 264,
            kl_threshold=0.06,
            weight_bits={"head": 4},
            mechanism=(
                "The champion plus `mark_static_address` on all 64 decode-cache tensors. "
                "One removes 953.55 MB/token from a matmul; the other removes inductor's "
                "per-call alignment check from 64 tensors on each of 128 decode steps, "
                "which is worth 1198 -> 1222 GB/s on identical bytes. Nothing in either "
                "touches what the other does."
            ),
            prediction="win",
            rationale=(
                "**This is batch 005's `026`, which has never run**: its precondition "
                "declined it on rental 40 and batch 006 did not carry it. Both halves are "
                "now measured alone -- 1.0791 on rental 40 and 1.0196 (about 1.4% net of "
                "that rental's +0.53% offset) on rental 42 -- and they are the two "
                "mechanisms in this project furthest apart: bytes in one kernel against "
                "dispatch work outside every kernel. Naive expectation is the product, "
                "0.493 ms and 0.094 ms off 6.70, which is **1.0960**; predicted "
                "**1.08-1.11**. It runs second because it is the batch's most likely "
                "champion and banking one early is what lets the slots behind it fail "
                "safely. The composition itself is the risk and it is a **tested** one: "
                "both installers replace the root model class, which silently discarded "
                "each other until batch 005 keyed the factories by base, and "
                "`test_composing_two_root_class_patches_keeps_both` has asserted since "
                "that both survive. Bars are the head's, unchanged: the static cache is "
                "bit-identical (264/264, 0.0 nats on rental 42) so it cannot move them, "
                "and a correctness result away from 022's would mean the head install did "
                "not survive the composition."
            ),
        ),
        Hypothesis(
            slug="037-int4-head-wide-tile",
            kernels=("tiled_int4_head_wide",),
            category="B",
            byte_share=0.1480,
            replaces=("decode_step",),
            correctness="approximate",
            correctness_positions=264,
            top1_threshold=240 / 264,
            kl_threshold=0.06,
            weight_bits={"head": 4},
            mechanism=(
                "The champion's kernel and site at BLOCK_N=128 with 8 warps instead of "
                "BLOCK_N=64 with 4: 1940 programs instead of 3880, a full 128 contiguous "
                "bytes of packed weight per row read instead of 64, and the same "
                "fp32 weight tile per thread because the warps double with the width."
            ),
            prediction="win",
            rationale=(
                "**The tile on this site is worth 2.3x and only two points have ever been "
                "measured in the decode step.** BLOCK_N=64 ran at 656 GB/s (rental 40) and "
                "BLOCK_N=256 at 282 (rental 42), and the ranked suspect for the second is "
                "register pressure rather than the grid: at BLOCK_N=256, BLOCK_K=64 and 4 "
                "warps the kernel materialises a 64 KB fp32 weight tile per program, which "
                "is 128 registers a thread before the accumulator. This pin is the point "
                "between them that keeps per-thread pressure at the champion's while "
                "doubling the bytes each row read covers. The head streams 317.85 MB "
                "against a baseline that moves 1271.40 MB there at 1282 GB/s, so the slot "
                "ties the champion at 656 GB/s, reaches **1.105 at 900** and **1.122 at "
                "1200**, against a ceiling of 1.1249. Predicted **1.09-1.12**. Two other "
                "outcomes are findings rather than nulls: ~1.079 means the tile is not "
                "what separates 656 from the baseline's 1198-1282, and **below 1.0 means "
                "the pressure suspect is wrong and the grid is what matters**, which "
                "points the next batch at BLOCK_N=32 rather than at another wide tile. "
                "Bars are 022's and must be met exactly: the launch geometry changes no "
                "arithmetic and SPLIT_K stays 1, so the summation order is unchanged and a "
                "different correctness number here is a tiling bug."
            ),
        ),
        Hypothesis(
            slug="038-int4-head-deep-pipe",
            kernels=("tiled_int4_head_deep",),
            category="B",
            byte_share=0.1480,
            replaces=("decode_step",),
            correctness="approximate",
            correctness_positions=264,
            top1_threshold=240 / 264,
            kl_threshold=0.06,
            weight_bits={"head": 4},
            mechanism=(
                "The champion's tile with `num_stages=5` instead of Triton's default 3: "
                "four loads of the packed weight in flight per program instead of two, "
                "paid for in shared memory rather than in registers."
            ),
            prediction="win",
            rationale=(
                "**The champion runs at 36% of the card and nobody has established what "
                "binds it.** 656 GB/s against 1792 peak, with 3880 programs -- 23 waves on "
                "170 SMs -- so it is not the grid. The arithmetic per byte is small enough "
                "to rule out the ALU on its own: the nibble unpack is about ten operations "
                "per byte, which is 3.2 G ops per token against a card that issues tens of "
                "T ops/s, and the int8 slot at the same site reached 847 GB/s with a "
                "fifth of that work -- one load and one convert per byte. What is left "
                "is latency: 20 dependent loop "
                "iterations, each waiting on a 4 KB tile, with a pipeline two deep. This "
                "slot is the one axis rental 42's search never varied alone -- "
                "`refine_pipeline` moved warps, BLOCK_K and depth together on a "
                "micro-benchmark whose ranking inverted in place. Predicted **1.08-1.12**, "
                "the same arithmetic as 037 because it is the same byte saving at a "
                "different rate. It is read **against 037**, not only against 035: if both "
                "win the two axes are independent and the next batch pins their "
                "combination; if this one wins and 037 does not, the kernel is "
                "latency-bound and the width is a distraction. Bars are 022's, exactly as "
                "in 037 and for the same reason."
            ),
        ),
        Hypothesis(
            slug="039-int4-head-and-conv",
            kernels=("tiled_int4_head", "fused_causal_conv"),
            category="B",
            byte_share=0.1486,
            replaces=("decode_step", "causal_conv"),
            correctness="approximate",
            correctness_positions=264,
            top1_threshold=240 / 264,
            kl_threshold=0.06,
            weight_bits={"head": 4},
            mechanism=(
                "The champion plus the four-tap causal convolution's cat, cuDNN call, silu "
                "and cache copy collapsed into one Triton kernel per linear-attention "
                "layer: 953.55 MB/token of weight traffic removed from one matmul, and 48 "
                "of the step's 508 launches removed from 24 layers."
            ),
            prediction="win",
            rationale=(
                "**The re-run of the largest unexplained number this project holds, with "
                "the one variable rental 42 changed put back.** `029` was this pair with "
                "the *tuned* head -- BLOCK_N=256, the tile that measured 282 GB/s alone -- "
                "and it returned **0.7937**, with the conv adding ~1.85 ms/token where it "
                "had saved 0.093 by itself. This slot is the same pair at the champion's "
                "tile, so it separates two readings that rental 42 could not: if it lands "
                "near **1.096** (0.493 + 0.093 ms off 6.70) the regression belonged to the "
                "tuned tile, and if it lands near 0.79 the regression belongs to composing "
                "these two installers and the `TORCH_LOGS=output_code` dump this rental "
                "takes of exactly this pair names it. Predicted **win, 1.08-1.11**, and "
                "the prediction is deliberately the optimistic branch: the conv installer "
                "patches `GatedDeltaNet` while the head replaces the root class, which is "
                "the same disjointness 036 relies on, and nothing in the conv's kernel "
                "reads the head's. **The honest confidence is about two in three**, which "
                "is why nothing ahead of this slot is gated on it and why it runs after "
                "the three that are not."
            ),
        ),
        Hypothesis(
            slug="040-int4-head-conv-cache",
            kernels=("tiled_int4_head", "fused_causal_conv", "static_decode_cache"),
            category="B",
            byte_share=0.1486,
            replaces=("decode_step", "causal_conv", "decode_cache"),
            correctness="approximate",
            correctness_positions=264,
            top1_threshold=240 / 264,
            kl_threshold=0.06,
            weight_bits={"head": 4},
            requires=Precondition(
                slug="039-int4-head-and-conv",
                floor=1.05,
                reason=(
                    "this slot is 039 plus a mechanism 036 has already measured composed "
                    "with the head, so if the head-and-conv pair has not cleared the "
                    "champion there is nothing here the batch does not already have: it "
                    "would re-measure the conv regression with a third install on top, "
                    "which is what batch 003 spent five slots doing"
                ),
            ),
            mechanism=(
                "All three measured wins at once: int4 on the tied head, the fused causal "
                "conv, and the decode cache allocated static. Bytes, launches and dispatch "
                "work, on three disjoint parts of the step."
            ),
            prediction="win",
            rationale=(
                "**This is batch 005's `027`, which has also never run**, and it is the "
                "batch's arithmetic maximum at the champion's tile: 6.70 - 0.493 - 0.093 - "
                "0.094 = 6.020 ms, or **1.113**. Predicted **1.10-1.14**. What makes it "
                "worth a slot rather than an inference is that two of the three pairs "
                "inside it are measured by the time it runs -- 036 and 039 -- so 040 "
                "against those two is the first three-way composition this project can "
                "*read*: if it falls short of 036 x 039 / 035 the shortfall is the "
                "interaction, isolated, with both pairs on disk to subtract. Two of the "
                "three installers replace the root class and the third patches "
                "`GatedDeltaNet`; all three survive together on the CPU fixture. Bars are "
                "the head's again -- conv and cache are both bit-identical -- so a miss is "
                "an install that did not survive, and the layer-1 checks name which one."
            ),
        ),
        Hypothesis(
            slug="041-wide-tile-and-cache",
            kernels=("tiled_int4_head_wide", "static_decode_cache"),
            category="B",
            byte_share=0.1480,
            replaces=("decode_step", "decode_cache"),
            correctness="approximate",
            correctness_positions=264,
            top1_threshold=240 / 264,
            kl_threshold=0.06,
            weight_bits={"head": 4},
            requires=Precondition(
                slug="037-int4-head-wide-tile",
                floor=1.08,
                reason=(
                    "this is 036 with the wide tile substituted for the champion's, so it "
                    "is only worth a slot if that tile beat the champion's; below 1.08 the "
                    "batch already holds the better version of this measurement in 036 and "
                    "the tile question has been answered by 037 on its own"
                ),
            ),
            mechanism=(
                "037's pinned tile composed with the static decode cache: the best "
                "measured head tile in the batch, plus the dispatch saving that shares "
                "nothing with it."
            ),
            prediction="win",
            rationale=(
                "The point of 037 is to find a better tile; the point of this slot is to "
                "**carry it into the composition that is otherwise the batch's champion**. "
                "If 037 reaches 1.10, this lands near **1.116** on the same arithmetic 036 "
                "uses, and predicted **1.10-1.15**. It is gated rather than unconditional "
                "because its whole content above 036 is the tile: with 037 below 1.08 this "
                "slot is a worse 036 and the three minutes are better spent on 042. Bars "
                "are the head's, for the third time and for the same reason -- neither "
                "ingredient changes an arithmetic operation."
            ),
        ),
        Hypothesis(
            slug="042-wide-tile-conv-cache",
            kernels=("tiled_int4_head_wide", "fused_causal_conv", "static_decode_cache"),
            category="B",
            byte_share=0.1486,
            replaces=("decode_step", "causal_conv", "decode_cache"),
            correctness="approximate",
            correctness_positions=264,
            top1_threshold=240 / 264,
            kl_threshold=0.06,
            weight_bits={"head": 4},
            requires=Precondition(
                slug="041-wide-tile-and-cache",
                floor=1.10,
                reason=(
                    "everything this slot adds to 041 is the conv, worth 0.093 ms of 6.70, "
                    "and everything it adds to 040 is the tile; below 1.10 on 041 neither "
                    "addition can carry the composition past what the batch has already "
                    "measured, and a declined 041 -- which is what a 037 below 1.08 gives "
                    "-- means the champion's tile stands and 040 is already this slot"
                ),
            ),
            mechanism=(
                "Everything in this batch that composes: the head at 037's tile, the fused "
                "causal conv, and the static decode cache. 97% of the step untouched, "
                "three disjoint mechanisms on the rest."
            ),
            prediction="win",
            rationale=(
                "**The batch's arithmetic maximum and its riskiest slot, so it runs last "
                "with everything cheaper already on disk.** It is 040 with the tile "
                "swapped, or 041 with the conv added, and both of those are measured by "
                "the time it runs -- so whatever it returns is readable as a difference "
                "rather than as a number on its own. Predicted **1.12-1.17** if 037 gave a "
                "faster tile, which is 040's 1.113 plus whatever 037 beat 035 by. The "
                "question it asks that nothing else here does: **the conv's win is a "
                "dispatch win and the static cache is also a dispatch win**, so if 042 "
                "minus 041 comes out below the conv's measured 0.093 ms the two are "
                "competing for the same microseconds, and the launch account this project "
                "has been keeping since rental 38 is not additive. Either reading is a "
                "finding and it costs one slot, because every ingredient is measured "
                "separately above it."
            ),
        ),
    ),
)


BATCHES: dict[str, Batch] = {
    BATCH_001.batch_id: BATCH_001,
    BATCH_002.batch_id: BATCH_002,
    BATCH_003.batch_id: BATCH_003,
    BATCH_004.batch_id: BATCH_004,
    BATCH_005.batch_id: BATCH_005,
    BATCH_006.batch_id: BATCH_006,
    BATCH_007.batch_id: BATCH_007,
}


def get_batch(batch_id: str) -> Batch:
    try:
        return BATCHES[batch_id]
    except KeyError:
        raise SystemExit(f"unknown batch {batch_id!r}; known batches: {sorted(BATCHES)}") from None
