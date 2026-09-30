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
    "BATCH_008",
    "BATCH_009",
    "BATCH_010",
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


BATCH_008 = Batch(
    batch_id="008-ingredients-and-barriers",
    description=(
        "Rental 43 ended with two facts and one admitted mistake, and this batch is built "
        "out of all three. **The card moved every measured effect to zero** -- two RTX "
        "5090s on the same memory clock, driver and torch ran the reference 1.61x apart, "
        "and on the slow one the champion, the static cache and the composition all "
        "measured nothing. **The conv regression was traced to the custom op rather than "
        "to the conv**: an opaque op is a fusion barrier, so inductor materialised 190 "
        "buffers where it had allocated 59 and recomputed a producer chain 24 times per "
        "token. And the batch **composed a mechanism it had not re-measured**, so its "
        "0.8111 cannot be read at all. "
        "So: every ingredient runs alone, today, before anything composes it -- the head, "
        "the conv, the cache, each on this card in this process. Then the experiment that "
        "follows from the barrier: `045` computes the same four taps as torch operations "
        "rather than as one opaque call, and `050` against `051` prices what the opacity "
        "costs when it is composed with the champion. The head's last untried tile "
        "direction and its two remaining encodings fill the cheap slots. "
        "Eleven slots. Two predict a loss and one predicts nothing decisive, which is the "
        "point: the barrier claim is falsifiable and the tile claim has a registered "
        "consequence either way."
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
                "batch is void. It has measured 1.0009, 1.0018, 1.0024, 0.9913, 1.0008, "
                "1.0053 and **1.0101** on the seven rentals that got this far, and the "
                "last of those is the largest offset and by far the widest band recorded "
                "-- on the rental where nothing else measured anything. **Read its sign "
                "and its IQR before reading any other slot**: on rental 43 the identity "
                "candidate removed 0.102 ms/token by installing nothing, which is more "
                "than the champion removed. "
                "This slot also now runs the card pre-flight: `card_baseline.card_report` "
                "compares the `compiled` column's achieved bandwidth here against every "
                "rental that has measured this GPU model, and says loudly if the card is "
                "an outlier. Rental 43's 845 GB/s against rental 40's 1283 would have been "
                "on the log at minute 27 instead of in the writeup. It reports and never "
                "decides: a slow card still produces valid within-slot ratios."
            ),
        ),
        Hypothesis(
            slug="043-int4-head",
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
                "The champion exactly as rentals 40 and 43 ran it: group-128 int4 on the "
                "tied LM head alone, at the heuristic tile -- BLOCK_N 64, BLOCK_K 64, "
                "SPLIT_K 1, 4 warps, 3 stages -- with the install-time tuner off. 1271.40 "
                "MB/token of bf16 weight becomes 317.85 MB of packed nibbles plus 19.87 MB "
                "of fp32 group scales, in one matmul that launches 3880 programs."
            ),
            prediction="win",
            rationale=(
                "**The incumbent, re-measured for the third time, and the reference every "
                "other slot in this batch is read against.** It is not a new hypothesis "
                "and it is not optional: `AGENT.md` section 6 forbids carrying a number "
                "across sessions, and rental 43 showed why in the strongest possible form "
                "-- the same kernel returned 1.0791 and 1.0161 on two cards, and on the "
                "second it removed 0.097 ms/token against an identity slot that removed "
                "0.102. Predicted **win, 1.02-1.09**, and the width of that range is the "
                "card rather than the kernel: the 1.1249x ceiling is fixed arithmetic and "
                "what a card collects against it is not. **The reading that matters is not "
                "the ratio but the ratio minus identity's.** If the head again saves "
                "nothing net of a slot that installs nothing, two of three cards say this "
                "champion's margin is a property of rental 40, and `LEADERBOARD.md` has to "
                "say so in the champion block rather than in a footnote. Bars are 022's "
                "own, met to the digit on both rentals (0.9318, 0.01674 nats); a third "
                "identical correctness result is also the evidence that the install is the "
                "same install."
            ),
        ),
        Hypothesis(
            slug="044-fused-causal-conv",
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
                "025 unchanged: the `cat`, the cuDNN `extern_kernels.convolution`, the "
                "`silu` and the `copy_` that advances the history, collapsed into one "
                "Triton custom op per linear-attention layer. 72 of the step's 508 "
                "launches become 24."
            ),
            prediction="win",
            rationale=(
                "**The ingredient batch 007 composed without measuring, which is the "
                "mistake that batch's writeup names as its own.** `025` won at 1.0144 on "
                "rental 40 and has not run since; `039` then composed it on rental 43 and "
                "returned 0.8111, on a card where every other mechanism measured zero. "
                "Those two facts cannot be separated without this slot, and it costs three "
                "minutes. Predicted **win, 1.005-1.02**: the saving is 48 of 508 launches "
                "plus the cuDNN dispatch premium, which is 0.08-0.24 ms of a 7.30 ms step "
                "-- the same arithmetic 025 was registered against, and the low end of it "
                "is inside the noise band. **On a slow card it shrinks**, because a fixed "
                "dispatch saving is a smaller fraction of a longer step, so an "
                "`inconclusive` here is not a refutation and the writeup must not read it "
                "as one. It is also the control for 045: same arithmetic, same bars, one "
                "opaque call against a fusible expression."
            ),
        ),
        Hypothesis(
            slug="045-inline-causal-conv",
            kernels=("inline_causal_conv",),
            category="A",
            byte_share=0.00055,
            replaces=("causal_conv",),
            correctness="approximate",
            correctness_positions=264,
            top1_threshold=260 / 264,
            kl_threshold=0.001,
            weight_bits={},
            mechanism=(
                "The same four taps at seq_len 1, written as torch operations -- four "
                "multiplies, a round to bf16, a silu, and a shifted history -- instead of "
                "`F.conv1d`. The expression is pointwise, so inductor may fuse it into the "
                "kernels it was stranded between, and **there is no opaque op for it to "
                "fence against**."
            ),
            prediction="win",
            rationale=(
                "**The experiment that decides whether entry 8's new law is the whole "
                "story.** Rental 43's dump of `039` showed the fused conv doing exactly "
                "what it promised -- all 24 cuDNN calls gone, `triton_poi_*` 89 to 41, the "
                "`cat` replaced, 24 fewer launches than the baseline -- while the step ran "
                "**19% slower**, because `torch.ops.deltaforge.fused_causal_conv_step` is "
                "a fusion barrier: 59 allocations became 190 and the linear-attention "
                "state reduction over (1, 32, 128, 128) was recomputed twice per layer. "
                "This slot removes the same cuDNN dispatch and the same `cat` **without "
                "erecting the barrier**, which is the one variable separating it from 044. "
                "Predicted **win, 1.005-1.03**, the same ceiling as 044 because it is the "
                "same saving; what differs is the bill. Three readings, all findings: "
                "**045 > 044** prices the opacity directly and makes the law quantitative; "
                "**045 = 044** says the barrier costs nothing when nothing else is "
                "installed and moves the whole effect into the composition, which 050 "
                "against 051 then measures; **045 < 044** says inductor's own pointwise "
                "schedule is worse here than one hand-written kernel, which would be the "
                "first evidence in this project that a Triton body beats inductor on a "
                "pointwise operation. Bars are 044's exactly, and this is the only "
                "candidate in the repository whose numerics the CPU suite verifies rather "
                "than assumes -- `inline_causal_conv_test` asserts bit-identity with "
                "`GatedDeltaNet._causal_conv` at rtol=0, atol=0 in bf16."
            ),
        ),
        Hypothesis(
            slug="046-static-decode-cache",
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
                "`mark_static_address` on all 64 decode-cache tensors. Inductor reads "
                "`static_input_idxs` twice: once for the cudagraph mutation check, which "
                "has refused twice, and once in the launch path, where a static input "
                "skips the per-call alignment test and the copy behind it -- 64 tensors on "
                "each of 128 decode steps."
            ),
            prediction="inconclusive",
            rationale=(
                "**Two cards, two answers, and this is the tiebreak.** `034` won at 1.0196 "
                "(IQR 0.00149) on rental 42 with the identity slot at 1.0053, so about "
                "1.4% net; on rental 43 the same mechanism inside `036` contributed 722 "
                "GB/s against the head's 717 -- zero, inside a 0.0133 IQR. The prediction "
                "is **inconclusive** and it is a real prediction rather than a hedge: a "
                "dispatch saving denominated in microseconds of CPU work should be *more* "
                "visible on a slower card, not less, and the fact that it was not is the "
                "one rental-43 result the card explanation does not obviously cover. "
                "Predicted **0.995-1.02**, straddling the band on purpose. It also carries "
                "the batch's only cudagraph counters: entry 8's actual hypothesis is 0 for "
                "2 with `cudagraph_nodes: 0` both times, and a third zero on torch 2.11 "
                "closes the version question that a laptop running 2.14 has already half "
                "answered. **A win here is a win for the alignment check, not for CUDA "
                "graphs**, and the record has to keep saying which."
            ),
        ),
        Hypothesis(
            slug="047-int4-head-narrow-tile",
            kernels=("tiled_int4_head_narrow",),
            category="B",
            byte_share=0.1480,
            replaces=("decode_step",),
            correctness="approximate",
            correctness_positions=264,
            top1_threshold=240 / 264,
            kl_threshold=0.06,
            weight_bits={"head": 4},
            mechanism=(
                "The champion's kernel, site and arithmetic at BLOCK_N=32 instead of 64: "
                "7760 programs instead of 3880, and 64 contiguous packed bytes per row "
                "read instead of 128."
            ),
            prediction="loss",
            rationale=(
                "**The last tile direction this site has, registered with a consequence "
                "for either answer.** Four points have now been measured in the decode "
                "step -- heuristic BLOCK_N 64 at 1.0791/1.0161, searched 256 at 0.9920, "
                "pinned 128 at 0.9343, pinned 64-with-5-stages at 0.9502 -- and rental 43 "
                "killed both of the ranked mechanisms: register pressure (037 held "
                "per-thread pressure constant and still lost 0.83 ms/token) and latency "
                "(038 deepened the pipeline, the axis that pays for a latency-bound loop, "
                "and lost 0.63). What survives is the wave-count reading, that the head is "
                "fast because it has programs. **Predicted `loss`**, at 0.97-1.01, because "
                "the champion already runs 23 waves on 170 SMs -- far past the point where "
                "more programs buy occupancy -- while halving the contiguous run per row "
                "read costs coalescing that is not free at 656 GB/s. The prediction is "
                "worth a slot precisely because of what it closes: if this loses, five "
                "measured points disagree with the heuristic in both directions, **the "
                "tile is not what holds this site below the baseline's byte rate**, and "
                "`docs/HYPOTHESES.md` entry 9 stops spending slots on tiles and starts "
                "spending them on a published int4 kernel as an unscored column. If it "
                "wins, the wave-count theory is the survivor and BLOCK_N=16 is next. Bars "
                "are 022's and must be met exactly: the geometry changes, the summation "
                "order does not, so a different correctness number here is a tiling bug."
            ),
        ),
        Hypothesis(
            slug="048-int8-head",
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
                "The same site at 8 bits, per-channel scale: 635.7 MB/token instead of "
                "1271.40, with one convert per byte instead of int4's nibble unpack. Ties "
                "the baseline at 588 GB/s."
            ),
            prediction="win",
            rationale=(
                "**Registered on rental 40, defective there, fixed, and never measured "
                "since -- so it is new code that has never run** (`AGENT.md` section 8 on "
                "declined and defective slots). `023` returned 1.0373 with a layer-1 "
                "relative error of 4511, because `_tiled_gemv_scaled_kernel` declared "
                "`SCALE` and `HAS_SCALE` and read neither. The missing operation is one FMA "
                "in an epilogue, so the *timing* was usable and the correctness was not; "
                "this slot makes both usable at once. Predicted **win, 1.01-1.05**: it "
                "moves 635.7 MB where the baseline moves 1271.40, and it achieved 847 GB/s "
                "at this site against int4's 656 -- a higher byte rate on twice the bytes, "
                "which is why it should win by less. **The ordering is the finding, not "
                "the ratio.** int4 ahead of int8 says the kernel is bandwidth-bound here "
                "and the nibble unpack is paid for; int8 ahead of int4 says it is still "
                "issue-bound, which is what batch 003 measured on the crowded sites and "
                "what rental 40 refuted on this one. Bars are 013 minus 012's measured "
                "point -- about 0.0001 nats and roughly one flip of 264 -- widened to "
                "256/264 and 0.003 for the per-channel scale."
            ),
        ),
        Hypothesis(
            slug="049-fp8-head",
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
                "The same site and the same bit width as 048, stored e4m3. On sm_120 the "
                "conversion happens inside the MMA pipeline rather than as an ALU "
                "instruction on the critical path, which is the tax batch 003 measured at "
                "**1.438x** for int8 against bf16 in the same kernel."
            ),
            prediction="win",
            rationale=(
                "**The third encoding at the one site where nothing else is binding, and "
                "the only one of the three that has never compiled.** `024` died on "
                "rental 40 because the kernel's masked load used `other=0`, which will not "
                "cast to e4m3; that is fixed and `kernel_contract_test` now refuses the "
                "shape of it on a CPU, but the fixed path has still never executed on a "
                "card, so it runs after 048 and carries the batch's real error risk. "
                "Predicted **win, 1.01-1.06**, bracketing 048: same bytes, and the whole "
                "difference between the two slots is the conversion tax, isolated at a "
                "site where the kernel is not grid-starved. **048 against 049 is the "
                "measurement**; either one against the baseline is a byte saving anyone "
                "would expect. Bars are looser than 048's -- e4m3 carries 3 mantissa bits "
                "against int8's effective 7 -- and derived from batch 003's two measured "
                "points (int8 0.0011 nats and 8/264 flips, int4 0.0919 and 38/264), not "
                "from priors about fp8, because priors are what failed four working slots "
                "in batch 003."
            ),
        ),
        Hypothesis(
            slug="050-int4-head-and-conv",
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
                "The champion plus the Triton custom-op conv: 953.55 MB/token removed from "
                "one matmul, and 48 of the step's 508 launches removed from 24 layers -- "
                "at the cost of an opaque op inductor cannot fuse across."
            ),
            prediction="loss",
            rationale=(
                "**The third run of the largest unexplained number in this project, and "
                "the first one whose ingredients are both measured in the same process.** "
                "It returned 0.7937 on rental 42 at the tuned tile and 0.8111 on rental 43 "
                "at the champion's, so the tile is not the cause; the dump named a "
                "structural one. **Predicted `loss`, 0.78-0.85**, and the prediction is "
                "the claim: if the fusion barrier is the mechanism, this regression is "
                "structural and therefore card-independent, and a third measurement near "
                "0.81 on a third card is what confirms that. A result near 1.09 -- the "
                "naive sum of two measured savings -- would refute the barrier reading "
                "outright and mean both prior numbers belonged to something neither dump "
                "nor arithmetic has found. **This slot is not gated and runs before 051 on "
                "purpose**: it is the control for the batch's payoff slot, and a "
                "composition whose ingredients and whose alternative are all measured in "
                "the same process is the thing batch 007 could not produce."
            ),
        ),
        Hypothesis(
            slug="051-int4-head-and-inline-conv",
            kernels=("tiled_int4_head", "inline_causal_conv"),
            category="B",
            byte_share=0.1486,
            replaces=("decode_step", "causal_conv"),
            correctness="approximate",
            correctness_positions=264,
            top1_threshold=240 / 264,
            kl_threshold=0.06,
            weight_bits={"head": 4},
            mechanism=(
                "The champion plus the same four taps expressed as fusible torch "
                "operations: the identical pair of savings as 050, with nothing in the "
                "graph inductor has to fence against."
            ),
            prediction="win",
            rationale=(
                "**The batch's payoff slot, and it is one variable away from 050.** Same "
                "site, same bytes, same launches removed, same bars -- the only difference "
                "is whether the conv arrives as an opaque custom op or as an expression "
                "inductor can schedule. Predicted **win, 1.03-1.10**: the champion's "
                "measured saving plus the conv's, with none of the 131 extra allocations "
                "and none of the 24 recomputed state reductions that the dump found in "
                "050's graph. **051 minus 050 is the price of opacity**, measured rather "
                "than argued, on a pair where every other variable is held fixed -- and it "
                "is the number entry 8's law is currently missing, since 24 extra "
                "reductions and 131 allocator calls do not add up to the +2.68 ms/token "
                "that was measured. If 051 also lands near 0.81 the barrier explanation is "
                "wrong and the conv simply does not compose with the head, which is a "
                "finding this batch can state because 043, 044 and 045 are all on disk by "
                "the time it runs. It is deliberately **not gated on 045**: a floor keyed "
                "to an absolute ratio would decline the batch's most informative slot on "
                "exactly the slow card that makes every absolute meaningless, which is how "
                "rental 43 lost three of its nine."
            ),
        ),
        Hypothesis(
            slug="052-int4-head-inline-conv-cache",
            kernels=("tiled_int4_head", "inline_causal_conv", "static_decode_cache"),
            category="B",
            byte_share=0.1486,
            replaces=("decode_step", "causal_conv", "decode_cache"),
            correctness="approximate",
            correctness_positions=264,
            top1_threshold=240 / 264,
            kl_threshold=0.06,
            weight_bits={"head": 4},
            requires=Precondition(
                slug="051-int4-head-and-inline-conv",
                floor=1.00,
                reason=(
                    "this slot is 051 plus a mechanism 046 has already measured alone, so "
                    "with 051 below 1.00 there is nothing here the batch does not hold: it "
                    "would add a third install on top of a composition that lost, which is "
                    "what batch 003 spent five slots doing. The floor is 1.00 rather than a "
                    "champion-beating bar because on a slow card every absolute collapses "
                    "toward it, and a floor that declines a slot for the card's reasons "
                    "rather than the hypothesis's is the failure mode rental 43 recorded"
                ),
            ),
            mechanism=(
                "Three disjoint mechanisms at once: bytes inside one matmul, launches "
                "inside 24 layers, and dispatch work outside every kernel -- with the conv "
                "in the form that does not cost its neighbours."
            ),
            prediction="win",
            rationale=(
                "**The batch's arithmetic maximum, and the first three-way composition "
                "this project could actually read.** Every one of its three pairs is "
                "measured above it in the same process -- 043, 046, 051 alone, and 051 is "
                "the head-and-conv pair -- so whatever it returns is a difference rather "
                "than a number: a shortfall against 051 times 046 divided by 043 is the "
                "interaction, isolated, with both halves on disk to subtract. Predicted "
                "**win, 1.04-1.12**. The question it asks that nothing else here does: the "
                "conv's saving is a dispatch saving and the static cache's is too, so if "
                "052 minus 051 comes out below what 046 measured alone, **the two are "
                "competing for the same microseconds** and the launch account this project "
                "has kept since rental 38 is not additive. Either reading is a finding. It "
                "runs last, with everything cheaper already on disk, and all three "
                "installers are asserted to survive together on the CPU fixture -- two "
                "replace the root class, keyed by base since batch 005, and the third "
                "patches `GatedDeltaNet`."
            ),
        ),
    ),
)


BATCH_009 = Batch(
    batch_id="009-visible-kernels",
    description=(
        "Rental 45 priced opacity and the number was enormous. `044-fused-causal-conv` and "
        "`045-inline-causal-conv` compute **the same function** -- both bit-identical to "
        "the reference -- and returned **0.7854 and 1.0765**, differing only in whether "
        "the arithmetic reached inductor as a `torch.library.custom_op` it must fence "
        "against or as operations it could schedule. "
        "**`tiled_gemv_int4`, the champion of `decode_step`, is a custom op too.** The dump "
        "shows the reference fusing the final RMSNorm *into* the lm_head matmul where our "
        "candidate cannot, so the head has real fusion to lose and nothing has measured "
        "what losing it costs. This batch runs the same experiment one level up, as three "
        "registrations of one program: the champion unchanged, the identical Triton kernel "
        "behind `torch.library.triton_op` so inductor can see it, and **no kernel at all** "
        "-- the dequantise-GEMV written in torch and handed whole to `max-autotune`. "
        "That third slot asks the question this project has never answered: is the "
        "hand-written kernel **necessary** here, or merely sufficient? Its gated sequel "
        "takes the same answer to the MLP, which is 52.75% of per-token bytes. "
        "Every ingredient runs alone before anything composes it, and the batch runs "
        "**15 scoring rounds instead of 5** -- rental 45 lost six slots to a band it could "
        "have narrowed for twenty seconds each."
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
                "batch is void. It has measured 1.0009, 1.0018, 1.0024, 0.9913, 1.0008, "
                "1.0053, 1.0101 and **0.9972** on the eight rentals that got this far. "
                "**Two things to read here before anything else.** Its *sign*: rental 43 "
                "carried +1.01% and rental 45 carried −0.28%, and a slot's margin means "
                "nothing until that is subtracted. And its *IQR*, which this batch is "
                "trying to shrink: 15 scoring rounds instead of 5, because rental 45's "
                "identity band was 0.0193 and six slots landed `inconclusive` inside "
                "bands of 0.019-0.151. If the identity IQR does not fall below ~0.01 "
                "here, more rounds are not the instrument and the next batch needs a "
                "different one. The card pre-flight also runs off this slot."
            ),
        ),
        Hypothesis(
            slug="053-inline-causal-conv",
            kernels=("inline_causal_conv",),
            category="A",
            byte_share=0.00055,
            replaces=("causal_conv",),
            correctness="approximate",
            correctness_positions=264,
            top1_threshold=264 / 264,
            kl_threshold=1e-06,
            weight_bits={},
            mechanism=(
                "The champion: the four-tap depthwise causal conv at seq_len 1 written as "
                "torch operations -- four multiplies, a round to bf16, a silu and a "
                "shifted history -- instead of `F.conv1d`. Inductor folds it into the "
                "kernels either side and all 24 cuDNN calls disappear."
            ),
            prediction="win",
            rationale=(
                "**The incumbent, re-measured, and the slot everything behind it is read "
                "against.** `AGENT.md` section 6 forbids carrying a number across "
                "sessions, and rental 45 is the rental that showed what happens when a "
                "composition's ingredient is not re-measured: three rentals attributed a "
                "20% loss to an interaction that did not exist. Predicted **win, "
                "1.03-1.09**. Rental 45 measured 1.0765 with an IQR of 0.0552 on a card "
                "that downclocked mid-batch; the same 15-round protocol that shrinks the "
                "identity band should shrink this one, and the honest expectation is that "
                "the **median moves less than the band does**. Below 1.02 would say the "
                "1.0765 had the card in it, which is exactly the claim rental 45 could "
                "not test on its own. Bars are **tightened from 045's**: it measured "
                "264/264 and 0.00000 nats, and `inline_causal_conv_test` asserts "
                "bit-identity with the reference at rtol=0, atol=0 on a CPU, so anything "
                "other than exact agreement here is a bug rather than a dtype."
            ),
        ),
        Hypothesis(
            slug="054-int4-head",
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
                "The champion of `decode_step`, unchanged: group-128 int4 on the tied LM "
                "head, heuristic tile, tuner off, installed through "
                "`torch.library.custom_op`. 1271.40 MB/token of bf16 weight becomes 317.85 "
                "of packed nibbles plus 19.87 of fp32 group scales."
            ),
            prediction="win",
            rationale=(
                "**The control for the two slots behind it, and the fourth measurement of "
                "a kernel whose headline number nobody can reproduce.** It returned "
                "1.0791 (rental 40), 1.0161 (43) and **1.0105 at an IQR of 0.0263** (45) "
                "-- the last on a healthy 1197 GB/s card with the identity slot at "
                "−0.28%, so net of calibration it saved about 1.3% and could not be "
                "resolved from zero. Predicted **win, 1.00-1.03**, and the point of the "
                "slot is **resolution rather than magnitude**: at 15 scoring rounds a real "
                "1.3% should clear its own band for the first time. Below 1.00 is the "
                "interesting outcome and would mean the 1.1249x byte ceiling is not being "
                "collected at all on modern cards. Bars are 022's own, met to the digit on "
                "three rentals (0.9318, 0.01674 nats) -- and **055 and 056 must reproduce "
                "them exactly**, because all three compute the same function and differ "
                "only in who writes the code."
            ),
        ),
        Hypothesis(
            slug="055-int4-head-triton-op",
            kernels=("int4_head_triton_op",),
            category="A",
            byte_share=0.1480,
            replaces=("decode_step",),
            correctness="approximate",
            correctness_positions=264,
            top1_threshold=240 / 264,
            kl_threshold=0.06,
            weight_bits={"head": 4},
            mechanism=(
                "The champion's Triton kernel, byte for byte, registered through "
                "`torch.library.triton_op` instead of `custom_op`: the launch enters the "
                "graph as a structured node whose inputs, outputs and mutation semantics "
                "inductor knows, rather than an opaque call it must fence against. Same "
                "tile, same arithmetic, same summation order, same class swap, same "
                "quantised weights."
            ),
            prediction="win",
            rationale=(
                "**The experiment rental 45 pointed at, one level up, with one variable.** "
                "The conv pair measured the price of opacity at **37%** on a mechanism "
                "whose kernel was never in doubt; this asks the same of the head, where "
                "the dump shows the reference fusing the final RMSNorm *into* the lm_head "
                "matmul (`triton_red_fused_..._mm_..._slice_t_view_43`) and the candidate "
                "unable to. Predicted **win, 1.02-1.06 — above 054, and by less than the "
                "conv's margin**, and the mechanism says why: the conv sat inside 24 "
                "layers between a fused producer and a fused consumer and was worth 24 "
                "call sites, where the head is **one** call site at the end of the model "
                "with one reduction beside it. Three readings, all findings. **055 > 054** "
                "prices the barrier at this site and makes the law quantitative twice. "
                "**055 = 054** says the barrier costs nothing where there is little to "
                "fuse across, which bounds the law usefully and is the outcome that stops "
                "the next batch rewriting every kernel. **055 < 054** would say "
                "`triton_op` carries its own overhead, which nothing here predicts. Bars "
                "are 054's and must be met exactly: the body is identical, so a different "
                "correctness number is a registration bug, and "
                "`visible_int4_head_test` already asserts on a CPU that the op is really "
                "registered rather than silently falling back to the bare function."
            ),
        ),
        Hypothesis(
            slug="056-int4-head-torch-dequant",
            kernels=("int4_head_torch_dequant",),
            category="A",
            byte_share=0.1480,
            replaces=("decode_step",),
            correctness="approximate",
            correctness_positions=264,
            top1_threshold=240 / 264,
            kl_threshold=0.06,
            weight_bits={"head": 4},
            mechanism=(
                "The same program with no kernel of ours in it at all: unpack the nibbles, "
                "apply the group scales, round the scaled weight to bf16 because the "
                "kernel does, accumulate in fp32 -- the expression "
                "`tiled_int4_correctness_checks` already uses as the champion's reference "
                "-- handed whole to `max-autotune`."
            ),
            prediction="loss",
            rationale=(
                "**The question this project has never answered: is the hand-written "
                "kernel necessary here, or merely sufficient?** `045` won by deleting a "
                "kernel, and the honest consequence is to ask the same of the one we are "
                "proudest of. Predicted **`loss`, 0.55-0.85**, on a bandwidth budget "
                "rather than a hunch: for this to win, inductor must fuse a *grouped* "
                "dequantisation into the prologue of a 248320-wide GEMV. If it instead "
                "materialises the bf16 weight -- 1271.40 MB/token, against a 96 MB L2 that "
                "cannot hold it -- the candidate moves that **plus** the 317.85 MB of "
                "packed nibbles it read to build it, and it cannot reach 1.0 however good "
                "the matmul is. **There is no middle outcome that is hard to read.** "
                "Entry 1's one datum for this shape is `010-int8-dequant-torch` at 0.9893 "
                "on rental 37, and it is not this experiment: that was int8 on all 248 "
                "layer projections, which batch 005 showed cannot separate a kernel from a "
                "site, and its bandwidth budget proved inductor did *not* materialise "
                "there. **A win here would be the largest result this project could "
                "produce and the most uncomfortable** -- it would mean the compiler "
                "collects the 1.1249x by itself and the champion's kernel earns nothing -- "
                "which is exactly why it is registered as a predicted loss before the "
                "rental. Bars are 054's, and the layer-2 numbers must come back identical "
                "(0.9318, 0.01674) because it is the same function; "
                "`visible_int4_head_test` asserts that equality on a CPU at rtol=0, atol=0."
            ),
        ),
        Hypothesis(
            slug="057-static-decode-cache",
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
                "`mark_static_address` on all 64 decode-cache tensors. Inductor reads "
                "`static_input_idxs` twice: once for the cudagraph mutation check, which "
                "has refused three times, and once in the launch path, where a static "
                "input skips the per-call alignment test and the copy behind it -- 64 "
                "tensors on each of 128 decode steps."
            ),
            prediction="win",
            rationale=(
                "**Two measurements agree and neither could resolve itself, which is what "
                "15 rounds are for.** It returned **1.0196 (IQR 0.00149)** on rental 42 "
                "and **1.0214 (IQR 0.0406, `inconclusive`)** on rental 45 -- the same "
                "number twice, once resolvable and once not. Predicted **win, 1.01-1.03**. "
                "This is the cheapest slot in the batch and it is here as an ingredient: "
                "`060` composes it, and rental 45's lesson is that a composition whose "
                "parts were not measured the same day is unreadable. It also carries the "
                "batch's cudagraph counters. **Entry 8's actual hypothesis is 0 for 3** -- "
                "`cudagraph_nodes: 0` every time, on torch 2.11 -- and a fourth zero "
                "settles that the mutation check is not reachable on this build, which is "
                "a whole-rental decision (bump torch) rather than a slot. **A win here is "
                "a win for the alignment check, not for CUDA graphs**, and the record has "
                "to keep saying which."
            ),
        ),
        Hypothesis(
            slug="058-conv-and-head",
            kernels=("inline_causal_conv", "tiled_int4_head"),
            category="B",
            byte_share=0.1486,
            replaces=("causal_conv", "decode_step"),
            correctness="approximate",
            correctness_positions=264,
            top1_threshold=240 / 264,
            kl_threshold=0.06,
            weight_bits={"head": 4},
            mechanism=(
                "Both current champions at once: the fusible causal conv and the int4 "
                "head. 953.55 MB/token removed from one matmul, and 24 cuDNN dispatches "
                "removed from 24 layers, on disjoint parts of the step."
            ),
            prediction="win",
            rationale=(
                "**The shipped configuration, measured.** `apply_champions` now installs "
                "exactly this pair, so a batch that never benchmarked it would be shipping "
                "an unmeasured default -- and both ingredients are measured above it in "
                "the same process, which is the rule rental 45 established by breaking it. "
                "Rental 45's `051` measured this pair at **1.0443 ± 0.1511** and its `052` "
                "measured the pair plus the static cache at **1.0747 ± 0.0127**; the "
                "candidate timings agreed and only the reference column differed, so the "
                "pair is probably worth ~1.07 and the batch could not say so. Predicted "
                "**win, 1.04-1.10**. The reading that matters is **058 against 053 and "
                "054 separately**: if it falls short of their product the two are "
                "competing, and rental 45 could not test that because `051`'s band "
                "swallowed the difference."
            ),
        ),
        Hypothesis(
            slug="059-conv-and-head-triton-op",
            kernels=("inline_causal_conv", "int4_head_triton_op"),
            category="B",
            byte_share=0.1486,
            replaces=("causal_conv", "decode_step"),
            correctness="approximate",
            correctness_positions=264,
            top1_threshold=240 / 264,
            kl_threshold=0.06,
            weight_bits={"head": 4},
            requires=Precondition(
                slug="055-int4-head-triton-op",
                floor=1.00,
                reason=(
                    "this slot is 058 with the visible head substituted for the opaque "
                    "one, so its entire content above 058 is whatever 055 showed the "
                    "registration is worth. With 055 below 1.00 the substitution is a "
                    "worse 058 and the batch already holds the better version; the floor "
                    "is 1.00 rather than a champion-beating bar because on a drifting card "
                    "every absolute collapses toward it, and declining a slot for the "
                    "card's reasons rather than the hypothesis's is rental 43's failure"
                ),
            ),
            mechanism=(
                "058 with the head registered through `triton_op`: the same two mechanisms "
                "with nothing in the graph inductor has to fence against."
            ),
            prediction="win",
            rationale=(
                "**If the barrier costs anything at the head, this is where the project "
                "banks it**, and it is one variable from 058 exactly as 055 is one "
                "variable from 054. Predicted **win, 1.06-1.13**: 058's measurement plus "
                "whatever 055 beat 054 by. The question it asks that 055 cannot: the head "
                "and the conv are adjacent in neither the graph nor the model, but both "
                "now let inductor schedule across them, so **059 minus 058 should equal "
                "055 minus 054** if the two barriers are independent. A shortfall means "
                "they are not, and that is a finding about how fusion regions compose that "
                "no launch census could produce. Bars are 054's; nothing here changes an "
                "arithmetic operation."
            ),
        ),
        Hypothesis(
            slug="060-conv-head-cache",
            kernels=("inline_causal_conv", "tiled_int4_head", "static_decode_cache"),
            category="B",
            byte_share=0.1486,
            replaces=("causal_conv", "decode_step", "decode_cache"),
            correctness="approximate",
            correctness_positions=264,
            top1_threshold=240 / 264,
            kl_threshold=0.06,
            weight_bits={"head": 4},
            requires=Precondition(
                slug="058-conv-and-head",
                floor=1.00,
                reason=(
                    "this is 058 plus a mechanism 057 has measured alone, so with 058 "
                    "below 1.00 there is nothing here the batch does not hold: it would "
                    "add a third install on top of a composition that lost, which is what "
                    "batch 003 spent five slots doing"
                ),
            ),
            mechanism=(
                "Three disjoint mechanisms: bytes inside one matmul, 24 cuDNN dispatches "
                "removed from 24 layers, and inductor's per-call alignment check removed "
                "from 64 tensors on 128 steps."
            ),
            prediction="win",
            rationale=(
                "**Rental 45's best measured candidate, re-measured with all three of its "
                "ingredients on disk the same day for the first time.** `052` returned "
                "**1.0747 at an IQR of 0.0127** -- the tightest win in that batch -- but "
                "its conv and head were measured in slots whose own bands were 0.055 and "
                "0.026, so the decomposition was not readable. Here 053, 054 and 057 all "
                "run alone above it. Predicted **win, 1.05-1.11**. The question: **the "
                "conv's saving is a dispatch saving and the static cache's is too**, so if "
                "060 minus 058 comes out below what 057 measured alone, the two are "
                "competing for the same microseconds and this project's launch accounting "
                "is not additive. Either reading is a finding, and it costs one slot "
                "because every ingredient is measured separately above it."
            ),
        ),
        Hypothesis(
            slug="061-int4-mlp-torch-dequant",
            kernels=("int4_mlp_torch_dequant",),
            category="B",
            byte_share=0.5275,
            replaces=("swiglu_mlp",),
            correctness="approximate",
            correctness_positions=264,
            top1_threshold=220 / 264,
            kl_threshold=0.10,
            weight_bits={"mlp": 4},
            requires=Precondition(
                slug="056-int4-head-torch-dequant",
                floor=1.02,
                reason=(
                    "this is the same question asked of a block 3.6x larger, and it is "
                    "only worth asking once the cheap version has shown that inductor "
                    "fuses a grouped dequantisation into a GEMV prologue at all. Below "
                    "1.02 on the head it demonstrably does not, and this slot would "
                    "re-measure a materialised weight at 96 sites instead of one -- which "
                    "is precisely what batches 003 and 004 spent two rentals doing"
                ),
            ),
            mechanism=(
                "The compiler-generated dequantise-GEMV on the 96 MLP projections: "
                "**52.75% of per-token bytes and a 1.6545x ceiling**, the largest "
                "homogeneous block in the model, at group-128 int4 with no hand-written "
                "kernel anywhere in it."
            ),
            prediction="win",
            rationale=(
                "**The prize, and it is only reachable down the branch 056 opens.** Three "
                "hand-written kernels have failed on these sites at ~66 GB/s across three "
                "rentals, rental 43 refuted register pressure and latency, and rental 45 "
                "closed the tile question at five measured points. What has never been "
                "tried is the possibility that **the right code for these sites is "
                "inductor's own**. Predicted **win, 1.25-1.55** *given the gate*: the MLP "
                "costs 4529.85 MB/token of 8587.80, group-128 int4 moves 0.2578 of that, "
                "and the ceiling is 1.6545x -- if the head slot showed the dequantisation "
                "fuses, the same fusion at 96 wider sites collects proportionally more, "
                "because the MLP's sites are 9216 and 2560 channels wide where the "
                "starved gates are 32. It runs last because it is the batch's riskiest "
                "and largest install, with everything cheaper already on disk. Bars are "
                "`030-int4-mlp`'s, which measured 0.04868 nats against this 0.10 after "
                "predicting ~0.048 -- the one place in this repository where a bar derived "
                "from a measured point has been checked twice."
            ),
        ),
    ),
)


BATCH_010 = Batch(
    batch_id="010-speculative-verify",
    description=(
        "What a k+1-token verify costs, measured with no drafter in it, and then two "
        "drafters that cost nothing. Spec: docs/superpowers/specs/"
        "2026-09-24-speculative-decoding-design.md."
    ),
    hypotheses=(
        Hypothesis(
            slug="000-identity",
            kernels=(),
            category="calibration",
            byte_share=0.0,
            mechanism="Installs nothing; calibrates the harness against the card of the hour.",
            prediction="identity",
            rationale=(
                "The identity champion must return 1.00 +- noise or every other number in "
                "the batch is void. It also reports the reference column's achieved "
                "bandwidth before any candidate runs, which is how rental 43's 800 GB/s "
                "card was recognised as a card rather than as a result."
            ),
        ),
        Hypothesis(
            slug="062-verify-inflation-k4",
            kernels=("rollback_state", "speculative_fixed_k4"),
            category="C",
            byte_share=0.0,
            mechanism=(
                "The speculative loop with a drafter that always proposes the same token, "
                "so acceptance is 0 by construction and the ratio is 1/gamma(4)."
            ),
            prediction="loss",
            rationale=(
                "This slot is an instrument, not a candidate. gamma is the whole downside "
                "of the hypothesis and nothing has ever measured it: +8.2% of bytes for the "
                "five-token KV and state traffic, +2.9% for the state versioning, and an "
                "unknown dispatch term because the linear-attention scan runs five steps "
                "per layer instead of one. Predicted 0.87-0.95. Above 1.25 in 1/ratio terms "
                "and blocks longer than 2 are dead, which is registered as a kill criterion "
                "in the spec rather than decided after the number. The divergence gate's "
                "ceiling is set at the measured bf16 ULP scale (0.28125, oracle_test.py:264) "
                "rather than at a round number: below that, the gate cannot pass a single "
                "divergence this reduction-order effect causes."
            ),
            correctness="sequence",
            divergence_gap_ceiling=0.3,
        ),
        Hypothesis(
            slug="063-verify-inflation-k2",
            kernels=("rollback_state", "speculative_fixed_k2"),
            category="C",
            byte_share=0.0,
            contrast_with="062-verify-inflation-k4",
            mechanism="The same instrument at k=2: how gamma scales with the block.",
            prediction="loss",
            rationale=(
                "One variable from 062, the block size. Traffic says gamma(2) ~ 1.06 "
                "against gamma(4) ~ 1.11, so this should lose about half as much. If it "
                "loses as much or more, gamma is dispatch rather than traffic and the "
                "spec's arithmetic is wrong in a way that matters more than the slot does. "
                "Ceiling set at the measured bf16 ULP scale, as 062's is, not at a round "
                "number."
            ),
            correctness="sequence",
            divergence_gap_ceiling=0.3,
        ),
        Hypothesis(
            slug="064-spec-ngram-k2",
            kernels=("rollback_state", "speculative_ngram_k2"),
            category="C",
            byte_share=0.0,
            contrast_with="063-verify-inflation-k2",
            mechanism=(
                "The same loop at k=2 with a prompt-lookup drafter: d=0, so the whole "
                "downside is gamma-1 and any acceptance above ~10% is a win."
            ),
            prediction="inconclusive",
            rationale=(
                "One variable from 063: the drafter. On the headline workload the prompt "
                "is torch.randint token ids, and an n-gram drafter over noise has no "
                "structure to find -- so the honest prediction here is that it lands within "
                "noise of 063, and the result worth having is the acceptance histogram "
                "rather than the ratio. The text workload is where this is a real question. "
                "Ceiling set at the measured bf16 ULP scale, as 062's is, not at a round "
                "number."
            ),
            correctness="sequence",
            divergence_gap_ceiling=0.3,
        ),
        Hypothesis(
            slug="065-spec-ngram-k4",
            kernels=("rollback_state", "speculative_ngram_k4"),
            category="C",
            byte_share=0.0,
            contrast_with="064-spec-ngram-k2",
            mechanism="The prompt-lookup drafter at k=4.",
            prediction="loss",
            rationale=(
                "A longer block costs more gamma and an n-gram drafter's acceptance decays "
                "fastest with depth, so at k=4 on random ids this should sit below 064. "
                "Registered as a loss so that a win is informative: it would mean "
                "acceptance is holding deeper than the drafter deserves, which is worth "
                "knowing before the int4 self-draft is built. Ceiling set at the measured "
                "bf16 ULP scale, as 062's is, not at a round number."
            ),
            requires=Precondition(
                slug="064-spec-ngram-k2",
                floor=0.0,
                versus="063-verify-inflation-k2",
                reason=(
                    "The n-gram drafter must at least not lose to the same loop with a "
                    "drafter that is wrong on purpose. If it does, drafting is costing more "
                    "than it saves at any block size and k=4 will not rescue it. The floor "
                    "is on the comparison because that is the proposition -- rental 46's "
                    "061 declined on an absolute ratio that meant nothing here."
                ),
            ),
            correctness="sequence",
            divergence_gap_ceiling=0.3,
        ),
    ),
)


BATCH_011 = Batch(
    batch_id="011-bytes-not-kernels",
    description=(
        "**Ten batches have answered the question this project registered, and the answer is "
        "no.** The premise was that a hand-written Triton kernel can beat what "
        "`torch.compile(mode='max-autotune')` generates on this decode path. Every kernel "
        "written here has now been measured against the compiler at the two sites most "
        "favourable to it, and the compiler won both: `045` beat `044` by **37%** on the "
        "causal conv, and `056` beat `054` by **3.2%** on the int4 head -- each time the "
        "*same function*, the win going to the version with no kernel of ours in it. Both "
        "champions are kernels this project deleted. "
        "**So this batch stops contesting codegen and starts contesting the program.** The "
        "compiler chooses instructions; it does not choose how many bytes the weights "
        "occupy, and it does not choose how many forward passes a token costs. Those two "
        "are still ours, and this batch measures both: the grouped int4 dequantise-GEMV "
        "that `056` proved inductor fuses, taken from 14.80% of per-token bytes to "
        "**52.75%** (`071`), **67.55%** (`072`) and **97.85%** (`074`) -- a dose-response "
        "ladder whose every rung is one install on top of a measured one -- and the "
        "**two-token verify** (`069`), which decides whether rental 54's 23% step at the "
        "seq=1 -> seq>1 boundary is real. "
        "It also measures **the pair this repository ships**: `apply_champions` installs "
        "`inline_causal_conv` beside `int4_head_torch_dequant` and no rental has ever "
        "benchmarked that combination -- rental 46's `058` composed the conv with the "
        "kernel that has since been retired. "
        "Every composition sits above its own ingredients in the same process, and every "
        "gate is on a **comparison** rather than an absolute ratio, which is the defect "
        "that cost rental 46 the largest slot in the batch."
    ),
    hypotheses=(
        Hypothesis(
            slug="000-identity",
            kernels=(),
            category="calibration",
            byte_share=0.0,
            mechanism=(
                "Install nothing. The candidate is the reference, so the ratio is the "
                "harness's own noise floor rather than a property of any kernel."
            ),
            prediction="identity",
            rationale=(
                "Must return 1.00 within the noise band or every other number here is "
                "void. Nine rentals have got this far: 1.0009, 1.0018, 1.0024, 0.9913, "
                "1.0008, 1.0053, 1.0101, 0.9972 and -- with 15 scoring rounds instead of "
                "5 -- **1.0002 at an IQR of 0.0001**, the tightest this project has "
                "recorded. Read its *sign* before any margin below it (rental 43 carried "
                "+1.01%, rental 45 −0.28%), and read the reference column's achieved "
                "bandwidth before any of them: 1282, 800, 1197 and 849 GB/s on four cards "
                "reporting the same clocks. **This slot is also the divisor for four gates "
                "in this batch**, which is new: every precondition here is a margin "
                "against this slot rather than an absolute ratio, so a drifting card "
                "cannot decline a hypothesis for reasons that have nothing to do with it."
            ),
        ),
        Hypothesis(
            slug="066-inline-causal-conv",
            kernels=("inline_causal_conv",),
            category="A",
            byte_share=0.00055,
            replaces=("causal_conv",),
            correctness="approximate",
            correctness_positions=264,
            top1_threshold=264 / 264,
            kl_threshold=1e-06,
            weight_bits={},
            mechanism=(
                "The champion of `causal_conv`: the four-tap depthwise convolution at "
                "seq_len 1 written as torch operations -- four multiplies, a round to "
                "bf16, a silu and a shifted history -- instead of `F.conv1d`. Inductor "
                "folds it into the kernels either side and all 24 cuDNN calls disappear."
            ),
            prediction="win",
            rationale=(
                "**An ingredient, measured alone, because three slots below compose it.** "
                "Measured 1.0765 (IQR 0.0552) on rental 45 and **1.0650** on rental 46 at "
                "15 rounds, bit-identical both times (264/264, 0.00000 nats). Predicted "
                "**win, 1.03-1.09**. It attacks 0.055% of the bytes and returns 6%, which "
                "is the one place in this repository where the byte ceiling is not the "
                "binding constraint: the saving is 24 cuDNN dispatches and the fusion "
                "either side of them, not traffic. Bars are `045`'s and are exact by "
                "construction -- `inline_causal_conv_test` asserts bit-identity with the "
                "reference at rtol=0, atol=0 on a CPU, so anything but 264/264 at 0 nats "
                "is a harness fault rather than a kernel property."
            ),
        ),
        Hypothesis(
            slug="067-int4-head-torch-dequant",
            kernels=("int4_head_torch_dequant",),
            category="A",
            byte_share=0.1480,
            replaces=("decode_step",),
            correctness="approximate",
            correctness_positions=264,
            top1_threshold=240 / 264,
            kl_threshold=0.06,
            weight_bits={"head": 4},
            mechanism=(
                "The champion of `decode_step`: the group-128 int4 dequantise-GEMV on the "
                "tied 248320 x 2560 head, written as torch operations -- unpack the "
                "nibbles, apply the group scales, round to bf16, accumulate in fp32 -- and "
                "handed whole to `max-autotune`, which fuses the entire unpack, the `mm`, "
                "the final RMSNorm and the residual add into one reduction kernel."
            ),
            prediction="win",
            rationale=(
                "**The ingredient every quantised slot below is read against, and the only "
                "in-batch evidence that inductor fuses a grouped dequantisation at all.** "
                "Measured **1.0171 (IQR 0.0130)** on rental 46 against a 1.1249x byte "
                "ceiling, with the hand-written kernel at 0.9851 in the same process. "
                "Predicted **win, 1.01-1.04**. Its margin over `000` is the quantity four "
                "gates below read, and that is the whole point of running it first: rental "
                "46 gated the MLP on this slot reaching **1.02 absolute** and declined the "
                "largest prize in the backlog by 0.3%, on a number that moves with how "
                "fast the card is that hour and with what fraction of the step the head "
                "happens to be. Bars are `054`'s and the layer-2 numbers must come back "
                "**0.9318 and 0.01674** -- five rentals have returned exactly that, and a "
                "different number means the weights or the rounding changed, not the card."
            ),
        ),
        Hypothesis(
            slug="068-champion-pair",
            kernels=("inline_causal_conv", "int4_head_torch_dequant"),
            category="B",
            byte_share=0.1486,
            replaces=("causal_conv", "decode_step"),
            contrast_with="067-int4-head-torch-dequant",
            correctness="approximate",
            correctness_positions=264,
            top1_threshold=240 / 264,
            kl_threshold=0.06,
            weight_bits={"head": 4},
            mechanism=(
                "Both champions at once, which is what `apply_champions` installs: 953.55 "
                "MB/token removed from one matmul, and 24 cuDNN dispatches removed from 24 "
                "layers, on disjoint parts of the step."
            ),
            prediction="win",
            rationale=(
                "**The shipped configuration, and it has never been benchmarked.** Rental "
                "46's `058` measured the conv with the *retired* Triton head, so the pair "
                "this repository actually assembles has no number at all -- an unmeasured "
                "default is the one thing a project like this may not ship. Predicted "
                "**win, 1.06-1.12**: `066` and `067` compose to ~1.08 if their savings are "
                "independent, which they should be -- one is bytes inside the last matmul "
                "and the other is dispatches inside 24 layers. **The reading that matters "
                "is 068 against 066 and 067 separately.** A shortfall would put this pair "
                "in the same class as `060`, where two dispatch savings competed for the "
                "same microseconds and the static cache was worth +1.5% alone and +0.0% on "
                "top of the conv. Bars are `067`'s: the conv changes no arithmetic."
            ),
        ),
        Hypothesis(
            slug="069-verify-inflation-k1",
            kernels=("rollback_state", "speculative_fixed_k1"),
            category="C",
            byte_share=0.0,
            correctness="sequence",
            divergence_gap_ceiling=0.3,
            mechanism=(
                "The speculative loop at block_size=1 with a drafter that always proposes "
                "the same token: acceptance is 0 by construction, every cycle runs one "
                "**two-token** verify, and the ratio is 1/gamma(1)."
            ),
            prediction="loss",
            rationale=(
                "**One slot that separates two models of rental 54's number, and they are "
                "10% apart.** gamma(2)=1.316 and gamma(4)=1.404 fit a line with slope 4.4% "
                "per token and intercept 1.184 -- but a one-token verify *is* ordinary "
                "decode at gamma=1.000, so either there is a **~23% step at the seq=1 -> "
                "seq>1 boundary itself** or the two points are on a curve. The step model "
                "predicts gamma(1)=1.272 and a ratio of **0.786**; a model with no step, "
                "through (1 token, 1.000) and (3 tokens, 1.316), predicts 1.158 and "
                "**0.864**. Rental 54's identity IQR was 0.0001 and these slots' were "
                "0.005, so the separation is 15 band-widths. Predicted **loss** either "
                "way, and the prediction that is on the record is the *number*: **0.786, "
                "the step**, because the step is what the suspects predict -- "
                "`reference.py`'s `if seq_len > 1` mask branch and a `k+1` autotune pool "
                "that the reference column never compiles are both **fixed costs of "
                "leaving seq=1**, not per-token ones. 0.864 would refute both and make the "
                "verify a traffic problem the spec's own model already prices. **This is "
                "the cheapest decisive slot in the batch**: no new arithmetic, no new "
                "kernel, and it governs whether entry 10 is alive."
            ),
        ),
        Hypothesis(
            slug="070-verify-inflation-k2",
            kernels=("rollback_state", "speculative_fixed_k2"),
            category="C",
            byte_share=0.0,
            contrast_with="069-verify-inflation-k1",
            correctness="sequence",
            divergence_gap_ceiling=0.3,
            mechanism="The same instrument at block_size=2: a three-token verify, gamma(2).",
            prediction="loss",
            rationale=(
                "**One variable from 069, and the only number in this batch that can be "
                "compared with another rental's.** Rental 54 measured 0.7596 for exactly "
                "this slot on an **RTX 4090**, where every candidate compile logged `No "
                "valid triton configs ... Required: 110592 Hardware limit: 101376` on the "
                "verify's `k+1` shape -- a shape the reference column never compiles. A "
                "5090 has more shared memory per SM and offers a wider pool, so **if this "
                "returns materially above 0.7596 on a 5090 the 23% step is partly that "
                "card's shared-memory limit and not this model's decode graph.** "
                "Predicted **loss, 0.76-0.85**, the range spanning both readings. Without "
                "this slot, 069's number could not be placed: 069 alone measures gamma(1) "
                "against a card nothing else has characterised, and the pair measures the "
                "*shape* of gamma on one card, which is the quantity the step model is "
                "about. Run it whatever 069 returns."
            ),
        ),
        Hypothesis(
            slug="071-int4-mlp-torch-dequant",
            kernels=("int4_mlp_torch_dequant",),
            category="B",
            byte_share=0.5275,
            replaces=("swiglu_mlp",),
            correctness="approximate",
            correctness_positions=264,
            top1_threshold=232 / 264,
            kl_threshold=0.06,
            weight_bits={"mlp": 4},
            requires=Precondition(
                slug="067-int4-head-torch-dequant",
                floor=0.0,
                versus="000-identity",
                reason=(
                    "the proposition this slot depends on is that a grouped int4 "
                    "dequantise-GEMV written in torch is worth something at a site inductor "
                    "compiles -- and that proposition is a *comparison* between the head "
                    "slot and the unmodified reference, measured in the same process on the "
                    "same card in the same interleaved rounds. Rental 46 put the floor on "
                    "the head's absolute ratio at 1.02, the head returned 1.0171 while "
                    "answering the question with an unambiguous yes, and this slot -- 52.75% "
                    "of per-token bytes, the largest prize in the backlog -- declined by "
                    "0.3% of a quantity that depends on the hour's card"
                ),
            ),
            mechanism=(
                "The construction `056` proved fuses, on the 96 MLP projections: **52.75% "
                "of per-token bytes at a 1.6545x ceiling**, the largest homogeneous block "
                "in the model, at group-128 int4 with no hand-written kernel in it."
            ),
            prediction="win",
            rationale=(
                "**The prize, and the first slot in eleven batches to attack it with code "
                "we did not write.** Three hand-written kernels have failed on these sites "
                "at ~66 GB/s across three rentals; rental 43 refuted register pressure and "
                "latency; rental 45 closed the tile at five measured points. What was "
                "never tried is that the right code for these sites is inductor's own. "
                "Predicted **win, 1.25-1.55** against a 1.6545x ceiling -- the head "
                "collected 1.0171 of 1.1249, 14% of its ceiling, and if this collects the "
                "same fraction it returns 1.09; if it collects what the *bytes* say, 1.45. "
                "The honest range spans them, and the wide sites are the reason to expect "
                "the upper half: gate_proj and up_proj are 9216 channels and down_proj "
                "2560, against the 32 that starved every kernel batches 003 and 004 "
                "measured. **A bandwidth budget reads a loss just as clearly.** If "
                "inductor materialises the dequantised weight at these sites, the "
                "candidate moves 4529.85 MB/token of bf16 *plus* the 1132.46 MB of packed "
                "nibbles it read to build it, and lands near 0.6 -- there is no middle "
                "outcome that is hard to read, which is what made `056` worth three "
                "minutes and makes this worth six. Bars are derived from a measured point: "
                "`030-int4-mlp` is the same weights, the same k-major grouping and the "
                "same rounding through a Triton kernel, and returned **237/264 agreement "
                "at 0.04868 nats** -- and `054` against `056` showed two authors of one "
                "function return layer-2 numbers identical to the digit, so a number far "
                "from 0.04868 here is a registration bug rather than a quantisation "
                "effect. `--dump-install int4_mlp_torch_dequant` is set for this rental, "
                "so whatever this slot returns, the generated code says why."
            ),
        ),
        Hypothesis(
            slug="072-int4-mlp-and-head",
            kernels=("int4_mlp_torch_dequant", "int4_head_torch_dequant"),
            category="B",
            byte_share=0.6755,
            replaces=("swiglu_mlp", "decode_step"),
            contrast_with="071-int4-mlp-torch-dequant",
            correctness="approximate",
            correctness_positions=264,
            top1_threshold=224 / 264,
            kl_threshold=0.09,
            weight_bits={"mlp": 4, "head": 4},
            requires=Precondition(
                slug="071-int4-mlp-torch-dequant",
                floor=0.0,
                versus="000-identity",
                reason=(
                    "this is the MLP install plus a second one measured alone two slots "
                    "above it, so with the MLP below the reference there is nothing here "
                    "the batch does not already hold -- adding a second install on top of a "
                    "composition that lost is what batch 003 spent five slots doing, and "
                    "batch 004 two more. The floor is on the margin over identity because "
                    "that is the proposition: did quantising the MLP in torch beat bf16 on "
                    "*this* card"
                ),
            ),
            mechanism=(
                "Both quantised installs at once: 67.55% of per-token bytes at 4 bits, "
                "**a 2.0269x ceiling** -- the MLP's 96 projections and the tied head, "
                "which between them are every wide matmul the decode step performs."
            ),
            prediction="win",
            rationale=(
                "**The rung that says whether the construction adds.** One variable from "
                "071, and the quantity is `072 - 071` against `067 - 000`: if the head's "
                "margin survives being stacked on the MLP's, the mechanism is byte-count "
                "and byte-counts add. If it does not, then something shared is saturating "
                "-- the same non-additivity `060` found for two dispatch savings, but for "
                "traffic, which would be a more interesting result than the ratio. "
                "Predicted **win, 1.35-1.75**. Bars come from two measured points and a "
                "superset: KL is roughly additive over independent perturbations, so "
                "0.04868 (`030`, the MLP) + 0.01674 (`022`, the head) ~ 0.065, and "
                "`014-int4-full` -- a strict superset of these sites -- measured **226/264 "
                "at 0.09185**, which bounds it from above. The bar sits at 224/264 and "
                "0.09, between the sum and the bound."
            ),
        ),
        Hypothesis(
            slug="073-conv-mlp-and-head",
            kernels=("inline_causal_conv", "int4_mlp_torch_dequant", "int4_head_torch_dequant"),
            category="B",
            byte_share=0.6761,
            replaces=("causal_conv", "swiglu_mlp", "decode_step"),
            contrast_with="072-int4-mlp-and-head",
            correctness="approximate",
            correctness_positions=264,
            top1_threshold=224 / 264,
            kl_threshold=0.09,
            weight_bits={"mlp": 4, "head": 4},
            requires=Precondition(
                slug="072-int4-mlp-and-head",
                floor=0.0,
                versus="000-identity",
                reason=(
                    "this is 072 with the conv added, and the conv's own value is measured "
                    "in 066 and 068. With 072 below the reference the batch already holds "
                    "every part of this slot separately and the composition would only "
                    "stack a third install on a loss"
                ),
            ),
            mechanism=(
                "The configuration this repository would ship if 071 wins: bytes removed "
                "from every wide matmul, and 24 cuDNN dispatches removed from 24 layers."
            ),
            prediction="win",
            rationale=(
                "**One variable from 072: the conv, whose value was measured alone in 066 "
                "and in one composition in 068.** Predicted **win, 1.40-1.85**, and the "
                "reading is `073 - 072` against `068 - 067`: the conv saved 24 dispatches "
                "next to a bf16 MLP, and this asks whether it still saves them next to an "
                "MLP whose own kernels changed. It is the promotable configuration, so it "
                "has to exist as a number before anything promotes it -- `058` is on the "
                "record as the last time this project shipped a pair no rental had run. "
                "Bars are 072's: the conv is bit-identical to the reference."
            ),
        ),
        Hypothesis(
            slug="074-int4-wide-and-head",
            kernels=("int4_wide_torch_dequant",),
            category="B",
            byte_share=0.9785,
            replaces=("decode_step",),
            contrast_with="072-int4-mlp-and-head",
            correctness="approximate",
            correctness_positions=264,
            top1_threshold=220 / 264,
            kl_threshold=0.12,
            weight_bits={"mlp": 4, "linear_attn": 4, "full_attn": 4, "head": 4},
            requires=Precondition(
                slug="072-int4-mlp-and-head",
                floor=0.0,
                versus="071-int4-mlp-torch-dequant",
                reason=(
                    "this slot's whole content above 072 is *more sites*, so the "
                    "proposition it rests on is that adding sites to this construction "
                    "keeps paying -- which is exactly what 072 minus 071 measures, in one "
                    "process on one card. If putting the head's sites on top of the MLP's "
                    "took the ratio down, putting another 2602 MB/token of narrower "
                    "projections on top will take it down further, and the batch will "
                    "already know why. The floor is on the margin between two slots rather "
                    "than on either one's absolute value, because a card that drifts moves "
                    "both together"
                ),
            ),
            mechanism=(
                "Every layer projection except the two 32-channel gates, plus the tied "
                "head, at group-128 int4 in torch: 200 sites, **97.85% of what the "
                "compiled column moves, a 3.7578x ceiling** -- the whole weight stream "
                "bar 7.86 MB/token."
            ),
            prediction="win",
            rationale=(
                "**The largest byte share this project can attack, and the riskiest install "
                "in the batch, so it runs last with everything cheaper already on disk.** "
                "Predicted **win, 1.5-2.4**. The 48 excluded sites are `in_proj_a` and "
                "`in_proj_b` -- 0.09% of the bytes and the whole of the numerical risk, "
                "because they feed an exponential through `A_log` -- so this buys 97.85% of "
                "the stream at none of that cost. **What it can find that 072 cannot**: "
                "the head is 248320 channels wide and the MLP 9216, and every site added "
                "here is narrower -- 2560, 4096, the full-attention projections. If the "
                "ratio falls below 072's, the fusion `056` discovered is **width-dependent**, "
                "and the boundary is a fact about inductor's GEMV prologue that this "
                "project could state in one table. That reading is worth a slot on its own. "
                "The one measurement of these sites at 4 bits is `014-int4-full` (rental "
                "37): **226/264 at 0.09185 nats, gate passed** -- through a row-major "
                "packing and a hand-written kernel running at 141 GB/s, so its correctness "
                "transfers and its ratio does not. Bars sit at 220/264 and 0.12, one "
                "measurement's headroom above that point rather than on top of it, and "
                "**this slot excludes the gates 014 included**, so it should be no worse."
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
    BATCH_008.batch_id: BATCH_008,
    BATCH_009.batch_id: BATCH_009,
    BATCH_010.batch_id: BATCH_010,
    BATCH_011.batch_id: BATCH_011,
}


def get_batch(batch_id: str) -> Batch:
    try:
        return BATCHES[batch_id]
    except KeyError:
        raise SystemExit(f"unknown batch {batch_id!r}; known batches: {sorted(BATCHES)}") from None
