# Start here

**This is the only file you need to read before starting.** Everything else is linked from
it, and nothing else is required reading until you need the detail it holds.

---

## 1. What this project is trying to prove

> A hand-written Triton kernel can beat what `torch.compile(mode="max-autotune")`
> generates, on the decode path of an open-weights LLM — and we can say **where**, **by how
> much**, and **why**, in advance.

The target is `Qwen/Qwen3.5-4B`, benchmarked at batch-1 decode. Every session rents one GPU
for at most two hours, measures a batch of hypotheses, records every result win or lose,
and destroys the instance.

**The last clause is the whole project.** "My kernel beat the compiler" is the premise the
entire inference-serving industry is built on — vLLM, SGLang and TensorRT-LLM are all
hand-written kernels — so re-proving it is not a finding. The finding is a correct,
mechanistic account of *where the compiler wins and where it cannot*, registered before the
measurement and then confirmed by it.

So: **state your prediction and your mechanism before you write code.** Being right in
advance is the result. A hypothesis you predicted would fail, that failed for the reason you
gave, is worth more than an unexplained win.

### 1.1 Ten batches in, the premise is answered, and the answer is no

Read this before filling a batch, because it decides what kind of hypothesis is worth a
slot.

The question above has now been put to the two sites in this model most favourable to a
hand-written kernel, and it lost both times **to the same program written as torch
operations**:

| pair | one variable | kernel | no kernel |
|---|---|---:|---:|
| `044` / `045` (rental 45) | the four taps as a `custom_op`, or as torch | 0.7854 | **1.0765** |
| `054` / `056` (rental 46) | the int4 dequantise-GEMV as a kernel, or as torch | 0.9851 | **1.0171** |

Both of this project's champions are kernels it **deleted**, and the mechanism is not a
mystery: a `custom_op` costs its own kernel *plus* everything inductor can no longer fuse
across it, and that second term is invisible at the call site. The two numbers also bound
it — 37% where the kernel sat between a fused producer and a fused consumer in 24 layers,
3.2% at one call site with one reduction beside it. **The barrier's cost and the fusion
opportunity are the same quantity**, so a hand-written kernel is cheapest exactly where it
has least to gain. That is close to a closed argument.

**What is not closed is the program.** The compiler picks instructions; it does not pick

* **how many bytes the weights occupy** — quantisation and layout are ours. **Read the
  correction in §1.2 before building on this**: rental 56 showed `056`'s fusion does not
  generalise past the one site it was measured at, so the 91.85% is not behind one fact, it
  is behind one fact *per site shape*.
* **how many forward passes a token costs** — speculative decoding changes it under a
  property of greedy decoding no scheduler can observe. γ is measured now (§5) and the open
  question is a 23% step, not the drafter.

So from batch 011 the rule is: **do not register a slot that writes our code where torch
already expresses the program.** Register a slot that hands the compiler a *different
program* — fewer bytes, fewer passes, a layout it cannot choose — and let it write the
kernel. `docs/HYPOTHESES.md` entry 1 is now a ladder of byte shares rather than a kernel
backlog.

**What would reopen the kernel line, stated in advance so it can happen.** A site where
inductor's generated kernel falls materially short of the roofline, with the dump to show
what it emitted — and the bar is now quantitative rather than rhetorical: a hand-written
replacement has to beat it by **more than the fusion it destroys**, which the table above
prices at 3-37% depending on what sits next to it. Every slot reports achieved GB/s beside
its ratio, so any batch can notice this; none has yet.

### 1.2 Correction (rental 56): the compiler's dequant fusion is site-dependent

§1.1 generalised from `056` — the int4 head written in torch, which inductor compiled into
one reduction carrying the whole grouped unpack with **no weight-sized buffer in the graph**
— to "write the quantisation in torch and inductor will fuse it". **That generalisation is
wrong, and the dump that cost nothing says so.**

At the 96 MLP projections, the same source expression produces a **pointwise kernel that
writes two complete `(2560, 9216)` fp32 weight tensors** and a **separate** reduction kernel
for the matmul. 94.4 MB a site, 34 such allocations in the graph, zero `extern_kernels.mm`:
inductor is not falling back to cuBLAS, it is choosing to materialise. That is
**18.12 GB/token** of extra traffic against a reference that moves 8.59, so
`071-int4-mlp-torch-dequant` would have returned **~0.37** against its registered 1.25-1.55.

The operands differ by 27x — the head's dequantised fp32 weight would be **2.54 GB**, the
MLP's is 94.4 MB — so the ranked reading is that **inductor materialises when it can afford
to, and the head's fusion was forced rather than chosen.**

**How to apply.** "Express it in torch and let the compiler fuse it" is not a law, it is a
measurement that holds at one site shape. Before extending a fusion result to a new site,
**dump the generated code at that site** and look for an allocation the size of the operand —
`--dump-install <kernel>` does it for the price of one step, needs no slot, and answered a
declined hypothesis here. And do not read fusion off a kernel *name*: inductor names a fused
node after its origins, so four kernels in rental 56's dump carry both the nibble unpack and
`mm` in their names and none of them performs a matmul. The allocation and the grid size are
the evidence.

**MEASURED, rental 57 (2026-10-02): the dtype is a no-op, and the reason closes this
section's own question.** Batch 012 ran the pair at two widths. The head: **1.0251** fp32
against **1.0218** bf16, margin −0.0033. The 96 MLP projections: **0.6390** against
**0.6452**, margin +0.0062. Both inside their IQRs, pointing opposite ways, with the
candidate columns at **458 against 458 GB/s** and layer 2 agreeing to five decimal places.

**Why, from the dump of the bf16 candidate:** the reference's graphs hold zero dequantised
weight buffers; the candidate's two hold **16 fp32 `(1, 2560, 9216)` buffers plus one
`down_proj`-shaped one each — 17 of 96 MLP sites — and zero bf16 buffers of either shape**,
with the unpack as a pointwise kernel and the matmuls as separate `mm` reductions, and zero
`extern_kernels.mm`. **Inductor picks its materialisation point upstream of `.to(x.dtype)`**,
so the cast is applied to a buffer already written at full fp32 width. A matmul's dtype
cannot move a decision made before the matmul. `flat.float() @ dense.float()` was never the
lever, and no spelling of this expression is.

**The affordability reading also fails as a predictor of direction**, but read it off the
*census*, not off the ratios: the 9216-wide output sites carry 16 of 64 buffers and
`down_proj` carries 1 of 32, so the cheaper narrower operand materialises **less**. `081` at
200 sites and 97.85% of the bytes returned **0.6150**, no worse than the MLP's 0.639.

**Count allocations; do not back-solve a multiplier from a ratio.** `012`'s writeup first
inferred "2.07x the quantised weight at the MLP, 1.64x at the wide sites" from the achieved
bandwidths and read the difference as evidence about fusion. The dump refutes that arithmetic:
17 operands account for **3.21 GB/token of the 8.25 GB** by which the candidate exceeds its own
byte model, and **~5 GB is unattributed** — the first thing the next session should count, for
free, from the dump already on disk.

**What is left of the 91.85%, stated precisely.** Not a dtype and not a kernel: the only
untested route is making the fp32 intermediate *never exist* — bf16 group scales, or an
unpack whose cast precedes the arithmetic — so there is no fp32 buffer for inductor to choose
to write. Everything else in this line is closed; see
`results/batches/012-the-matmul-dtype/`.

## 2. The arithmetic that decides whether your hypothesis is worth trying

Run this first. It needs no GPU, no checkpoint, and no money:

```sh
uv run python docs/roofline.py
```

It prints where every byte goes in one decode step. The headline result:

| What moves | share of per-token bytes |
|---|---:|
| Weights, streamed once | **91.85%** |
| GQA `repeat_interleave` materialisation | **6.23%** |
| Recurrent state | 1.10% |
| KV cache read | 0.78% |
| SwiGLU intermediates | 0.026% |
| Norm + residual | 0.018% |

**A hypothesis cannot beat the share of bytes it touches.** That is the single most useful
sentence in this repo. A perfect, infinitely fast fused RMSNorm buys 0.018% — below the
harness's own noise band, so the hypothesis is not unlikely, it is *unmeasurable*. Five
hypotheses are in the graveyard for exactly this reason, closed by arithmetic rather than by
renting a GPU.

Batch-1 decode is a **weight-streaming** problem, not a kernel problem. Once a kernel
streams its data once it is at the roofline, and inductor is good at reaching the roofline
on simple memory-bound work — it emits Triton, so you would be hand-writing the kernel it
already generates. You beat it by **moving fewer bytes** (quantisation, layout, removing a
materialised intermediate) or by **running a different algorithm** (chunked scans,
online-softmax attention, split-K) — never by writing a tighter elementwise loop.

`docs/HYPOTHESES.md` ranks the open backlog by this measure and records why each graveyard
entry is dead.

## 3. Read what you are trying to beat

```sh
TORCH_LOGS=output_code python -m deltaforge.cli bench --weights … 2>&1 | tee inductor.txt
```

This prints the exact Triton inductor generated. **Do this before writing a kernel.** You
cannot claim to beat code you have not read, and you cannot explain why you won without it.
Also confirm both columns actually compiled. A silent graph break drags `compiled` toward
eager and inflates every ratio in your favour, which is the first thing a sceptical reader
checks — and rental 35 proved it cuts the other way just as easily, when the *candidate*
stopped compiling and six hypotheses reported ratios near 0.15 that measured nothing (§8).
Every slot record now carries `graphs_compiled`; a `0` there means the ratio beside it is
not a comparison.

## 4. A session, start to finish

**A rental measures a batch of 7-12 hypotheses, not one.** The fixed cost of a rental —
container image, torch, a 9.32 GB checkpoint, the GPU suite, and one `max-autotune` compile
of the reference — is about 19 minutes, and each additional hypothesis costs 3-6. Those are
measured on rentals 34-35, not estimated. Testing one per rental pays that fixed cost to buy
a single measurement, and nine rentals were billed that way without producing a number. `docs/BATCHES.md` has the arithmetic and the workflow;
`docs/superpowers/specs/2026-09-06-batched-hypotheses-design.md` is the design.

1. **Read `LEADERBOARD.md`** — current champion and every attempt so far.
2. **Read the graveyard in `docs/HYPOTHESES.md`.** Do not re-run a dead end — but note
   that "unmeasurable, not worth a rental" is *not* a dead end any more (see §4.1).
3. **Fill a batch.** 7-12 hypotheses in `src/deltaforge/batches.py`, each stating its
   mechanism, its category, its share of per-token bytes, and — the part that matters —
   **its predicted outcome and the reasoning behind it, committed before the rental.**
   Ordering is load-bearing: identity champion first, cheapest and most diagnostic next,
   riskiest last.
4. **Branch `batch/NNN-slug`.**
5. **Write and test the kernels locally.** No GPU needed; GPU time buys validation and
   measurement, not development.
   ```sh
   uv sync --extra dev
   uv run pytest          # batches_test.py proves every hypothesis installs and changes something
   uv run ruff check . && uv run ruff format --check .
   ```
   Each kernel goes in `src/deltaforge/kernels/`, registers itself, and adds **an installer
   keyed by its kernel name** to `deltaforge.model.INSTALLERS` and its checks to
   `CHECK_BUILDERS` in the same commit. Both tables are keyed by kernel, not by the
   operation replaced, because several kernels routinely attack the same operation.
6. **Dry-run the money machinery**, which spends nothing:
   ```sh
   remote/run_remote.sh --dry-run --session-id smoke --batch NNN-slug
   ```
7. **Run it.** In the foreground, if you can sit with it:
   ```sh
   remote/run_remote.sh --session-id "$SESSION" --batch "NNN-slug"
   ```
   In the background — which is what an agent session actually does — go through the
   launcher, never a hand-rolled `nohup ... &`:
   ```sh
   remote/launch.sh --log "$LOG" --session-id "$SESSION" --batch "NNN-slug"
   tail -n +1 -F "$LOG"                      # attach a monitor: +1, never -n 0
   remote/launch.sh --status --log "$LOG"    # 0 finished, 1 failed or killed, 2 running
   ```
   `launch.sh` guarantees the log ends with one `[deltaforge] [launch] run exited N`
   line and that the exit status lands in `$LOG.status`, so a run that ends before
   anyone is watching still says so. **A launch is not a rental**: `--status` and a
   non-empty instance list are what say a rental is up. See §8.
8. **Record the outcome** (§6). Every slot, win, loss or error.
9. **Update the docs before you finish** (§6.1). The writeup is part of the session, not a
   follow-up to it.

The benchmark runs four columns. `compiled` against `candidate_compiled` is the score —
identical `max-autotune` treatment, the only difference being who wrote the kernel.
`compiled_nocudagraphs` is opt-in via `--columns all`.

### 4.1 What batching changes about which hypotheses are worth trying

The ceiling arithmetic in §2 still decides what can *win*. What it no longer decides is
what is worth *measuring*.

Five hypotheses sit in the graveyard closed as "unmeasurable": their ceiling is below the
noise band, so a whole rental to measure one was not worth it. **That was an argument about
cost, and batching dissolves it.** At three minutes a slot, a measured null carrying a real
ratio and a real IQR from a real card beats an arithmetic prediction of a null — and the
two are not the same claim, because a measurement also contains launch overhead, CUDA-graph
behaviour, and whatever inductor actually emitted.

So: still rank by byte share, still refuse to *promote* on a noise-band margin, but stop
using "the ceiling is too small to measure" as a reason not to fill a slot. Fill the slot.

### 4.2 The batch's own guarantees

* **A failing slot costs a slot, not the rental.** Every hypothesis runs in its own
  try/except; an exception is recorded as `error` with its traceback and the batch
  continues.
* **Records are written as each slot finishes**, and teardown pulls results *before*
  destroying the instance, so a late failure cannot take the earlier slots with it.
* **The batch stops itself** before a slot it cannot finish, so the watchdog never has to.
* **The identity champion runs first and must return 1.00 ± noise.** If it does not, the
  harness is measuring something other than the kernel under test and **every other number
  in that batch is void** — say so in the writeup rather than reporting them as findings.

### If this is the first session that ever gets a GPU

**No measurement exists yet.** Batch 001 is built for exactly this: it opens with the
identity champion and the weight-value oracle runs before it. Do not add a kernel to the
front of a batch to "get a result faster" — an uncalibrated result is not a result.

## 5. Money and safety

**Start by re-validating the reference, before anything else costs money.** On 2026-09-10
`test_reference_greedy_decode_matches_the_oracle_token_for_token` **failed** — it had
passed on 2026-09-07. The logits oracle and the mRoPE test still pass, so the architecture
facts in §8 are still corroborated and the disagreement is a single argmax at token 2.

The suspect was that `transformers` was installed from a *floor* (`>=5.16,<6`) rather than
a pin, so the oracle could move on its own. It is now `transformers==5.16.1`, which is
correct regardless — an unpinned oracle is a defect, because the oracle *is* HuggingFace and
its version is an input to the experiment rather than a dependency of it.

**But it was not the cause, and rental 28 proved that.** 5.17.0 was released 2026-09-09, so
the floor already resolved to 5.16.1 on 2026-09-07, the day the test passed. Rental 28
installed 5.16.1 explicitly and failed *byte-identically* to rental 27 — same index, same
two token ids — on different hardware. Model code, test, prompt, checkpoint revision, torch,
triton and python are all unchanged between the passing and failing runs.

**So the live question is whether the gate itself can hold**: exact equality over 32
sequential bf16 argmaxes, where commit `5722aaa` already found and fixed three other bounds
in the same file that no correct bf16 implementation could meet. The deciding number is the
top-two logit gap at the diverging step against the 1-2 ULP the two models disagree by
anyway; `test_report_the_first_greedy_divergence` measures it and writes
`results/diagnostics/oracle-divergence.json` without asserting anything.

Until that number exists, **do not promote either reading**, and every number downstream of
the reference remains inadmissible. See `docs/GPU-ACCESS.md` (blocker 13) and
`results/batches/001-calibration/README.md`.

**The compile cost is now measured** (2026-09-14, rentals 34-35), and `docs/BATCHES.md`
carries the numbers: ~19 minutes of fixed cost, a reference `max-autotune` compile of
**268 s cold / 57 s warm**, and **173-376 s per slot**. A full nine-slot batch ran in 44
minutes. The ~40-minute figure this section used to carry measured an unrolled prefill scan
that nothing times — see §8. `002-compile-cost` was written to obtain these numbers and
batch 001 produced them instead.

**The compile cache now comes home.** Torch's fx-graph and autotune caches are on by
default but write to `/tmp` on a box that gets destroyed, so every rental this project has
run compiled cold. `run_remote.sh` points them at `/workspace/df-cache`, pulls that
directory back into `cache/compile/<gpu>-<torch>-<cuda>/` before destroying the instance,
and pushes it up again next time the same card comes back. A warm cache changes how long
compilation takes, not what it emits, and both scored columns share it.

**At 180 cumulative billed minutes — three hours — this session may not start another
batch.** `run_remote.sh` checks before each run and exits 3 when the session is spent. When that
happens: destroy any live instance, then finish recording and writing up the work you
already did — none of which needs a GPU — and stop. Do not start "just one more attempt".

**And it refuses to rent at all when even three hours cannot fit one hypothesis** — exit
5, before provisioning. The minimum is two slots: the identity champion calibrates the
harness but scores nothing, so a rental that fits only that has bought no science. Nine
rentals were billed here without producing a number, and the cheapest of those failures
would have been not renting. The estimate comes from `cache/compile/<key>/phases.env` and
falls back to the deliberately pessimistic cold numbers in `batch.COLD_PHASE_ESTIMATES`.
**Until 2026-09-14 it always fell back**, whatever had been measured: the gate runs before
provisioning and `DF_CACHE_KEY` is read off the rented box, so the path it built was always
`cache/compile/unknown/`. `df_phase_estimates` in `remote/lib.sh` now surveys the cards that
have measured something and takes the most pessimistic value for each phase, because the
gate cannot know which card the market will give it.

The gate was an hour, then 90 minutes, then two, and is now three because one hypothesis on
a **cold compile cache** does not fit in two — see
`docs/superpowers/specs/2026-09-10-compile-cost-and-memory-design.md` §4.1. It is not
licence to spend longer: the gate is a ceiling, not a target. Vast bills by the minute and
the run destroys itself when the batch ends, so a warm session still pays for the ~40
minutes it uses, and the real budget control is the $45 month-to-date gate.

The check happens before a run, never during one: a benchmark executing at minute 179
finishes normally, because killing it would waste the money already spent and leave nothing
recorded in exchange. The 210-minute watchdog is a backstop for hangs and is always set
above the gate — **if it ever fires, that is a fault and the writeup must say so.**

**Inside a run, a slot that runs away costs a slot rather than the rental.** Every slot
carries a wall-clock cap (`SlotBudget.cap_for`), and a slot that overruns is recorded as
`error` with a `SlotTimeout`. The first two slots are exempt, because capping them would
defeat the guarantee the cap protects.

`VAST_API_KEY` lives in a gitignored `.env` at the repo root and reaches curl through a
config file on stdin, so it never enters argv, a file, or a log. **Never** print it, commit
it, or put it in a PR body. CI greps tracked files for it. The same applies to the optional
`DOCKER_LOGIN_USER` / `DOCKER_LOGIN_TOKEN`; request bodies go to curl through a `0600` temp
file rather than `argv`, and `remote/scripts_test.py` asserts that.

**If a run hangs before sshd answers, read `docs/GPU-ACCESS.md` before renting again.**
**Fifty-six instances created, fifty-six destroyed, $11.489 lifetime, zero leaked. Both
champions are kernels this project deleted rather than wrote.**

**Rental 56 (2026-10-01): batch 011 ran six of ten slots and the two findings are both
corrections.** The MLP ladder declined on a card running at **0.34x**, and the dump refuted
its central prediction anyway (§1.2). γ is confirmed as a **step** on a second card and
generation — 91% at the seq=1 → seq>1 boundary here against the 4090's 23%, with a two-token
verify batch 010 never ran — and batch 010's suspect 3 is dead, because a 5090 reports the
same 101376-byte shared-memory limit. **The shipped pair is finally measured at 1.2179**,
16% above the product of its ingredients, flagged cross-slot and unconfirmed. And **the card
variable has a name**: the host driver separates seven identity slots where clock, GPU model
and toolchain do not, so `DF_MAX_CUDA` now refuses a too-new driver at the offer line.
$0.769, and $1.147 across the session's two rentals. See
[`results/batches/011-bytes-not-kernels/`](results/batches/011-bytes-not-kernels/).

**Rental 57 (2026-10-02): batch 012 ran six of eight slots for $0.388 and the dtype line is
closed.** §1.2 has the finding and the dump. Three things worth carrying forward. **The
52.75% prize is executed at last** — `077-int4-mlp-torch-dequant`, registered a win twice
(`061`, `071`) and declined unexecuted both times, ran ungated and returned **0.6390 at an
IQR of 0.0060**; its registered 1.25-1.55 is refuted by measurement now, not by a dump. **The
head is at its best ever measured**, `075` at **1.0251** against 1.0171 (r46) and 0.9971
(r56), on a 1220 GB/s card the `DF_MAX_CUDA` gate won by rejecting the CUDA 13.3 branch.
**And a zero-floor margin gate passes on noise**: `081` was gated on `078 - 077 >= 0.0` and
cleared it by **+0.0062 against an IQR of 0.0259**. A margin floor belongs above the IQR of
the slots it reads; `batch.py` neither enforces that nor warns. Power telemetry also landed
for the first time (426 W of 500, throttle `0x0`), which rules out blocker 18's suspect on
this host.

**Batch 012 is registered (2026-10-02): `012-the-matmul-dtype`, eight slots, and it is
three pairs one operator apart.** §1.2 has the mechanism and the ranked prediction. Two
things in it are deliberate and are the lesson of rental 46 applied in both directions.
**`077-int4-mlp-torch-dequant` runs ungated** — 52.75% of per-token bytes, registered a win
twice and executed neither time, now registered a **loss at 0.33-0.45** and run as the
control the batch is read against; a declined slot is an unexecuted slot. And
**`081-int4-wide-torch-dequant-bf16` is gated on the dtype margin `078 - 077`, not on `078`
beating the reference**, because on the ranked branch `078` is a loss and declining 97.85% of
the model's bytes for that would repeat rental 46 exactly. Only `079` and `080`, whose
content is profitability rather than mechanism, are gated on beating the reference. Run it
with `--dump-install int4_mlp_torch_dequant_bf16`.

**Batch 011 is registered (2026-09-30): `011-bytes-not-kernels`, ten slots.** It is the
first batch built on §1.1: nothing in it writes a kernel. The pair this repository *ships*
(`068`) is measured for the first time, then the construction `056` proved fuses climbs a
byte-share ladder — 52.75% (`071`), 67.55% (`072`), 97.85% (`074`) — and two slots
(`069`, `070`) settle whether rental 54's 23% step at the seq=1 → seq>1 boundary is real.
**Every gate in it is a margin between two slots rather than an absolute ratio**, which is
the defect that declined the largest hypothesis in the backlog on rental 46. Run it with
`--dump-install int4_mlp_torch_dequant`: the MLP slot's generated code is what says why,
whichever way it goes.

Two pre-rental fixes went with it, both latent rental-voiders no CPU test would have shown
before: `int4_mlp_torch_dequant` was registered against `decode_step`, which made it
**impossible to compose with the head** (one champion per operation) — the Triton kernel on
the same 96 sites has always said `swiglu_mlp`; and `DF_MIN_CUDA` still defaulted to
**12.8**, the value that is 3 for 3 on `Error 804` (§8, and `docs/GPU-ACCESS.md` blocker 11).

**Rental 55 (2026-09-30) bought nothing: $0.378 for 55.32 minutes, batch 011 never reached
slot 0.** `fetch-weights` sat silent at **4/6 files** and the step guard gave up after
2400 s — the same signature rental 47 recorded at 8:07, which is `huggingface_hub` retrying
a CAS error internally and without a word. **The guard bounds silence, and this is the one
step here whose healthy state is silent**: `snapshot_download`'s bar ticks once per
completed file and the last two files are the safetensors shards. `fetch-weights` now
watches its destination's byte total instead, exits 3 when it stops growing
(`--stall-seconds`, default 300), and takes Xet out of the path by default because both
recorded stalls were CAS retries. See `docs/GPU-ACCESS.md` blocker 21.

**Rental 54 (2026-09-30): `gamma` exists.** A `k+1`-token verify costs **1.316 at k=2 and
1.404 at k=4**, and the two points say the cost is a **step** — ~23% for entering the
seq>1 path at all, then 4.4% per token. Batch 010's two registered kill criteria were both
crossed and both are **withdrawn as underived**: the spec's own formula with the measured
`gamma` still shows 1.34x and 1.45x. A kill criterion is a prediction; derive it and
re-substitute before writing "dead". See
[`results/batches/010-speculative-verify/`](results/batches/010-speculative-verify/).

**That session also cost $0.811 across eight rentals, seven of which measured nothing** —
three to one bug (blocker 19: a remote step lived exactly as long as its ssh connection, and
had been misfiled against hosts twice). Remote steps now run detached via `remote/step.sh`.

**Rental 46 (2026-09-23): the compiler beat our hand-written kernel at the one site we had
ever won on.** Three registrations of one program ran in one process — the champion's
`torch.library.custom_op` (**0.9851, a loss**), the same kernel behind `triton_op`
(**errored: it does not trace on torch 2.11**), and the dequantise-GEMV written in **torch
with no kernel of ours in it** (**1.0171, a win**). Same weights, same function, layer 2
identical to the digit. **`tiled_int4_head` is retired and `int4_head_torch_dequant` is
champion of `decode_step`.** The dump refutes the registered prediction outright: one
reduction kernel carries `__rshift__`, `bitwise_and`, the `mm`, the final RMSNorm and the
residual add together, and **no weight-sized buffer exists in the graph** — inductor fuses
a grouped dequantisation into a 248320-wide GEMV prologue, which this repository assumed
for nine batches it could not. See §8, and `results/batches/009-visible-kernels/`.

**The shipped pair has never been benchmarked.** `apply_champions` now installs
`inline_causal_conv` beside `int4_head_torch_dequant`; `058` composed the conv with the
*retired* kernel. Measure that pair before anything else.

**Rental 45 (2026-09-23) produced the largest margin ever measured here against a
same-day reference, and it contains no Triton at all.** `045-inline-causal-conv` writes the
four-tap causal convolution as torch operations inductor can fuse and returned **1.0765**,
bit-identical to the reference (264/264, 0.00000 nats), on a candidate column reaching
**1219 GB/s — 68% of vendor peak**. Its control, `044-fused-causal-conv`, is the *same
arithmetic* wrapped in a `torch.library.custom_op` and returned **0.7854** in the same
process. **One variable, 37%** — and it also means the conv regression rentals 42 and 43
blamed on composition was never a composition effect: the conv loses 21% alone. See §8.

**And batch 008 re-measured the incumbent on a healthy card: 1.0105, `inconclusive`.**
The reference ran at 1197 GB/s and the identity slot at **−0.28%**, so rental 43's excuse
does not apply. Two of three cards put the int4 head at ~1%; the 7.91% belongs to rental
40, and `LEADERBOARD.md` says so in the champion block. The 1.1249x byte ceiling is
arithmetic and stands.

Batch 005 (2026-09-19) produced this
project's first two wins: `022-int4-head` at **1.0791, IQR 0.00034** — group-128 int4 on
the tied LM head alone — and `025-fused-causal-conv` at **1.0144** — a number **two later rentals contradict**, at 0.7854 alone and 0.7963 composed, and which nothing has explained; that kernel is now retired in favour of `045`. Batch 006 (2026-09-20)
added a third, `034-static-cache-cudagraphs` at **1.0196**, and refuted its own premise:
see §8 on the tuner.

**Batch 007 (2026-09-20) re-measured the champion and got 1.0161.** Same kernel, same
site, same heuristic tile — on a card that ran the reference at **800 GB/s where rental
40's ran at 1282**, with the identity slot carrying **+1.01%** and the head removing 0.097
ms/token against identity's 0.102. On that card the int4 head and the static decode cache
both measured **zero**. Two RTX 5090s reporting the same memory clock, driver and torch
differed by **1.61x** on the reference. **The card is an uncontrolled variable the size of
the effects being measured**, so read every ratio against the identity slot from the same
rental, and never quote one without naming its rental. Batch 003 (2026-09-16) produced seven admissible ratios, all losses, and batch
004 (2026-09-17) rewrote the kernel that lost hardest and lost harder, 0.2801 → 0.1934.

**What changed between losing by 5x and winning was not the kernel.** Batches 003 and 004
installed on all 248 layer projections at once; batch 005 installed on one site. The same
kernel family achieves 228-319 GB/s averaged over the projections and **656 GB/s on the
head**, which launches 3880 programs where `in_proj_a` launches four. Before concluding
anything about a kernel, check whether the sites you measured it on could fill the card.

**The tile question is closed, and five measured points closed it.** Batch 008's
`047-int4-head-narrow-tile` pinned BLOCK_N=32 — the wave-count theory, the last direction
left after rental 43 killed register pressure and latency — registered in advance as a
predicted `loss`, and it lost at **0.9617**. The heuristic nobody chose on purpose is the
best of five points measured in the decode step. **Stop spending slots on tiles here**; the
live direction is §8's observation that `tiled_gemv_int4` is itself a custom op.

**But the tile is not the difference, and batch 006 spent a rental establishing it.** The
obvious next step from that paragraph — the layer projections are starved, so give them a
better tile — was searched on the card across BLOCK_N, SPLIT_K, BLOCK_K, warps and
pipeline depth. The MLP came back at **67 GB/s** against batch 003's 65 at a completely
different tile, and the search made the *head* 2.3x worse. **This kernel family is ~66
GB/s on the layer projections and ~656 on the head, and three rentals have now failed to
move it.** The next idea there has to be a different algorithm, and the cheapest
instrument is a published int4 kernel as an unscored column.

**The baseline is characterised, and the number in this paragraph used to be wrong twice
over.** Rental 38's `TORCH_LOGS=output_code` dump shows inductor folds the GQA head
expansion into index arithmetic, so the compiled column never moves the 570.43 MB/token the
roofline attributes to it. Against the **8587.80 MB/token it actually moves** it runs at
**1177 GB/s, 65.7% of a 5090's 1792 GB/s peak**. The earlier "1308 GB/s, 73%" counted the
folded expansion *and* mixed SI megabytes with binary gigabytes per second. That baseline is
also **generated Triton with no cuBLAS anywhere**, with the residual add and RMSNorm fused
inside its matmuls, and it runs with **no CUDA graphs**. See
`results/batches/004-bandwidth-bound-gemv/README.md`.

**And it dispatches 508 kernel launches per decoded token.** The decode graph is fully
unrolled, so the dump can simply be counted: 483 `triton_*.run(...)` call sites plus 25
`extern_kernels`. Against 4.79 ms at vendor peak and 5.3-6.4 at an achievable one, the
0.9-2.5 ms residue in that 7.30 is **1.8-4.9 µs a launch** — which is what inductor's
Python launch path costs when nothing is captured into a graph. That is `docs/HYPOTHESES.md`
entry 8, and it is the first hypothesis in this repository whose ceiling does not depend on
a hand-written kernel being good.

Rental 35 (2026-09-14) is where the infrastructure chain ended: it ran all nine slots of
batch 001, the oracle gate passed and the identity champion measured 1.0018, but dynamo hit
`recompile_limit` inside slot 2 and six ratios were void. Batch 003 closed that. Memory has
been flat across both batches — 8.07 GiB of 31.36 at bf16, 12.9 GiB with int8 copies
resident. `docs/GPU-ACCESS.md` records every blocker, how each was fixed, and which fixes
are *proven on a GPU* rather than merely believed.

## 6. Recording the outcome — the part that matters

Sessions are only worth anything if they compound. **A loss is recorded as carefully as a
win**, with the mechanism that failed and why — not a restatement of the result, the
*cause*.

1. `results/hypotheses/NNN-slug.json` — written by the harness, win or lose.
2. `LEADERBOARD.md` — a row with the ratio and outcome.
3. `docs/HYPOTHESES.md` — a graveyard entry if it lost, or an updated open entry if the
   result was inconclusive.
4. Confirm `ledger/spend.jsonl` reflects the run.

**Promotion is not "it was faster."** Promote to champion only if the candidate's median
ratio exceeds the incumbent's by more than the interquartile spread of the scoring rounds
(`BenchResult.iqr_ratio`). A margin inside the noise band is **inconclusive**: it neither
promotes nor graveyards, and the hypothesis stays open for a cleaner measurement. Recording
a noise-band result as a win is how a leaderboard becomes fiction.

**Never carry a number across sessions.** Every session rents different physical hardware,
so the score is a ratio measured inside one process on one card, and the reigning champion
is re-benchmarked every time.

**Do not fabricate, estimate, or placeholder any number.** If a run produced no measurement,
the record says so.

### 6.1 Close the session by updating the docs

The records above say what a *hypothesis* measured. They do not say what the *session*
learned, and the prose docs are what the next session reads first. So the last step of
every session — it needs no GPU, and it still works after the session gate has stopped
everything else — is a pass over the docs, adding what this session proved and fixing what
it proved wrong.

| Doc | Update it when |
|---|---|
| `LEADERBOARD.md` | Anything ran. Champion, plus a row per attempt, win or lose. |
| `docs/HYPOTHESES.md` | A hypothesis died (graveyard entry, with the *cause* rather than the result), or its ceiling or ranking changed. |
| `results/batches/NNN-slug/README.md` | A batch ran. What is now settled, where it died, what is still open, what it cost. **Void batches are written up too** — `results/batches/001-calibration/` is the worked example. |
| `docs/BATCHES.md` | A rental measured something the cost arithmetic had only estimated. An estimate a rental has contradicted is worse than no estimate. |
| `docs/GPU-ACCESS.md` | A run failed before the benchmark. A blocker row, and whether its fix is *proven on a GPU* or merely believed. |
| `docs/ARCHITECTURE.md` | The model turned out to work differently than written — and add the test that catches it next time. |
| `AGENT.md` §8 | A trap cost this session real time and would cost the next one the same. |
| `README.md` | The headline result, or any claim it makes, is no longer true. It faces outward and is the first file to go stale. |

**"If necessary" is a real test, not politeness.** Write an entry only if a future session
would *act differently* for having read it. A changelog of what you did is not that — the
git log already holds it. Prefer amending an existing entry to appending a new one, and
delete what a measurement has superseded.

Two things are never optional. **A claim this session disproved gets fixed in the same
pass**, and **anything still unproven says so out loud** — `docs/GPU-ACCESS.md` keeps a
"Proven?" column and answers `no` in it for exactly this reason. A doc that quietly carries
a wrong number spends the next rental.

Commit the writeup on the same branch as the work it describes. A session that measured
something and did not write it down has spent money to produce nothing, which is the
failure mode this project exists to avoid.

## 7. Where things are

| Path | What it is |
|---|---|
| `docs/roofline.py` | **Run first.** Where the bytes go; the ceiling on any hypothesis. |
| `docs/HYPOTHESES.md` | The ranked backlog and the graveyard, with mechanisms. |
| `docs/BATCHES.md` | **How a batch works** and what filling one requires. |
| `src/deltaforge/batches.py` | The batch manifests, with every prediction registered in advance. |
| `src/deltaforge/batch.py` | Batch model, outcome arithmetic, deadline policy. No torch. |
| `src/deltaforge/fusion.py` | **Pre-rental instruments.** What inductor generated, parsed; which registrations are opaque. `cli fusion` runs both. |
| `src/deltaforge/batch_run.py` | The GPU-side batch loop: compile once, isolate every slot. |
| `docs/ARCHITECTURE.md` | Resolved model facts. Read before writing any kernel. |
| `LEADERBOARD.md` | Champion and every attempt. |
| `src/deltaforge/reference.py` | **The baseline.** Never contains a custom kernel. |
| `src/deltaforge/config.py` | Model configs and the `tiny_config()` CPU test fixture. |
| `src/deltaforge/weights.py` | Checkpoint → model; explicit name map, loud on anything unmapped. |
| `src/deltaforge/model.py` | Assembly, champion installation, greedy decode. |
| `src/deltaforge/kernels/` | Kernels and the registry. |
| `src/deltaforge/harness/bench.py` | Interleaved A/B/A timing. Knows nothing about Qwen or Triton. |
| `src/deltaforge/harness/correctness.py` | Both gates. Reports magnitudes, never a bare bool. |
| `src/deltaforge/ledger.py` | Spend record and both budget gates, in Python. |
| `remote/run_remote.sh` | The one entry point for a GPU run. |
| `docs/GPU-ACCESS.md` | **Read if a run hangs before sshd.** The container-pull blocker and its fix. |
| `remote/lib.sh` | The same two gates in awk, plus the Vast API and lifecycle helpers. |
| `docs/superpowers/specs/2026-08-29-deltaforge-design.md` | The original design spec. Historical. |
| `RESULTS.md` | **The outward-facing write-up**, for a reader who was not here. Headline numbers, mechanisms, method, open questions. |
| `site/index.html` | `RESULTS.md` as one standalone HTML page. The `gh-pages` branch serves a copy at <https://lovemanna.github.io/deltaforge/>; `site/README.md` has the three commands that publish a change. |

Tests are colocated as `<module>_test.py`; `testpaths` is `src` and `remote`.

**This repository has been public since 2026-10-03**, and a page built from it is served on
github.io. Three things follow. Anything committed from here is world-readable the moment it
is pushed, `ledger/spend.jsonl` and `docs/GPU-ACCESS.md` included — the CI secret-scan job in
`.github/workflows/ci.yml` is now load-bearing rather than belt-and-braces. A number this
project has retracted is visible until the prose carrying it is fixed, so §6.1's "fix what
this session proved wrong" is the difference between a stale line and a public one. And
`RESULTS.md` and `site/index.html` carry the same headline numbers in the same order as
`LEADERBOARD.md`: a rental that moves a champion moves all three, or the public page starts
contradicting the record.

## 7a. What the first working session actually learned (2026-09-07)

Two lessons that cost rentals to learn and will cost them again if forgotten.

### A correctness bound written for fp32 is not a bound in bf16

Four separate tolerances in `oracle_test.py` were absolute fp32-era numbers — `5e-2` on
logits, `1e-3` on RoPE cos/sin — applied to **bf16** tensors. bf16 carries 8 mantissa bits,
so at the |logit| ≈ 30 this model produces, **one ULP is already 0.25**. Those bounds sat
below the representable granularity of the dtype: no correct implementation could ever meet
them, and each one looked exactly like a model bug.

The tell was that two independent tests reported *precisely* `0.28125` = 9/32. A real cache
bug does not reproduce a full-forward-vs-HuggingFace difference to the bit.

**So: score against the tensor's own scale, not an absolute epsilon**, and where a token
sequence is the thing that actually matters, assert the tokens. `reference.py` was right
every time; the tests were wrong four times.

### Batch mode is memory-bound before it is time-bound

`ReferenceModel(config).to("cuda")` allocates a full fresh 8.4 GB of parameters *before*
`load_state_dict(assign=True)` rebinds them to the reference's and frees the duplicates.
Peak is 16.8 GB of weights for a model needing 8.4.

Harmless once, before anything is compiled. **Fatal in a batch**, where the reference's
compiled state is already resident — every slot of batch 001 died there on a 32 GB card,
tens of MB short. Candidates are now built under `torch.device("meta")`; note that
non-persistent buffers (`rotary_emb.inv_freq`) are absent from a `state_dict` and must be
rebound explicitly or the first forward dies on a meta tensor.

**The general point: adding a hypothesis to a batch adds resident state, not just time.**
Before widening a batch, check the memory headroom, not only the clock.

### A guard's premise can expire (2026-09-10)

The stall guard destroyed a **healthy** rental at the moment its container pull *succeeded*.
It watches `status_msg` and treats a frozen message as a stuck pull — sound, but only while
bytes are still moving. Once the last layer reports `Download complete`, verification and
extraction emit no updates, so a healthy box necessarily goes quiet in exactly the window
before it becomes reachable.

Blocker 7 was a guard firing *before* the thing it guarded was ever reached. This is the
same error inverted: a guard firing *after* that thing had already succeeded. So the
question to ask of any guard is not only "did the thing it protects get reached?" but
**"is the signal it reads still meaningful at the moment it fires?"** The loop already
encoded that reasoning one branch lower — an empty `status_msg` was exempted because
punishing a quiet host "would destroy healthy instances at 300s for the crime of being
quiet" — and the settled-pull case is the same argument with a different cause.

**And put toolchain checks in front of the expensive downloads.** Rental 27 spent 88 billed
minutes — image, torch, a 9.32 GB checkpoint, a full GPU suite — to discover a missing
`python3-dev` that a one-second probe catches. `g++` was already checked early; the headers
now are too. Anything that can fail after a gigabyte has been paid for should be tested
before it.

## 8. Things that will bite you

**A launch is not a rental, and silence is not health.** Rental 29 (2026-09-11) was
announced as "up and monitored" and then reported nothing for a day. It had exited 4
three seconds after launch — no RTX 5090 or 4090 met the filters, so `provision.sh`
refused correctly, created nothing and billed nothing. The report never came because the
monitor was attached with `tail -n 0 -f` *six and a half seconds after the process had
already ended*: `-n 0` discards the backlog, so it waited forever on a file that would
never grow again. Nothing was wrong remotely and nothing was wrong in the scripts. The
observation was wrong, and a correct refusal became a lost day.

Three rules, all enforced by `remote/launch.sh` and its tests in `remote/scripts_test.py`:

- **Launch through `remote/launch.sh`**, so the log always ends with a terminal
  `[deltaforge] [launch] run exited N` marker and the status lands in a file.
- **Attach with `tail -n +1 -F`, never `tail -n 0 -f`.** Replaying from line 1 costs
  nothing and is the whole difference between seeing a fast failure and hanging on one.
- **Never infer a rental from a launch.** "provisioning..." prints *before* the offer
  search. Confirm with `--status` and a live instance before saying a rental is up.

An exit-4 no-offer refusal is a market transient, not a fault: the fix is to widen
`--max-rate` deliberately, or to try again later, not to route around the gate.

**A bare `rm` after a command is not cleanup.** `df_api` writes request bodies to a 0600
temp file so a registry token never reaches argv, where `ps` would show it — but the
removal used to sit on the line *after* curl, which `set -e` walks straight past when the
request fails. Forty of those accumulated in `/tmp`, each holding the Docker token. The
removal is now armed by a trap *before* the request, inside its own subshell: an EXIT
trap set in the function directly would replace `run_remote.sh`'s `trap 'df_teardown'
EXIT` and disarm teardown, turning a leaked temp file into a leaked GPU. Cleanup for
anything sensitive goes on a trap, armed before the thing that can fail.

**Never modify `reference.py` to accommodate a kernel.** It is the definition of the
baseline; changing it invalidates every stored result. `reference_purity_test.py` enforces
by AST inspection that it contains no Triton, no FlashAttention and no
`F.scaled_dot_product_attention` — SDPA *is* one of the kernels under test. Kernels attach
through `deltaforge.model.INSTALLERS`, never by editing the reference. (Fixing the reference
because it *models the checkpoint wrongly* is a different thing and is correct to do.)

Model-level traps, all asserted by tests so CI catches them before the GPU does:

- **`head_dim` is 256, not `hidden_size / num_heads` (160).** The projections are not
  square. Most likely early bug.
- **RMSNorm scales by `1 + weight`** — the checkpoint stores zero-centred norm weights.
  Reading it as plain `weight` gives an all-zero activation and a model that still runs.
- **The gated RMSNorm inside Gated DeltaNet is the opposite**: plain `weight`, and it
  normalises *before* gating.
- **The attention output gate is config-driven.** Qwen3.5 omits `output_gate_type` and means
  sigmoid; Qwen3.8 declares `swish`. Hardcoding either gives a model that runs and emits
  plausible logits on the other checkpoint.
- **RoPE is partial** (64 of 256 dims) and **mRoPE-interleaved**.
- **The recurrent state is fp32** even though weights are bf16.
- **`q_proj` is doubled** for the output gate, and the split is per head, not a split of the
  flat projection.
- **The Gated DeltaNet projection layout differs from Qwen3-Next**: four separate
  projections, no head interleaving to undo.

**Compiling something nothing measures.** `gated_delta_rule` scans the sequence with a
Python `for t in range(seq_len)`, which dynamo unrolls into the graph — ~22 FX nodes per
token per linear-attention layer, measurable on a CPU with the tiny config. At the
benchmark's 2048-token context across 24 such layers that is **~1.08M nodes**, handed to
inductor under `max-autotune`, once per column. Four rentals died there. None of it was ever
timed: `run_interleaved` excludes every `setup` from the measured region by design, and the
prefill is setup. The prefill now runs eager.

The general form: **before optimising a cost, check that anything measures it.** The cheapest
version of that check is free and needs no GPU — a counting `torch.compile` backend that
returns the graph module and reports `len(gm.graph.nodes)`.

**Dynamo stops compiling, and says so only in a warning.** Its `recompile_limit` (default 8)
is per *code object*. Every batch slot compiles a fresh candidate against the same
`ReferenceModel.forward`, so slot N is cache entry N, and past the limit dynamo runs that
code object **eagerly for the rest of the process**. Rental 35 tripped it inside slot 2:
six hypotheses attacking six unrelated operations then returned ratios between 0.146 and
0.157, with tight IQRs, all of them measuring eager against compiled.

Nothing failed. The ratios looked like results. `recompile_limit_for` in `batch_run.py` now
raises the limit to cover the batch, and every slot record carries `graphs_compiled` — a `0`
there means the number beside it is not a comparison. **Widening a batch means widening that
limit too.**

**And so does widening what one slot's candidate does.** `recompile_limit_for`'s budget
assumed two cache entries per slot — a candidate calling the reference's `forward` at one
shape, plus its dynamic variant. A speculative-decoding candidate calls it at *two*: a plain
one-token continuation and a `k+1`-token verify. Batch 010 (2026-09-26) registered four such
slots and sat at 18 entries against an estimated need of 16-20. It now takes
`entries_per_slot` and `_entries_per_slot_for` reads it off the batch, so a decode-loop batch
gets three and every older batch keeps the budget its record was measured under. **Before
trusting a slot's ratio, check what shapes its candidate actually calls, not just how many
slots share the batch** — and note the marker that works: a loop candidate is identified by
its registered `impl`, not by `replaces`, because `replaces` names the operation and fifteen
ordinary module-swap kernels name `decode_step` too.

**"The flop waste is free" is an argument about a memory-bound kernel, and it only holds if
the kernel is one.** Batch 004's GEMV padded M to 16 so `tl.dot` could carry the partial sums
in the MMA accumulator, on the stated ground that arithmetic intensity at batch-1 decode is
~2 flop/byte against a machine balance near 150 — so 15 wasted rows of every 16 cost nothing.
That is true of a kernel at the memory wall. This one achieved **228 GB/s against a baseline
at 1177**, so whatever was binding it, bandwidth was not, and the padding multiplied the work
on it. The rewrite came out **1.40× slower than the naive kernel it replaced.** Before
trading flops for a structural win, check which resource the kernel is actually spending its
time on — and if you do not know, the ablation is a slot, not a rewrite.

**You are not competing with cuBLAS, and you are not competing with a bare matmul.**
`extern_kernels` in the compiled decode graph is called for convolution and nothing else:
under `max-autotune` inductor generates Triton for **every** matmul here. And those kernels
are fused — the one for `in_proj_a`/`in_proj_b` does the residual add, the RMSNorm and *both*
projections in one pass over the hidden state. A hand-written GEMV replaces the matmul alone,
so the norm and residual become separate kernels again and two launches replace one, at
**248 projection sites per decode step**. Read the dump before costing a kernel: the fusion
you are giving up is not in any roofline.

**Match a config to a manifest by shape, never by `name`.** `from_hf_config` sets `name` from
the checkpoint's `model_type` (`'qwen3_5'`), not from the transcription in `MODELS`
(`'qwen3.5-4b'`). A name-keyed lookup therefore resolves exactly the configs built by calling
a factory — every CPU test — and fails on every config a rental actually has. Batch 004
shipped no achieved bandwidth at all for this, which was the one feature it had been built to
add. The guard held: it reported nothing rather than another checkpoint's byte count.

**A gate is only a gate if it can resolve the thing it measures.** Batch 003 failed four
slots that were working correctly, on a top-1 agreement bar of 0.98 over **264** teacher-forced
positions. Agreement there quantises to 1/264 = 0.0038, and `013-int8-full` missed its bar by
**0.000303 — eight hundredths of one token**. A threshold an order of magnitude finer than one
sample is not a decision procedure. The binomial 95% CI on the measured 8/264 is roughly
[1.3%, 5.9%], so the bar and the measurement were never distinguishable.

Two rules follow. **Prefer a continuous statistic**: mean KL has no resolution floor, behaved
perfectly across int8 (0.0011 nats) and int4 (0.0919), and would have passed every slot that
deserved to pass. **And derive a bar from something this repository has measured**, not from
general knowledge — batch 003's 0.98 came from priors about int8 being mild. It is mild; it
still flips 3% of argmaxes on this checkpoint, because the top-2 logit gaps are narrow, and
`test_report_the_first_greedy_divergence` already computes exactly that distribution for free.

**Do not gate a kernel `exact` unless it is bit-identical by construction.** `009` computes
the same *function* as the reference — bf16 in, fp32 accumulate, bf16 out — and matched 1 of
5 prompts. Summing K in a different order from cuBLAS lands one bf16 ULP away, and one ULP
flips an argmax on this model. Computing the same function and producing the same bits are
different properties, and only the identity champion has the second one. This is the third
time the project has paid for it (§7a, rental 35's slots 001 and 002, now `009`).

**The corollary for a gate that measures a gap rather than equality: size the ceiling to the
dtype, or the gate cannot pass anything.** `oracle_test.py:264` puts 1-2 bf16 ULP on this
checkpoint at `0.28125` — that is the smallest logit gap a rounding difference can produce,
not a rare one. Speculative decoding's `divergence_gap_ceiling` was drafted at `0.02`, about
1/14th of that, which made its `SequenceCheck.passed` false for every divergence the
technique can produce and turned a gate built to distinguish "reduction order" from "bug"
into the exact-equality gate it exists to replace. Caught in review, before any rental;
shipped at `0.3`. Before setting *any* tolerance ceiling on this checkpoint's bf16 outputs,
compute the dtype's ULP at the relevant magnitude first — the same arithmetic that makes an
`exact` gate wrong makes an under-sized approximate one wrong too.

**A shared correctness reference is only shared if the implementations share their rounding.**
`010`'s layer 1 was the batch's only layer-1 failure and it is not a kernel: `Int8DequantLinear`
rounds the dequantised weight to bf16 before the matmul, as any real PyTorch implementation
would, while the reference both it and `012` were checked against dequantises in fp32. The
check reported a relative error of 0.45 for a computation doing exactly what it should.

**Removing bytes from a kernel that is not memory-bound makes it slower, and the control is
the only thing that can tell you which you have.** Batch 003's kernels got slower as they
moved less: 26.90 → 38.68 → 42.49 ms/token for 9158 → 5588 → 2850 MB/token, achieving 332,
141 and 65 GB/s against the compiled baseline's **1308 GB/s, 73% of peak**. The cause was in
the source all along — `acc += tl.sum(w * x[None, :], axis=1)` is a cross-lane reduction run
*once per K-iteration*, 20 times for K=2560 and 72 for K=9216, where the standard form
accumulates a tile and reduces once.

Without `009-gemv-bf16-control` — the same kernel on unquantised weights, moving exactly
cuBLAS's bytes — the honest reading of `012` at 0.1962 would have been "weight-only
quantisation does not help", which is false and would have closed the backlog's best
hypothesis for the wrong reason. **Put the mechanism's control in the batch, and put it
early**: five of batch 003's seven slots were determined the moment `009` returned 0.2801,
and nothing in the framework could act on that. See `docs/BATCHES.md` on conditional slots.

**A declined slot is an unexecuted code path, not a verified one.** Batch 004's
preconditions declined five slots and saved 16 billed minutes, which was right. What came
with the saving is that `_tiled_gemv_scaled_kernel` shipped, passed CI and review, and had
**never run** — so batch 005 discovered on a rented box that it declared `SCALE` and
`HAS_SCALE` and read neither, returning unscaled int8 dot products at a layer-1 relative
error of **4511**. The same kernel's `other=0` will not cast to e4m3, so the fp8 slot did
not compile at all. **A Triton body is a string a GPU compiles; the CPU suite cannot see
into it**, and `kernels/kernel_contract_test.py` now checks what the AST *can* see — a
parameter declared and never read, and a dtype-polymorphic masked load whose `other` only
one dtype accepts. When a batch picks up a kernel an earlier batch declined, treat it as
new code.

**A candidate that replaces the root model class pays a full `max-autotune` recompile.**
The identity slot's candidate reuses the reference's compiled code and its first warmup
round takes ~860 ms; on rental 40 `021`'s took **71.7 s** and `025`'s **218 s**, because a
new class is a new dynamo code object. Both rounds are discarded warmup so no ratio is
affected — but it is 1-4 minutes a slot, it is not in `COLD_PHASE_ESTIMATES`, and it is
easy to mistake for a hang.

**Two installers that both swap the root class will silently discard each other.** The
tiled LM head and the static decode cache each work by replacing `type(model)` with a
`ReferenceModel` subclass. Anchored at `ReferenceModel`, whichever ran second dropped the
first — and nothing would have caught it: `_build_candidate` asks only whether *any* module
class changed, which is still true, so a composed slot would compile, pass correctness and
return a plausible ratio for a candidate holding one of the two kernels it claims. Both
factories now subclass whatever the model already is, keyed by base. **Before composing two
installers, check what each one actually replaces** — two that patch disjoint submodules
compose for free, and two that patch the same attribute do not.

**A micro-benchmark of one site does not rank tiles for the step that site is part of.**
Batch 006 built an install-time autotuner — the honest answer to competing with
`max-autotune`, which measures its tile where we were deriving ours from a comment about
SM counts. It timed each site in a loop on an idle card with L2 flushed between
iterations, and on the champion's own site it reported **1639 GB/s, faster than the whole
compiled model achieves**, then chose BLOCK_N=256. That tile ran at **282 GB/s in the
decode step** where the heuristic's BLOCK_N=64 had run at 656: the same kernel, the same
bytes, **2.3x slower**, and the slot fell from 1.0791 to 0.9920.

The direction is the tell. Fewer, fatter programs win a tight loop over one kernel on an
idle card, where launch and scheduling dominate; they lose a step where 507 other kernels
have already shaped the cache and the clock. **A tuner is only as good as the thing it
times**, and the thing worth timing here is the decode step, not the site. `DF_TILE_TUNE`
defaults off; the search space and the rounds are kept for a tuner that measures the right
thing. A plausibility guard (`IMPLAUSIBLE_GBPS`) now refuses a measurement implying more
bandwidth than exists — and note that it would **not** have caught this one, because 1639
GB/s is possible on paper. An impossible number can be refused by arithmetic; an
unrepresentative one cannot.

**The hand-written kernel can lose to the compiler on the same program, and at the head it
did.** `054-int4-head` (a `custom_op`) measured **0.9851** and `056-int4-head-torch-dequant`
— the identical function written as torch operations — measured **1.0171**, same process,
same weights, layer 2 identical. Two costs compound, and only the first was predicted: the
custom op forfeits the RMSNorm fusion the reference welds into the lm_head matmul, **and**
the register-level nibble unpack it exists for is something inductor emits inline anyway.
The dump is unambiguous — one reduction kernel carrying `__rshift__`, `bitwise_and`, `mm`,
`mean`, `rsqrt`, and **no weight-sized buffer in the graph** against a registered prediction
that 1271.40 MB/token would be materialised.

**So the question to ask of any kernel in this repository is no longer "is it fast?" but
"would inductor write this if I expressed it as torch?"** Both champions here answer that
by not being kernels.

**Dispatch savings do not add.** `057-static-decode-cache` measured **+1.5% alone** and
**+0.0% composed with the fused causal conv** — `058` and `060` returned the same 7.251
ms/token from separate runs, with round-0 compiles of 186 s and 54 s to prove they were
separate. Both mechanisms remove CPU-side dispatch work and they compete for the same
microseconds. Batch 008 reported the cache adding 3% there, inside a band of **0.1511**;
at rental 46's bands it is zero. **A launch census predicts neither the sign nor the size.**

**Resolution is buyable, and it changes answers rather than error bars.** A batch runs
`cli.BATCH_ROUNDS` = 17 rounds (15 scoring) rather than 7. On the same host drifting the
same way, IQRs fell from 0.0074-0.1511 to 0.0072-0.0213 and `inconclusive` slots from six
of eleven to one of ten. **Three of batch 009's findings were unavailable at five rounds.**
Budget a batch round at **~18 s**, not the ~2 s its `median_ms` suggests: `run_interleaved`
excludes `setup` from the timed region and the 2048-token prefill is setup, so most of a
round's wall clock is invisible to the number the bench reports.

**A registration the CPU suite can see is not a registration that traces.** `055` asserted
on a laptop that `torch.ops.deltaforge.tiled_gemv_int4_visible` existed — it did — and died
on the box at trace time, because `torch.library.triton_op` runs the body under
`FakeTensorMode` to build its fake implementation and our Triton launch reached
`.data_ptr()` instead of being intercepted by `wrap_triton`. Tracing needs Triton, so no
CPU test could have caught it. **The cheap fix is to `torch.compile` each newly registered
module for one step inside the dump step that already runs**, before the batch spends a
slot on it.

**A precondition must name the proposition its slot depends on.** `061-int4-mlp-torch-dequant`
— 52.75% of per-token bytes, the largest prize in the backlog — declined because its
ingredient returned **1.0171 against a floor of 1.02**. The floor's stated purpose was "has
the compiler shown it can fuse a grouped dequantisation at all", which the dump answers yes
and which `056` against `054` answers at **+3.2%**. The floor was well-resolved and about
the wrong quantity. Batch 003's bar was finer than its statistic; this one measured
something else entirely. **Where the proposition is a comparison, the floor belongs on the
comparison** — and `batch.Precondition` can only express "slug ≥ float", which is now a
known limitation rather than an accident.

**The cost of an opaque custom op is 21% of the whole decode step, and the fix is to stop
writing one.** `044-fused-causal-conv` and `045-inline-causal-conv` compute **the same
function** — the record proves it rather than asserting it: both are bit-identical to the
reference, layer 1 at relative error `0.0` and layer 2 at 264/264 and 0.00000 nats. One
arrives as a `torch.library.custom_op`; the other as four torch multiplies. They measured
**0.7854 and 1.0765** in one process on one card. The dump says exactly what differs: the
inline candidate removes all 24 `extern_kernels.convolution` calls and adds **nothing** —
the pointwise launch count is unchanged at 89, because inductor folded the taps into
kernels that already existed — and runs **297 reductions against the reference's 298**,
where the custom op needed **321**.

So before wrapping a kernel in a custom op, ask whether the operation is *pointwise enough
that inductor would simply fuse it* — and if it is, write the torch and let it. The
corollary is uncomfortable and is the next thing worth a slot: **`tiled_gemv_int4`, the
champion of `decode_step`, is itself a custom op**, and has never been measured in a form
the scheduler can see into.

**And the allocation count is not the mechanism.** Rental 43 ranked 59 → 190 allocations
beside a 19% loss. The winning inline candidate allocates **232**. It was correlated; the
duplicated reduction was causal.

**"They do not compose" is a claim that needs the ingredients measured, and three rentals
made it without them.** Rentals 42 and 43 attributed a 20% loss to an interaction between
the int4 head and the fused conv, and batch 007's writeup called it the largest unexplained
number in the project. **There is no interaction.** The conv alone measures 0.7854 and the
pair 0.7963 — +0.011 apart, inside both IQRs. The three-minute slot that settles it is the
ingredient, placed *before* the composition. `batches_test` now asserts that structurally:
every kernel in a composed slot must be measured alone, earlier in the same batch.

**A card can pass the pre-flight and then downclock.** Rental 45's SM clock fell **2910 →
2400 MHz at slot 4** and stayed there; the memory clock never moved and the reference
column drifted **7.17 → 7.92 ms/token inside one rental**. Interleaved rounds divide that
out of every ratio, so no slot is void — what it costs is **resolution**. IQRs ran 0.0074
to 0.1511 against rental 40's 0.00034, and **six of eleven slots came back `inconclusive`
on effects that are probably real**. `card_baseline.card_report` cannot catch this: a
pre-flight tests the card you were given, not the card you will still have in forty
minutes. The instrument that answers it is more scoring rounds when the IQR is wide.

**Read an `inconclusive` as "this rental could not resolve it", never as "this is zero".**
`051` is the worked example: its candidate rounds are tight (921-985 ms) and its
*reference* rounds carry 1110 and 1179 ms outliers, which is what produced an IQR of
0.1511. `052` ran the same candidate three minutes later against a clean reference and
returned **1.0747 at an IQR of 0.0127**.

**The compile cache now costs more than it saves, and its cost grows while its saving does
not.** Rental 45 spent **~33 of its 115 billed minutes pushing 2.2 GB** at ~1.05 MB/s to
buy a warm compile worth **268 s cold → 57 s warm**, and the teardown pull then timed out
so nothing came home. A warm compile saves the same 3.5 minutes whether the directory holds
200 MB or 2.2 GB. `run_remote.sh` now refuses a push above `DF_CACHE_MAX_PUSH_MB` (512).
**Watch the fixed cost, not only the slot count**: 72 minutes of fixed cost bought 39
minutes of measurement on that rental, the worst ratio recorded here.

**~~Two kernels that each win alone can lose badly together~~ — RETRACTED on rental 45.
They never did, and the reason it looked that way is that nobody measured the ingredient.**
`029` (rental 42, 0.7937) and `039` (rental 43, 0.8111) composed the int4 head with the
fused causal conv, and two writeups reasoned about how the two installers interfere. Batch
008 measured the conv **alone** on the same card in the same process: **0.7854**, against
**0.7963** for the pair — +0.011 apart, inside both IQRs. **The conv simply loses 21% on
its own**, the head adds nothing to that and subtracts nothing from it, and there is no
interaction to explain. The surviving instruction is the one below about ingredients, and
it is now asserted rather than remembered:
`batches_test.test_every_ingredient_of_every_008_composition_is_measured_alone_first`
requires every kernel in a composed slot to have a single-kernel slot earlier in the same
batch. Measure a composition, never infer it — and **measure its parts first, in the same
process**, or the composition's number is not attributable to the composition.

**`graphs_compiled: 0` does not mean the candidate ran eager.** The counter is a delta on
dynamo's `unique_graphs`, so a **guard-passing cache hit scores zero** — and after a slot
that compiled an identical graph, zero is the *healthy* value. Batch 007's two pinned-tile
slots both recorded it and both tripped the "nothing compiled, this is eager against
compiled" warning; both were fine. The tile is written into `_TUNED` at install time and
read as a launch parameter, so it is not in the traced graph and dynamo correctly reused
the previous slot's. **The evidence that separates the two cases is already recorded:**
round 0 of a genuinely compiling candidate takes 60-70 s (rental 43: 70437 ms and 63639 ms)
where a cache hit takes the same ~1.5 s as every other round; and rental 35's real fallback
ran candidates **6.8x slow at ratios of 0.146-0.157**, not within 7% of the reference. The
counter is still worth keeping — it caught rental 35 — but read the round-0 time before
believing its warning.

**A composition slot is worth nothing unless its ingredients were measured in the same
process.** Batch 007 existed because rental 42 carried `022`'s number across sessions. It
fixed that for the head — `035` re-measured the champion and is the reason the rest of the
batch is readable — and then **made the identical mistake with the conv**, carrying `025`'s
1.0144 from rental 40 and never re-measuring the conv alone. So `039`'s 0.8111 cannot
separate "this composition is bad" from "the conv is bad on this card", on a rental where
every other mechanism measured zero. Both numbers were equally one rental old; only one was
under suspicion, and suspicion is not the criterion. **If a slot composes N mechanisms, the
batch needs all N measured alone that day, or the composition's number means nothing.**

**Batch 008 enforced it and it paid immediately**: the conv alone measured 0.7854, which
retracted a composition effect three rentals had been reasoning about, and cost one
three-minute slot placed before the composition instead of after it.

**Install-time state in a module global outlives the slot that set it.** `scoped_registry`
exists because the kernel registry is process-wide and a batch runs every slot in one
process; `_TUNED` in `tiled_gemv.py` is the same shape one level down. It maps
``(kind, N, K)`` to a tile, it is written at install time, and batch 007 pins tiles there
deliberately -- so a slot that pinned BLOCK_N=128 would leave it in force for the next
slot, which would then report a ratio for a candidate its manifest does not describe.
`_install_head` now *clears* that key when no tile is named, and a CPU test asserts that an
unpinned slot after a pinned one gets the heuristic back. **Before putting anything at
module scope that an installer writes, ask what the slot after this one reads.**

**A slot can win while its mechanism never fires, and only a counter can tell you.**
`034-static-cache-cudagraphs` returned **1.0196, IQR 0.00149**, bit-identical, with
`cudagraph_nodes: 0`. The hypothesis it was built for — CUDA-graphing the decode step — is
still untested at 0 for 2. The 2.0% came from the *other* thing `mark_static_address`
does: a static input skips inductor's per-call alignment check, 64 tensors on each of 128
steps. Reporting that as "CUDA graphs are worth 2%" would have been false in both
directions, and the only reason it is not in this file as a finding is that the slot
record carries the node count beside the ratio.

**And record the reason, not only the counter.** Batch 005 had `cudagraph_skips: 127` and
spent a section of its writeup ranking three suspects it could not separate;
`cudagraphs_during` now captures the message inductor logs beside the counter, which named
`mutated inputs (64 instances)` with all 64 tensors marked and printed
`Recording cudagraph tree for symint key 2049, 2050, …` — confirming both halves of the
diagnosis in one run. **When a guard fires, log what it read, not just that it fired.**

**Two of batch 005's ranked suspects were refuted on a laptop, for nothing.** Running the
tiny config under `TORCH_LOGS=cudagraph_static_inputs` prints `Adding static input pos 5
for source L['cache'].layers[0].conv`, which settles that `mark_static_address` reaches
`static_input_indices` through a plain Python object. Before booking a rental to
investigate a dynamo or inductor behaviour, **check whether the tiny CPU config plus a
`TORCH_LOGS` artifact already answers it.** Most of the compile-time machinery is
device-independent.

**A byte model that a rental has corrected still has to be corrected in the code.** Rental
38 established that the compiled columns move 8587.80 MB/token rather than the roofline's
9158.23, and `LEADERBOARD.md` and `docs/HYPOTHESES.md` were fixed. `decode_bytes_per_token`
was not, and it is what the bench divides time by — so the feature built to stop computing
GB/s by hand would have shipped every achieved bandwidth 6.6% low. `traffic` still carries
the row, because eager really does perform that copy and `docs/roofline.py` documents eager;
`decode_bytes_per_token` defaults to the compiled total and takes `compiled=False` for the
other. **Fixing the prose and leaving the arithmetic is half a fix**, and the half that is
left is the one a rental reads.

**A reclaim can have a premise that was never true.** `release_compiled_state` called
`reset_cudagraph_trees` between slots, its docstring asserting the reference would
"re-record on the next slot's first warmup call". It does not: the shutdown is permanent for
an already-recorded callable, and inductor's trees are per *device*, not per model. Rental 34
calibrated in slot 0 and then lost all eight scoring slots to
`AssertionError: Running CUDAGraph after shutdown`, 23s each.

It stayed invisible for 33 rentals because it can only fire *between* two completed slots,
and until rental 34 no slot had ever completed. Blocker 7 was a guard firing before its
subject was reached; blocker 9 one firing after its subject had succeeded; this is a third
shape — **a cleanup whose stated justification had never been tested at all.** When a
comment explains why something is safe, check whether anything ever exercised it.

## 9. Environments — the one thing that surprises people

**Local and CI (CPU).** `pyproject.toml` pins torch to PyTorch's **CPU index** via
`[tool.uv.sources]`, because the acceptance bar is that `uv run pytest` passes on a clean
CPU-only checkout and the default PyPI wheel drags ~3 GB of `nvidia-*` packages in.

**The rented box (GPU).** `run_remote.sh` installs torch from PyTorch's CUDA CDN and then
`pip install --no-deps -e .` on top, so the CPU pin can never downgrade the box.

`uv.lock` is gitignored: one lock file cannot serve both environments.

Anything needing a GPU or the checkpoint carries the `gpu` and `weights` pytest markers and
**skips on a real condition** — no CUDA, or no checkpoint — rather than being faked, so it
starts running by itself once both are present.

## 10. Maintaining this file

This is project memory, not a changelog. Add something only when it is durable knowledge
almost every future session needs and the code does not already show for itself. Prefer a
pointer to the authoritative file, command or test over copying a detail that will drift. If
something here turns out to be wrong, fix it in the same pass as the work that revealed it.
