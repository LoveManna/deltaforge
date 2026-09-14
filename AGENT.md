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
Also confirm the baseline actually compiled — a silent graph break drags `compiled` toward
eager and inflates every ratio in your favour, which is the first thing a sceptical reader
checks.

## 4. A session, start to finish

**A rental measures a batch of 7-12 hypotheses, not one.** The fixed cost of a rental —
container image, torch, a 9.32 GB checkpoint, the GPU suite, and one `max-autotune` compile
of the reference — is about 15 minutes. Each additional hypothesis costs 2-4. Testing one
per rental pays that 15 minutes to buy a single measurement, and nine rentals were billed
that way without producing a number. `docs/BATCHES.md` has the arithmetic and the workflow;
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

**Then know what a compile costs.** The 15-minutes-fixed + 2-4-per-slot split in §4 and
`docs/BATCHES.md` is an *estimate that has never been measured*, and the only evidence so
far contradicts it: rental 22 spent ~40 minutes in one slot's cold `max-autotune` compile
and the session gate ended the run. Until that number is known, a batch's size is a guess.
`002-compile-cost` — three slots whose product is the clock — is written and has **never
started**: five rentals on 2026-09-10 all died before the batch loop. See
`results/batches/001-calibration/README.md` and
`results/batches/002-compile-cost/README.md`.

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
would have been not renting. The estimate comes from the last rental on the same card when
there is one (`cache/compile/<key>/phases.env`) and from the deliberately pessimistic cold
numbers in `batch.COLD_PHASE_ESTIMATES` when there is not.

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
**Thirty-two rentals have been billed on this project and none has produced a number.**
The early ones died on an anonymous Docker Hub pull and the blockers behind it; those are
fixed. As of 2026-09-13 the failure has moved all the way to the end: the oracle gate
passes on three independent hosts, correctness gates pass, memory is flat — and slot 0
still cannot finish a cold `max-autotune` compile inside a session. `docs/GPU-ACCESS.md`
records every blocker, how each was fixed, and which fixes are *proven on a GPU* rather
than merely believed.

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

Tests are colocated as `<module>_test.py`; `testpaths` is `src` and `remote`.

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
