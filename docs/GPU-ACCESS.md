# GPU access: a chain of blockers, each hiding behind the last

Thirty-seven rentals have been billed on this project, **$6.157 lifetime, zero leaked.**
**The chain is finished.** The harness has been calibrated since rental 34, rental 35 ran a
full batch, and rental 37 closed the last blocker in the table: batch 003 returned **seven
admissible ratios with nothing voided**. What stands between this project and a champion is
now a kernel, not a blocker. Each rental that got further than its predecessor did so by
exposing the next problem in the chain:

| | Blocker | Found by | Fixed by | Proven? |
|---|---|---|---|---|
| 1 | Anonymous Docker Hub pulls stall from vast egress ranges | rentals 1-8 | registry credentials — **configured, and proven on rental 28** | yes |
| 2 | The image refuses the account's ssh key | rental 9 | injecting the key via `PUBLIC_KEY` **and** `onstart` | yes |
| 3 | The image has `python3` but no `python` | rental 10 | establishing the interpreter first, and `python -m pip` | yes |
| 4 | `accelerate` absent, so the HF oracle cannot be constructed | rental 13 | installing it, and verifying the import out loud | yes |
| 5 | Four `oracle_test.py` bounds were fp32 absolutes on bf16 tensors | rental 14 | scoring relative to the tensor's own scale | yes |
| 6 | Candidate construction double-allocates 8.4 GB of weights | rental 16 | building candidates on `torch.device("meta")` | yes |
| 7 | Readiness read `cur_state` (the rental contract) instead of `actual_status` (the container) | rentals 18-20 | gating on `actual_status` alone | yes |
| 8 | The benchmark OOMs at warmup with the default four columns | rental 21 | `--columns compiled,candidate_compiled` | yes — nine slots on rental 35 never passed 8.07 GiB of 31.36 |
| 9 | **The stall guard destroys instances whose pull has just finished** | rental 24 | `df_pull_settled`: a settled pull is bounded by the readiness timeout, not the stall budget | yes |
| 10 | A phantom ask traps the deterministic offer search | 3 refused creates, 2026-09-10 | the create step walks the N cheapest candidates instead of dying on `.[0]` | tests; exercised on rental 28, not stressed |
| 11 | Host driver older than the torch build → CUDA `Error 804` | rental 26 | `DF_MIN_CUDA` (12.8), matching the cu128 index | **no — recurred on rental 33** |
| 12 | The ghcr image ships no Python headers, so Triton's JIT shim will not build | rental 27 | installing `python3-dev` beside `g++` | yes — the GPU suite passed on rentals 34 and 35 |
| 13 | The oracle's greedy decode no longer matches HuggingFace | rental 27 | resolved without a named cause — agrees on rentals 30-32, 34, 35 | yes, five hosts, zero tie-breaks |
| 14 | A cold `max-autotune` compile never finishes inside a session | rentals 22, 30-32 | the unrolled prefill scan is not compiled; it is untimed setup | yes — 6980.9s unfinished → 267.5s → 57.4s warm |
| 15 | `reset_cudagraph_trees` between slots tears down the reference columns | rental 34 | stop resetting the trees; the pool it reclaimed was unmeasurable | yes — rental 35 ran all nine slots |
| 16 | Dynamo's `recompile_limit` (8) silently makes a batch time **eager** candidates | rental 35 | `recompile_limit_for`, plus `graphs_compiled` in every slot record | **yes** — rental 37 |
| 17 | The repo sync ships the 1.4 GB compile cache, and has no retry when that drops | rental 39 | `--exclude=cache`, plus `df_retry` and `rsync --partial` | **yes — rental 40 synced clean and its fixed cost fell from ~30 min to ~13** |
| 18 | **A host can run the reference 1.61x slow with nothing in the environment to show it** | rental 43 | `card_baseline.card_report` — the identity slot's achieved bandwidth against every rental that measured this GPU model, said in the run log at slot 0 | **partly — it reported correctly on rental 45 (1197 GB/s, "In family"), and it cannot see blocker 19** |
| 19 | **A card can pass the pre-flight and then downclock mid-rental** | rental 45 | nothing — ratios survive it, resolution does not | **no.** SM clock fell 2910 → 2400 MHz at slot 4 and held; the reference drifted 7.17 → 7.92 ms/token and six of eleven slots came back `inconclusive` |
| 20 | **The compile-cache push costs an order of magnitude more than the compile it saves** | rental 45 | `DF_CACHE_MAX_PUSH_MB` (512) refuses an oversized push and compiles cold instead | **yes — rental 46, same host, fixed cost ~72 min → ~22** |
| — | ~~Some hosts never answer sshd at all~~ **Withdrawn — this was blocker 7** | rentals 11, 17 | — | n/a |

Blockers 1-9 and 12-15 are fixed and proven on a GPU. **Blocker 11 regressed** — it was
recorded as proven on rental 27 and rental 33 disproved it. **Blocker 18 is open and
unmitigated**, and it is the first one here that costs *correctness of conclusions* rather
than a rental.

### Blocker 18 — the card is an uncontrolled variable the size of the effect

Rental 43 rented an RTX 5090 that ran the reference at **10.73 ms/token and 800 GB/s**
where rental 40's ran at **6.70 and 1282**. Everything the environment record captures says
the two cards are the same:

| | rental 40 | rental 42 | **rental 43** |
|---|---:|---:|---:|
| reference ms/token | **6.70** | 7.17 | **10.73** |
| reference achieved | **1282 GB/s** | 1198 GB/s | **800 GB/s** |
| memory clock | 13801 MHz | 13801 MHz | **13801 MHz** |
| SM clock | 2827 MHz | 2902 MHz | **2955 MHz** (highest of the three) |
| driver | 580.159.03 | 580.159.03 | 580.159.04 |
| torch / triton | 2.11.0+cu128 / 3.6.0 | same | same |
| host platform | `5.15.0-181-generic` | `5.15.0-185-generic` | **`6.10.0-hiveos`** |
| host reliability | — | — | 0.9808 (lowest accepted) |

**Nothing here is a diagnosis.** A persistent power limit on a mining host is the obvious
suspect and this rental captured no power telemetry to test it with. The clocks are read
at capture time, unlocked, so they rule out little. What is established is the consequence:
on that card **the int4 head and the static decode cache both measured zero**, against
+7.91% and +1.96% on their own rentals, and the identity champion itself carried +1.01%
with an IQR of 0.0109 — ten times any previous rental's.

**What to do about it, cheapest first.**

1. **Capture power.** `nvidia-smi --query-gpu=power.draw,power.limit,enforced.power.limit,clocks_throttle_reasons.active`
   into the environment block costs one command and would test the suspect outright.
2. **Say it out loud at slot 0.** `000-identity` already reports the reference column's
   achieved bandwidth before any kernel runs — 845 GB/s here. Comparing that against what
   this GPU model has recorded before, and logging it loudly, costs nothing.
3. **Probably do not abort.** A slow card still produces valid within-slot ratios, and
   rental 43's most valuable result (`039` and its dump) came off this one. The failure
   was not renting it; it was reading absolute predictions derived from another card's
   clock as though they transferred.

**Blocker 16 is closed.** It was recorded here as "the only thing between this project and
its first admissible ratio", and that was right: rental 37 raised the limit to 22 for a
seven-slot batch, every slot reported `graphs_compiled: 3`, and all seven ratios are
compiled-against-compiled. Batch 003 is the first batch in this project with no voided slot.

The counter is the durable part of that fix, not the limit. A raised limit stops *this*
failure; `graphs_compiled` in every slot record is what makes the next one visible, because
a candidate dynamo has stopped compiling still produces a median, an IQR and a ratio that
look like results. **Widening a batch means widening the limit** — `recompile_limit_for`
derives it from the slot count, so that happens automatically, but a batch that ever reports
a `0` there must say so rather than report the number beside it.

Blockers 14, 15 and 16 arrived in that order on one day, and could not have arrived in any
other: 14 stopped any slot from finishing, which hid 15 (it only fires *between* two slots),
and 15 emptied the batch after slot 0, which hid 16 (it only fires once several candidates
have compiled). Each rental bought exactly one layer.

**Blocker 10 now has a real fix rather than a manual escape.** `select_offers` emits the
`DF_OFFER_CANDIDATES` cheapest offers (default 5) instead of only `.[0]`, and the create
step walks them, treating a refusal as "try the next" rather than as fatal. A refusal costs
nothing — no instance exists, so nothing is billed — which is what makes walking strictly
better than dying and making a human pass `--exclude-machines`. Two tests stand up a stub
API: one refuses the two cheapest asks and asserts the third is created and is the offer
the ledger names; the other refuses everything and asserts exit 4, an explanation, and no
ledger row. Neither needs a GPU, so "proven" here means proven in CI.

**Blocker 13 is closed, and its cause was never found.** The oracle has agreed
token-for-token on five independent hosts since (rentals 30, 31, 32, 34, 35) with
`tie_break_count: 0`. What follows is the investigation as it stood, kept because the
refutation in it is the part worth reading.

**Blocker 13's leading explanation was tested on rental 28 and refuted.**

`transformers` was installed as `>=5.16,<6` — a floor, not a pin — and the oracle *is*
HuggingFace, so its version is an input to the experiment rather than a dependency of it.
That is a real defect and it is now pinned. It was not the cause.

The arithmetic that looked damning does not survive contact with the dates. 5.17.0 was
released **2026-09-09**, so the floor resolved to **5.16.1 on 2026-09-07** — the day the
test passed — and to 5.17.0 only on 2026-09-10. Rental 28 installed 5.16.1 explicitly,
verified it in the log, and got:

```
At index 2 diff: 11540 != 1528
```

**Byte-identical to rental 27**, which ran 5.17.0 on different hardware. Neither the
library version nor the card moves the result by one token.

What is left after that is a short list, and everything on it is constant:

| input | 2026-09-06 (passed) | 2026-09-12 (failed) |
|---|---|---|
| `reference.py`, `model.py`, `config.py`, `weights.py` | unchanged since 2026-09-06 | identical |
| `oracle_test.py` and its prompt | unchanged since 2026-09-06 | identical |
| `transformers` | 5.16.1 | 5.16.1 |
| checkpoint `Qwen/Qwen3.5-4B` | last modified 2026-03-02 | identical |
| torch / triton / python | 2.11.0+cu128 / 3.6.0 / 3.12.3 | identical |
| GPU | RTX 5090 | RTX 4090 (rental 27's 5090 diverged identically) |

So the question is no longer what moved. It is **whether exact token equality over 32
sequential bf16 argmaxes was ever a gate that could hold** — the same question commit
`5722aaa` answered "no" for three other bounds in this same file on 2026-09-06, while
deliberately leaving this one alone until the oracle could adjudicate.

`test_report_the_first_greedy_divergence` now measures the deciding number and writes it to
`results/diagnostics/oracle-divergence.json`, which teardown pulls home. It asserts nothing
and changes no gate. See `results/batches/001-calibration/README.md` for what the number
means either way. **Do not promote a reading before it arrives.**

**Blocker 9 is the one to internalise, because it was self-inflicted.** The guard fired on
a healthy box at the exact moment its pull *succeeded*. See
`results/batches/002-compile-cost/README.md` for the full account: it is blocker 7's lesson
inverted, and the loop already contained the correct reasoning one branch lower.

This file records each one so a future session spends its money on kernels rather than
rediscovering them.

Read it if `run_remote.sh` hangs at `waiting for instance to start` or
`instance is running; waiting for sshd`, or dies with `command not found`.

**The pattern worth internalising:** every one of these was cheap to fix and expensive to
*notice*. Two of the three announced themselves only as a stall or a warning. Anything on a
rented box that can fail quietly needs something that says so out loud.

---

## The symptom

```
[deltaforge] created instance 49706056
[deltaforge] waiting for instance to start (status: loading | Pulling from pytorch/pytorch)
[deltaforge] waiting for instance to start (status: loading | Pulling fs layer)
[deltaforge] waiting for instance to start (status: loading | Pulling fs layer)
... for twenty minutes, then the readiness timeout, then teardown
```

Provisioning works. The offer filter works. The API key works. The instance is created,
billed, and destroyed cleanly. It simply never becomes usable.

## The evidence that identified it

| # | Date | Image | Registry | Outcome |
|---|---|---|---|---|
| 1–5 | 2026-09-03 | `pytorch/pytorch` (14.1 GB devel) | Docker Hub | never pulled |
| 6 | 2026-09-03 | `pytorch/pytorch` (14.1 GB devel) | Docker Hub | 20 min, still "Pulling" |
| 7 | 2026-09-03 | `pytorch/pytorch` (4.26 GB runtime) | Docker Hub | 20 min, still "Pulling" |
| 8 | 2026-09-03 | `vastai/base-image` (2.5 GB mini) | Docker Hub | "Pulling fs layer", never advanced |

Four machines. Three images. Sizes from 2.5 GB to 14.1 GB. **One registry.**

The earlier writeup called this "two registries" and it was wrong: `pytorch/pytorch` and
`vastai/base-image` are both Docker Hub. Once that is corrected the pattern is unambiguous —
*every failure was an anonymous Docker Hub pull*, and nothing else has ever been tried.

Docker Hub's unauthenticated pull limits are applied per source IP and are aggressive
against datacenter ranges. Vast hosts pull anonymously unless told otherwise, and they share
a small number of egress ranges, which is why four "different" machines behave identically.
Layers that resolve a manifest and then stall at `Pulling fs layer` is that failure's exact
signature.

### The test that confirmed it

Rental 9, 2026-09-05, session `regtest-20260905T154252Z`, instance 49975299 — an RTX 5090
at $0.356/hr, identical filters to the eight failures, **one variable changed**: the image
came from `ghcr.io` instead of Docker Hub.

```
[deltaforge] created instance 49975299
   t+ 44s  actual=loading  | 2741c81b500d: Pull complete
   t+180s  actual=running  | success, running ghcr.io/ai-dock/base-image_v2-cuda-...-22.04/ssh
```

**The image pulled in about three minutes.** Eight Docker Hub attempts across four machines
and three image sizes never got a single layer to "Pull complete"; the first non-Docker-Hub
attempt landed the whole image on the first try. Same account, same credit balance, same
`paid_verified: 0.0`. The registry is the variable that matters.

Billed 4.57 minutes, $0.0271, destroyed cleanly, nothing leaked.

### The theory this replaces

An earlier note blamed the account never having made a payment (`paid_verified: 0.0`,
`has_billing: false`, `billing_creditonly: 1`). **Rental 9 rules it out.** The account state
was unchanged and the pull succeeded, so whatever those fields mean, they do not prevent
pulling an image.

## The fix

Two independent changes, either of which is sufficient. Both are implemented.

### 1. Authenticate the pull (preferred — keeps the image choice free)

A free Docker Hub account raises the pull limit by an order of magnitude. Vast passes
credentials to `docker login` on the host through the create request's `image_login` field.

```sh
# In the gitignored .env at the repo root, alongside VAST_API_KEY:
DOCKER_LOGIN_USER=yourdockerhubusername
DOCKER_LOGIN_TOKEN=dckr_pat_...        # a read-only Personal Access Token, not a password
```

Nothing else changes. `provision.sh` picks them up automatically and logs only the username:

```
[deltaforge] registry credentials loaded for user yourname (token never logged)
```

With no credentials present the change is inert and the create request is byte-identical to
before:

```
[deltaforge] no registry credentials; images will be pulled anonymously
```

**Secret handling.** The token reaches Vast and nothing else. Adding it forced one real
change: request bodies used to go to curl as `--data "$body"`, which puts them in `argv`
where any user on the box can read them with `ps`. Bodies now travel through a `0600`
temp file. `remote/scripts_test.py` asserts both that no `--data` invocation uses the argv
form and that loading credentials never prints the token.

### 2. Pull from a registry that is not Docker Hub

```sh
remote/run_remote.sh --session-id "$SESSION" \
                     --image ghcr.io/ai-dock/base-image:v2-cuda-12.4.1-base-22.04
```

Vast supports `ghcr.io`, `nvcr.io`, `mcr.microsoft.com` and `lscr.io` — 86 of the 2048
public templates on the platform use one (66 `ghcr.io`, 15 `nvcr.io`, 4 `mcr`, 1 `lscr`),
and they carry `use_ssh: true`, so this is a supported path rather than a trick. Note that
`docker.io/...` is still Docker Hub and does not count.

The image needs very little: `sshd`, Python ≥ 3.10, and `apt`. It does **not** need a
preinstalled PyTorch or a matching CUDA toolkit — `run_remote.sh` installs torch from
`download.pytorch.org` (a CDN with no anonymous pull limit) and those wheels ship their own
CUDA runtime. The GPU driver comes from the host.

### 3. Stop paying for a pull that is stuck

The expensive part of this whole episode was not the failure, it was **how long each failure
took to notice**. Every one of the eight rentals ran the full readiness timeout and was
billed for it, because a stalled pull and a slow pull look identical if all you do is wait.

They are distinguishable: a pull that is merely slow keeps rewriting `status_msg` with new
byte counts, and a stuck one repeats the same line forever. `run_remote.sh` now tracks that:

```sh
DF_PULL_STALL_SECONDS=300   # default
```

If `status_msg` has not changed in that long while the instance is still not reachable, the
run dies, the trap destroys the instance, and the error names both remedies above. That
turns a 20-minute $0.12 loss into a ~5-minute $0.03 one, which is what makes testing another
image cheap enough to be worth doing.

## The second blocker, found behind the first

The ghcr.io container pulled, started, and ran sshd — and then **rejected our key**:

```
debug1: Connection established.
debug1: Server host key: ssh-ed25519 SHA256:Vigt/5VlFr3BmEO7fYib22S1/9F46lPmxBZde3qQ5kg
debug1: Offering public key: /home/anthony/.ssh/deltaforge_vast ED25519 ...
root@ssh4.vast.ai: Permission denied (publickey).
```

Vast's API insisted the key was fine — `POST /instances/49975299/ssh/` returned
`"SSH key already associated with instance."` The association is real; the *image* is what
does not honour it. Vast injects the account key in a way its own `vastai/*` images consume,
and an `ghcr.io/ai-dock` image provisions `authorized_keys` by its own convention instead.

This is why the first eight failures hid it: no image ever got far enough to refuse a login.

**Fixed by injecting the key ourselves, two ways, so image family stops mattering:**

* `env: -e PUBLIC_KEY="ssh-ed25519 ..."` — what the ai-dock and RunPod-style images read.
* an `onstart` append to `/root/.ssh/authorized_keys` — covers everything else.

A public key is not a secret, so both are safe to put in a create request.

**And made cheap to detect.** `Permission denied (publickey)` is a permanent failure: the
image is answering the port and refusing us, and waiting cannot fix it. The probe loop now
reads the ssh error and dies immediately instead of paying out the readiness timeout.

## What the stall guard got wrong the first time

Worth recording, because it is the same mistake in miniature. The stall detector was added
to the *outer* readiness poll — and rental 9 never spent a second there. `cur_state` went to
`running` almost immediately, the run entered the *inner* sshd-probe loop, and that loop had
no progress check at all: it would have waited the full 1200 s. The rental was cut short by
hand.

Both loops now share the `DF_PULL_STALL_SECONDS` budget. A guard that covers only the path
you happened to imagine is not a guard.

## What to do next time this happens

1. Read `status_msg` in the log line — it is the only thing that distinguishes the cases.
2. **Stuck on a Docker Hub pull:** credentials are now in `.env` (since 2026-09-11), so
   check the provision log says `registry credentials loaded for user …` before assuming
   the pull was anonymous. If it did and the pull still stalls, `--image` something on
   `ghcr.io` or `nvcr.io`; that path is proven to work.
3. **`refused the ssh key`:** the image honours neither `PUBLIC_KEY` nor `authorized_keys`.
   **Rental 44 (2026-09-23) shows this advice was half right.** It hit exactly this error
   on `vastai/base-image:cuda-12.9-mini-py312-2026-08-28` — the image rentals 34-43 all
   used successfully — so on that evidence it is the **host**, not the image. Excluding
   machine 148117 and relaunching got a working box on the first try, and the batch ran.
   So: exclude the machine and retry *first*; change the image only if a second host
   refuses the same way. The refusal cost 6.23 minutes and $0.0500, which is the cheapest
   diagnostic in this table.
4. Record the machine id in `--exclude-machines` so the offer search does not hand you the
   same host again. This is for a host that *rents and then misbehaves* — a dead ask is
   handled automatically now (blocker 10).
5. **`no_such_ask` / HTTP 400 from the create endpoint:** the offer was listed but is not
   rentable, usually because someone took it between the search and the create. Nothing was
   billed, and **this now resolves itself**: the create step walks up to
   `DF_OFFER_CANDIDATES` offers in price order and only gives up when every one of them
   refuses. A single `offer … refused` warning followed by a successful create is the
   system working, not a fault. If you see the walk exhaust itself, the market is tight —
   raise `--offer-candidates` or widen `--max-rate` (blocker 10).
6. **`Error 804: forward compatibility was attempted on non supported HW`:** the host's
   driver is older than the torch build we install. Forward-compat packages are
   data-centre-only and these are GeForce cards. `DF_MIN_CUDA` filters on the offer's
   advertised `cuda_max_good`, which **is not the same thing as what the driver can run**:
   rental 33 (machine 44927) advertised exactly 12.8, passed the filter, and died on 804
   anyway, while rental 34 advertised 13.0 and was fine. So the floor is a filter on a
   *claim*, and a host sitting exactly on it is the risky case rather than the safe one.
   The cheap response is `--exclude-machines <id>` and move on — the log prints that line
   with the machine id at selection time for exactly this. A real fix would probe the
   driver before paying for torch and a 9.32 GB checkpoint, which is the same lesson
   rental 27 taught about `python3-dev` (blocker 11).
7. **`fatal error: Python.h: No such file or directory`:** the image ships no Python
   headers and Triton cannot build its JIT shim. `python3-dev` is now installed beside
   `g++`; if this recurs, that step failed or ran too late (blocker 12).
8. **The run says it pulled a compile cache — check that one arrived.** `sync.sh` exits 0
   on a failed pull by design, so teardown now decides from what is on disk. A log claiming
   warmth when the directory is empty would misattribute a cold compile, which is the one
   number batch 002 exists to measure.

## What a successful run costs, once it gets through

About 15 GB has to land on every rented box, because nothing persists between sessions:

| | Size | From | At the market median link (775 Mbps) |
|---|---:|---|---|
| Container image | ~2.5 GB | ghcr.io | ~3 min (measured) |
| PyTorch + CUDA libs | ~3 GB | download.pytorch.org | ~1-3 min |
| Model weights | 9.32 GB | HuggingFace | ~2-5 min |
| transformers, tokenizers, pytest | ~0.1 GB | PyPI | <1 min |

Three things were changed to keep that from eating the 60-minute session budget:

* **`huggingface_hub[hf_transfer]`** for the checkpoint. The old path was one HTTP
  connection, sequential, with no resume — an interruption 8 GB into a 9.3 GB shard started
  that shard again. `--no-hf-transfer` still selects it, and a missing library or a failed
  fast download falls back automatically rather than ending a paid run.
* **Shared parameter tensors** between the reference and candidate models. They were loaded
  independently: two 9.3 GB reads from disk in each of the three commands that build models,
  and 16.8 GB resident on a 32 GB card. Now 8.4 GB and one read.
* **Four benchmark columns by default** instead of five. Each `max-autotune` column is a
  full compilation of a 32-layer model, and `compiled_nocudagraphs` stopped being
  load-bearing once the scoring column gained CUDA graphs of its own.

**`transformers` is now pinned** (`>=5.16,<6`). The GPU suite runs the weight-value oracle
through it, and an unpinned version that dropped the Qwen3.5 architecture would kill a run
after every gigabyte above had already been paid for. Confirmed on 2026-09-05 that
`qwen3_5` is a supported architecture.

## The third blocker, found behind the second

Rental 50119910, 2026-09-07 — the first rental in this project's history to get **past
sshd**. The injected key was accepted, the repo synced, `apt` and all three pip installs
succeeded, and then:

```
bash: line 1: python: command not found
```

The vast and ai-dock images ship **`python3` and `pip`, but no `python`.** Every remote
step after the installs used a bare `python`, so the run died there — having already paid
for the container pull, the 3 GB torch download and every pip install. Billed 9.38 minutes,
$0.0557, destroyed cleanly.

**Fixed** by establishing the interpreter as the *first* thing in the environment step
rather than discovering it as the twentieth:

```sh
command -v python >/dev/null 2>&1 || ln -sf "$(command -v python3)" /usr/local/bin/python
python --version
```

and by running every install as `python -m pip` so pip cannot belong to a different
interpreter than the benchmark does. `remote/scripts_test.py` asserts both orderings.

### And a silent one in the same log

```
WARNING: huggingface-hub 1.30.0 does not provide the extra 'hf-transfer'
```

The `hf_transfer` extra was dropped in huggingface-hub 1.30, so
`pip install 'huggingface_hub[hf_transfer]'` **warns and installs nothing** — leaving the
9.32 GB checkpoint on the single-connection, no-resume downloader, on billed wall-clock
time. A warning is not a failure, so nothing would ever have reported this except the
clock. `hf_transfer` is now its own package and the import is checked out loud.

**The lesson, which is the same one as the stall guard's:** every step that can fail
quietly on a rented box needs something that says so. The expensive failures on this
project have not been the loud ones.

## What each rental has cost so far

| # | Date | Got as far as | Billed | Cost |
|---|---|---|---|---:|
| 1-8 | 2026-09-03 | container pull (Docker Hub) | 94.2 min | $0.4783 |
| 9 | 2026-09-05 | container started, sshd refused the key | 4.57 min | $0.0271 |
| 10 | 2026-09-07 | **past sshd**, installs done, no `python` | 9.38 min | $0.0557 |
| 11 | 2026-09-07 | host never answered sshd (stall guard fired) | 5.63 min | $0.0334 |
| 12 | 2026-09-07 | cancelled by hand — same bad host reselected | 0.67 min | $0.0040 |
| 13 | 2026-09-07 | **checkpoint down**, oracle blocked on `accelerate` | 10.70 min | $0.0683 |
| 14 | 2026-09-07 | **oracle ran** — reference matches HF token-for-token | 11.08 min | $0.0707 |
| 15 | 2026-09-07 | one bf16 tolerance left | 10.38 min | $0.0663 |
| 16 | 2026-09-07 | **GPU suite passed; batch ran** — 9 slots, all OOM | 17.35 min | $0.1107 |
| 17 | 2026-09-07 | host never answered sshd | 5.07 min | $0.0324 |
| 18 | 2026-09-08 | blocker 7 — killed while still `loading` | 5.55 min | $0.0354 |
| 19 | 2026-09-08 | blocker 7 again, different machine | 5.42 min | $0.0346 |
| 20 | 2026-09-08 | blocker 7, now logged as `status: loading` | 5.90 min | $0.0382 |
| 21 | 2026-09-08 | **batch ran** — candidates built, 8 OOM at bench, 1 ImportError | 25.14 min | $0.1548 |
| 22 | 2026-09-08 | 2-column batch; ~40 min in one cold max-autotune compile | 56.68 min | $0.3492 |
| 23 | 2026-09-10 | Docker Hub pull refused — blocker 1 recurred on the default image | 8.17 min | $0.0402 |
| 24 | 2026-09-10 | **healthy, and destroyed by our own stall guard** — blocker 9 | 5.65 min | $0.0310 |
| 25 | 2026-09-10 | host reported `GPU error, unable to start instance` | 11.78 min | $0.0687 |
| 26 | 2026-09-10 | **past sshd and torch**, CUDA `Error 804` — blocker 11 | 6.08 min | $0.0353 |
| 27 | 2026-09-10 | **GPU suite ran** — 11 fail on `Python.h`, oracle diverges | 88.55 min | $0.6242 |
| 28 | 2026-09-12 | **authenticated pull worked**; oracle diverges *identically* on a 4090 with the pin | 7.12 min | $0.0478 |
| 29 | 2026-09-13 | refused: no offer met the filters. Reported nothing for a day — blocker 12 | 0 min | $0 |
| 30 | 2026-09-13 | **oracle gate passes**; host dropped ssh mid-slot-0 (exit 255) | 69.72 min | $0.5448 |
| 31 | 2026-09-13 | **root cause found**: 2 slots, `BackendCompilerFailed` from live autograd — blocker 13 | 131.03 min | $0.8069 |
| 32 | 2026-09-13 | crash fixed; slot 0 hit its 6980.9s cap. 47 MB of compile cache came home | 176.80 min | $1.3878 |
| 33 | 2026-09-14 | CUDA `Error 804` on a host advertising exactly 12.8 — blocker 11 recurred | 8.90 min | $0.0602 |
| 34 | 2026-09-14 | **the compile finished (267.5s) and the harness calibrated (1.0009)**; 8 slots lost to blocker 15 | 30.15 min | $0.2192 |
| 35 | 2026-09-14 | **all nine slots ran**; calibrated 1.0018; 6 slots void to blocker 16 | 52.65 min | $0.3829 |
| 36 | 2026-09-16 | cancelled by hand before the batch started | 2.08 min | $0.0142 |
| 37 | 2026-09-16 | **batch 003: seven admissible ratios, nothing voided**, seven losses | 55.55 min | $0.3786 |
| 38 | 2026-09-17 | **batch 004: two slots, five declined**; the `output_code` dump ran | 39.65 min | $0.2883 |
| 39 | 2026-09-19 | ssh dropped mid-repo-sync, before torch — blocker 17 | 3.53 min | $0.0257 |
| 40 | 2026-09-19 | **batch 005: the project's first two wins**, 1.0791 and 1.0144 | 32.32 min | $0.2350 |
| 41 | 2026-09-19 | RTX 4090 advertising exactly 12.8 — CUDA `Error 804`, blocker 11 three for three | 5.13 min | $0.0337 |
| 42 | 2026-09-19 | **batch 006: the tile is not the difference**; `034` won at 1.0196 | 50.82 min | $0.4373 |
| 43 | 2026-09-20 | **batch 007: the champion is card-dependent** — blocker 18 | 42.67 min | $0.3287 |
| 44 | 2026-09-23 | container started, **refused the account ssh key** — blocker 2 recurred on one host | 6.23 min | $0.0500 |
| 45 | 2026-09-23 | **batch 008: eleven slots, a new champion at 1.0765, and a custom op priced at 21%** | 115.52 min | $0.9427 |
| 46 | 2026-09-23 | **batch 009: the compiler beat our kernel at the head**; fixed cost ~22 min after the cache guard | 79.00 min | $0.6446 |

Forty rentals, $6.706, **zero leaked instances** — every one destroyed cleanly by the
trap, including two cancelled mid-flight with SIGTERM.

**Rental 28 is the cheapest informative rental yet**, and worth reading against rental 27.
Both reached the GPU suite and died at the same assertion; 27 cost $0.6242 and 28 cost
$0.0478. The difference is not luck — 27 paid the full readiness timeout and a cold
checkpoint fetch behind a blocker that had not been diagnosed, while 28 pulled an
authenticated image in about a minute and failed fast on a question it had been sent to
ask. A rental that knows what it is testing is an order of magnitude cheaper than one that
is finding out.

Three further attempts on 2026-09-10 were refused by the API before an instance existed
(`no_such_ask`) and cost nothing. They are not rentals and are not counted here, but they
were blocker 10: the price-ordered search was deterministic, so it re-selected the same
dead ask every time until the machine was excluded by hand. Since 2026-09-11 the create
step walks the next candidate instead; rental 28 created on its first candidate, so the
walk has been exercised but never yet stressed by a refusal on a live market.

**Rental 27 is the most expensive single rental this project has run**, and worth reading
as a cost lesson rather than a failure: 88 billed minutes bought the image, torch, a 9.32 GB
checkpoint and a full GPU suite, and the thing that stopped it — a missing `python3-dev` —
would have cost a minute had it been checked before the checkpoint download rather than
after. Toolchain checks belong in front of the expensive downloads. `g++` already was;
the headers now are too.

The "some hosts never answer sshd" rate this table used to report was **blocker 7**, not the
market. Rentals 11, 17, 18, 19 and 20 all died to it. `--exclude-machines` is still worth
having, but it was treating a symptom: rentals 21 and 22 landed on machine 144172, which
rental 20's predecessor would have excluded as dead.

**A note on cancelling a run.** `run_remote.sh` traps TERM, but POSIX `sh` defers a trap
until the current foreground command returns — and during a batch that command is an `ssh`
that can sit for the better part of an hour. Signalling the script alone does nothing
visible. Kill the `ssh` child as well; the step then returns, the trap fires, and teardown
still pulls results before destroying.

## The seventh blocker: waiting for the contract instead of the container

Three rentals on 2026-09-08 (50314339, 50314728, 50316163) died identically: "instance is
running", then twelve `waiting for sshd` polls, then destroyed at the 300s stall budget with
`Connection timed out` to the vast ssh proxy. Two different machines, two different proxy
hosts. The obvious reading was the one already in this file — some hosts never answer sshd —
and it was wrong.

**The tell was in a line that did not exist yet.** The sshd probe loop never logged what the
instance was doing, so 300s of "waiting for sshd" carried no information. Adding the
instance's own status to that line answered it on the next rental:

```
[deltaforge] instance is running; waiting for sshd (status: loading | no status_msg)
```

The instance was `loading`. The poller had declared it ready anyway.

### The two fields

| Field | What it means | When it says `running` |
|---|---|---|
| `cur_state` | the **rental contract** — this instance is rented and meant to run | from the moment it is created |
| `actual_status` | the **container** — null, then `loading`, then `running` | when the container is actually up |

The readiness poll asked for `.actual_status // .cur_state`. In jq that falls back when the
left side is null — which is exactly the window before the container has started. So the
fallback fired **only** in the case where it was guaranteed to be wrong, and readiness was
declared on the first poll of every rental this project has ever run.

The evidence was sitting in the logs the whole time: **zero** `waiting for instance to start`
lines. The outer poll loop never completed a single iteration. The image pull, the container
start, and the ssh probe were all happening at once, and only the probe was being timed.

### Why it looked like a host problem

Because it is invisible on a host that has the image cached. There the container comes up in
well under the budget and everything works — which is what rentals 13-16 were. On a host that
must pull ~2.5 GB first, the probe spends its entire budget against a container that does not
exist yet, and the run reports the host as dead.

**Rentals 11 and 17 are almost certainly this, not bad hosts.** The "2 in 17 rentals go to
hosts that never answer sshd, budget for it" note that used to be here is withdrawn: it was a
real pattern with the wrong cause attached, and it made a code bug look like a cost of doing
business. That is the expensive kind of wrong — it argues against investigating.

### The fix, and the guard it broke

Gate on `actual_status` alone; an absent value is "not ready yet", never "ready".

That exposed a second problem. The outer stall guard watches `status_msg` for progress, and
these hosts populate no `status_msg` at all. An always-empty message is indistinguishable
from a frozen one, so the guard would now destroy healthy instances at 300s for being quiet.
The Docker Hub hangs it was built for all *did* report one, so the guard keeps working where
it has a signal and defers to the readiness deadline where it does not. The container state
is folded into the progress key, so `null -> loading -> running` counts as the progress it is.

The inner sshd loop keeps a flat budget. That is correct there and only there: by the time it
runs, the container is genuinely up and sshd is the only thing still missing.

**The pattern, again:** every blocker in this file was cheap to fix and expensive to notice,
and this one hid behind a plausible story about flaky hosts. A wrong explanation that
predicts the observation is worse than no explanation, because it ends the investigation.

## The twelfth blocker: a launch that reported nothing

Rental 29 exited 4 three seconds after launch — no offer met the filters, so
`provision.sh` refused correctly, created nothing and billed nothing. The session then
reported nothing at all for a day. The monitor had been attached with `tail -n 0 -f`
**6.5 seconds after the process already ended**, and `-n 0` discards the backlog, so it
waited forever on a file that would never grow again.

**Fixed, and proven:** `remote/launch.sh` always terminates the log with
`[deltaforge] [launch] run exited N` and writes the status to a file; `--status` answers
"is it still going?" at any moment. Proven on a GPU in the weakest possible sense and the
strongest: rental 30's *first* launch attempt hit the identical exit-4 refusal and
announced itself in seconds instead of vanishing.

## The thirteenth blocker: the benchmark ran with autograd enabled

`harness/bench.py` had no `no_grad` anywhere; `harness/correctness.py:120` did. So the
correctness gates passed and only the benchmark failed. Rental 31, slot 0
(`000-identity`, which installs nothing and *is* the reference): **3162s then
`BackendCompilerFailed`**, the fp32 recurrent state's in-place update tripping autograd's
version counter. Slot 1 failed identically in 3206s with a different kernel installed —
the signature of a fault in the shared timing path.

Inductor was compiling the backward graph as well as the forward. That also killed the
one-core theory for rental 22's ~40-minute compile: rental 30 gave ~52 min on 64 cores,
rental 31 gave ~53 min per slot on **256**.

**Fixed in `f461025`** (`_inference_context()` around warmup, setups and timed calls),
and **proven on a GPU by rental 32**: no `BackendCompilerFailed`, correctness gates
passed, memory flat at 8.07 GiB.

**Not fixed: the compile cost itself, which is worse than believed.** Rental 31's 3162s
was *time-until-crash*, not a completed compile, so nothing ever established that one
finishes in ~53 min. With the crash gone, rental 32's slot 0 ran the **full 6980.9s cap
without completing** `max-autotune` on the reference and candidate columns. The 40-minute
figure in `docs/BATCHES.md` measured a backward pass nothing needed; the real forward-only
cost is still unmeasured and is now the single thing standing between this project and a
number.

**What rental 32 did bank:** 47 MB / 1800 inductor and triton entries, compiled *without*
autograd and therefore reusable, against 312 KB from rental 31. Inductor caches per
kernel, so a timed-out compile still makes progress. The next rental on this card starts
genuinely warm, and whether that is enough is the next thing to measure.

## The seventeenth blocker: a dropped socket on the one step with nothing to lose yet

Rental 39 died three and a half minutes in, at the first thing that touches the network
after sshd answers:

```
[deltaforge] syncing repo up to root@ssh2.vast.ai:/workspace/deltaforge
client_loop: send disconnect: Broken pipe
rsync: [sender] write error: Broken pipe (32)
rsync error: error in socket IO (code 10)
```

**Nothing was wrong with the box, the scripts, or the batch.** The host was running, sshd
had answered, and an ssh connection dropped mid-transfer. Teardown behaved correctly:
results pulled (there were none), instance destroyed, $0.0257 billed, nothing leaked.

Worth fixing anyway, because of *where* it sits. The repo sync runs before torch, before
the 9.32 GB checkpoint and before the GPU suite, so a failure there has nothing to lose and
the whole fixed cost still to pay — a rental that dies at minute 3 has to be paid for again
from minute 0. `df_retry` in `remote/lib.sh` now retries a transfer three times with
exponential backoff, and both syncs pass `--partial` so a retry resumes rather than
restarting.

**It is deliberately not applied to anything that creates or destroys an instance.**
Retrying a lifecycle call is how a project ends up paying for two rentals and knowing about
one. The retry covers transfers, which are idempotent, and nothing else.

**And it exposed what the transfer actually was.** The repo is **9.5 MB**. The compile
cache pulled off previous rentals is **1.4 GB**, it lives in `cache/compile/<key>/` inside
the checkout, and `EXCLUDES` did not name it — so every rental since one first came home
has shipped it twice: once buried in the repo sync to `/workspace/deltaforge/cache`, where
nothing looks for it, and once properly via `cache-up` to `/workspace/df-cache`, where the
run points torch. It is gitignored, which is exactly why nobody saw it: `git status` is
silent about it and rsync does not read `.gitignore`. The sync now excludes it, with a test.

Two things this cost beyond the four cents. The teardown's cache pull ran with
`DF_CACHE_KEY` still unset — the key is read off the box, and the box had not got that far
— so it created an empty `cache/compile/unknown/`, which is the directory
`df_phase_estimates` surveys. And the immediate retry with `--exclude-machines 140887`
**refused with exit 4**: that host was the only offer meeting the filters at the time, so
excluding it emptied the market. The run that succeeded went back to the same machine with
the retry in place, which is the right order — the fix addresses the failure, and widening
`--max-rate` to buy a different host would have been paying to route around a socket.

## Current status

**Access is solved.** Blockers 1-5 and 7 are fixed and proven on a GPU; blocker 6's
meta-device fix is proven too — every slot on rental 21 built its candidate and passed
correctness, which is exactly what rental 16 could not do.

**No benchmark number exists yet.** What stands between the project and its first
measurement is no longer the rental path:

1. **The benchmark OOMs with the default four columns** — 30.71 GiB of 31.36, eight slots,
   all at warmup. Construction is not the cost: weights are 7.83 GiB and both columns
   together reach 8.07 GiB. The two eager columns are the suspects and
   `run_remote.sh --columns compiled,candidate_compiled` drops them, but that configuration
   has not yet been observed to survive warmup.
2. **A cold `max-autotune` compile takes ~40 minutes**, not the 3-4 the batch cost model
   assumes. That is now the binding constraint on how many hypotheses fit a rental, and it
   should be measured before another batch is filled.

Both are questions about the benchmark, not about access. That is a different project than
the one this file has been documenting.


## Rental 42 (2026-09-20): three launches before a card, and blocker 11 is now three for three

| Launch | What happened | Cost |
|---|---|---|
| 1 | RTX 4090, machine 27290, advertised `cuda_max_good` **exactly 12.8**. Reached the box, installed torch, died at the first CUDA call: `Error 804: forward compatibility was attempted on non supported HW`. | **5.13 min, $0.0337** |
| 2 | `DF_MIN_CUDA=12.9`, machine excluded. Vast answered the 4090 offer search with **HTTP 429**; zero offers; exit 4. | nothing created |
| 3 | Same filters, no 429, and a genuinely empty 4090 pool at 12.9 under $0.45. Exit 4. | nothing created |
| 4 | `--max-rate 0.65`: five RTX 5090 offers, took one at **$0.5163/hr, CUDA 13.0, 32607 MB, reliability 0.997**. Ran the batch and destroyed cleanly. | 50.82 min, $0.4373 |

**Blocker 11, Proven? — the filter is now derived from three data points rather than one.**

| rental | advertised `cuda_max_good` | outcome |
|---|---|---|
| 33 | 12.8 | `Error 804` |
| 34 | 13.0 | fine |
| 42 | 12.8 | `Error 804` |

A host sitting *exactly on* the floor is the failing case, every time it has been tried.
`DF_MIN_CUDA=12.9` is therefore the correct filter for a cu128 torch build — but it
**emptied the 4090 pool at the default $0.45 ceiling**, so the two knobs are coupled:
tighten the CUDA floor and the rate ceiling has to move with it. Raising `--max-rate` to
0.65 produced five 5090 offers immediately, at $0.5163/hr against the $0.4363 rentals 38
and 40 paid. A 50-minute rental at that rate is $0.43; the failed 4090 cost $0.03. **Pay
the 18% and take the card that works.**

**An exit-4 refusal is not always a market transient.** `AGENT.md` §8 says "widen
`--max-rate` deliberately, or try again later" — and launches 2 and 3 are the two cases
that advice conflates. Launch 2 was an HTTP 429 on the search: retrying was right, and
nothing about the filters was wrong. Launch 3 was a real empty pool: retrying would have
failed forever and moving a filter was the only fix. **The `curl: (22) ... error: 429`
line is the whole difference**, and it is already in the log; read it before deciding
which response a no-offer refusal wants.
