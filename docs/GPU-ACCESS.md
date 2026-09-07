# GPU access: three blockers, each hiding behind the last

Ten rentals have been billed on this project. **None has yet produced a benchmark number**,
and each one that got further than its predecessor did so by exposing the next problem in
the chain:

| | Blocker | Found by | Fixed by | Proven? |
|---|---|---|---|---|
| 1 | Anonymous Docker Hub pulls stall from vast egress ranges | rentals 1-8 | registry credentials, or a non-Docker-Hub image | yes |
| 2 | The image refuses the account's ssh key | rental 9 | injecting the key via `PUBLIC_KEY` **and** `onstart` | yes |
| 3 | The image has `python3` but no `python` | rental 10 | establishing the interpreter first, and `python -m pip` | yes |
| 4 | `accelerate` absent, so the HF oracle cannot be constructed | rental 13 | installing it, and verifying the import out loud | yes |
| 5 | Four `oracle_test.py` bounds were fp32 absolutes on bf16 tensors | rental 14 | scoring relative to the tensor's own scale | yes |
| 6 | Candidate construction double-allocates 8.4 GB of weights | rental 16 | building candidates on `torch.device("meta")` | **no** |
| — | Some hosts never answer sshd at all | rentals 11, 17 | the 300s stall guard; re-run with `--exclude-machines` | n/a |

All are fixed. **Blocker 6's fix has not been tested on a GPU** — the session's 90-minute
gate arrived first. That is where the next session starts.

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
2. **Stuck on a Docker Hub pull:** add `DOCKER_LOGIN_USER` / `DOCKER_LOGIN_TOKEN` to `.env`,
   or `--image` something on `ghcr.io` or `nvcr.io`. The latter is proven to work.
3. **`refused the ssh key`:** the image honours neither `PUBLIC_KEY` nor `authorized_keys`.
   Pick a different image rather than fighting it.
4. Record the machine id in `--exclude-machines` so the deterministic, price-ordered offer
   search does not hand you the same host again.

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

Seventeen rentals, $0.561, **zero leaked instances** — every one destroyed cleanly by the
trap, including one cancelled mid-flight with SIGTERM.

**Two of seventeen rentals died to hosts that never answered sshd**, on different machines,
with `reliability2 > 0.98` in both cases. That rate is worth knowing: budget for it, keep
the stall guard, and use `--exclude-machines` (provisioning now logs the machine id and the
exact flag to re-run with).

## Current status

The default `vastai/base-image` on Docker Hub **pulls, starts, and accepts the injected
key** — rental 50119910 on 2026-09-07 got through all three and on to the pip installs. The
key injection through `PUBLIC_KEY` plus the `onstart` append is therefore **proven**, not
merely written; that was the open question this file previously flagged.

What is still unproven is everything past the checkpoint download: no rental has yet
reached the GPU test suite, the correctness gates or a benchmark.

The default image is still `vastai/base-image` on Docker Hub, because with credentials in
`.env` that is the better choice: it is purpose-built for Vast and its key handling already
works. Switch with `--image` if you have no Docker Hub account.
