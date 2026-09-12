#!/bin/sh
# Search Vast.ai for an instance matching the hard filters, then create it.
#
# Filters are not preferences. On-demand only (an interruptible instance that dies
# mid-sweep wastes more than it saves), host reliability > 0.98, exactly one GPU, and an
# hourly-rate ceiling. RTX 5090 preferred for its memory bandwidth, RTX 4090 as fallback.
#
# The month-to-date budget gate is checked before anything is created, and the ledger row
# is written *before* the instance is used, so a session that dies mid-run still leaves a
# record of what it started.
#
# --dry-run exercises every branch of this logic — the gate, the offer filter, the
# selection, the ledger write — against a committed offers fixture, without contacting
# the create endpoint and without spending anything.

set -eu

DF_SCRIPT_PATH="$0"
unset CDPATH
DF_REPO_ROOT=$(cd -- "$(dirname -- "$DF_SCRIPT_PATH")/.." && pwd)
export DF_REPO_ROOT
. "$DF_REPO_ROOT/remote/lib.sh"

# Vast's own small base image, plus torch from PyTorch's CDN in the prep step.
#
# The spec asks for a preinstalled-PyTorch image so a run does not pay for a dependency
# install. Measured on the market, that trade runs the other way: billing starts at
# instance creation, and `pytorch/pytorch` pulls from Docker Hub stalled on every host
# tried — 20 minutes without landing 4.26 GB, and the same for 14.1 GB. Two rentals ended
# with the image still "Pulling". `vastai/*` images come from the registry Vast hosts
# mirror and cache, and the mini tag is 2.5 GB; torch then comes from
# download.pytorch.org, a CDN with no anonymous pull limit, in about a minute.
DF_IMAGE="${DF_IMAGE:-vastai/base-image:cuda-12.9-mini-py312-2026-08-28}"

# Machines that have already cost a rental without producing a result. The market is
# ordered by price and deterministic, so without this the next run lands on exactly the
# same failing host.
DF_EXCLUDE_MACHINES="${DF_EXCLUDE_MACHINES:-}"

DF_MAX_RATE="${DF_MAX_RATE:-0.45}"
DF_GPU="${DF_GPU:-RTX 5090}"
DF_FALLBACK_GPU="${DF_FALLBACK_GPU:-RTX 4090}"
DF_MIN_RELIABILITY="${DF_MIN_RELIABILITY:-0.98}"
DF_MIN_GPU_RAM="${DF_MIN_GPU_RAM:-24000}"
# The host driver must be able to run the wheels run_remote.sh installs, which come from
# PyTorch's cu128 index. CUDA's forward-compatibility packages exist for exactly this gap
# but are supported only on data-centre cards, never on the consumer GeForce parts this
# project rents -- so on a 4090 an older driver is a hard stop, not a slow path. Rental 25
# reached the box, installed torch, and died at the first CUDA call with "Error 804:
# forward compatibility was attempted on non supported HW". Keep this in step with the
# --index-url in run_remote.sh.
DF_MIN_CUDA="${DF_MIN_CUDA:-12.8}"
# Two filters bought with rented time rather than reasoning. The cheapest single RTX 5090
# on the market was an unverified consumer host that never finished pulling the container
# image across three provisioning attempts, and whose ssh proxy was unreachable from
# here. Every minute of that is billed. A run downloads a ~9 GB image and a ~9 GB
# checkpoint before it computes anything, so link speed is a cost input, not a nicety.
DF_REQUIRE_VERIFIED="${DF_REQUIRE_VERIFIED:-1}"
DF_MIN_INET_DOWN="${DF_MIN_INET_DOWN:-300}"
DF_DISK_GB="${DF_DISK_GB:-40}"
DF_MAX_MINUTES="${DF_MAX_MINUTES:-210}"
DF_LEDGER="${DF_LEDGER:-$DF_REPO_ROOT/ledger/spend.jsonl}"
DF_MTD_LIMIT="${DF_MTD_LIMIT:-45}"
DF_SESSION_ID="${DF_SESSION_ID:-}"
DF_HYPOTHESIS="${DF_HYPOTHESIS:-}"
DF_STATE_FILE="${DF_STATE_FILE:-$DF_REPO_ROOT/.deltaforge-instance}"
DF_OFFERS_FILE="${DF_OFFERS_FILE:-}"
# How many offers the create step may walk before giving up. A listed-but-unrentable ask
# is refused instantly and costs nothing, so trying a few is cheap; the ceiling exists so
# a market-wide outage fails fast instead of grinding through every offer on the platform.
DF_OFFER_CANDIDATES="${DF_OFFER_CANDIDATES:-5}"

usage() {
    cat <<'EOF'
Usage: remote/provision.sh [options]

  --dry-run                 Exercise every gate and the full selection path without
                            contacting the create endpoint. Uses the committed offers
                            fixture unless --offers-file is given.
  --session-id ID           Session identifier recorded in the ledger (required).
  --hypothesis SLUG         Hypothesis this instance is for. Empty for a baseline run.
  --gpu NAME                Preferred GPU name (default: RTX 5090).
  --fallback-gpu NAME       Fallback GPU name (default: RTX 4090).
  --max-rate USD            Hourly rate ceiling (default: 0.45).
  --exclude-machines IDS    Comma-separated machine ids to skip.
  --max-minutes N           Estimated ceiling written to the ledger (default: 210).
  --ledger PATH             Ledger file (default: ledger/spend.jsonl).
  --mtd-limit USD           Month-to-date refusal threshold (default: 45).
  --offers-file PATH        Read offers from a file instead of the API.
  --offer-candidates N      How many offers the create step may try, in price order,
                            before giving up (default: 5). A listed-but-unrentable ask
                            is refused instantly, so walking a few costs nothing.
  --state-file PATH         Where to write the created instance's details.
  -h, --help                This message.

Exit codes: 0 ok, 1 error, 3 refused by the month-to-date budget gate,
            4 no offer met the filters, or every candidate refused the create.
EOF
}

while [ $# -gt 0 ]; do
    case "$1" in
        --dry-run)       DF_DRY_RUN=1 ;;
        --session-id)    DF_SESSION_ID="$2"; shift ;;
        --hypothesis)    DF_HYPOTHESIS="$2"; shift ;;
        --gpu)           DF_GPU="$2"; shift ;;
        --fallback-gpu)  DF_FALLBACK_GPU="$2"; shift ;;
        --max-rate)      DF_MAX_RATE="$2"; shift ;;
        --max-minutes)   DF_MAX_MINUTES="$2"; shift ;;
        --ledger)        DF_LEDGER="$2"; DF_LEDGER_EXPLICIT=1; shift ;;
        --mtd-limit)     DF_MTD_LIMIT="$2"; shift ;;
        --offers-file)   DF_OFFERS_FILE="$2"; shift ;;
        --state-file)    DF_STATE_FILE="$2"; shift ;;
        --image)         DF_IMAGE="$2"; shift ;;
        --exclude-machines) DF_EXCLUDE_MACHINES="$2"; shift ;;
        --offer-candidates) DF_OFFER_CANDIDATES="$2"; shift ;;
        -h|--help)       usage; exit 0 ;;
        *)               df_die "unknown option: $1 (try --help)" ;;
    esac
    shift
done
export DF_DRY_RUN

[ -n "$DF_SESSION_ID" ] || df_die "--session-id is required: the session gate cannot work without it"
df_require_cmd awk date

# ---------------------------------------------------------------------------
# Gate 1: month-to-date budget. Checked before anything is created.
# ---------------------------------------------------------------------------

MTD=$(df_ledger_month_to_date "$DF_LEDGER")
if [ "$DF_DRY_RUN" = "1" ] && [ "${DF_LEDGER_EXPLICIT:-0}" != "1" ]; then
    # Gate read above against the real record; rows below go to a scratch copy.
    DF_LEDGER=$(df_dryrun_ledger "$DF_LEDGER")
fi
df_log "month-to-date spend: \$$MTD (refusal threshold \$$DF_MTD_LIMIT)"
if df_ge "$MTD" "$DF_MTD_LIMIT"; then
    df_die "REFUSED by month-to-date budget gate: \$$MTD >= \$$DF_MTD_LIMIT. \
No instance was created. Wait for the next calendar month or raise the ceiling deliberately."
fi

# ---------------------------------------------------------------------------
# Search
# ---------------------------------------------------------------------------

# `type` is a bare string in this API; passing it as an object like the other filters is
# rejected. Verified against the live API on 2026-08-30.
offers_query() {
    _verified=true
    [ "$DF_REQUIRE_VERIFIED" = "1" ] || _verified=false
    printf '{"gpu_name":{"eq":"%s"},"num_gpus":{"eq":1},"rentable":{"eq":true},"reliability2":{"gt":%s},"dph_total":{"lte":%s},"verified":{"eq":%s},"inet_down":{"gt":%s},"type":"on-demand","order":[["dph_total","asc"]],"limit":64}' \
        "$1" "$DF_MIN_RELIABILITY" "$DF_MAX_RATE" "$_verified" "$DF_MIN_INET_DOWN"
}

fetch_offers() {
    _gpu="$1"
    if [ -n "$DF_OFFERS_FILE" ]; then
        cat "$DF_OFFERS_FILE"
    elif [ "$DF_DRY_RUN" = "1" ]; then
        df_log "(dry-run) using committed offers fixture instead of POST /bundles/"
        cat "$DF_REPO_ROOT/remote/fixtures/offers.json"
    else
        df_api POST "/bundles/" "$(offers_query "$_gpu")"
    fi
}

# Re-apply every filter client-side. The server is asked for the right thing, but a
# budget control that trusts a remote filter is not a control.
#
# Emits up to `DF_OFFER_CANDIDATES` rows in price order, not one. Returning only the
# cheapest made the search a trap: it is deterministic and price-ordered, so an ask that is
# listed but unrentable was selected, refused with `no_such_ask`, and then selected again
# on the next attempt, forever. `--exclude-machines` was the manual escape. Walking the
# list is the automatic one.
select_offers() {
    _gpu="$1"
    fetch_offers "$_gpu" | jq -r --arg gpu "$_gpu" \
        --argjson maxrate "$DF_MAX_RATE" \
        --argjson minrel "$DF_MIN_RELIABILITY" \
        --argjson minram "$DF_MIN_GPU_RAM" \
        --argjson mincuda "$DF_MIN_CUDA" \
        --argjson mindown "$DF_MIN_INET_DOWN" \
        --argjson wantverified "$DF_REQUIRE_VERIFIED" \
        --arg excluded "$DF_EXCLUDE_MACHINES" \
        --argjson limit "$DF_OFFER_CANDIDATES" '
        .offers // []
        | map(select(
            .gpu_name == $gpu
            and (.num_gpus // 0) == 1
            and (.rentable // false) == true
            and ((.is_bid_only // false) | not)
            and (.reliability2 // 0) > $minrel
            and (.dph_total // 1e9) <= $maxrate
            and (.gpu_ram // 0) >= $minram
            # A driver older than the torch build cannot run it: forward compatibility is
            # a data-centre-only feature and these are GeForce cards.
            and (((.cuda_max_good // 0) | tonumber? // 0) >= $mincuda)
            and (.inet_down // 0) > $mindown
            # The query filters on `verified`, but the offer objects come back with the
            # field null, so a client-side `== true` rejects the entire market. Reject an
            # explicit false and let absent mean "the server already filtered on it".
            and ($wantverified == 0 or (.verified != false))
            and ((.machine_id // 0) as $m
                 | ($excluded | split(",") | map(select(length > 0)) | index($m | tostring)) == null)
          ))
        | sort_by(.dph_total)
        | .[0:$limit]
        | .[]
        | [.id, .gpu_name, .dph_total, .reliability2, .gpu_ram, (.cuda_max_good // "?"),
           (.machine_id // "?")]
        | @tsv'
}

df_require_cmd jq
if [ "$DF_DRY_RUN" != "1" ] || [ -z "$DF_OFFERS_FILE" ]; then
    if ! df_load_api_key; then
        if [ "$DF_DRY_RUN" = "1" ]; then
            df_log "(dry-run) no API key present; continuing, since no endpoint is contacted"
        else
            df_die "no API key: set VAST_API_KEY or add it to $DF_REPO_ROOT/.env"
        fi
    fi
fi

# Optional, and inert when absent.
df_load_registry_login || true

DF_CANDIDATES_FILE=$(umask 077; mktemp "${TMPDIR:-/tmp}/df-offers.XXXXXX") || df_die "mktemp failed"
trap 'rm -f "$DF_CANDIDATES_FILE"' EXIT INT TERM

for gpu in "$DF_GPU" "$DF_FALLBACK_GPU"; do
    [ -n "$gpu" ] || continue
    df_log "searching for on-demand $gpu, 1 GPU, reliability > $DF_MIN_RELIABILITY, <= \$$DF_MAX_RATE/hr"
    # stderr is deliberately NOT suppressed: it carries the dry-run fixture notice and
    # whatever the API said when a search fails, and a silent search is how you spend a
    # session wondering why the market looks empty.
    select_offers "$gpu" > "$DF_CANDIDATES_FILE" || true
    [ -s "$DF_CANDIDATES_FILE" ] && break
    df_warn "no $gpu offer met the filters"
done

if [ ! -s "$DF_CANDIDATES_FILE" ]; then
    df_log "no offer met the filters for any of: $DF_GPU, $DF_FALLBACK_GPU"
    exit 4
fi

DF_CANDIDATE_COUNT=$(wc -l < "$DF_CANDIDATES_FILE" | tr -d ' ')
df_log "$DF_CANDIDATE_COUNT candidate offer(s) in price order; trying each until one is created"

# ---------------------------------------------------------------------------
# Create, then record. The ledger row is written before the instance is used.
# ---------------------------------------------------------------------------

# The public half of the key run_remote.sh authenticates with. Vast associates the account
# key with every instance, but that only reaches images built to its conventions: an
# `ghcr.io/ai-dock` image pulled and ran and then answered ssh with "Permission denied
# (publickey)", which cost a rental. Injecting the key ourselves works for any image family
# -- `PUBLIC_KEY` is what the ai-dock and RunPod-style images read, and appending to
# authorized_keys covers everything else. A public key is not a secret.
df_public_key() {
    _pub="${DF_SSH_KEY:-$HOME/.ssh/deltaforge_vast}.pub"
    [ -f "$_pub" ] && tr -d '\n' < "$_pub"
}

create_body() {
    set +x
    _pubkey=$(df_public_key)
    _onstart="touch ~/.no_auto_tmux"
    if [ -n "$_pubkey" ]; then
        _onstart="$_onstart; mkdir -p /root/.ssh; chmod 700 /root/.ssh; grep -qF '$_pubkey' /root/.ssh/authorized_keys 2>/dev/null || echo '$_pubkey' >> /root/.ssh/authorized_keys; chmod 600 /root/.ssh/authorized_keys"
    fi
    _onstart="$_onstart; shutdown -h +$DF_MAX_MINUTES"
    # `image_login` is the field Vast passes to `docker login` on the host before pulling.
    # Present only when credentials were found; the token is never logged, and the body
    # reaches curl through a 0600 file rather than argv (see df_api in lib.sh).
    _env=""
    [ -n "$_pubkey" ] && _env=$(printf ',"env":"-e PUBLIC_KEY=\\"%s\\""' "$_pubkey")
    if [ -n "${DF_REGISTRY_LOGIN:-}" ]; then
        printf '{"client_id":"me","image":"%s","disk":%s,"runtype":"ssh","onstart":"%s","image_login":"%s"%s}' \
            "$DF_IMAGE" "$DF_DISK_GB" "$_onstart" "$DF_REGISTRY_LOGIN" "$_env"
    else
        printf '{"client_id":"me","image":"%s","disk":%s,"runtype":"ssh","onstart":"%s"%s}' \
            "$DF_IMAGE" "$DF_DISK_GB" "$_onstart" "$_env"
    fi
}

# Attempt one create. Prints the instance id on stdout and returns 0, or explains the
# refusal on stderr and returns non-zero so the caller can try the next candidate.
#
# curl runs with --fail-with-body so the API's reason survives an error status -- but a
# plain `RESPONSE=$(df_api ...)` assignment hands that status straight to `set -e`, and
# the run dies on a bare `curl: (22)` one line before the message written to explain it.
# A create was refused with HTTP 400 and the log said nothing about why. Taking the status
# in a `||` list keeps `set -e` out of it, so the body reaches the log.
#
# This used to `df_die` on a refusal, which is what made a phantom ask fatal: the search is
# price-ordered and deterministic, so the next run picked the same unrentable offer and
# died the same way. Returning a status instead is what lets the caller walk on.
try_create_instance() {
    _offer_id="$1"
    if ! _resp=$(df_api PUT "/asks/$_offer_id/" "$(create_body)"); then
        df_warn "offer $_offer_id refused: $(printf '%s' "$_resp" | jq -c '.' 2>/dev/null || printf '%s' "$_resp")"
        return 1
    fi
    _iid=$(printf '%s' "$_resp" | jq -r '.new_contract // empty')
    if [ -z "$_iid" ]; then
        df_warn "offer $_offer_id returned no contract: $(printf '%s' "$_resp" | jq -c '.' 2>/dev/null || printf '%s' "$_resp")"
        return 1
    fi
    printf '%s' "$_iid"
}

# Walk the candidates in price order. A refusal costs nothing -- no instance exists, so
# nothing is billed -- which is why trying the next one is strictly better than dying and
# making a human pass --exclude-machines.
#
# `while read ... done < file` rather than a pipeline: a pipeline runs the loop in a
# subshell and the chosen OFFER_* values would not survive it.
INSTANCE_ID=""
DF_ATTEMPT=0
while IFS= read -r _row; do
    [ -n "$_row" ] || continue
    DF_ATTEMPT=$((DF_ATTEMPT + 1))

    OFFER_ID=$(printf '%s' "$_row" | cut -f1)
    OFFER_GPU=$(printf '%s' "$_row" | cut -f2)
    OFFER_RATE=$(printf '%s' "$_row" | cut -f3)
    OFFER_REL=$(printf '%s' "$_row" | cut -f4)
    OFFER_RAM=$(printf '%s' "$_row" | cut -f5)
    OFFER_CUDA=$(printf '%s' "$_row" | cut -f6)
    # The machine id, not the offer id, is what --exclude-machines takes. Logging only the
    # offer id made the documented remedy for a host that burns a rental — "record the
    # machine id in --exclude-machines so the deterministic, price-ordered search does not
    # hand you the same host again" — impossible to actually carry out.
    OFFER_MACHINE=$(printf '%s' "$_row" | cut -f7)

    df_log "candidate $DF_ATTEMPT/$DF_CANDIDATE_COUNT: trying offer $OFFER_ID on machine $OFFER_MACHINE (\$$OFFER_RATE/hr)"

    # Belt and braces: the ceiling is re-checked after selection, in case a filter was
    # loosened upstream by an edit that looked harmless. Still fatal rather than skipped —
    # an over-ceiling offer here means the filter is broken, and walking past it would hide
    # a budget control that has stopped working.
    if df_ge "$OFFER_RATE" "$(awk -v r="$DF_MAX_RATE" 'BEGIN { printf "%.6f", r + 0.000001 }')"; then
        df_die "selected offer rate \$$OFFER_RATE exceeds the ceiling \$$DF_MAX_RATE; refusing"
    fi

    if df_dry "would PUT /asks/$OFFER_ID/ with image $DF_IMAGE, disk ${DF_DISK_GB}GB"; then
        INSTANCE_ID="dryrun-$(df_now_epoch)"
        break
    fi

    if INSTANCE_ID=$(try_create_instance "$OFFER_ID"); then
        df_log "created instance $INSTANCE_ID"
        break
    fi
    INSTANCE_ID=""
done < "$DF_CANDIDATES_FILE"

if [ -z "$INSTANCE_ID" ]; then
    df_warn "all $DF_CANDIDATE_COUNT candidate offer(s) refused the create."
    df_warn "No instance exists and nothing was billed. Raise --offer-candidates, widen"
    df_warn "--max-rate, or wait for the market to move."
    exit 4
fi

# "selected offer N" is the canonical line, and it belongs to the offer that actually
# produced an instance -- never to one we merely tried. A candidate that was refused was
# not selected, and a log that says otherwise would make the walk unreadable.
df_log "selected offer $OFFER_ID on machine $OFFER_MACHINE: $OFFER_GPU, \$$OFFER_RATE/hr, reliability $OFFER_REL, ${OFFER_RAM}MB, CUDA $OFFER_CUDA"
df_log "if this host burns the rental: re-run with --exclude-machines $OFFER_MACHINE"

df_ledger_append_provision "$DF_LEDGER" "$DF_SESSION_ID" "$INSTANCE_ID" \
    "$OFFER_GPU" "$OFFER_RATE" "$DF_MAX_MINUTES" "$DF_HYPOTHESIS" \
    "offer $OFFER_ID machine $OFFER_MACHINE reliability $OFFER_REL"
df_log "ledger row appended before use: instance $INSTANCE_ID at \$$OFFER_RATE/hr"

# The remote-side shutdown timer above is the second line of defence; the local watchdog
# started by run_remote.sh is the first. A remote that hangs cannot defeat its own kill
# switch, because the kill switch does not run on it.
mkdir -p "$(dirname "$DF_STATE_FILE")"
# Values are quoted: GPU names contain spaces, and this file is sourced by run_remote.sh.
cat > "$DF_STATE_FILE" <<EOF
DF_INSTANCE_ID='$INSTANCE_ID'
DF_INSTANCE_GPU='$OFFER_GPU'
DF_INSTANCE_RATE='$OFFER_RATE'
DF_INSTANCE_OFFER='$OFFER_ID'
DF_INSTANCE_START_EPOCH='$(df_now_epoch)'
DF_SESSION_ID='$DF_SESSION_ID'
DF_HYPOTHESIS='$DF_HYPOTHESIS'
EOF
df_log "instance details written to $DF_STATE_FILE"
printf '%s\n' "$INSTANCE_ID"
