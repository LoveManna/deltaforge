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

# Verified 2026-08-30 on Docker Hub: this tag exists and matches the spec's
# "PyTorch 2.11 + CUDA 12.8 preinstalled" requirement, which keeps us from paying for a
# long dependency install on every provision.
DF_IMAGE="${DF_IMAGE:-pytorch/pytorch:2.11.0-cuda12.8-cudnn9-devel}"

# Live market check on 2026-08-30 returned 6+ single RTX 5090s at $0.32-$0.35/hr. The
# ceiling is set from that observation, not from the spec's older $0.45-0.60 estimate.
DF_MAX_RATE="${DF_MAX_RATE:-0.45}"
DF_GPU="${DF_GPU:-RTX 5090}"
DF_FALLBACK_GPU="${DF_FALLBACK_GPU:-RTX 4090}"
DF_MIN_RELIABILITY="${DF_MIN_RELIABILITY:-0.98}"
DF_MIN_GPU_RAM="${DF_MIN_GPU_RAM:-24000}"
# Two filters bought with rented time rather than reasoning. The cheapest single RTX 5090
# on the market was an unverified consumer host that never finished pulling the container
# image across three provisioning attempts, and whose ssh proxy was unreachable from
# here. Every minute of that is billed. A run downloads a ~9 GB image and a ~9 GB
# checkpoint before it computes anything, so link speed is a cost input, not a nicety.
DF_REQUIRE_VERIFIED="${DF_REQUIRE_VERIFIED:-1}"
DF_MIN_INET_DOWN="${DF_MIN_INET_DOWN:-300}"
DF_DISK_GB="${DF_DISK_GB:-40}"
DF_MAX_MINUTES="${DF_MAX_MINUTES:-90}"
DF_LEDGER="${DF_LEDGER:-$DF_REPO_ROOT/ledger/spend.jsonl}"
DF_MTD_LIMIT="${DF_MTD_LIMIT:-45}"
DF_SESSION_ID="${DF_SESSION_ID:-}"
DF_HYPOTHESIS="${DF_HYPOTHESIS:-}"
DF_STATE_FILE="${DF_STATE_FILE:-$DF_REPO_ROOT/.deltaforge-instance}"
DF_OFFERS_FILE="${DF_OFFERS_FILE:-}"

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
  --max-minutes N           Estimated ceiling written to the ledger (default: 90).
  --ledger PATH             Ledger file (default: ledger/spend.jsonl).
  --mtd-limit USD           Month-to-date refusal threshold (default: 45).
  --offers-file PATH        Read offers from a file instead of the API.
  --state-file PATH         Where to write the created instance's details.
  -h, --help                This message.

Exit codes: 0 ok, 1 error, 3 refused by the month-to-date budget gate,
            4 no offer met the filters.
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
select_offer() {
    _gpu="$1"
    fetch_offers "$_gpu" | jq -r --arg gpu "$_gpu" \
        --argjson maxrate "$DF_MAX_RATE" \
        --argjson minrel "$DF_MIN_RELIABILITY" \
        --argjson minram "$DF_MIN_GPU_RAM" \
        --argjson mindown "$DF_MIN_INET_DOWN" \
        --argjson wantverified "$DF_REQUIRE_VERIFIED" '
        .offers // []
        | map(select(
            .gpu_name == $gpu
            and (.num_gpus // 0) == 1
            and (.rentable // false) == true
            and ((.is_bid_only // false) | not)
            and (.reliability2 // 0) > $minrel
            and (.dph_total // 1e9) <= $maxrate
            and (.gpu_ram // 0) >= $minram
            and (.inet_down // 0) > $mindown
            and ($wantverified == 0 or (.verified // false) == true)
          ))
        | sort_by(.dph_total)
        | .[0]
        | if . == null then empty
          else [.id, .gpu_name, .dph_total, .reliability2, .gpu_ram, (.cuda_max_good // "?")]
               | @tsv
          end'
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

OFFER=""
for gpu in "$DF_GPU" "$DF_FALLBACK_GPU"; do
    [ -n "$gpu" ] || continue
    df_log "searching for on-demand $gpu, 1 GPU, reliability > $DF_MIN_RELIABILITY, <= \$$DF_MAX_RATE/hr"
    OFFER=$(select_offer "$gpu" || true)
    [ -n "$OFFER" ] && break
    df_warn "no $gpu offer met the filters"
done

if [ -z "$OFFER" ]; then
    df_log "no offer met the filters for any of: $DF_GPU, $DF_FALLBACK_GPU"
    exit 4
fi

OFFER_ID=$(printf '%s' "$OFFER" | cut -f1)
OFFER_GPU=$(printf '%s' "$OFFER" | cut -f2)
OFFER_RATE=$(printf '%s' "$OFFER" | cut -f3)
OFFER_REL=$(printf '%s' "$OFFER" | cut -f4)
OFFER_RAM=$(printf '%s' "$OFFER" | cut -f5)
OFFER_CUDA=$(printf '%s' "$OFFER" | cut -f6)
df_log "selected offer $OFFER_ID: $OFFER_GPU, \$$OFFER_RATE/hr, reliability $OFFER_REL, ${OFFER_RAM}MB, CUDA $OFFER_CUDA"

# Belt and braces: the ceiling is re-checked after selection, in case a filter was
# loosened upstream by an edit that looked harmless.
if df_ge "$OFFER_RATE" "$(awk -v r="$DF_MAX_RATE" 'BEGIN { printf "%.6f", r + 0.000001 }')"; then
    df_die "selected offer rate \$$OFFER_RATE exceeds the ceiling \$$DF_MAX_RATE; refusing"
fi

# ---------------------------------------------------------------------------
# Create, then record. The ledger row is written before the instance is used.
# ---------------------------------------------------------------------------

create_body() {
    printf '{"client_id":"me","image":"%s","disk":%s,"runtype":"ssh","onstart":"%s"}' \
        "$DF_IMAGE" "$DF_DISK_GB" \
        "touch ~/.no_auto_tmux; shutdown -h +$DF_MAX_MINUTES"
}

if df_dry "would PUT /asks/$OFFER_ID/ with image $DF_IMAGE, disk ${DF_DISK_GB}GB"; then
    INSTANCE_ID="dryrun-$(df_now_epoch)"
else
    RESPONSE=$(df_api PUT "/asks/$OFFER_ID/" "$(create_body)")
    INSTANCE_ID=$(printf '%s' "$RESPONSE" | jq -r '.new_contract // empty')
    [ -n "$INSTANCE_ID" ] || df_die "instance creation failed: $(printf '%s' "$RESPONSE" | jq -c '.' 2>/dev/null || printf '%s' "$RESPONSE")"
    df_log "created instance $INSTANCE_ID"
fi

df_ledger_append_provision "$DF_LEDGER" "$DF_SESSION_ID" "$INSTANCE_ID" \
    "$OFFER_GPU" "$OFFER_RATE" "$DF_MAX_MINUTES" "$DF_HYPOTHESIS" \
    "offer $OFFER_ID reliability $OFFER_REL"
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
