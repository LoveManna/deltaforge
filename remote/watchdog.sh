#!/bin/sh
# Wall-clock kill switch for a rented instance.
#
# This runs on the LOCAL machine, not the rented box. That placement is the whole point:
# a remote that hangs, wedges its network, or spins in a kernel launch cannot defeat a
# kill switch that does not run on it. `provision.sh` also sets a remote-side `shutdown`
# timer as a second line of defence, but that one *can* be defeated by the failure it
# exists to catch, so it is the backup and this is the primary.
#
# This is a backstop, not a routine control. The 120-minute session gate is what should
# normally end a session. If this watchdog ever fires, something went wrong and the
# session must report that rather than treat it as a normal ending — hence the marker
# line it writes.

set -eu

unset CDPATH
DF_REPO_ROOT=$(cd -- "$(dirname -- "$0")/.." && pwd)
export DF_REPO_ROOT
. "$DF_REPO_ROOT/remote/lib.sh"

DF_TIMEOUT_MINUTES="${DF_TIMEOUT_MINUTES:-150}"
DF_TIMEOUT_SECONDS=""
DF_INSTANCE_ID=""
DF_POLL_SECONDS="${DF_POLL_SECONDS:-15}"
DF_CANCEL_FILE=""
DF_LEDGER="${DF_LEDGER:-$DF_REPO_ROOT/ledger/spend.jsonl}"
DF_SESSION_ID="${DF_SESSION_ID:-}"
DF_INSTANCE_RATE="${DF_INSTANCE_RATE:-0}"
DF_INSTANCE_GPU="${DF_INSTANCE_GPU:-}"

usage() {
    cat <<'EOF'
Usage: remote/watchdog.sh --instance-id ID [options]

  --instance-id ID        Instance to destroy when the timer expires (required).
  --timeout-minutes N     Wall-clock limit (default: 150), above the 120-minute gate.
  --timeout-seconds N     Same, in seconds. Overrides --timeout-minutes; used by tests.
  --poll-seconds N        How often to check for cancellation (default: 15).
  --cancel-file PATH      If this file appears, the run finished normally: exit quietly
                          without destroying anything.
  --ledger PATH           Ledger to reconcile into if the watchdog fires.
  --session-id ID         Session id recorded on the reconciliation row.
  --rate USD              Hourly rate, for the reconciliation row.
  --gpu NAME              GPU model, for the reconciliation row.
  --dry-run               Run the full timer and teardown path without calling destroy.
  -h, --help              This message.

Exit codes: 0 cancelled normally, 2 the watchdog fired and destroyed the instance.
EOF
}

while [ $# -gt 0 ]; do
    case "$1" in
        --instance-id)      DF_INSTANCE_ID="$2"; shift ;;
        --timeout-minutes)  DF_TIMEOUT_MINUTES="$2"; shift ;;
        --timeout-seconds)  DF_TIMEOUT_SECONDS="$2"; shift ;;
        --poll-seconds)     DF_POLL_SECONDS="$2"; shift ;;
        --cancel-file)      DF_CANCEL_FILE="$2"; shift ;;
        --ledger)           DF_LEDGER="$2"; shift ;;
        --session-id)       DF_SESSION_ID="$2"; shift ;;
        --rate)             DF_INSTANCE_RATE="$2"; shift ;;
        --gpu)              DF_INSTANCE_GPU="$2"; shift ;;
        --dry-run)          DF_DRY_RUN=1 ;;
        -h|--help)          usage; exit 0 ;;
        *)                  df_die "unknown option: $1 (try --help)" ;;
    esac
    shift
done
export DF_DRY_RUN

[ -n "$DF_INSTANCE_ID" ] || df_die "--instance-id is required"

if [ -z "$DF_TIMEOUT_SECONDS" ]; then
    DF_TIMEOUT_SECONDS=$(awk -v m="$DF_TIMEOUT_MINUTES" 'BEGIN { printf "%d", m * 60 }')
fi

START=$(df_now_epoch)
DEADLINE=$((START + DF_TIMEOUT_SECONDS))
df_log "watchdog armed for instance $DF_INSTANCE_ID: hard destroy in ${DF_TIMEOUT_SECONDS}s"

while :; do
    if [ -n "$DF_CANCEL_FILE" ] && [ -f "$DF_CANCEL_FILE" ]; then
        df_log "watchdog cancelled: run finished normally, instance $DF_INSTANCE_ID left to its owner"
        exit 0
    fi
    NOW=$(df_now_epoch)
    [ "$NOW" -ge "$DEADLINE" ] && break
    REMAINING=$((DEADLINE - NOW))
    if [ "$REMAINING" -lt "$DF_POLL_SECONDS" ]; then
        sleep "$REMAINING"
    else
        sleep "$DF_POLL_SECONDS"
    fi
done

ELAPSED_MIN=$(awk -v s="$(df_now_epoch)" -v t="$START" 'BEGIN { printf "%.2f", (s - t) / 60 }')
df_warn "WATCHDOG FIRED after ${ELAPSED_MIN} minutes for instance $DF_INSTANCE_ID."
df_warn "This is a fault, not a normal ending: the 120-minute session gate should have"
df_warn "stopped the session first. Report it in the session writeup."
df_vast_destroy "$DF_INSTANCE_ID"

if ! df_dry "would append watchdog reconciliation row to $DF_LEDGER"; then
    df_ledger_append_destroy "$DF_LEDGER" "$DF_SESSION_ID" "$DF_INSTANCE_ID" \
        "$DF_INSTANCE_GPU" "$DF_INSTANCE_RATE" "$ELAPSED_MIN" "" "destroyed by watchdog"
fi
exit 2
