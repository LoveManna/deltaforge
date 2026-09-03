#!/bin/sh
# The one entry point a session calls.
#
#   session gate -> provision -> watchdog -> sync up -> correctness -> bench
#                -> pull results -> destroy
#
# The entire body runs inside a trap on EXIT/INT/TERM, so a crash, a failed benchmark, a
# Ctrl-C, or an unhandled error still destroys the instance and still reconciles the
# ledger. An agent that forgets to tear down burns about $13/day; that must not be
# possible to do by accident.
#
# --dry-run runs every step of this logic, including both budget gates and the teardown
# trap, without contacting the create or destroy endpoints and without spending anything.
# --simulate-failure STAGE forces a failure at a named stage, which is how the tests prove
# the trap really does tear down on the unhappy path.

set -eu

unset CDPATH
DF_REPO_ROOT=$(cd -- "$(dirname -- "$0")/.." && pwd)
export DF_REPO_ROOT
. "$DF_REPO_ROOT/remote/lib.sh"

DF_SESSION_ID="${DF_SESSION_ID:-session-$(date -u +%Y%m%dT%H%M%SZ)}"
DF_HYPOTHESIS="${DF_HYPOTHESIS:-}"
DF_LEDGER="${DF_LEDGER:-$DF_REPO_ROOT/ledger/spend.jsonl}"
DF_SESSION_LIMIT_MINUTES="${DF_SESSION_LIMIT_MINUTES:-60}"
DF_WATCHDOG_MINUTES="${DF_WATCHDOG_MINUTES:-90}"
DF_REMOTE_DIR="${DF_REMOTE_DIR:-/workspace/deltaforge}"
DF_WEIGHTS_DIR="${DF_WEIGHTS_DIR:-/workspace/qwen3.5-4b}"
DF_STATE_FILE="${DF_STATE_FILE:-$DF_REPO_ROOT/.deltaforge-instance}"
DF_SIMULATE_FAILURE="${DF_SIMULATE_FAILURE:-}"
DF_SSH_READY_TIMEOUT="${DF_SSH_READY_TIMEOUT:-600}"
DF_PROVISION_ARGS=""

usage() {
    cat <<'EOF'
Usage: remote/run_remote.sh [options]

  --dry-run                 Run every stage, both budget gates and the teardown trap
                            without contacting the create/destroy endpoints.
  --session-id ID           Session identifier (default: session-<UTC timestamp>).
  --hypothesis SLUG         Hypothesis being tested. Empty means a baseline run.
  --ledger PATH             Ledger file (default: ledger/spend.jsonl).
  --session-limit N         Session GPU-time soft gate, in minutes (default: 60).
  --watchdog-minutes N      Hard watchdog timeout (default: 90).
  --max-rate USD            Hourly rate ceiling, passed to provision.sh.
  --simulate-failure STAGE  Force a failure at: provision, sync, correctness, bench, pull.
                            For testing the teardown path.
  -h, --help                This message.

Exit codes: 0 ok, 1 error (teardown still ran), 3 refused by a budget gate.
EOF
}

while [ $# -gt 0 ]; do
    case "$1" in
        --dry-run)           DF_DRY_RUN=1 ;;
        --session-id)        DF_SESSION_ID="$2"; shift ;;
        --hypothesis)        DF_HYPOTHESIS="$2"; shift ;;
        --ledger)            DF_LEDGER="$2"; DF_LEDGER_EXPLICIT=1; shift ;;
        --session-limit)     DF_SESSION_LIMIT_MINUTES="$2"; shift ;;
        --watchdog-minutes)  DF_WATCHDOG_MINUTES="$2"; shift ;;
        --max-rate)          DF_PROVISION_ARGS="$DF_PROVISION_ARGS --max-rate $2"; shift ;;
        --simulate-failure)  DF_SIMULATE_FAILURE="$2"; shift ;;
        --state-file)        DF_STATE_FILE="$2"; shift ;;
        -h|--help)           usage; exit 0 ;;
        *)                   df_die "unknown option: $1 (try --help)" ;;
    esac
    shift
done
export DF_DRY_RUN

# Make the paths absolute before anything uses them.
#
# This is not tidiness. POSIX `.` searches $PATH when its argument contains no slash, so
# sourcing a relative state file like `--state-file mystate` fails with "not found" — and
# the failure lands *after* the instance has been created. Teardown would then see an
# empty DF_INSTANCE_ID, conclude nothing was provisioned, and leave the GPU running,
# which is the exact leak this script exists to prevent.
df_absolute() {
    case "$1" in
        /*) printf '%s\n' "$1" ;;
        *)  printf '%s/%s\n' "$PWD" "$1" ;;
    esac
}
DF_STATE_FILE=$(df_absolute "$DF_STATE_FILE")
DF_LEDGER=$(df_absolute "$DF_LEDGER")

# A rehearsal reads the real ledger but must not write to it.
if [ "$DF_DRY_RUN" = "1" ] && [ "${DF_LEDGER_EXPLICIT:-0}" != "1" ]; then
    DF_LEDGER=$(df_dryrun_ledger "$DF_LEDGER")
    df_log "(dry-run) ledger seeded from the real record; writes go to $DF_LEDGER"
fi

DF_INSTANCE_ID=""
DF_INSTANCE_GPU=""
DF_INSTANCE_RATE="0"
DF_INSTANCE_START_EPOCH=""
DF_WATCHDOG_PID=""
DF_CANCEL_FILE="$(dirname "$DF_STATE_FILE")/.deltaforge-watchdog-cancel"
DF_TEARDOWN_DONE=0

# ---------------------------------------------------------------------------
# Teardown. Installed before anything is created, and idempotent, because it can be
# reached from a normal exit, an error, or a signal.
# ---------------------------------------------------------------------------

df_teardown() {
    _status=$?
    [ "$DF_TEARDOWN_DONE" = "1" ] && return 0
    DF_TEARDOWN_DONE=1
    set +e

    df_log "[teardown] running (exit status $_status)"

    if [ -n "$DF_WATCHDOG_PID" ]; then
        : > "$DF_CANCEL_FILE" 2>/dev/null
        kill "$DF_WATCHDOG_PID" 2>/dev/null
        wait "$DF_WATCHDOG_PID" 2>/dev/null
        df_log "[teardown] watchdog $DF_WATCHDOG_PID stopped"
    fi

    if [ -n "$DF_INSTANCE_ID" ]; then
        df_log "[teardown] destroying instance $DF_INSTANCE_ID"
        df_vast_destroy "$DF_INSTANCE_ID"

        _minutes=0
        if [ -n "$DF_INSTANCE_START_EPOCH" ]; then
            _minutes=$(awk -v s="$(df_now_epoch)" -v t="$DF_INSTANCE_START_EPOCH" \
                'BEGIN { printf "%.2f", (s - t) / 60 }')
        fi
        df_ledger_append_destroy "$DF_LEDGER" "$DF_SESSION_ID" "$DF_INSTANCE_ID" \
            "$DF_INSTANCE_GPU" "$DF_INSTANCE_RATE" "$_minutes" "$DF_HYPOTHESIS" \
            "reconciled at teardown exit $_status"
        df_log "[teardown] ledger reconciled: ${_minutes} minutes at \$$DF_INSTANCE_RATE/hr"
    else
        df_log "[teardown] no instance was created; nothing to destroy"
    fi

    rm -f "$DF_CANCEL_FILE" "$DF_STATE_FILE" 2>/dev/null
    df_log "[teardown] complete"
    return 0
}

trap 'df_teardown' EXIT
trap 'df_log "interrupted"; exit 130' INT
trap 'df_log "terminated"; exit 143' TERM

df_stage_should_fail() {
    [ "$DF_SIMULATE_FAILURE" = "$1" ] || return 1
    df_warn "simulated failure at stage: $1"
    return 0
}

# ---------------------------------------------------------------------------
# Gate 2: session GPU-time soft gate.
#
# Checked BEFORE a run starts, never during one. A benchmark already executing at minute
# 59 finishes normally: killing it halfway would waste the money already spent on it, and
# leave nothing recorded in exchange.
#
# This is the gate expected to fire in ordinary operation. It exists to kill the "just one
# more attempt" pattern that otherwise runs a session into the hard watchdog with work in
# flight.
# ---------------------------------------------------------------------------

SESSION_MINUTES=$(df_ledger_session_minutes "$DF_LEDGER" "$DF_SESSION_ID")
df_log "session $DF_SESSION_ID has used ${SESSION_MINUTES} billed GPU minutes (limit ${DF_SESSION_LIMIT_MINUTES})"
if df_ge "$SESSION_MINUTES" "$DF_SESSION_LIMIT_MINUTES"; then
    df_log "REFUSED by the session GPU-time gate: ${SESSION_MINUTES} >= ${DF_SESSION_LIMIT_MINUTES} minutes."
    df_log "Do not start another hypothesis. Destroy any live instance, then finish"
    df_log "recording, writing up and merging the work already completed - none of which"
    df_log "needs a GPU - and end the session."
    exit 3
fi

# ---------------------------------------------------------------------------
# Provision
# ---------------------------------------------------------------------------

if df_stage_should_fail provision; then
    df_die "provisioning failed (simulated)"
fi

df_log "provisioning..."
PROVISION_FLAGS=""
[ "$DF_DRY_RUN" = "1" ] && PROVISION_FLAGS="--dry-run"
# shellcheck disable=SC2086
sh "$DF_REPO_ROOT/remote/provision.sh" $PROVISION_FLAGS \
    --session-id "$DF_SESSION_ID" \
    --hypothesis "$DF_HYPOTHESIS" \
    --ledger "$DF_LEDGER" \
    --max-minutes "$DF_WATCHDOG_MINUTES" \
    --state-file "$DF_STATE_FILE" \
    $DF_PROVISION_ARGS >/dev/null

[ -f "$DF_STATE_FILE" ] || df_die "provision.sh did not write $DF_STATE_FILE"
# shellcheck disable=SC1090
. "$DF_STATE_FILE"
df_log "instance $DF_INSTANCE_ID ($DF_INSTANCE_GPU) at \$$DF_INSTANCE_RATE/hr is now this script's responsibility"

# ---------------------------------------------------------------------------
# Watchdog: armed immediately after creation, before any work is attempted.
# ---------------------------------------------------------------------------

rm -f "$DF_CANCEL_FILE"
WATCHDOG_FLAGS=""
[ "$DF_DRY_RUN" = "1" ] && WATCHDOG_FLAGS="--dry-run"
# shellcheck disable=SC2086
sh "$DF_REPO_ROOT/remote/watchdog.sh" $WATCHDOG_FLAGS \
    --instance-id "$DF_INSTANCE_ID" \
    --timeout-minutes "$DF_WATCHDOG_MINUTES" \
    --cancel-file "$DF_CANCEL_FILE" \
    --ledger "$DF_LEDGER" \
    --session-id "$DF_SESSION_ID" \
    --rate "$DF_INSTANCE_RATE" \
    --gpu "$DF_INSTANCE_GPU" &
DF_WATCHDOG_PID=$!
df_log "watchdog started (pid $DF_WATCHDOG_PID, ${DF_WATCHDOG_MINUTES} minute hard limit)"

# ---------------------------------------------------------------------------
# Wait for SSH, then sync
# ---------------------------------------------------------------------------

wait_for_ssh() {
    if df_dry "would poll GET /instances/$DF_INSTANCE_ID/ until ssh is reachable"; then
        DF_SSH_HOST="root@dry-run.invalid"
        DF_SSH_PORT="22"
        return 0
    fi
    _deadline=$(( $(df_now_epoch) + DF_SSH_READY_TIMEOUT ))
    while [ "$(df_now_epoch)" -lt "$_deadline" ]; do
        _info=$(df_api GET "/instances/$DF_INSTANCE_ID/" 2>/dev/null || true)
        _status=$(printf '%s' "$_info" | jq -r '.instances.actual_status // empty' 2>/dev/null || true)
        if [ "$_status" = "running" ]; then
            DF_SSH_HOST="root@$(printf '%s' "$_info" | jq -r '.instances.ssh_host')"
            DF_SSH_PORT=$(printf '%s' "$_info" | jq -r '.instances.ssh_port')
            df_log "instance is running: $DF_SSH_HOST port $DF_SSH_PORT"
            return 0
        fi
        df_log "waiting for instance to start (status: ${_status:-unknown})"
        sleep 10
    done
    df_die "instance $DF_INSTANCE_ID did not become reachable within ${DF_SSH_READY_TIMEOUT}s"
}

if [ "$DF_DRY_RUN" != "1" ]; then
    df_load_api_key || df_die "no API key available for instance polling"
fi
wait_for_ssh

if df_stage_should_fail sync; then
    df_die "sync failed (simulated)"
fi
SYNC_FLAGS=""
[ "$DF_DRY_RUN" = "1" ] && SYNC_FLAGS="--dry-run"
# shellcheck disable=SC2086
sh "$DF_REPO_ROOT/remote/sync.sh" up $SYNC_FLAGS \
    --host "$DF_SSH_HOST" --port "$DF_SSH_PORT" --remote-dir "$DF_REMOTE_DIR"

# ---------------------------------------------------------------------------
# Remote work
# ---------------------------------------------------------------------------

# The image already ships torch 2.11 / CUDA 12.8 and a matching Triton, so DeltaForge is
# installed on top of it with --no-deps. Re-resolving torch here would either waste
# several minutes of paid time or, worse, replace the CUDA build with a CPU one.
remote_sh() {
    if df_dry "would run on the instance: $*"; then
        return 0
    fi
    ssh -p "$DF_SSH_PORT" -o StrictHostKeyChecking=accept-new "$DF_SSH_HOST" \
        "cd '$DF_REMOTE_DIR' && $*"
}

df_log "preparing the remote environment"
remote_sh "pip install --no-deps -e . && pip install safetensors transformers tokenizers"
remote_sh "python -m deltaforge.cli fetch-weights --dest '$DF_WEIGHTS_DIR'"

if df_stage_should_fail correctness; then
    df_die "correctness gate failed (simulated)"
fi
df_log "running correctness gates"
remote_sh "python -m deltaforge.cli correctness --weights '$DF_WEIGHTS_DIR' --session-id '$DF_SESSION_ID' --hypothesis '$DF_HYPOTHESIS'"

if df_stage_should_fail bench; then
    df_die "benchmark failed (simulated)"
fi
df_log "running the benchmark"
remote_sh "python -m deltaforge.cli bench --weights '$DF_WEIGHTS_DIR' --session-id '$DF_SESSION_ID' --hypothesis '$DF_HYPOTHESIS' --instance-id '$DF_INSTANCE_ID' --hourly-rate '$DF_INSTANCE_RATE'"

if df_stage_should_fail pull; then
    df_die "pulling results failed (simulated)"
fi
# shellcheck disable=SC2086
sh "$DF_REPO_ROOT/remote/sync.sh" down $SYNC_FLAGS \
    --host "$DF_SSH_HOST" --port "$DF_SSH_PORT" --remote-dir "$DF_REMOTE_DIR"

df_log "run complete; teardown follows"
exit 0
