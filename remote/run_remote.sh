#!/bin/sh
# The one entry point a session calls.
#
#   session gate -> provision -> watchdog -> sync up
#                -> batch (or: correctness -> bench)
#                -> pull results -> destroy
#
# Teardown pulls results before it destroys, so a failure late in a long batch does not
# take the slots that already succeeded with it.
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
# The batch of hypotheses to measure on this rental. A rental's fixed cost — image pull,
# torch, a 9.32 GB checkpoint, the GPU suite, and one max-autotune compile of the
# reference — is about 15 minutes, and each additional hypothesis costs 2-4. Testing one
# per rental pays that 15 minutes to buy a single measurement; nine rentals have been
# billed on this project and none produced a number. Empty means the old
# single-hypothesis path, which is still supported and is a batch of one.
DF_BATCH="${DF_BATCH:-}"
# Which benchmark columns the batch measures. Empty means the CLI default
# (eager, compiled, candidate, candidate_compiled).
#
# Every column is a live model state on the card: its own KV cache, and for a compiled one
# its own CUDA-graph pool. The reference's columns are built once and held for the whole
# batch, so they are resident while every candidate compiles on top of them. Rental 21 lost
# all nine slots to an OOM at 30.7 GiB of 31.4 with the four-column default, and only two
# of those columns score -- `compiled` and `candidate_compiled`. Dropping the two eager
# diagnostic columns is the cheapest way to fit a 32 GB card.
DF_COLUMNS="${DF_COLUMNS:-}"
DF_LEDGER="${DF_LEDGER:-$DF_REPO_ROOT/ledger/spend.jsonl}"
# 180 rather than 120: the cold-cache arithmetic in
# docs/superpowers/specs/2026-09-10-compile-cost-and-memory-design.md §4.1 puts one
# hypothesis -- the identity slot plus one kernel slot -- at 83-129 minutes, and 120 minus
# the 12-minute reserve left 108. The pre-flight check below refuses to rent at all when
# these two numbers disagree, so they are not independent: raising the check without
# raising this would refuse every run.
#
# The gate is a ceiling, not a spend commitment. Vast bills by the minute and the run
# destroys itself when the batch ends, so a warm session still pays for the ~40 minutes it
# uses; the month-to-date $45 gate is the real budget control. Three hours at $0.356/hr is
# $1.07.
DF_SESSION_LIMIT_MINUTES="${DF_SESSION_LIMIT_MINUTES:-180}"
# Always above the session gate: the gate is what should end a run, and a watchdog firing
# is a reportable fault. Raising the gate without raising this would make the backstop the
# routine control.
DF_WATCHDOG_MINUTES="${DF_WATCHDOG_MINUTES:-210}"
# How much of the session gate the batch may spend on slots, leaving the rest for setup
# and teardown. The batch stops itself before a slot it cannot finish, so the watchdog
# never has to — AGENT.md treats a watchdog firing as a reportable fault.
DF_BATCH_RESERVE_MINUTES="${DF_BATCH_RESERVE_MINUTES:-12}"
DF_REMOTE_DIR="${DF_REMOTE_DIR:-/workspace/deltaforge}"
# The checkpoint under test. Kept small on purpose: the benchmark holds a bf16 baseline
# and a candidate in one process on one card, so the model must leave room for both plus
# three max-autotune compilations. See docs/ARCHITECTURE.md on why not a newer Qwen.
DF_MODEL="${DF_MODEL:-Qwen/Qwen3.5-4B}"
DF_WEIGHTS_DIR="${DF_WEIGHTS_DIR:-/workspace/$(printf '%s' "${DF_MODEL#*/}" | tr 'A-Z' 'a-z')}"
DF_STATE_FILE="${DF_STATE_FILE:-$DF_REPO_ROOT/.deltaforge-instance}"
DF_SIMULATE_FAILURE="${DF_SIMULATE_FAILURE:-}"
# A fresh instance pulls a ~9 GB container image before sshd exists. 600s was not enough
# headroom for that on a well-connected host and turned a slow pull into a lost rental.
DF_SSH_READY_TIMEOUT="${DF_SSH_READY_TIMEOUT:-1200}"
# Abort a rental whose container image is not making progress. Eight rentals were spent
# discovering that a stalled pull is indistinguishable from a slow one if you only wait:
# every one of them ran the full readiness timeout and was billed for it. `status_msg`
# carries pull progress, so a message that has not changed in this many seconds while the
# instance is still not running means the pull is stuck, not slow. Failing here turns a
# 20-minute loss into a 5-minute one and makes testing another image cheap.
DF_PULL_STALL_SECONDS="${DF_PULL_STALL_SECONDS:-300}"
# Ceiling on the single longest remote step. `max-autotune` compiles three columns and
# can run away on a large graph; without a bound the run would sit there until the hard
# watchdog fired, and a watchdog firing is a reportable fault rather than a normal ending.
# Exceeding this fails the step cleanly, with teardown and the results already pulled.
DF_BENCH_TIMEOUT="${DF_BENCH_TIMEOUT:-2400}"
# A batch's single remote step is the entire measurement run, not one benchmark, so it
# gets its own ceiling. The batch stops itself at its deadline long before this; this is
# the backstop for a step that has stopped making progress at all. It has to stay clear of
# the largest deadline the gate can hand out — a 180-minute gate minus a few minutes of
# setup and the 12-minute reserve is already ~155 — or this backstop would become the thing
# that ends healthy batches.
DF_BATCH_TIMEOUT="${DF_BATCH_TIMEOUT:-9600}"
# Torch's fx-graph and autotune caches are enabled by default but write to
# /tmp/torchinductor_<user> on a box that gets destroyed, so every rental this project has
# ever run compiled cold -- and rental 22 spent ~40 minutes doing it. Point them somewhere
# we can pull home, and key the local copy by GPU and toolchain, because inductor keys its
# entries the same way and a 4090's cache buys a 5090 nothing.
DF_REMOTE_CACHE="${DF_REMOTE_CACHE:-/workspace/df-cache}"
DF_LOCAL_CACHE="${DF_LOCAL_CACHE:-$DF_REPO_ROOT/cache/compile}"
export DF_LOCAL_CACHE
DF_CACHE_PULL_TIMEOUT="${DF_CACHE_PULL_TIMEOUT:-180}"
# Filled in once the box has told us what it is. Until then a cache cannot be matched.
DF_CACHE_KEY="${DF_CACHE_KEY:-}"
# `nproc` on the box, not here: rental 22 showed a single inductor worker alongside python
# at 129% CPU, which is what a one-core view of the machine looks like. If a container
# really does report one core, that alone explains a 40-minute compile.
DF_COMPILE_ENV="TORCHINDUCTOR_CACHE_DIR=$DF_REMOTE_CACHE/inductor TRITON_CACHE_DIR=$DF_REMOTE_CACHE/triton TORCHINDUCTOR_COMPILE_THREADS=\$(nproc)"
DF_PROVISION_ARGS=""

usage() {
    cat <<'EOF'
Usage: remote/run_remote.sh [options]

  --dry-run                 Run every stage, both budget gates and the teardown trap
                            without contacting the create/destroy endpoints.
  --session-id ID           Session identifier (default: session-<UTC timestamp>).
  --hypothesis SLUG         Single hypothesis to test. Empty means a baseline run.
  --batch ID                Measure a whole batch of hypotheses on this one rental
                            (e.g. 001-calibration). Amortises the ~15 minute fixed
                            cost across 7-12 measurements instead of one. Mutually
                            exclusive with --hypothesis.
  --columns LIST            Comma-separated benchmark columns, or 'all'. Each column
                            is a resident model state on the card; the scoring pair
                            is compiled,candidate_compiled.
  --model REPO_ID           Checkpoint to benchmark (default: Qwen/Qwen3.5-4B).
  --ledger PATH             Ledger file (default: ledger/spend.jsonl).
  --session-limit N         Session GPU-time soft gate, in minutes (default: 180).
  --watchdog-minutes N      Hard watchdog timeout (default: 210). Keep it above the gate.
  --max-rate USD            Hourly rate ceiling, passed to provision.sh.
  --image REF               Container image. Use a non-Docker-Hub registry to test
                            whether a stalled pull is a Docker Hub rate limit.
  --exclude-machines IDS    Comma-separated machine ids to skip (ones that already
                            cost a rental without producing a result).
  --simulate-failure STAGE  Force a failure at: provision, sync, correctness, bench, pull.
                            For testing the teardown path.
  -h, --help                This message.

Exit codes: 0 ok, 1 error (teardown still ran), 3 refused by a budget gate,
            5 refused because the session gate cannot fit one hypothesis.

An errored hypothesis inside a batch is a recorded result, not a failed run: the batch
continues and exits 0. Only a failure of the rental itself is exit 1.
EOF
}

while [ $# -gt 0 ]; do
    case "$1" in
        --dry-run)           DF_DRY_RUN=1 ;;
        --session-id)        DF_SESSION_ID="$2"; shift ;;
        --hypothesis)        DF_HYPOTHESIS="$2"; shift ;;
        --batch)             DF_BATCH="$2"; shift ;;
        --columns)           DF_COLUMNS="$2"; shift ;;
        --model)             DF_MODEL="$2"
                             DF_WEIGHTS_DIR="/workspace/$(printf '%s' "${DF_MODEL#*/}" | tr 'A-Z' 'a-z')"
                             shift ;;
        --ledger)            DF_LEDGER="$2"; DF_LEDGER_EXPLICIT=1; shift ;;
        --session-limit)     DF_SESSION_LIMIT_MINUTES="$2"; shift ;;
        --watchdog-minutes)  DF_WATCHDOG_MINUTES="$2"; shift ;;
        --max-rate)          DF_PROVISION_ARGS="$DF_PROVISION_ARGS --max-rate $2"; shift ;;
        --image)             DF_PROVISION_ARGS="$DF_PROVISION_ARGS --image $2"; shift ;;
        --exclude-machines)  DF_PROVISION_ARGS="$DF_PROVISION_ARGS --exclude-machines $2"; shift ;;
        --simulate-failure)  DF_SIMULATE_FAILURE="$2"; shift ;;
        --state-file)        DF_STATE_FILE="$2"; shift ;;
        -h|--help)           usage; exit 0 ;;
        *)                   df_die "unknown option: $1 (try --help)" ;;
    esac
    shift
done
export DF_DRY_RUN

if [ -n "$DF_BATCH" ] && [ -n "$DF_HYPOTHESIS" ]; then
    df_die "--batch and --hypothesis are mutually exclusive: --batch '$DF_BATCH' names a manifest of hypotheses and --hypothesis '$DF_HYPOTHESIS' names one. Pick whichever the run means; guessing would put the wrong label on every record it writes."
fi

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
DF_SSH_HOST=""
DF_SSH_PORT="22"
DF_SSH_READY=0
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

    # Rescue the results before the box goes away.
    #
    # The trap could only ever destroy, so a failure after the last successful sync took
    # every measurement with it — survivable when a run was 10 minutes and one hypothesis,
    # not when it is two hours and nine. Batch mode writes each slot's record the moment
    # that slot finishes, which is only worth anything if something fetches them.
    #
    # Best-effort and fully guarded: this must never be able to prevent the destroy below.
    # A leaked instance costs about $13/day; a lost result costs one rental.
    # `DF_SSH_READY` rather than merely `DF_SSH_HOST`: a host that never answered has
    # nothing to pull, and trying anyway spends the connect timeout on an instance that is
    # still billing. Rental 50121263 died exactly there, and paid 20s for the discovery.
    if [ -n "$DF_INSTANCE_ID" ] && [ "${DF_SSH_READY:-0}" = "1" ] && [ "$DF_DRY_RUN" != "1" ]; then
        df_log "[teardown] pulling results before destroying"
        # Hard-bounded: this runs while the instance is still billing, and no amount of
        # results is worth an unbounded wait before the destroy call.
        if timeout "${DF_TEARDOWN_PULL_TIMEOUT:-180}" \
            sh "$DF_REPO_ROOT/remote/sync.sh" down \
            --host "$DF_SSH_HOST" --port "${DF_SSH_PORT:-22}" \
            --remote-dir "$DF_REMOTE_DIR" 2>&1; then
            df_log "[teardown] results pulled"
        else
            df_warn "[teardown] could not pull results (exit $?); destroying anyway"
        fi
    fi

    # The compile cache, after the results and before the destroy. Same rule as the pull
    # above -- the trap cannot rsync from a dead box -- and the same guards: bounded, and
    # unable to prevent the destroy. Results come first because a lost measurement costs a
    # rental and a cold cache costs 40 minutes, once.
    if [ -n "$DF_INSTANCE_ID" ] && [ "${DF_SSH_READY:-0}" = "1" ] && [ "$DF_DRY_RUN" != "1" ]; then
        df_log "[teardown] pulling the compile cache"
        if timeout "$DF_CACHE_PULL_TIMEOUT" \
            sh "$DF_REPO_ROOT/remote/sync.sh" cache-down \
            --host "$DF_SSH_HOST" --port "${DF_SSH_PORT:-22}" \
            --remote-dir "$DF_REMOTE_CACHE" --cache-key "${DF_CACHE_KEY:-unknown}" 2>&1; then
            # sync.sh exits 0 even when nothing came back, so ask the disk rather than the
            # exit status: claiming warmth that is not there misreports the one number
            # this batch exists to measure.
            if df_cache_is_warm "$DF_LOCAL_CACHE/${DF_CACHE_KEY:-unknown}"; then
                df_log "[teardown] compile cache pulled; the next rental on this card starts warm"
            else
                df_log "[teardown] no compile cache came back; the next rental on this card compiles cold"
                rmdir "$DF_LOCAL_CACHE/${DF_CACHE_KEY:-unknown}" 2>/dev/null || true
            fi
        else
            df_warn "[teardown] could not pull the compile cache (exit $?); destroying anyway"
        fi
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

# Pre-flight: can this session finish one hypothesis at all?
#
# The gate above refuses a session that is spent. This refuses one that is not spent enough
# -- a session with time to rent, compile, and then stop before scoring anything. Nine
# rentals have been billed on this project without producing a number, and the cheapest of
# those failures would have been not renting.
#
# The minimum is two slots: the identity champion calibrates the harness but scores no
# hypothesis, so a rental that fits only that has bought no science. The estimates come
# from the last rental on this card when there was one, and from the deliberately
# pessimistic cold numbers in `batch.py` when there was not -- being wrong optimistically
# here costs a whole rental.
DF_PHASE_SETUP_S="${DF_PHASE_SETUP_S:-1500}"
DF_PHASE_REFERENCE_COMPILE_S="${DF_PHASE_REFERENCE_COMPILE_S:-2400}"
DF_PHASE_SLOT_S="${DF_PHASE_SLOT_S:-1980}"
# `DF_CACHE_KEY` is read off the rented box and so is still empty here -- this gate runs
# before anything is provisioned. Asking for `$DF_LOCAL_CACHE/${DF_CACHE_KEY:-unknown}/`
# therefore missed on every rental this project has run, and the "measured" branch below
# had never once been taken. `df_phase_estimates` surveys the cards that *have* measured
# something and takes the most pessimistic value for each phase.
DF_PHASES_MEASURED=$(df_phase_estimates "$DF_LOCAL_CACHE" "${DF_CACHE_KEY:-}")
if [ -n "$DF_PHASES_MEASURED" ]; then
    eval "$DF_PHASES_MEASURED"
    df_log "pre-flight: measured phase costs from $DF_LOCAL_CACHE -- $(printf '%s' "$DF_PHASES_MEASURED" | tr '\n' ' ')"
fi
DF_NEEDED_MIN=$(awk -v s="$DF_PHASE_SETUP_S" -v c="$DF_PHASE_REFERENCE_COMPILE_S" \
    -v t="$DF_PHASE_SLOT_S" -v r="$DF_BATCH_RESERVE_MINUTES" \
    'BEGIN { printf "%.1f", (s + c + 2 * t) / 60 + r }')
DF_AVAILABLE_MIN=$(awk -v l="$DF_SESSION_LIMIT_MINUTES" -v u="$SESSION_MINUTES" \
    'BEGIN { printf "%.1f", l - u }')
if df_ge "$DF_NEEDED_MIN" "$DF_AVAILABLE_MIN"; then
    df_warn "REFUSED: the session gate cannot fit one hypothesis."
    df_warn "needs ${DF_NEEDED_MIN} minutes (setup + reference compile + two slots + reserve),"
    df_warn "has ${DF_AVAILABLE_MIN}. Raise --session-limit, or start a new session."
    df_warn "Renting anyway would buy a compile and no measurement, which is how the first"
    df_warn "nine rentals were spent."
    exit 5
fi
df_log "pre-flight: ${DF_AVAILABLE_MIN} minutes available; one hypothesis needs ${DF_NEEDED_MIN}"

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
    if df_dry "would poll GET /api/v1/instances/ until $DF_INSTANCE_ID is running and ssh answers"; then
        DF_SSH_HOST="root@dry-run.invalid"
        DF_SSH_PORT="22"
        return 0
    fi
    _deadline=$(( $(df_now_epoch) + DF_SSH_READY_TIMEOUT ))
    _dumped=0
    _last_msg=""
    _msg_changed_at=$(df_now_epoch)
    _settled_logged=0
    while [ "$(df_now_epoch)" -lt "$_deadline" ]; do
        _listing=$(df_api_v1 GET "/instances/" 2>/dev/null || true)
        _row=$(df_instance_row "$_listing" "$DF_INSTANCE_ID")
        # `actual_status` is the *container* state: null, then "loading", then "running".
        # `cur_state` is the *contract* state and reads "running" from the moment the
        # instance is created, before a byte of the image has landed.
        #
        # Falling back from one to the other was therefore backwards. The fallback fired
        # exactly when `actual_status` had not been populated yet -- which is precisely
        # when the instance is not ready -- so readiness was declared on the first poll of
        # every rental, and the ssh probe ran against a container that did not exist. Five
        # rentals were written off as "the host never answered sshd" on the strength of
        # it. The committed fixture encodes the same trap: `actual_status: null` with
        # `cur_state: "running"` is a freshly created instance, not a live one.
        #
        # So gate on the container state alone. An absent `actual_status` is "not ready
        # yet", not "ready"; the stall guard below and the readiness deadline bound how
        # long that can go on.
        _actual=$(printf '%s' "$_row" | jq -r '.actual_status // empty' 2>/dev/null || true)
        _contract=$(printf '%s' "$_row" | jq -r '.cur_state // empty' 2>/dev/null || true)
        _status="${_actual:-${_contract:+starting}}"
        if [ "$_actual" = "running" ]; then
            _host=$(printf '%s' "$_row" | jq -r '.ssh_host // empty')
            _port=$(printf '%s' "$_row" | jq -r '.ssh_port // empty')
            if [ -n "$_host" ] && [ -n "$_port" ]; then
                DF_SSH_HOST="root@$_host"
                DF_SSH_PORT="$_port"
                df_log "instance is running: $DF_SSH_HOST port $DF_SSH_PORT"
                # "running" means the container started, not that sshd is accepting yet.
                # Probe until it answers, so the first real command is not the thing that
                # discovers the connection is not up.
                _ssh_started=$(df_now_epoch)
                _ssh_last_msg=""
                _ssh_msg_changed_at=$(df_now_epoch)
                while [ "$(df_now_epoch)" -lt "$_deadline" ]; do
                    # shellcheck disable=SC2086
                    _probe=$(ssh $DF_SSH_ID -p "$DF_SSH_PORT" -o StrictHostKeyChecking=accept-new \
                        -o ConnectTimeout=15 -o BatchMode=yes "$DF_SSH_HOST" true 2>&1) && {
                        df_log "ssh is answering"
                        DF_SSH_READY=1
                        return 0
                    }
                    # A rejected key is a permanent failure, not a slow boot: the image is
                    # answering on the port and refusing us. Waiting cannot fix it, so do
                    # not pay out the readiness timeout discovering that. Cost one rental.
                    case "$_probe" in
                        *"Permission denied"*|*"publickey"*)
                            df_die "instance $DF_INSTANCE_ID refused the ssh key: ${_probe##*$(printf '\n')}. The image does not honour the Vast account key. See docs/GPU-ACCESS.md; provision.sh injects the key through PUBLIC_KEY and authorized_keys, so an image ignoring both needs its own handling."
                            ;;
                    esac
                    # Same stall budget as the pull, and now the same progress signal.
                    #
                    # This loop used to be blind: it counted a flat 300s from the moment
                    # `cur_state` said `running` and destroyed the instance regardless of
                    # what the container was doing. Two rentals died to that on
                    # 2026-09-08 -- both reported `running` on the *first* poll, before a
                    # single byte of the image had landed, so the whole budget was spent
                    # on hosts that were still starting normally. A guard that cannot see
                    # progress cannot tell a dead host from a slow one, which is the exact
                    # mistake the outer loop was already fixed for.
                    #
                    # So track `status_msg` here too: a container still rewriting it is
                    # making progress and gets the full readiness timeout, while one that
                    # has gone static for the stall budget is destroyed early as before.
                    _ssh_row=$(df_instance_row "$(df_api_v1 GET "/instances/" 2>/dev/null || true)" "$DF_INSTANCE_ID")
                    _ssh_state=$(printf '%s' "$_ssh_row" | jq -r '.actual_status // empty' 2>/dev/null || true)
                    _ssh_msg=$(printf '%s' "$_ssh_row" | jq -r '.status_msg // empty' 2>/dev/null | tr -d '\n' | cut -c1-70)
                    # `_ssh_msg_changed_at` starts at `_ssh_started`, so a host that
                    # reports no `status_msg` gets exactly the old flat budget -- which is
                    # the right answer *here*, unlike in the outer loop, because the
                    # container is genuinely running by this point and sshd is the only
                    # thing still missing. A host that does report progress gets extended.
                    if [ -n "$_ssh_msg" ] && [ "$_ssh_msg" != "$_ssh_last_msg" ]; then
                        _ssh_last_msg="$_ssh_msg"
                        _ssh_msg_changed_at=$(df_now_epoch)
                    elif [ $(( $(df_now_epoch) - _ssh_msg_changed_at )) -ge "$DF_PULL_STALL_SECONDS" ]; then
                        df_die "instance $DF_INSTANCE_ID has been running for $(( $(df_now_epoch) - _ssh_started ))s without sshd answering and has not progressed in ${DF_PULL_STALL_SECONDS}s (status: ${_ssh_state:-unknown} | ${_ssh_msg:-no status_msg}). Destroying rather than paying out the ${DF_SSH_READY_TIMEOUT}s timeout. Last probe: ${_probe:-no output}"
                    fi
                    df_log "instance is running; waiting for sshd (status: ${_ssh_state:-unknown}${_ssh_msg:+ | $_ssh_msg})"
                    sleep 10
                done
                df_die "instance $DF_INSTANCE_ID started but ssh never answered within ${DF_SSH_READY_TIMEOUT}s"
            fi
        fi
        if [ "${DF_DEBUG_POLL:-0}" = "1" ]; then
            df_log "raw row: $(printf '%s' "$_row" | head -c 2500)"
        fi
        if [ -z "$_row" ] && [ "$_dumped" = "0" ]; then
            # One raw dump the first time the instance is not in its own listing. Ten
            # minutes of "status: unknown" with nothing to look at is what made the last
            # API change cost a provisioning cycle to diagnose.
            df_warn "instance $DF_INSTANCE_ID not present in the listing; raw response follows"
            printf '%s\n' "$_listing" | head -c 600 >&2
            printf '\n' >&2
            _dumped=1
        fi
        # `status_msg` carries the image-pull progress. Without it, a slow pull and a
        # dead host look identical from here, and telling them apart cost two rentals.
        _msg=$(printf '%s' "$_row" | jq -r '.status_msg // empty' 2>/dev/null | tr -d '\n' | cut -c1-70)
        df_log "waiting for instance to start (status: ${_status:-unknown}${_msg:+ | $_msg})"

        # Stall detection. A pull that is merely slow keeps rewriting status_msg with new
        # byte counts; a pull that is stuck repeats the same line forever. Distinguishing
        # them is what the eight lost rentals paid for, so it is checked rather than
        # waited out.
        #
        # The guard needs a progress signal to mean anything. The Docker Hub hangs it was
        # built for all reported one ("Pulling fs layer", repeated forever), but some
        # hosts populate no `status_msg` at all -- and an always-empty message looks
        # identical to a frozen one, so applying the budget to it would destroy healthy
        # instances at 300s for the crime of being quiet. Where there is no signal, say so
        # and let the readiness deadline do the bounding instead of inventing a verdict.
        # The container state is folded into the key so null -> loading -> running counts
        # as the progress it is.
        _progress="$_status|$_msg"
        if [ "$_progress" != "$_last_msg" ]; then
            _last_msg="$_progress"
            _msg_changed_at=$(df_now_epoch)
        elif [ -z "$_msg" ]; then
            : # no progress signal from this host; bounded by DF_SSH_READY_TIMEOUT alone
        elif df_pull_settled "$_msg"; then
            # The download finished; verification, extraction and container start emit no
            # status_msg updates, so this message freezes because the pull SUCCEEDED. Same
            # bounding as the empty case: DF_SSH_READY_TIMEOUT, not the stall budget.
            if [ "$_settled_logged" = "0" ]; then
                df_log "pull has finished downloading; extraction reports no progress, so the readiness timeout bounds it from here"
                _settled_logged=1
            fi
        elif [ $(( $(df_now_epoch) - _msg_changed_at )) -ge "$DF_PULL_STALL_SECONDS" ]; then
            df_die "instance $DF_INSTANCE_ID has not progressed in ${DF_PULL_STALL_SECONDS}s (status: ${_status:-unknown} | ${_msg:-no status_msg}). Image pull is stuck, not slow: destroying rather than paying out the ${DF_SSH_READY_TIMEOUT}s timeout. Try --image on another registry, or add DOCKER_LOGIN_USER/DOCKER_LOGIN_TOKEN to .env."
        fi
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
    # shellcheck disable=SC2086
    ssh $DF_SSH_ID -p "$DF_SSH_PORT" -o StrictHostKeyChecking=accept-new "$DF_SSH_HOST" \
        "cd '$DF_REMOTE_DIR' && $*"
}

# Like `remote_sh`, but the box's answer comes back on stdout instead of being logged. Used
# for the two facts only the box knows: what card this is, and how many cores it will admit
# to. Never used for a step whose *effect* matters -- a dry run must not silently skip work.
remote_capture() {
    # shellcheck disable=SC2086
    ssh $DF_SSH_ID -p "$DF_SSH_PORT" -o StrictHostKeyChecking=accept-new "$DF_SSH_HOST" \
        "cd '$DF_REMOTE_DIR' && $*" 2>/dev/null
}

df_log "preparing the remote environment"
# g++ AND the Python headers, for the same reason: Triton JIT-compiles a small C shim
# (`cuda_utils.c`) the first time a kernel runs, and that shim does `#include <Python.h>`.
# The vastai image ships the headers, the ghcr ai-dock image does not, and the difference
# is invisible until a kernel actually runs -- rental 26 paid 88 billed minutes to reach a
# GPU suite where every Triton test died on `fatal error: Python.h: No such file or
# directory`. Installed here, before the checkpoint download and the suite, so an
# incomplete toolchain costs a minute rather than a rental.
remote_sh "command -v g++ >/dev/null 2>&1 || (apt-get update -qq && apt-get install -y -qq g++)"
remote_sh "python3 -c 'import sysconfig, os, sys; sys.exit(0 if os.path.exists(os.path.join(sysconfig.get_paths()[\"include\"], \"Python.h\")) else 1)' || (apt-get update -qq && apt-get install -y -qq python3-dev)"
# Guarantee a `python` on PATH before anything tries to use one.
#
# Cost one rental (50119910, 9.38 min, $0.0557, 2026-09-07): the ai-dock/vast images ship
# `python3` and `pip` but no `python`, so every remote step after the installs died with
# `bash: python: command not found` — after the container pull, the torch download and the
# pip installs had all been paid for. The interpreter is the first thing to establish,
# before it is the twentieth thing to discover.
remote_sh "command -v python >/dev/null 2>&1 || ln -sf \"\$(command -v python3)\" /usr/local/bin/python"
remote_sh "python --version"
# torch from PyTorch's own CDN rather than baked into the image: see the note on DF_IMAGE
# in provision.sh. The cu128 wheel brings its matching Triton with it.
remote_sh "python -m pip install --quiet torch --index-url https://download.pytorch.org/whl/cu128"
# Getting the 9.32 GB checkpoint down fast, on billed wall-clock time.
#
# The history here is two deprecations deep, so it is written out rather than rediscovered.
# `huggingface_hub[hf_transfer]` was the fast path; the extra was dropped in
# huggingface-hub 1.30, which *warns and installs nothing* -- a warning is not a failure,
# so nothing reported it except the clock. Installing `hf_transfer` as its own package
# fixed that, and then the box told us the rest: hf_transfer itself is now superseded by
# Xet, and `HF_HUB_ENABLE_HF_TRANSFER` is ignored. `HF_XET_HIGH_PERFORMANCE` is the current
# knob. Both are set in `cli.py`; whichever the installed version honours, one of them wins.
#
# `accelerate` is REQUIRED, not optional. Without it `transformers.from_pretrained` refuses
# any `device_map` outright, so the weight-value oracle -- the test that decides whether the
# reference interprets the checkpoint correctly, and therefore whether any number
# downstream of it means anything -- cannot even be constructed. Cost rental 50121911:
# 10.70 min and a full checkpoint download to reach an error that says "pip install
# accelerate".
#
# `transformers` is pinned EXACTLY, and the exactness is the point.
#
# It used to be `>=5.16,<6`, a floor chosen to stop the Qwen3.5 architecture disappearing
# mid-run. A floor cannot do that job, because the oracle is HuggingFace: the assertion
# compares our tokens to *theirs*, so their version is an input to the experiment and a
# floor lets that input move between rentals without anything in this repo changing.
#
# It did move. `test_reference_greedy_decode_matches_the_oracle_token_for_token` passed on
# 2026-09-07 and failed on 2026-09-10 at a single argmax (`index 2: 11540 != 1528`), with
# the logits oracle and the mRoPE test still passing. The floor resolved to 5.16.1 on the
# first date and to 5.17.0 on the second -- 5.17.0 was released 2026-09-09, between the two
# runs. Pinning to the version that was actually validated is what makes the reference a
# fixed thing to measure against.
#
# Raising this pin is a deliberate act that re-opens the validation question: bump it, and
# the next rental's oracle result is what says whether the reference still holds.
remote_sh "python -m pip install --quiet --no-deps -e . && python -m pip install --quiet safetensors 'transformers==5.16.1' tokenizers pytest huggingface_hub hf_transfer accelerate"
remote_sh "python -c 'import accelerate, transformers; print(\"accelerate\", accelerate.__version__, \"transformers\", transformers.__version__)'"
remote_sh "python -c \"import torch, triton; print('torch', torch.__version__, 'triton', triton.__version__, 'cuda', torch.version.cuda, torch.cuda.get_device_name(0))\""

# The cache is only ever reused on the hardware and toolchain that built it, so the key is
# read off the box rather than assumed. A key we cannot read means no reuse, not a failure.
if [ "$DF_DRY_RUN" = "1" ]; then
    DF_CACHE_KEY="dryrun-cache-key"
else
    DF_CACHE_KEY=$(remote_capture "python -c \"import torch,re;print(re.sub(r'[^A-Za-z0-9]+','',torch.cuda.get_device_name(0))+'-'+torch.__version__.split('+')[0]+'-cu'+str(torch.version.cuda))\"" 2>/dev/null | tr -d '\r' | tail -1)
fi
[ -n "$DF_CACHE_KEY" ] || DF_CACHE_KEY="unknown"
df_log "compile cache key: $DF_CACHE_KEY"
if [ "$DF_DRY_RUN" != "1" ]; then
    df_log "compile workers: $(remote_capture 'nproc' | tr -d '\r' | tail -1) cores reported by the box"
fi
# The direction is sync.sh's first positional argument, so flags come after it.
CACHE_FLAGS=""
[ "$DF_DRY_RUN" = "1" ] && CACHE_FLAGS="--dry-run"
# shellcheck disable=SC2086
sh "$DF_REPO_ROOT/remote/sync.sh" cache-up $CACHE_FLAGS \
    --host "$DF_SSH_HOST" --port "${DF_SSH_PORT:-22}" \
    --remote-dir "$DF_REMOTE_CACHE" --cache-key "$DF_CACHE_KEY" || \
    df_warn "could not send the compile cache up; this rental compiles cold"
remote_sh "python -m deltaforge.cli fetch-weights --model '$DF_MODEL' --dest '$DF_WEIGHTS_DIR'"

# The GPU-marked tests skip themselves on a CPU machine, so this is the first place they
# ever run. `oracle_test.py` is the weight-value oracle against HuggingFace: until it has
# passed, the reference is only proven structurally correct, and every number downstream
# of it would be measuring an unvalidated model. It runs before the gates, not after.
if df_stage_should_fail gputests; then
    df_die "GPU test suite failed (simulated)"
fi
df_log "running the GPU test suite: weight-value oracle and kernel numerics"
remote_sh "DELTAFORGE_WEIGHTS_DIR='$DF_WEIGHTS_DIR' python -m pytest -m gpu -q"

if [ -n "$DF_BATCH" ]; then
    # Batch mode runs the gates and the benchmark per hypothesis inside one process, so
    # the reference is loaded once and compiled once for the whole batch. Splitting it
    # into separate correctness and bench steps as the single-hypothesis path does would
    # pay both of those costs twice.
    if df_stage_should_fail correctness; then
        df_die "batch failed (simulated)"
    fi

    # The batch must stop itself before the watchdog does. Its deadline is the session
    # gate minus what is already spent minus a reserve for teardown and the final pull.
    DF_ELAPSED_MINUTES=$(awk -v s="$(df_now_epoch)" -v t="${DF_INSTANCE_START_EPOCH:-$(df_now_epoch)}" \
        'BEGIN { printf "%.2f", (s - t) / 60 }')
    DF_BATCH_DEADLINE=$(awk -v now="$(df_now_epoch)" -v limit="$DF_SESSION_LIMIT_MINUTES" \
        -v used="$SESSION_MINUTES" -v elapsed="$DF_ELAPSED_MINUTES" -v reserve="$DF_BATCH_RESERVE_MINUTES" \
        'BEGIN { printf "%d", now + (limit - used - elapsed - reserve) * 60 }')
    df_log "batch $DF_BATCH: deadline in $(awk -v d="$DF_BATCH_DEADLINE" -v n="$(df_now_epoch)" \
        'BEGIN { printf "%.1f", (d - n) / 60 }') minutes (session limit ${DF_SESSION_LIMIT_MINUTES}, \
already used ${SESSION_MINUTES}, this rental ${DF_ELAPSED_MINUTES}, reserve ${DF_BATCH_RESERVE_MINUTES})"

    df_log "running batch $DF_BATCH (remote step ceiling ${DF_BATCH_TIMEOUT}s)"
    # expandable_segments costs nothing and buys back the allocator fragmentation that a
    # sequence of max-autotune compilations leaves behind. It is not a fix for genuinely
    # not fitting -- see --columns for that -- but the OOM messages asked for it by name.
    remote_sh "$DF_COMPILE_ENV PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True timeout ${DF_BATCH_TIMEOUT} python -m deltaforge.cli batch --model '$DF_MODEL' --weights '$DF_WEIGHTS_DIR' --session-id '$DF_SESSION_ID' --batch '$DF_BATCH' --deadline-epoch '$DF_BATCH_DEADLINE' --instance-id '$DF_INSTANCE_ID' --hourly-rate '$DF_INSTANCE_RATE' --phases-env '$DF_REMOTE_CACHE/phases.env'${DF_COLUMNS:+ --columns '$DF_COLUMNS'}"
else
    if df_stage_should_fail correctness; then
        df_die "correctness gate failed (simulated)"
    fi
    df_log "running correctness gates"
    remote_sh "$DF_COMPILE_ENV python -m deltaforge.cli correctness --model '$DF_MODEL' --weights '$DF_WEIGHTS_DIR' --session-id '$DF_SESSION_ID' --hypothesis '$DF_HYPOTHESIS'"

    # Pull what has been recorded so far, before the longest and riskiest step. The
    # correctness record is the expensive part of this run — it needed the checkpoint on a
    # GPU — and a benchmark that fails or is killed by the watchdog must not take it down
    # with it.
    # shellcheck disable=SC2086
    sh "$DF_REPO_ROOT/remote/sync.sh" down $SYNC_FLAGS \
        --host "$DF_SSH_HOST" --port "$DF_SSH_PORT" --remote-dir "$DF_REMOTE_DIR"

    if df_stage_should_fail bench; then
        df_die "benchmark failed (simulated)"
    fi
    df_log "running the benchmark (remote step ceiling ${DF_BENCH_TIMEOUT}s)"
    remote_sh "$DF_COMPILE_ENV timeout ${DF_BENCH_TIMEOUT} python -m deltaforge.cli bench --model '$DF_MODEL' --weights '$DF_WEIGHTS_DIR' --session-id '$DF_SESSION_ID' --hypothesis '$DF_HYPOTHESIS' --instance-id '$DF_INSTANCE_ID' --hourly-rate '$DF_INSTANCE_RATE'"
fi

if df_stage_should_fail pull; then
    df_die "pulling results failed (simulated)"
fi
# shellcheck disable=SC2086
sh "$DF_REPO_ROOT/remote/sync.sh" down $SYNC_FLAGS \
    --host "$DF_SSH_HOST" --port "$DF_SSH_PORT" --remote-dir "$DF_REMOTE_DIR"

df_log "run complete; teardown follows"
exit 0
