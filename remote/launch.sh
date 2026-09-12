#!/bin/sh
# Start a rental in the background and leave evidence that it started.
#
# Rental 29 (2026-09-11) is why this exists. It was launched with a hand-rolled
# `nohup ... &`, exited 4 three seconds later because no offer met the filters, and
# reported nothing at all: the monitor was attached with `tail -n 0 -f` six and a half
# seconds after the process had already ended, so it sat forever on a file that would
# never grow again. Nothing was billed and nothing was broken remotely. The run was
# right; the observation was wrong, and the session went silent for a day.
#
# Two properties prevent that, and both belong to the launcher rather than to whoever
# remembers the right tail flags at three in the morning:
#
#   * **The log always ends with one terminal marker line**, whatever happened, so a
#     reader arriving after the run is over still learns that it is over. It is prefixed
#     `[deltaforge]` so the usual monitor filter catches it.
#   * **The exit status is written to a file**, so "is it still going?" is a question
#     that can be answered at any moment, by anyone, without having watched from the
#     start. Silence is never evidence of health.
#
# The pid file is written only by the parent and the status file only by the child, so
# the two never race for the same bytes -- which is the bug this script is about.
#
# Attach monitors with `tail -n +1 -F`, never `tail -n 0 -f`. Replaying a log from its
# first line costs nothing and is the whole difference between seeing a fast failure and
# hanging on one.

set -eu

unset CDPATH
DF_REPO_ROOT=$(cd -- "$(dirname -- "$0")/.." && pwd)
export DF_REPO_ROOT
. "$DF_REPO_ROOT/remote/lib.sh"

DF_LAUNCH_LOG=""
DF_LAUNCH_STATUS_ONLY=0
DF_LAUNCH_TARGET="${DF_LAUNCH_TARGET:-$DF_REPO_ROOT/remote/run_remote.sh}"

usage() {
    cat <<'EOF'
Usage: remote/launch.sh --log PATH [run_remote.sh options...]
       remote/launch.sh --status --log PATH

  --log PATH    Where to write the run's output. Required. Two siblings are written
                next to it: PATH.pid while it runs, PATH.status when it ends.
  --status      Report on the launch that owns PATH instead of starting a new one.
  -h, --help    This message.

Every option after these is passed through to run_remote.sh untouched.

Exit codes, launching:  0 started.
Exit codes, --status:   0 the run finished successfully, 1 it failed or was killed
                        without recording anything, 2 it is still running.
EOF
}

while [ $# -gt 0 ]; do
    case "$1" in
        --log)     DF_LAUNCH_LOG="$2"; shift ;;
        --status)  DF_LAUNCH_STATUS_ONLY=1 ;;
        -h|--help) usage; exit 0 ;;
        --)        shift; break ;;
        *)         break ;;
    esac
    shift
done

[ -n "$DF_LAUNCH_LOG" ] || df_die "--log PATH is required (try --help)"
DF_STATUS_FILE="$DF_LAUNCH_LOG.status"
DF_PID_FILE="$DF_LAUNCH_LOG.pid"

# ---------------------------------------------------------------------------
# --status: answer the question rental 29 had no way to ask
# ---------------------------------------------------------------------------

if [ "$DF_LAUNCH_STATUS_ONLY" -eq 1 ]; then
    if [ -f "$DF_STATUS_FILE" ]; then
        _code=$(cat "$DF_STATUS_FILE")
        df_log "run finished with exit status $_code (log: $DF_LAUNCH_LOG)"
        [ "$_code" = "0" ] || exit 1
        exit 0
    fi

    [ -f "$DF_PID_FILE" ] || df_die "no launch owns $DF_LAUNCH_LOG: nothing was launched with this log"

    _pid=$(cat "$DF_PID_FILE")
    if kill -0 "$_pid" 2>/dev/null; then
        df_log "run still running (pid $_pid, log: $DF_LAUNCH_LOG)"
        exit 2
    fi

    # The child writes its status as its last act, so a pid that has just gone may beat
    # the status file by a few milliseconds. Look again before calling it a death.
    sleep 1
    if [ -f "$DF_STATUS_FILE" ]; then
        _code=$(cat "$DF_STATUS_FILE")
        df_log "run finished with exit status $_code (log: $DF_LAUNCH_LOG)"
        [ "$_code" = "0" ] || exit 1
        exit 0
    fi

    df_warn "pid $_pid is gone and never recorded an exit status: the run was killed."
    df_warn "Anything it created may still be live. Check the instance list before relaunching."
    exit 1
fi

# ---------------------------------------------------------------------------
# Launching
# ---------------------------------------------------------------------------

[ -x "$DF_LAUNCH_TARGET" ] || df_die "launch target is not executable: $DF_LAUNCH_TARGET"

DF_LOG_DIR=$(dirname -- "$DF_LAUNCH_LOG")
mkdir -p -- "$DF_LOG_DIR" || df_die "cannot create log directory $DF_LOG_DIR"

if [ -f "$DF_PID_FILE" ] && [ ! -f "$DF_STATUS_FILE" ]; then
    _prior=$(cat "$DF_PID_FILE")
    if kill -0 "$_prior" 2>/dev/null; then
        df_die "a launch (pid $_prior) is already using $DF_LAUNCH_LOG; use --status, or pick another --log"
    fi
fi

: > "$DF_LAUNCH_LOG" || df_die "cannot write $DF_LAUNCH_LOG"
rm -f -- "$DF_STATUS_FILE" "$DF_PID_FILE"

# Ignore SIGHUP across the fork, so the run survives the shell that started it. The
# child inherits the ignored disposition; the parent restores its own below.
trap '' HUP

{
    set +e
    "$DF_LAUNCH_TARGET" "$@" >> "$DF_LAUNCH_LOG" 2>&1
    _code=$?
    set -e
    printf '[deltaforge] [launch] run exited %d\n' "$_code" >> "$DF_LAUNCH_LOG"
    # Written last, and atomically, so the file's existence means "this is final".
    printf '%d\n' "$_code" > "$DF_STATUS_FILE.tmp"
    mv -f "$DF_STATUS_FILE.tmp" "$DF_STATUS_FILE"
} < /dev/null > /dev/null 2>&1 &

DF_CHILD_PID=$!
trap - HUP

printf '%d\n' "$DF_CHILD_PID" > "$DF_PID_FILE.tmp"
mv -f "$DF_PID_FILE.tmp" "$DF_PID_FILE"

df_log "launched pid $DF_CHILD_PID; log: $DF_LAUNCH_LOG"
df_log "watch it with:  tail -n +1 -F '$DF_LAUNCH_LOG'"
df_log "ask it with:    remote/launch.sh --status --log '$DF_LAUNCH_LOG'"
df_log "A launch is not a rental. Confirm an instance exists before saying one is up."
