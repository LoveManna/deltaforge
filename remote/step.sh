#!/bin/sh
# Run one step on the box, detached from the ssh session that asked for it.
#
# Why this exists. Until 2026-09-29 every remote step was `ssh host "command"`, which ties
# a paid hour of GPU work to the life of one TCP connection: the batch's stdout *is* the
# ssh channel, so a dropped connection kills the measurement. Four rentals died that way in
# one evening (47, 48 and 50 on `closed by remote host`, at three different hosts and three
# different gateways, in three different stages) for $0.528 and not one recorded slot, and
# the same failure had already cost rentals 30 and 39. Keepalives were tried first and are
# not enough: rental 50 carried them and still lost the session 32 minutes in, during slot
# 0's benchmark.
#
# So the step's lifetime stops depending on the connection. `start` launches it under
# `setsid` with its output going to a file on the box; `poll` reads that file from wherever
# the caller has got to. A dropped ssh now costs one reconnect, and the work keeps running.
#
# The one thing this must never do is run a paid step twice. `start` records `started`
# before it launches anything, so a launch whose ssh died *after* the box took the request
# is not repeated when the caller retries -- which is what makes the caller's retry safe.
set -eu

_action=${1:-}
_dir=${2:-}
if [ -z "$_action" ] || [ -z "$_dir" ]; then
    echo "usage: step.sh start|poll DIR [OFFSET]" >&2
    exit 2
fi

case "$_action" in
start)
    mkdir -p "$_dir"
    if [ ! -f "$_dir/started" ]; then
        : > "$_dir/log"
        # Written before the launch, never after: see the header.
        touch "$_dir/started"
        setsid sh -c "sh '$_dir/cmd.sh' >> '$_dir/log' 2>&1; echo \$? > '$_dir/status'" \
            </dev/null >/dev/null 2>&1 &
    fi
    ;;
poll)
    _offset=${3:-0}

    # Status first, then size, then the bytes -- in that order for a reason. A status file
    # means the step is over and its log is final, so a size read after it is final too.
    # Reading size first and status second could report "finished" alongside a size that
    # missed the last write, and the caller would never ask for those bytes again.
    _status=running
    if [ -f "$_dir/status" ]; then
        _status=$(tr -d ' \n' < "$_dir/status")
    fi

    _size=0
    if [ -f "$_dir/log" ]; then
        _size=$(wc -c < "$_dir/log" | tr -d ' ')
    fi

    # The marker goes *before* the bytes, and that ordering is load-bearing. A step's log
    # routinely ends mid-line -- the weights fetch emits a `\r` progress bar -- so a marker
    # printed after the payload would land on the end of that partial line, where the
    # caller's line-based parse cannot see it. The caller would then read no offset and no
    # status, treat the step as still running, and hang until the stall budget fired.
    # Printed first, the marker is always a whole line terminated by this printf.
    printf '__DF_STEP__ offset=%s status=%s\n' "$_size" "$_status"

    # Bounded by the size read above rather than by the end of the file, so the offset in
    # that marker is exactly how far the caller has been shown. `tail -c +N` is 1-based.
    if [ "$_size" -gt "$_offset" ]; then
        head -c "$_size" "$_dir/log" | tail -c "+$((_offset + 1))" || true
    fi
    ;;
*)
    echo "step.sh: unknown action '$_action'" >&2
    exit 2
    ;;
esac
