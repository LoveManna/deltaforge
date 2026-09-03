#!/bin/sh
# Move the repo up to the rented box and the results back down.
#
# Excludes are deliberate: the virtualenv and caches are host-specific, the weights are
# 9 GB and are fetched on the instance instead, and .env must never leave this machine.

set -eu

unset CDPATH
DF_REPO_ROOT=$(cd -- "$(dirname -- "$0")/.." && pwd)
export DF_REPO_ROOT
. "$DF_REPO_ROOT/remote/lib.sh"

DF_SSH_HOST="${DF_SSH_HOST:-}"
DF_SSH_PORT="${DF_SSH_PORT:-22}"
DF_REMOTE_DIR="${DF_REMOTE_DIR:-/workspace/deltaforge}"
DF_DIRECTION=""

usage() {
    cat <<'EOF'
Usage: remote/sync.sh (up|down) --host USER@HOST [options]

  up                    Push the repo to the instance.
  down                  Pull results/ back from the instance.
  --host USER@HOST      SSH target (required unless --dry-run).
  --port N              SSH port (default: 22).
  --remote-dir PATH     Remote checkout location (default: /workspace/deltaforge).
  --dry-run             Show what would transfer without contacting the host.
  -h, --help            This message.
EOF
}

[ $# -gt 0 ] || { usage; exit 1; }
case "$1" in
    up|down) DF_DIRECTION="$1"; shift ;;
    -h|--help) usage; exit 0 ;;
    *) df_die "first argument must be 'up' or 'down'" ;;
esac

while [ $# -gt 0 ]; do
    case "$1" in
        --host)       DF_SSH_HOST="$2"; shift ;;
        --port)       DF_SSH_PORT="$2"; shift ;;
        --remote-dir) DF_REMOTE_DIR="$2"; shift ;;
        --dry-run)    DF_DRY_RUN=1 ;;
        -h|--help)    usage; exit 0 ;;
        *)            df_die "unknown option: $1 (try --help)" ;;
    esac
    shift
done
export DF_DRY_RUN

if [ "$DF_DRY_RUN" != "1" ]; then
    [ -n "$DF_SSH_HOST" ] || df_die "--host is required"
    df_require_cmd rsync ssh
fi

RSYNC_SSH="ssh -p $DF_SSH_PORT -o StrictHostKeyChecking=accept-new -o ConnectTimeout=20"

# .env carries the API key and must never be transferred. Listing it here is belt and
# braces: it is gitignored too, but rsync does not read .gitignore.
EXCLUDES="
--exclude=.git
--exclude=.venv
--exclude=.env
--exclude=__pycache__
--exclude=*.pyc
--exclude=.pytest_cache
--exclude=.ruff_cache
--exclude=models
--exclude=weights
--exclude=*.safetensors
--exclude=.deltaforge-instance
"

case "$DF_DIRECTION" in
    up)
        if df_dry "would rsync $DF_REPO_ROOT/ -> $DF_SSH_HOST:$DF_REMOTE_DIR/ (excluding .env, .git, .venv, weights)"; then
            exit 0
        fi
        df_log "syncing repo up to $DF_SSH_HOST:$DF_REMOTE_DIR"
        # shellcheck disable=SC2086
        ssh -p "$DF_SSH_PORT" -o StrictHostKeyChecking=accept-new "$DF_SSH_HOST" \
            "mkdir -p '$DF_REMOTE_DIR'"
        # shellcheck disable=SC2086
        rsync -az --delete $EXCLUDES -e "$RSYNC_SSH" \
            "$DF_REPO_ROOT/" "$DF_SSH_HOST:$DF_REMOTE_DIR/"
        ;;
    down)
        if df_dry "would rsync $DF_SSH_HOST:$DF_REMOTE_DIR/results/ -> $DF_REPO_ROOT/results/"; then
            exit 0
        fi
        df_log "pulling results down from $DF_SSH_HOST:$DF_REMOTE_DIR/results/"
        mkdir -p "$DF_REPO_ROOT/results"
        # No --delete here: a failed pull must never erase results already recorded.
        rsync -az -e "$RSYNC_SSH" \
            "$DF_SSH_HOST:$DF_REMOTE_DIR/results/" "$DF_REPO_ROOT/results/"
        ;;
esac
df_log "sync $DF_DIRECTION complete"
