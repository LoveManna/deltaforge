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
DF_CACHE_KEY="${DF_CACHE_KEY:-}"
DF_LOCAL_CACHE="${DF_LOCAL_CACHE:-$DF_REPO_ROOT/cache/compile}"
DF_DIRECTION=""

usage() {
    cat <<'EOF'
Usage: remote/sync.sh (up|down) --host USER@HOST [options]

  up                    Push the repo to the instance.
  down                  Pull results/ back from the instance.
  cache-up              Push this GPU's compile cache to the instance, if we have one.
  cache-down            Pull the compile cache back, so the next rental starts warm.
  --host USER@HOST      SSH target (required unless --dry-run).
  --port N              SSH port (default: 22).
  --remote-dir PATH     Remote checkout location, or cache location for cache-*.
  --cache-key KEY       Compatibility key for the cache: <gpu>-<torch>-<cuda>. A cache is
                        only ever reused on the hardware and toolchain that built it.
  --dry-run             Show what would transfer without contacting the host.
  -h, --help            This message.
EOF
}

[ $# -gt 0 ] || { usage; exit 1; }
case "$1" in
    up|down|cache-up|cache-down) DF_DIRECTION="$1"; shift ;;
    -h|--help) usage; exit 0 ;;
    *) df_die "first argument must be one of: up, down, cache-up, cache-down" ;;
esac

while [ $# -gt 0 ]; do
    case "$1" in
        --host)       DF_SSH_HOST="$2"; shift ;;
        --port)       DF_SSH_PORT="$2"; shift ;;
        --remote-dir) DF_REMOTE_DIR="$2"; shift ;;
        --cache-key)  DF_CACHE_KEY="$2"; shift ;;
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

RSYNC_SSH="ssh $DF_SSH_ID -p $DF_SSH_PORT -o StrictHostKeyChecking=accept-new -o ConnectTimeout=20"

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
        # shellcheck disable=SC2086
        ssh $DF_SSH_ID -p "$DF_SSH_PORT" -o StrictHostKeyChecking=accept-new "$DF_SSH_HOST" \
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
    cache-up)
        # Torch keys its fx-graph and autotune entries by hardware and toolchain, so a
        # cache built on another card buys nothing and costs the transfer. The key is the
        # directory name, and a miss simply starts cold.
        [ -n "$DF_CACHE_KEY" ] || df_die "--cache-key is required for cache-up"
        DF_CACHE_SRC="$DF_LOCAL_CACHE/$DF_CACHE_KEY"
        if [ ! -d "$DF_CACHE_SRC" ]; then
            df_log "no local compile cache for $DF_CACHE_KEY; this rental compiles cold"
            exit 0
        fi
        if df_dry "would rsync $DF_CACHE_SRC/ -> $DF_SSH_HOST:$DF_REMOTE_DIR/"; then
            exit 0
        fi
        df_log "sending the $DF_CACHE_KEY compile cache up"
        # shellcheck disable=SC2086
        ssh $DF_SSH_ID -p "$DF_SSH_PORT" -o StrictHostKeyChecking=accept-new "$DF_SSH_HOST" \
            "mkdir -p '$DF_REMOTE_DIR'"
        rsync -az -e "$RSYNC_SSH" "$DF_CACHE_SRC/" "$DF_SSH_HOST:$DF_REMOTE_DIR/"
        ;;
    cache-down)
        [ -n "$DF_CACHE_KEY" ] || df_die "--cache-key is required for cache-down"
        DF_CACHE_DEST="$DF_LOCAL_CACHE/$DF_CACHE_KEY"
        if df_dry "would rsync $DF_SSH_HOST:$DF_REMOTE_DIR/ -> $DF_CACHE_DEST/"; then
            exit 0
        fi
        df_log "pulling the compile cache into $DF_CACHE_DEST"
        mkdir -p "$DF_CACHE_DEST"
        # No --delete: a partial pull must not erase a cache that already works. An empty
        # remote cache is not a failure either -- the first rental has nothing to send.
        rsync -az -e "$RSYNC_SSH" \
            "$DF_SSH_HOST:$DF_REMOTE_DIR/" "$DF_CACHE_DEST/" || {
            df_warn "compile cache pull failed; the next rental compiles cold"
            exit 0
        }
        ;;
esac
df_log "sync $DF_DIRECTION complete"
