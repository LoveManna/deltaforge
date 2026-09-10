#!/bin/sh
# Shared helpers for the remote lifecycle scripts.
#
# POSIX sh, sourced rather than executed. These run *before* any Python environment
# exists, so nothing here may depend on the package being installed.
#
# Dependency split, on purpose:
#   * The budget gates use `awk` only. They are the safety-critical path and must work
#     on any machine, with no optional tooling.
#   * Parsing Vast's offer JSON uses `jq`. Hand-rolling a JSON parser for an arbitrary
#     API response in awk would be less trustworthy than requiring one small binary.
#
# Secret handling: VAST_API_KEY is read from a gitignored .env and passed to curl through
# a config file on **stdin** (`curl -K -`). It therefore never appears in argv (so never
# in `ps`), never in a file on disk, and never in a log. `set -x` is explicitly disabled
# inside the functions that touch it.

DF_API_HOST="${DF_API_HOST:-https://console.vast.ai}"
DF_API_BASE="${DF_API_BASE:-$DF_API_HOST/api/v0}"
# Instance *listing* moved to v1: `GET /api/v0/instances/` now answers
# "deprecated_endpoint", and `GET /api/v0/instances/<id>/` answers `{"instances": null}`
# for a live instance rather than failing, so a poller built on it waits out its whole
# timeout while reporting an unknown status. Offers, create and destroy are still v0.
DF_API_BASE_V1="${DF_API_BASE_V1:-$DF_API_HOST/api/v1}"
DF_DRY_RUN="${DF_DRY_RUN:-0}"

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

df_log()  { printf '[deltaforge] %s\n' "$*" >&2; }
df_warn() { printf '[deltaforge] WARNING: %s\n' "$*" >&2; }
df_die()  { printf '[deltaforge] ERROR: %s\n' "$*" >&2; exit 1; }

df_dry() {
    if [ "$DF_DRY_RUN" = "1" ]; then
        printf '[deltaforge] (dry-run) %s\n' "$*" >&2
        return 0
    fi
    return 1
}

df_repo_root() {
    # The directory containing remote/, resolved without relying on readlink -f.
    unset CDPATH
    _libdir=$(cd -- "$(dirname -- "$1")" && pwd)
    cd -- "$_libdir/.." && pwd
}

df_require_cmd() {
    for _cmd in "$@"; do
        command -v "$_cmd" >/dev/null 2>&1 || df_die "required command not found: $_cmd"
    done
}

# The rented box is reached with a dedicated key, not the operator's default identity:
# an ephemeral marketplace instance should never be handed a key that opens anything
# else. Generate it once with
#   ssh-keygen -t ed25519 -f ~/.ssh/deltaforge_vast -N ""
# and register the public half on the Vast account. Falls back to ssh's own defaults when
# the file is absent, so nothing here breaks on a machine that has not set one up.
DF_SSH_KEY="${DF_SSH_KEY:-$HOME/.ssh/deltaforge_vast}"
DF_SSH_ID=""
[ -f "$DF_SSH_KEY" ] && DF_SSH_ID="-i $DF_SSH_KEY -o IdentitiesOnly=yes"

df_now_epoch() { date -u +%s; }
df_now_iso()   { date -u +%Y-%m-%dT%H:%M:%SZ; }
df_this_month(){ date -u +%Y-%m; }

# Has this `status_msg` stopped carrying a byte-moving progress signal?
#
# The stall budget's premise is that a live pull keeps rewriting status_msg with new byte
# counts, so a frozen message means a stuck one. That premise expires the moment the last
# layer finishes downloading: checksum verification, extraction and container start emit
# no further updates, and the message freezes *because the pull succeeded*. Rental 24 was
# destroyed at 300s on "Verifying Checksum ... Download complete" -- the guard firing at
# precisely the point where the thing it guards had stopped being possible.
#
# Same reasoning the poll already applies to an empty status_msg: where the signal cannot
# mean anything, say so and let DF_SSH_READY_TIMEOUT do the bounding instead of inventing
# a verdict. A genuinely stuck extraction then costs the readiness timeout rather than the
# stall budget, which is the price of not destroying healthy rentals.
df_pull_settled() {
    case "$1" in
        *"Download complete"*|*"Verifying Checksum"*|*Extracting*|*"Pull complete"*|*"Already exists"*)
            return 0 ;;
    esac
    return 1
}

# ---------------------------------------------------------------------------
# Credentials
# ---------------------------------------------------------------------------

# df_env_value NAME [FILE]
# One value out of the gitignored .env, on stdout. Never logged by this function; every
# caller is responsible for logging only the length or a boolean, never the value.
df_env_value() {
    set +x
    _name="$1"
    _file="${2:-$DF_REPO_ROOT/.env}"
    [ -f "$_file" ] || return 0
    awk -v name="$_name" '
        index($0, name "=") == 1 || $0 ~ "^[[:space:]]*" name "[[:space:]]*=" {
            sub(/^[^=]*=[[:space:]]*/, "", $0)
            gsub(/^"|"$/, "", $0)
            gsub(/^'"'"'|'"'"'$/, "", $0)
            sub(/[[:space:]]+$/, "", $0)
            print $0
            exit
        }' "$_file"
}

# Loads the API key into DF_API_KEY. Never printed, never written anywhere.
df_load_api_key() {
    set +x
    _env_file="${1:-$DF_REPO_ROOT/.env}"
    DF_API_KEY="${VAST_API_KEY:-}"
    if [ -z "$DF_API_KEY" ]; then
        DF_API_KEY=$(df_env_value VAST_API_KEY "$_env_file")
    fi
    if [ -n "$DF_API_KEY" ]; then
        df_log "API key loaded (${#DF_API_KEY} characters, value never logged)"
        return 0
    fi
    return 1
}

# Loads Docker registry credentials into DF_REGISTRY_LOGIN, in the `docker login`
# argument form Vast expects for a create request's `image_login` field.
#
# Why this exists: every container image this project failed to pull came from Docker
# Hub, and Vast hosts pull anonymously by default. Docker Hub's unauthenticated pull
# limits are applied per source IP and are aggressive against datacenter ranges, which is
# consistent with layers stalling at "Pulling fs layer" on four different machines. A
# free Docker Hub account raises the limit by an order of magnitude.
#
# Optional: with no credentials set, this is inert and the create request is unchanged.
df_load_registry_login() {
    set +x
    _env_file="${1:-$DF_REPO_ROOT/.env}"
    DF_REGISTRY_LOGIN=""
    _user="${DOCKER_LOGIN_USER:-$(df_env_value DOCKER_LOGIN_USER "$_env_file")}"
    _token="${DOCKER_LOGIN_TOKEN:-$(df_env_value DOCKER_LOGIN_TOKEN "$_env_file")}"
    if [ -n "$_user" ] && [ -n "$_token" ]; then
        DF_REGISTRY_LOGIN="-u $_user -p $_token"
        df_log "registry credentials loaded for user $_user (token never logged)"
        return 0
    fi
    df_log "no registry credentials; images will be pulled anonymously"
    return 1
}

# df_api METHOD PATH [BODY]
# Emits the raw response body on stdout.
df_api()    { _df_api_base="$DF_API_BASE";    _df_api_call "$@"; }
df_api_v1() { _df_api_base="$DF_API_BASE_V1"; _df_api_call "$@"; }

_df_api_call() {
    set +x
    _method="$1"; _path="$2"; _body="${3:-}"
    [ -n "$DF_API_KEY" ] || df_die "no API key: set VAST_API_KEY or put it in $DF_REPO_ROOT/.env"

    if [ -n "$_body" ]; then
        # The body goes through a 0600 temp file, not `--data "$_body"`: a create request
        # can carry a registry token, and argv is world-readable through `ps`.
        _body_file=$(umask 077; mktemp "${TMPDIR:-/tmp}/df-body.XXXXXX") || df_die "mktemp failed"
        printf '%s' "$_body" > "$_body_file"
        printf 'header = "Authorization: Bearer %s"\nheader = "Content-Type: application/json"\n' \
            "$DF_API_KEY" \
        | curl --silent --show-error --fail-with-body --max-time 60 \
               --config - --request "$_method" --data "@$_body_file" "$_df_api_base$_path"
        _rc=$?
        rm -f "$_body_file"
        return $_rc
    else
        printf 'header = "Authorization: Bearer %s"\n' "$DF_API_KEY" \
        | curl --silent --show-error --fail-with-body --max-time 60 \
               --config - --request "$_method" "$_df_api_base$_path"
    fi
}

# ---------------------------------------------------------------------------
# Ledger
#
# Flat JSONL, parsed with awk. See src/deltaforge/ledger.py, which implements the same
# two gates in Python; remote/scripts_test.py asserts the two agree on shared fixtures.
# ---------------------------------------------------------------------------

DF_AWK_JGET='
function jget(line, key,   pat, start, rest, i, ch, out) {
    pat = "\"" key "\":"
    start = index(line, pat)
    if (start == 0) return ""
    rest = substr(line, start + length(pat))
    if (substr(rest, 1, 1) == "\"") {
        rest = substr(rest, 2)
        i = index(rest, "\"")
        if (i == 0) return ""
        return substr(rest, 1, i - 1)
    }
    out = ""
    for (i = 1; i <= length(rest); i++) {
        ch = substr(rest, i, 1)
        if (ch == "," || ch == "}") break
        out = out ch
    }
    return out
}
'

# df_ledger_month_to_date LEDGER [MONTH] -> dollars on stdout
#
# An instance that was provisioned but never reconciled counts at its estimated ceiling.
# Counting it at zero would make the gate fail open exactly when an instance is still
# running and still costing money.
#
# The watchdog (spec 8.3) and the teardown trap (spec 8.5) can each reconcile the same
# instance, and the append-only ledger cannot retract either row, so one reconciled
# destroy is counted per instance, at the LARGEST reported cost. Must agree with
# month_to_date_usd in src/deltaforge/ledger.py — this awk path is what provision.sh
# actually gates on.
df_ledger_month_to_date() {
    _ledger="$1"
    _month="${2:-$(df_this_month)}"
    [ -f "$_ledger" ] || { printf '0.000000\n'; return 0; }
    awk -v MONTH="$_month" "$DF_AWK_JGET"'
    /^[[:space:]]*[#]/ { next }
    /^[[:space:]]*$/   { next }
    {
        ev  = jget($0, "event")
        iid = jget($0, "instance_id")
        if (ev == "provision") {
            if (!(iid in prov)) {
                prov[iid] = 1
                pmonth[iid] = substr(jget($0, "ts"), 1, 7)
                pest[iid] = jget($0, "estimated_ceiling_usd") + 0
            }
        } else if (ev == "destroy") {
            c = jget($0, "actual_cost_usd")
            if (c != "" && c != "null") {
                if (!(iid in dcount) || c + 0 > dmax[iid]) {
                    dmax[iid] = c + 0
                    dmonth[iid] = substr(jget($0, "ts"), 1, 7)
                }
                dcount[iid]++
            }
        }
    }
    END {
        total = 0
        for (i in prov) {
            if (dcount[i] > 0) { if (dmonth[i] == MONTH) total += dmax[i] }
            else if (pmonth[i] == MONTH) total += pest[i]
        }
        for (i in dcount) if (!(i in prov) && dmonth[i] == MONTH) total += dmax[i]
        printf "%.6f\n", total
    }' "$_ledger"
}

# df_ledger_session_minutes LEDGER SESSION_ID [NOW_EPOCH] -> minutes on stdout
#
# An instance still running counts at its wall-clock elapsed time, so a session cannot
# dodge the gate by simply not tearing down. Double-reconciled instances are deduped at
# the largest reported duration, as in df_ledger_month_to_date.
df_ledger_session_minutes() {
    _ledger="$1"
    _session="$2"
    _now="${3:-$(df_now_epoch)}"
    [ -f "$_ledger" ] || { printf '0.000000\n'; return 0; }
    awk -v SESSION="$_session" -v NOW="$_now" "$DF_AWK_JGET"'
    /^[[:space:]]*[#]/ { next }
    /^[[:space:]]*$/   { next }
    {
        if (jget($0, "session_id") != SESSION) next
        ev  = jget($0, "event")
        iid = jget($0, "instance_id")
        if (ev == "provision") {
            if (!(iid in prov)) { prov[iid] = 1; pts[iid] = jget($0, "ts_epoch") + 0 }
        } else if (ev == "destroy") {
            m = jget($0, "actual_minutes")
            if (m != "" && m != "null") {
                if (!(iid in dcount) || m + 0 > dmax[iid]) dmax[iid] = m + 0
                dcount[iid]++
            }
        }
    }
    END {
        total = 0
        for (i in prov) {
            if (dcount[i] > 0) total += dmax[i]
            else { d = (NOW - pts[i]) / 60; if (d < 0) d = 0; total += d }
        }
        for (i in dcount) if (!(i in prov)) total += dmax[i]
        printf "%.6f\n", total
    }' "$_ledger"
}

# df_dryrun_ledger REAL_LEDGER -> scratch ledger path on stdout
#
# A dry run has to exercise the real month-to-date gate, which means READING the real
# ledger, while its synthetic provision/destroy rows must never reach it: they would
# inflate the budget reading for every later session and corrupt the spend the project
# reports. So seed a scratch copy from the real ledger and let the rehearsal read and
# write that instead. An explicitly supplied --ledger is always honoured as given.
df_dryrun_ledger() {
    _real="$1"
    _scratch="${DF_DRYRUN_LEDGER:-${DF_REPO_ROOT:-.}/.deltaforge-dryrun-spend.jsonl}"
    mkdir -p "$(dirname "$_scratch")" 2>/dev/null || true
    if [ -f "$_real" ]; then
        cp "$_real" "$_scratch"
    else
        : > "$_scratch"
    fi
    printf '%s\n' "$_scratch"
}

# df_instance_row JSON INSTANCE_ID -> that instance's object from a v1 listing, or empty.
#
# `.instances` is a list in v1, and ids come back as numbers, so the comparison is made
# on strings to avoid a jq type error when the id is quoted either way.
df_instance_row() {
    printf '%s' "$1" | jq -c --arg id "$2" '.instances[]? | select((.id|tostring) == $id)' 2>/dev/null || true
}

# Float comparison without bc, which is not universally installed.
# df_ge A B -> true when A >= B
df_ge() { awk -v a="$1" -v b="$2" 'BEGIN { exit !(a + 0 >= b + 0) }'; }
df_mul() { awk -v a="$1" -v b="$2" 'BEGIN { printf "%.6f\n", a * b }'; }
df_div() { awk -v a="$1" -v b="$2" 'BEGIN { printf "%.6f\n", a / b }'; }

df_json_escape() {
    # Ledger string fields must stay flat: no quotes, braces, commas or backslashes.
    printf '%s' "$1" | tr -d '"{},\\' | tr -s '[:space:]' ' '
}

# df_ledger_append_provision LEDGER SESSION INSTANCE GPU RATE EST_MINUTES HYPOTHESIS NOTE
df_ledger_append_provision() {
    _ledger="$1"; _session="$2"; _iid="$3"; _gpu="$4"; _rate="$5"
    _est_min="$6"; _hyp="${7:-}"; _note="${8:-}"
    _ceiling=$(awk -v r="$_rate" -v m="$_est_min" 'BEGIN { printf "%.6f", r * m / 60 }')
    mkdir -p "$(dirname "$_ledger")"
    printf '{"ts":"%s","ts_epoch":%s,"event":"provision","session_id":"%s","instance_id":"%s","gpu_model":"%s","hourly_rate_usd":%s,"estimated_ceiling_usd":%s,"estimated_minutes":%s,"actual_minutes":null,"actual_cost_usd":null,"hypothesis":"%s","note":"%s"}\n' \
        "$(df_now_iso)" "$(df_now_epoch)" \
        "$(df_json_escape "$_session")" "$(df_json_escape "$_iid")" \
        "$(df_json_escape "$_gpu")" "$_rate" "$_ceiling" "$_est_min" \
        "$(df_json_escape "$_hyp")" "$(df_json_escape "$_note")" \
        >> "$_ledger"
}

# df_ledger_append_destroy LEDGER SESSION INSTANCE GPU RATE ACTUAL_MINUTES HYPOTHESIS NOTE
df_ledger_append_destroy() {
    _ledger="$1"; _session="$2"; _iid="$3"; _gpu="$4"; _rate="$5"
    _minutes="$6"; _hyp="${7:-}"; _note="${8:-}"
    _cost=$(awk -v r="$_rate" -v m="$_minutes" 'BEGIN { printf "%.6f", r * m / 60 }')
    mkdir -p "$(dirname "$_ledger")"
    printf '{"ts":"%s","ts_epoch":%s,"event":"destroy","session_id":"%s","instance_id":"%s","gpu_model":"%s","hourly_rate_usd":%s,"estimated_ceiling_usd":0,"estimated_minutes":0,"actual_minutes":%s,"actual_cost_usd":%s,"hypothesis":"%s","note":"%s"}\n' \
        "$(df_now_iso)" "$(df_now_epoch)" \
        "$(df_json_escape "$_session")" "$(df_json_escape "$_iid")" \
        "$(df_json_escape "$_gpu")" "$_rate" "$_minutes" "$_cost" \
        "$(df_json_escape "$_hyp")" "$(df_json_escape "$_note")" \
        >> "$_ledger"
}

# ---------------------------------------------------------------------------
# Instance lifecycle
# ---------------------------------------------------------------------------

# df_vast_destroy INSTANCE_ID
# Idempotent by intent: destroying an already-destroyed instance is a success, because
# the teardown path must never fail in a way that leaves the caller unsure.
df_vast_destroy() {
    _iid="$1"
    [ -n "$_iid" ] || { df_warn "df_vast_destroy called with no instance id"; return 0; }
    if df_dry "would DELETE /instances/$_iid/ (no endpoint contacted)"; then
        return 0
    fi
    if df_api DELETE "/instances/$_iid/" >/dev/null 2>&1; then
        df_log "destroyed instance $_iid"
    else
        df_warn "destroy call for instance $_iid did not succeed; verify manually at https://cloud.vast.ai/instances/"
    fi
    return 0
}
