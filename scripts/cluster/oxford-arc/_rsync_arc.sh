#!/usr/bin/env bash
# Shared rsync helpers for laptop ↔ Oxford ARC.
# Sourced by pull_from_arc.sh / push_to_arc.sh. Do not run directly.
#
# Syncs every DVC-tracked path (from repo *.dvc outs) plus gitignored
# scripts/cluster/oxford-arc/env.sh. Run on the laptop (VPN or ProxyJump).
#
# Transfers every selected path in one rsync (one SSH password prompt).
#
# Optional env:
#   ARC_HOST     default htc-login.arc.ox.ac.uk
#   ARC_PROJECT  ARC share name (required; set in env.sh)
#   ARC_REMOTE   if set, used as-is (user@host:/abs/path/to/pe-hub)
#   ARC_USER     else first arg, else prompt
#   DRY_RUN=1    rsync --dry-run
#   DELETE=1     rsync --delete (dest extras removed)
#   LIST=1       print paths and exit
#   ONLY=a,b     only these relative paths (alias: env → env.sh;
#                bundles: pridict2-repro → reproduction weights + state;
#                scratch-weights → preferred scratch-benchmark weight dirs)
#   EXTRA=a,b    append these paths (or bundles) to the DVC set
#   SKIP=a,b     skip these relative paths

set -euo pipefail

if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    echo "Error: source this file from pull_from_arc.sh or push_to_arc.sh" >&2
    exit 1
fi

_RSYNC_ARC_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${_RSYNC_ARC_DIR}/../../.." && pwd)"
ENV_REL="scripts/cluster/oxford-arc/env.sh"
_PRIDICT2_REPRO_MANIFEST="scripts/experiments/pridict2-reproduction/supplementary_artifacts.txt"
_SCRATCH_WEIGHTS_MANIFEST="scripts/experiments/scratch-benchmark/weight_artifacts.txt"

_arc_rsync_usage() {
    local cmd="$1"
    cat <<EOF
Usage: ${cmd} [ARC_USER]

Run on the laptop. One rsync (one SSH password) of DVC-tracked folders plus ${ENV_REL}.

  DRY_RUN=1     show what would transfer
  LIST=1        print relative paths and exit
  DELETE=1      remove dest files that are not on the source
  ONLY=path,..  only these repo-relative paths (ONLY=env for env.sh;
                ONLY=pridict2-repro / ONLY=scratch-weights for weight bundles)
  EXTRA=path,.. append paths/bundles to the default DVC set
  SKIP=path,..  skip these repo-relative paths
  ARC_HOST      default htc-login.arc.ox.ac.uk
  ARC_PROJECT   ARC share name (required; set in env.sh)
  ARC_REMOTE    override user@host:/data/<project>/<user>/pe-hub

Off-campus: ProxyJump gateway.arc.ox.ac.uk for htc-login in ~/.ssh/config.
EOF
}

_arc_normalize_item() {
    local item="${1#"${1%%[![:space:]]*}"}"
    item="${item%"${item##*[![:space:]]}"}"
    case "${item}" in
        env|env.sh) echo "${ENV_REL}" ;;
        *) echo "${item}" ;;
    esac
}

# Expand CSV tokens; known bundles become many repo-relative paths.
# Prints one relative path per line (no kinds).
_arc_expand_csv_paths() {
    local csv="$1"
    [[ -z "${csv}" ]] && return 0
    local -a items
    local item line
    IFS=',' read -ra items <<< "${csv}"
    for item in "${items[@]}"; do
        item="$(_arc_normalize_item "${item}")"
        [[ -z "${item}" ]] && continue
        case "${item}" in
            pridict2-repro|pridict2_repro|pridict2-reproduction)
                if [[ ! -f "${REPO_ROOT}/${_PRIDICT2_REPRO_MANIFEST}" ]]; then
                    echo "Error: missing ${_PRIDICT2_REPRO_MANIFEST}" >&2
                    return 1
                fi
                while IFS= read -r line || [[ -n "${line}" ]]; do
                    [[ -z "${line}" || "${line}" =~ ^[[:space:]]*# ]] && continue
                    # first whitespace-separated field
                    echo "${line%%[[:space:]]*}"
                done < "${REPO_ROOT}/${_PRIDICT2_REPRO_MANIFEST}"
                ;;
            scratch-weights|scratch_weights|scratch-benchmark-weights)
                if [[ ! -f "${REPO_ROOT}/${_SCRATCH_WEIGHTS_MANIFEST}" ]]; then
                    echo "Error: missing ${_SCRATCH_WEIGHTS_MANIFEST}" >&2
                    echo "  Run: python3 scripts/experiments/scratch-benchmark/publish_preferred_weights.py" >&2
                    return 1
                fi
                while IFS= read -r line || [[ -n "${line}" ]]; do
                    [[ -z "${line}" || "${line}" =~ ^[[:space:]]*# ]] && continue
                    echo "${line%%[[:space:]]*}"
                done < "${REPO_ROOT}/${_SCRATCH_WEIGHTS_MANIFEST}"
                ;;
            *)
                echo "${item}"
                ;;
        esac
    done
}

_arc_csv_has_bundle_only() {
    # True if EVERY token in ONLY is a known bundle (no plain paths mixed in
    # that would still need the DVC inventory). Used to skip DVC listing.
    local csv="${1:-}"
    [[ -z "${csv}" ]] && return 1
    local -a items
    local item
    local any=0
    IFS=',' read -ra items <<< "${csv}"
    for item in "${items[@]}"; do
        item="$(_arc_normalize_item "${item}")"
        [[ -z "${item}" ]] && continue
        any=1
        case "${item}" in
            pridict2-repro|pridict2_repro|pridict2-reproduction) ;;
            scratch-weights|scratch_weights|scratch-benchmark-weights) ;;
            *) return 1 ;;
        esac
    done
    [[ "${any}" -eq 1 ]]
}

_arc_in_csv() {
    local rel="$1" csv="$2"
    [[ -z "${csv}" ]] && return 1
    local item
    while IFS= read -r item; do
        [[ -z "${item}" ]] && continue
        if [[ "${rel}" == "${item}" ]]; then
            return 0
        fi
    done < <(_arc_expand_csv_paths "${csv}")
    return 1
}

_arc_should_sync() {
    local rel="$1"
    if [[ -n "${ONLY:-}" ]] && ! _arc_in_csv "${rel}" "${ONLY}"; then
        return 1
    fi
    if [[ -n "${SKIP:-}" ]] && _arc_in_csv "${rel}" "${SKIP}"; then
        return 1
    fi
    return 0
}

_arc_guess_kind() {
    local rel="$1"
    if [[ "${rel}" == *.json || "${rel}" == *.tsv || "${rel}" == *.txt || "${rel}" == *.yaml || "${rel}" == *.yml || "${rel}" == *.sh ]]; then
        echo file
    else
        echo dir
    fi
}

# macOS ships openrsync (advertised as 2.6.9-compatible). It has no
# --info=progress2, --protect-args, or --ignore-missing-args. Prefer a
# Homebrew GNU rsync when present (brew install rsync).
_arc_find_rsync() {
    local c
    for c in /opt/homebrew/bin/rsync /usr/local/bin/rsync "$(command -v rsync 2>/dev/null)"; do
        [[ -n "${c}" && -x "${c}" ]] || continue
        if "${c}" --help 2>&1 | grep -q -- '--info'; then
            printf '%s\n' "${c}"
            return 0
        fi
    done
    command -v rsync
}

_arc_rsync_help_has() {
    grep -q -- "$1" <<< "${_ARC_RSYNC_HELP}"
}

# Prints "file|dir<TAB>relative/path" for each DVC out, then env.sh,
# then EXTRA=/bundle expansions.
_arc_list_sync_entries() {
    local py
    py="$(command -v python3 2>/dev/null || command -v python)"

    # Bundle-only ONLY=: skip DVC inventory; emit expanded paths only.
    if _arc_csv_has_bundle_only "${ONLY:-}"; then
        local rel
        while IFS= read -r rel; do
            [[ -z "${rel}" ]] && continue
            printf '%s\t%s\n' "$(_arc_guess_kind "${rel}")" "${rel}"
        done < <(_arc_expand_csv_paths "${ONLY}")
        return 0
    fi

    "${py}" - "${REPO_ROOT}" <<'PY'
from pathlib import Path
import os
import re
import sys

root = Path(sys.argv[1])
# *.dvc files sit beside the artifact dir; do not walk into the artifacts.
skip_dir_names = {
    ".git",
    ".dvc",
    "node_modules",
    "__pycache__",
    "reference",
    "exported",
    "standardized",
    "formatted",
    "catalog",
    "weights",
    "results",
    "slurm_output",
    "checkpoints",
    "artifacts",
}
seen = set()

def kind_from_block(block: str) -> str:
    if re.search(r"(?m)^\s*nfiles:\s*\d+", block) or re.search(
        r"(?m)^\s*md5:\s*\S+\.dir\s*$", block
    ):
        return "dir"
    return "file"

dvc_files = []
for dirpath, dirnames, filenames in os.walk(root):
    dirnames[:] = [d for d in dirnames if d not in skip_dir_names]
    for name in filenames:
        if name.endswith(".dvc"):
            dvc_files.append(Path(dirpath) / name)

for dvc in sorted(dvc_files):
    text = dvc.read_text(encoding="utf-8")
    parent = dvc.parent.relative_to(root)
    for match in re.finditer(r"(?m)^\s*path:\s*(.+?)\s*$", text):
        raw = match.group(1).strip().strip("'\"")
        if parent == Path("."):
            rel = raw
        else:
            rel = f"{parent.as_posix()}/{raw}"
        if rel in seen:
            continue
        seen.add(rel)
        start = max(0, match.start() - 400)
        kind = kind_from_block(text[start : match.end()])
        print(f"{kind}\t{rel}")
PY
    printf 'file\t%s\n' "${ENV_REL}"

    if [[ -n "${EXTRA:-}" ]]; then
        local rel
        while IFS= read -r rel; do
            [[ -z "${rel}" ]] && continue
            printf '%s\t%s\n' "$(_arc_guess_kind "${rel}")" "${rel}"
        done < <(_arc_expand_csv_paths "${EXTRA}")
    fi
}

arc_rsync_main() {
    local direction="$1"
    shift
    local cmd="pull_from_arc.sh"
    [[ "${direction}" == push ]] && cmd="push_to_arc.sh"

    if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
        _arc_rsync_usage "${cmd}"
        return 0
    fi

    ARC_HOST="${ARC_HOST:-htc-login.arc.ox.ac.uk}"
    : "${ARC_PROJECT:?Set ARC_PROJECT in env.sh to your ARC share name under /data/}"

    if [[ -n "${1:-}" ]]; then
        ARC_USER="$1"
    elif [[ -z "${ARC_USER:-}" && "${LIST:-0}" != "1" ]]; then
        read -r -p "ARC username: " ARC_USER
    fi
    if [[ "${LIST:-0}" != "1" ]]; then
        : "${ARC_USER:?ARC username required}"
    fi

    if [[ -z "${ARC_REMOTE:-}" && -n "${ARC_USER:-}" ]]; then
        ARC_REMOTE="${ARC_USER}@${ARC_HOST}:/data/${ARC_PROJECT}/${ARC_USER}/pe-hub"
    fi

    local rsync_bin
    rsync_bin="$(_arc_find_rsync)"
    _ARC_RSYNC_HELP="$("${rsync_bin}" --help 2>&1 || true)"

    # --relative + /./ keeps repo-relative layout in one SSH session.
    RSYNC_FLAGS=(-avh --partial --relative --no-owner --no-group
                 --exclude '.DS_Store' --exclude '__pycache__/')
    if _arc_rsync_help_has '--info'; then
        RSYNC_FLAGS+=(--info=progress2)
    else
        RSYNC_FLAGS+=(--progress)
    fi
    if _arc_rsync_help_has '--protect-args'; then
        RSYNC_FLAGS+=(--protect-args)
    fi
    if [[ "${DRY_RUN:-0}" == "1" ]]; then
        RSYNC_FLAGS+=(--dry-run)
    fi
    if [[ "${DELETE:-0}" == "1" ]]; then
        RSYNC_FLAGS+=(--delete)
    fi
    if _arc_rsync_help_has '--ignore-missing-args'; then
        RSYNC_FLAGS+=(--ignore-missing-args)
    fi

    local kinds=() rels=() line kind rel
    while IFS= read -r line; do
        [[ -z "${line}" ]] && continue
        kind="${line%%$'\t'*}"
        rel="${line#*$'\t'}"
        _arc_should_sync "${rel}" || continue
        kinds+=("${kind}")
        rels+=("${rel}")
    done < <(_arc_list_sync_entries)

    if [[ ${#rels[@]} -eq 0 ]]; then
        echo "No paths to sync (check ONLY=/SKIP=)." >&2
        return 1
    fi

    echo "Direction: ${direction}"
    echo "Laptop:    ${REPO_ROOT}"
    echo "ARC:       ${ARC_REMOTE:-"(unset; LIST=1)"}"
    echo "Paths:"
    local i
    for i in "${!rels[@]}"; do
        echo "  - ${rels[$i]} (${kinds[$i]})"
    done
    if [[ "${DELETE:-0}" == "1" ]]; then
        echo "DELETE=1: extra files on the destination will be removed."
    fi
    if [[ "${DRY_RUN:-0}" == "1" ]]; then
        echo "DRY_RUN=1: no files will be written."
    fi

    if [[ "${LIST:-0}" == "1" ]]; then
        return 0
    fi
    : "${ARC_REMOTE:?Set ARC_USER or ARC_REMOTE}"

    local src_args=() dest
    if [[ "${direction}" == pull ]]; then
        dest="${REPO_ROOT}/"
        for i in "${!rels[@]}"; do
            src_args+=("${ARC_REMOTE}/./${rels[$i]}")
        done
    else
        dest="${ARC_REMOTE}/"
        local keep_rels=() keep_kinds=()
        for i in "${!rels[@]}"; do
            if [[ ! -e "${REPO_ROOT}/${rels[$i]}" ]]; then
                echo "skip (missing locally): ${rels[$i]}"
                continue
            fi
            src_args+=("${REPO_ROOT}/./${rels[$i]}")
            keep_rels+=("${rels[$i]}")
            keep_kinds+=("${kinds[$i]}")
        done
        rels=("${keep_rels[@]}")
        kinds=("${keep_kinds[@]}")
    fi

    if [[ ${#src_args[@]} -eq 0 ]]; then
        echo "Nothing to sync (all sources missing locally)." >&2
        return 1
    fi

    echo ""
    echo "One rsync, one SSH password prompt."
    if ! _arc_rsync_help_has '--info'; then
        echo "Using ${rsync_bin} (macOS openrsync). Per-file --progress instead of GNU --info=progress2."
        echo "Optional: brew install rsync  # GNU rsync 3.x, overall progress bar"
    fi
    "${rsync_bin}" "${RSYNC_FLAGS[@]}" "${src_args[@]}" "${dest}"

    echo ""
    echo "Done (${direction})."
}
