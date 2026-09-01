#!/usr/bin/env bash
# Install pedb + peen CLIs (editable), enable bash/zsh tab completion, print usage.
#
# Usage (from repo root or anywhere):
#   ./scripts/setup-python-env.sh        # first time: create Python 3.11 env
#   conda activate pe-hub                # or pedb / venv
#   ./scripts/install-clis.sh
#
# Requires Python 3.11 (see environment.yml). Skip OptiPrime deps: SKIP_OPTIPRIME=1
# Skip completion with: SKIP_CLI_COMPLETION=1 ./scripts/install-clis.sh
#
# Tab completion (macOS/Linux bash/zsh): prefers conda activate.d so hooks load
# on env activate; falls back to ~/.zshrc or ~/.bashrc when writable.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

# shellcheck source=scripts/python-env.sh
source "${REPO_ROOT}/scripts/python-env.sh"

if ! PYTHON="$(pe_hub_resolve_python)"; then
    echo "Error: no Python interpreter found." >&2
    exit 1
fi

PY_REAL="$("${PYTHON}" -c 'import sys; print(sys.executable)')"
PY_PREFIX="$("${PYTHON}" -c 'import sys; print(sys.prefix)')"

echo "======================================"
echo "Install PE CLIs (editable)"
echo "======================================"
echo "Repo:    ${REPO_ROOT}"
echo "Python:  ${PY_REAL}"
echo "Prefix:  ${PY_PREFIX}"
echo "Target:  Python ${PE_HUB_PYTHON_VERSION} (see environment.yml)"
echo ""

pe_hub_refuse_system_python "${PY_REAL}" || exit 1
pe_hub_require_python_version "${PYTHON}" || exit 1
pe_hub_warn_if_no_virtual_env
pe_hub_export_pip_constraint "${REPO_ROOT}"

# Vendor model code lives in git submodules under vendor/models/.
sync_vendor_submodules() {
    if [[ "${SKIP_SUBMODULES:-0}" == "1" ]]; then
        echo "Skipping vendor submodules (SKIP_SUBMODULES=1)."
        echo ""
        return 0
    fi
    if [[ ! -f "${REPO_ROOT}/.gitmodules" ]]; then
        return 0
    fi
    if ! git -C "${REPO_ROOT}" rev-parse --git-dir >/dev/null 2>&1; then
        echo "Warning: no git checkout — cannot init vendor/models submodules." >&2
        echo "  If you rsynced without vendor sources, clone with submodules or sync again." >&2
        echo "" >&2
        return 0
    fi

    echo "Initializing vendor model submodules (vendor/models/) ..."
    if ! git -C "${REPO_ROOT}" submodule update --init --recursive; then
        echo "Error: git submodule update failed." >&2
        echo "  DeepPrime/OPED/PRIDICT2 URLs use git@github.com — need GitHub SSH access," >&2
        echo "  or rewrite to HTTPS: git config --global url.https://github.com/.insteadOf git@github.com:" >&2
        echo "  Skip only if sources are already present: SKIP_SUBMODULES=1 ./scripts/install-clis.sh" >&2
        exit 1
    fi

    local missing=0
    local name
    for name in deepprime oped pridict2 optiprime; do
        if [[ ! -d "${REPO_ROOT}/vendor/models/${name}" ]] || \
           [[ -z "$(find "${REPO_ROOT}/vendor/models/${name}" -mindepth 1 -maxdepth 1 ! -name '.git' 2>/dev/null | head -1)" ]]; then
            echo "Error: vendor/models/${name} is empty after submodule update." >&2
            missing=1
        fi
    done
    if [[ "${missing}" -ne 0 ]]; then
        exit 1
    fi
    echo "Vendor submodules ready."
    echo ""
}
sync_vendor_submodules

echo "Installing pe-common, pe-db (pedb), pe-ensemble (peen) ..."
"${PYTHON}" -m pip install -e "${REPO_ROOT}/packages/pe-common"
"${PYTHON}" -m pip install -e "${REPO_ROOT}/services/pe-db"
"${PYTHON}" -m pip install -e "${REPO_ROOT}/services/pe-ensemble[library]"
if [[ "${SKIP_OPTIPRIME:-0}" != "1" ]]; then
    bash "${REPO_ROOT}/scripts/install-optiprime-deps.sh"
else
    echo "Skipping OptiPrime deps (SKIP_OPTIPRIME=1)."
fi
# Scripts land in the env bin; ensure we look there even if PATH is stale.
ENV_BIN="$("${PYTHON}" -c 'import sys, pathlib; print(pathlib.Path(sys.executable).parent)')"
export PATH="${ENV_BIN}:${PATH}"

echo ""
echo "Verifying console scripts ..."
if ! command -v pedb >/dev/null 2>&1; then
    echo "Error: pedb not on PATH after install (looked in ${ENV_BIN})." >&2
    exit 1
fi
if ! command -v peen >/dev/null 2>&1; then
    echo "Error: peen not on PATH after install (looked in ${ENV_BIN})." >&2
    exit 1
fi

echo "  pedb -> $(command -v pedb)"
echo "  peen -> $(command -v peen)"
echo ""

# --- bash/zsh tab completion (argcomplete); non-fatal on permission errors ---
install_cli_completion() {
    if ! command -v register-python-argcomplete >/dev/null 2>&1; then
        echo "Warning: register-python-argcomplete not found; skip tab completion." >&2
        return 1
    fi

    local shell_name marker block conda_activate_d target rc dir
    shell_name="$(basename "${SHELL:-bash}")"
    marker="# pe-hub CLI tab completion (pedb / peen)"

    # Dual bash/zsh block. For zsh, compinit must run before argcomplete's compdef.
    block="$(cat <<EOF
${marker}
if command -v register-python-argcomplete >/dev/null 2>&1; then
  if [[ -n "\${ZSH_VERSION:-}" ]]; then
    autoload -Uz compinit
    if ! typeset -f compdef >/dev/null 2>&1; then
      compinit -C 2>/dev/null || compinit
    fi
  fi
  eval "\$(register-python-argcomplete pedb)"
  eval "\$(register-python-argcomplete pe-db)"
  eval "\$(register-python-argcomplete peen)"
  eval "\$(register-python-argcomplete pe-ensemble)"
fi
EOF
)"

    append_or_skip() {
        local dest="$1"
        dir="$(dirname "${dest}")"
        mkdir -p "${dir}"
        if [[ ! -w "${dir}" ]] || { [[ -e "${dest}" ]] && [[ ! -w "${dest}" ]]; }; then
            echo "Warning: cannot write ${dest} (permission denied)." >&2
            return 1
        fi
        if [[ -f "${dest}" ]] && grep -qF "${marker}" "${dest}" 2>/dev/null; then
            echo "Tab completion already installed in ${dest}."
            return 0
        fi
        {
            echo ""
            echo "${block}"
        } >> "${dest}"
        echo "Tab completion hooks appended to ${dest}"
        return 0
    }

    echo "======================================"
    echo "PE CLI tab completion (argcomplete)"
    echo "======================================"

    conda_activate_d=""
    if [[ -n "${CONDA_PREFIX:-}" && -d "${CONDA_PREFIX}" ]]; then
        conda_activate_d="${CONDA_PREFIX}/etc/conda/activate.d"
    fi

    if [[ -n "${conda_activate_d}" ]]; then
        target="${conda_activate_d}/pe-hub-cli-completion.sh"
        if append_or_skip "${target}"; then
            echo "Completion loads on: conda activate $(basename "${CONDA_PREFIX}")"
            echo "This shell: source ${target}"
            echo "  or: conda deactivate && conda activate $(basename "${CONDA_PREFIX}")"
            echo ""
            return 0
        fi
    fi

    if [[ "${shell_name}" == "zsh" ]]; then
        rc="${HOME}/.zshrc"
    else
        rc="${HOME}/.bashrc"
    fi
    if append_or_skip "${rc}"; then
        echo "Run: source ${rc}"
        echo ""
        return 0
    fi

    echo "Tab completion not installed (shell rc not writable)." >&2
    echo "  Fix: sudo chown \"\$(whoami)\" ${rc} && ./scripts/install-clis.sh" >&2
    echo "  Or this session: eval \"\$(register-python-argcomplete pedb)\"" >&2
    echo "                   eval \"\$(register-python-argcomplete peen)\"" >&2
    echo "" >&2
    return 1
}

if [[ "${SKIP_CLI_COMPLETION:-}" == "1" ]]; then
    echo "Skipping tab completion (SKIP_CLI_COMPLETION=1)."
    echo ""
else
    install_cli_completion || true
fi

echo "======================================"
echo "pedb — PE Database CLI"
echo "======================================"
echo "Catalog / data (no HTTP server needed):"
echo "  pedb init                         # seed + export + standardize"
echo "  pedb seed                         # catalog tables only"
echo "  pedb export [--study NAME]"
echo "  pedb standardize [--force]"
echo "  pedb filter --format deepprime --dataset library2 \\"
echo "    --cell-line HEK293T --pe-system PE2max --split-strategy holdout_3"
echo "  pedb studies | datasets | datasheets | scaffolds | statistics"
echo "  pedb formats"
echo "  pedb plugins reload"
echo ""
echo "Tip: use editable install (this script) or set DATA_ROOT to the repo datasets/"
echo "     so the CLI hits ${REPO_ROOT}/datasets/catalog/pe_database.db"
echo ""
pedb --help || true
echo ""

echo "======================================"
echo "peen — PE Ensemble CLI"
echo "======================================"
echo "Train / tune / evaluate / ensemble (uses pe_db.library in-process):"
echo "  peen models"
echo "  peen devices"
echo "  peen train --model deepprime --dataset-name library2 \\"
echo "    --dataset library2 --cell-line HEK293T --pe-system PE2max --device auto"
echo "  peen tune --model deepprime --dataset-name library2 \\"
echo "    --study deepprime --dataset library2 --n-trials 5 --device auto"
echo "  peen evaluate --model deepprime --weights <weight-id> --sync"
echo "  peen ensemble --ensemble-name demo \\"
echo "    --member deepprime:w1 --member oped:w2 --sync"
echo "  peen methods | weights --model deepprime"
echo "  peen jobs --kind train | peen logs --kind train <job-id>"
echo ""
peen --help || true
echo ""

echo "======================================"
echo "Done"
echo "======================================"
echo "Also available: pe-db / pe-ensemble (same entry points as pedb / peen)."
echo "Tab completion: conda activate.d or shell rc (skip: SKIP_CLI_COMPLETION=1)."
echo "  Reload: conda deactivate && conda activate <env>   # or source ~/.zshrc"
echo "  Try: pedb <TAB>  /  peen t<TAB>"
echo "Portal still needs the FastAPI servers (./scripts/start-all.sh)."
echo ""
