#!/usr/bin/env bash
# Shared Python environment helpers for PE-Hub install scripts.
#
# Source from other scripts:
#   # shellcheck source=scripts/python-env.sh
#   source "${REPO_ROOT}/scripts/python-env.sh"
#
# PE-Hub targets Python 3.11 (matches OptiPrime upstream and vendor JAX stack).
# Python 3.13+ breaks rs3 (Rule Set 3) because it pins scikit-learn 1.0.2.

: "${PE_HUB_PYTHON_VERSION:=3.11}"

pe_hub_resolve_python() {
    if [[ -n "${PYTHON:-}" ]]; then
        echo "${PYTHON}"
        return 0
    fi
    if [[ -n "${CONDA_PREFIX:-}" && -x "${CONDA_PREFIX}/bin/python" ]]; then
        echo "${CONDA_PREFIX}/bin/python"
        return 0
    fi
    if [[ -n "${VIRTUAL_ENV:-}" && -x "${VIRTUAL_ENV}/bin/python" ]]; then
        echo "${VIRTUAL_ENV}/bin/python"
        return 0
    fi
    if command -v python3 >/dev/null 2>&1; then
        command -v python3
        return 0
    fi
    if command -v python >/dev/null 2>&1; then
        command -v python
        return 0
    fi
    return 1
}

pe_hub_python_version_string() {
    local py="$1"
    "$py" -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}")'
}

pe_hub_python_minor_string() {
    local py="$1"
    "$py" -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")'
}

pe_hub_refuse_system_python() {
    local py_real="$1"
    case "${py_real}" in
        /Library/Developer/CommandLineTools/*|/usr/bin/python*|/System/*)
            echo "Error: refusing system/Command Line Tools Python: ${py_real}" >&2
            echo "" >&2
            echo "Use a writable conda or venv environment (Python ${PE_HUB_PYTHON_VERSION}):" >&2
            echo "  ./scripts/setup-python-env.sh" >&2
            echo "  conda activate pe-hub    # or your env name" >&2
            echo "  ./scripts/install-clis.sh" >&2
            return 1
            ;;
    esac
    return 0
}

pe_hub_require_python_version() {
    local py="$1"
    local minor ver
    minor="$(pe_hub_python_minor_string "${py}")"
    ver="$(pe_hub_python_version_string "${py}")"

    if [[ "${minor}" == "${PE_HUB_PYTHON_VERSION}" ]]; then
        return 0
    fi

    echo "Error: Python ${PE_HUB_PYTHON_VERSION} is required (found ${ver})." >&2
    echo "" >&2
    if [[ "${minor}" == "3.13" || "${minor}" == "3.14" ]]; then
        echo "Python 3.13+ is not supported: OptiPrime's rs3 dependency pins scikit-learn 1.0.2," >&2
        echo "which cannot be built on modern Python releases." >&2
        echo "" >&2
    fi
    echo "Create the supported environment, then re-run install:" >&2
    echo "  ./scripts/setup-python-env.sh" >&2
    echo "  conda activate pe-hub" >&2
    echo "  ./scripts/install-clis.sh" >&2
    echo "" >&2
    echo "Or with an existing conda base:" >&2
    echo "  conda create -n pe-hub python=${PE_HUB_PYTHON_VERSION} -y" >&2
    echo "  conda activate pe-hub" >&2
    return 1
}

pe_hub_warn_if_no_virtual_env() {
    if [[ -z "${CONDA_PREFIX:-}" && -z "${VIRTUAL_ENV:-}" ]]; then
        echo "Warning: CONDA_PREFIX / VIRTUAL_ENV not set." >&2
        echo "If install fails with Permission denied, activate conda/venv and retry." >&2
        echo "" >&2
    fi
}

pe_hub_export_pip_constraint() {
    local repo_root="$1"
    local constraints="${repo_root}/requirements/constraints.txt"
    if [[ -f "${constraints}" ]]; then
        export PIP_CONSTRAINT="${constraints}"
    fi
}
