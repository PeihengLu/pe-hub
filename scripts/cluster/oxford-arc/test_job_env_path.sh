#!/usr/bin/env bash
# Regression: module Anaconda python must not win over CONDA_PREFIX after pin.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
JOB_ENV_SOURCE_LIB=1
# shellcheck source=./job_env.sh
source "${ROOT}/job_env.sh"

tmpdir="$(mktemp -d)"
trap 'rm -rf "${tmpdir}"' EXIT

mkdir -p "${tmpdir}/module/bin" "${tmpdir}/env/bin" "${tmpdir}/other/bin"
printf '#!/bin/bash\necho module\n' > "${tmpdir}/module/bin/python"
printf '#!/bin/bash\necho env\n' > "${tmpdir}/env/bin/python"
printf '#!/bin/bash\necho other\n' > "${tmpdir}/other/bin/python"
chmod +x "${tmpdir}/module/bin/python" "${tmpdir}/env/bin/python" "${tmpdir}/other/bin/python"

orig_path="${PATH}"
PATH="${tmpdir}/module/bin:${tmpdir}/env/bin:${tmpdir}/other/bin:${orig_path}"
export PATH
if [[ "$(command -v python)" != "${tmpdir}/module/bin/python" ]]; then
    echo "setup failed: expected module python first" >&2
    exit 1
fi

_pin_conda_prefix_bin "${tmpdir}/env"
if [[ "$(command -v python)" != "${tmpdir}/env/bin/python" ]]; then
    echo "pin failed: command -v python is $(command -v python)" >&2
    exit 1
fi
if [[ "$(python)" != "env" ]]; then
    echo "pin failed: python ran $(python)" >&2
    exit 1
fi

# Second pin stays first and does not duplicate forever.
_pin_conda_prefix_bin "${tmpdir}/env"
case "${PATH}" in
    "${tmpdir}/env/bin:${tmpdir}/env/bin:"*)
        echo "pin duplicated env bin on PATH: ${PATH}" >&2
        exit 1
        ;;
esac

echo "ok: _pin_conda_prefix_bin prefers CONDA_PREFIX over module python"
