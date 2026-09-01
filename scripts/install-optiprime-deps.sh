#!/usr/bin/env bash
# Install OptiPrime vendor dependencies (JAX stack + rs3 Rule Set 3).
#
# rs3 pins scikit-learn<=1.0.2, which does not build on Python 3.13+.
# We install rs3 with --no-deps and rely on requirements/constraints.txt for
# lightgbm<=3.3.5 and a modern scikit-learn on Python 3.11.
#
# Called automatically from scripts/install-clis.sh.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

# shellcheck source=scripts/python-env.sh
source "${REPO_ROOT}/scripts/python-env.sh"

if ! PYTHON="$(pe_hub_resolve_python)"; then
    echo "Error: no Python interpreter found." >&2
    exit 1
fi
pe_hub_require_python_version "${PYTHON}" || exit 1
pe_hub_export_pip_constraint "${REPO_ROOT}"

echo "Installing OptiPrime dependencies (JAX stack, rs3) ..."
"${PYTHON}" -m pip install -e "${REPO_ROOT}/services/pe-ensemble[optiprime]"

# rs3 declares scikit-learn<=1.0.2; install without pulling that pin.
"${PYTHON}" -m pip install rs3 --no-deps

# LightGBM 4.x regressor pickles from rs3 need _n_classes on newer lightgbm.
"${PYTHON}" - <<'PY'
from pathlib import Path
import inspect
import rs3

seq_py = Path(inspect.getfile(rs3)).parent / "seq.py"
text = seq_py.read_text()
needle = (
    "    model = joblib.load(os.path.join(os.path.dirname(__file__), 'RuleSet3.pkl'))\n"
    "    return model"
)
patch = (
    "    model = joblib.load(os.path.join(os.path.dirname(__file__), 'RuleSet3.pkl'))\n"
    "    # RuleSet3.pkl targets LightGBM 3.x; 4.x predict() requires _n_classes.\n"
    "    if getattr(model, '_n_classes', None) is None:\n"
    "        model._n_classes = 1\n"
    "    return model"
)
if "model._n_classes = 1" in text:
    print("rs3 LightGBM patch already applied")
elif needle in text:
    seq_py.write_text(text.replace(needle, patch, 1))
    print(f"Patched {seq_py} for LightGBM 4.x compatibility")
else:
    print("Warning: could not auto-patch rs3/seq.py; OptiPrime RS3 scoring may fail")
PY

echo "Verifying OptiPrime imports ..."
"${PYTHON}" - <<'PY'
import importlib

for name in ("jax", "flax", "chex", "optax", "rs3"):
    importlib.import_module(name)

from rs3.seq import predict_seq

seq = "ATCGATCGATCGATCGATCGATCGATCGAT"
pred = predict_seq([seq], sequence_tracr="Chen2013")
assert len(pred) == 1
print("OptiPrime dependency check OK (rs3 predict_seq)")
PY
