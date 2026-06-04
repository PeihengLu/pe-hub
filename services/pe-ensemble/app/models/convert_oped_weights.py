"""Convert a legacy OPED full-pickle model into a portable state_dict.

OPED was originally distributed as a full ``torch.save(model)`` pickle, which
embeds the module path where the model class was defined (e.g. ``train_model``
or ``models``) and the PyTorch version used at save time. Such files break
across PyTorch versions and code refactors.

This one-shot utility loads such a pickle (mapping the legacy module paths to
the vendored OPED code) and writes out the model's ``state_dict`` instead. The
resulting ``*_weights.pt`` file is just tensor data keyed by layer name and
loads on any modern PyTorch, which is what :class:`OPEDModelWrapper` consumes.

Usage::

    python -m app.models.convert_oped_weights <input_pickle.pt> <output_weights.pt>

You normally only need this if the bundled ``*_weights.pt`` is missing or you
have produced a new full-pickle checkpoint that must be made portable.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

from .vendor_path import resolve_vendor_models_path

# Legacy module names that OPED full-pickles may reference. We alias them to the
# vendored implementation so unpickling can resolve the model class.
_LEGACY_MODULE_ALIASES = ("train_model", "models")


def _install_legacy_module_aliases() -> None:
    vendor_root = resolve_vendor_models_path()
    if str(vendor_root) not in sys.path:
        sys.path.insert(0, str(vendor_root))

    from oped.pegRNA_PredictingCodes import train_model as oped_train_model

    for alias in _LEGACY_MODULE_ALIASES:
        sys.modules.setdefault(alias, oped_train_model)


def convert(input_path: str, output_path: str) -> str:
    """Load a full-pickle OPED model and write its state_dict to ``output_path``."""
    _install_legacy_module_aliases()

    # weights_only=False is required because we are intentionally unpickling a
    # full module. Only run this on checkpoints you trust.
    obj = torch.load(input_path, map_location="cpu", weights_only=False)

    if isinstance(obj, dict) and all(torch.is_tensor(v) for v in obj.values()):
        state_dict = obj  # already a state_dict; just re-save portably
    elif isinstance(obj, torch.nn.Module):
        state_dict = obj.state_dict()
    else:
        raise TypeError(
            f"Unsupported checkpoint type {type(obj)!r}; expected a torch.nn.Module "
            "or a state_dict."
        )

    out = Path(output_path).expanduser()
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(state_dict, out)
    return str(out)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", help="Path to the legacy OPED full-pickle checkpoint")
    parser.add_argument("output", help="Destination path for the state_dict (*_weights.pt)")
    args = parser.parse_args(argv)

    out = convert(args.input, args.output)
    print(f"Wrote OPED state_dict to {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
