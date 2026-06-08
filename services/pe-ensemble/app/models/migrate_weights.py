"""One-time migration of vendor pretrained weights into WEIGHTS_ROOT.

Moves weight files out of vendor submodule trees and registers them in the
central weights registry. Legacy full-pickle OPED artifacts are left behind.
"""
from __future__ import annotations

import argparse
import shutil
import sys
from glob import glob
from pathlib import Path
from typing import Iterable, List

from .vendor_path import resolve_vendor_models_path
from .weights_registry import (
    _entry_dir,
    rebuild_index,
    register,
    register_from_directory,
    weights_root,
)


def _pridict2_compact_id(run_dir: Path, trained_root: Path) -> str:
    parts = [p for p in run_dir.relative_to(trained_root).parts if p != "train_val"]
    return "__".join(parts)


def _entry_registered(model: str, weight_id: str) -> bool:
    entry = _entry_dir(model, weight_id)
    return entry.is_dir() and (entry / "manifest.json").is_file()


def _migrate_deepprime(*, dry_run: bool) -> List[str]:
    models_root = resolve_vendor_models_path("deepprime", "models", "DeepPrime")
    base_dir = models_root / "DeepPrime_base"
    mean_src = base_dir / "mean.csv"
    std_src = base_dir / "std.csv"
    if not mean_src.is_file() or not std_src.is_file():
        raise FileNotFoundError(f"DeepPrime normalization files missing under {base_dir}")

    migrated: List[str] = []
    for entry in sorted(models_root.iterdir()):
        if not entry.is_dir():
            continue
        pt_files = sorted(entry.glob("*.pt"))
        if not pt_files:
            continue

        weight_id = entry.name
        if _entry_registered("deepprime", weight_id):
            print(f"[skip] deepprime/{weight_id} already registered")
            continue

        dest = _entry_dir("deepprime", weight_id)
        print(f"[deepprime] {weight_id} -> {dest}")
        if dry_run:
            migrated.append(weight_id)
            continue

        def populate(dest_dir: Path, variant: Path = entry) -> None:
            for pt in sorted(variant.glob("*.pt")):
                shutil.move(str(pt), str(dest_dir / pt.name))
            shutil.copy2(mean_src, dest_dir / "mean.csv")
            shutil.copy2(std_src, dest_dir / "std.csv")

        register(
            "deepprime",
            weight_id=weight_id,
            label=weight_id,
            source="vendor",
            format_name="deepprime_ensemble",
            metadata={
                "provenance": {"vendor_origin": str(entry)},
            },
            populate=lambda d, v=entry: populate(d, v),
            rebuild=False,
        )
        migrated.append(weight_id)
    return migrated


def _migrate_oped(*, dry_run: bool) -> List[str]:
    model_root = resolve_vendor_models_path(
        "oped", "pegRNA_PredictingCodes", "Model_Trained"
    )
    migrated: List[str] = []
    for weights_file in sorted(model_root.glob("*_weights.pt")):
        weight_id = weights_file.stem
        if _entry_registered("oped", weight_id):
            print(f"[skip] oped/{weight_id} already registered")
            continue

        dest = _entry_dir("oped", weight_id)
        print(f"[oped] {weights_file.name} -> {dest}/weights.pt")
        if dry_run:
            migrated.append(weight_id)
            continue

        def populate(dest_dir: Path, src: Path = weights_file) -> None:
            shutil.move(str(src), str(dest_dir / "weights.pt"))

        register(
            "oped",
            weight_id=weight_id,
            label=weight_id,
            source="vendor",
            format_name="oped_state_dict",
            metadata={
                "provenance": {"vendor_origin": str(weights_file)},
            },
            populate=lambda d, s=weights_file: populate(d, s),
            rebuild=False,
        )
        migrated.append(weight_id)
    return migrated


def _migrate_pridict2(*, dry_run: bool) -> List[str]:
    trained_root = resolve_vendor_models_path("pridict2", "trained_models")
    migrated: List[str] = []
    for run_dir in sorted(trained_root.glob("*/*/train_val/run_*")):
        if not (run_dir / "model_statedict").is_dir() or not (run_dir / "config").is_dir():
            continue

        weight_id = _pridict2_compact_id(run_dir, trained_root)
        if _entry_registered("pridict2", weight_id):
            print(f"[skip] pridict2/{weight_id} already registered")
            continue

        dest = _entry_dir("pridict2", weight_id)
        print(f"[pridict2] {run_dir} -> {dest}")
        if dry_run:
            migrated.append(weight_id)
            continue

        register_from_directory(
            "pridict2",
            run_dir,
            weight_id=weight_id,
            label=weight_id.replace("__", " / "),
            source="vendor",
            format_name="pridict2_run",
            metadata={
                "provenance": {"vendor_origin": str(run_dir)},
            },
            move=True,
            rebuild=False,
        )
        migrated.append(weight_id)

        # Remove empty parent dirs left after moving run contents.
        for parent in (run_dir, run_dir.parent, run_dir.parent.parent):
            if parent.is_dir() and not any(parent.iterdir()):
                parent.rmdir()

    return migrated


def migrate_all(*, dry_run: bool = False) -> dict:
    print(f"WEIGHTS_ROOT={weights_root()}")
    if dry_run:
        print("DRY RUN — no files will be moved")

    results = {
        "deepprime": _migrate_deepprime(dry_run=dry_run),
        "oped": _migrate_oped(dry_run=dry_run),
        "pridict2": _migrate_pridict2(dry_run=dry_run),
    }
    if not dry_run:
        rebuild_index()
        print(f"Registered {sum(len(v) for v in results.values())} weight set(s).")
    return results


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Migrate vendor weights into WEIGHTS_ROOT.")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print planned moves without modifying files.",
    )
    args = parser.parse_args(list(argv) if argv is not None else None)
    migrate_all(dry_run=args.dry_run)
    return 0


if __name__ == "__main__":
    sys.exit(main())
