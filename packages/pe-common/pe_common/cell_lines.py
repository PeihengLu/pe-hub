"""Cell-line name aliases shared by PE-DB filters and eval drivers.

PRIDICT library-diverse stores HEK293T as ``hek``; DeepPrime / DeepPE / OptiPrime
store the same line as ``hek293t``. Treat those as one cell line.
"""
from __future__ import annotations

from typing import Iterable, Optional

# Filename/DB keys that refer to the same biological line → canonical key.
_CANONICAL = {
    "hek": "hek293t",
    "hek293t": "hek293t",
}

# Canonical datasheet key → preferred PRIDICT2 decoder heads (vendor
# ``decoder_<HEAD>.pkl``), in order. K562-MLH1dn prefers a dedicated head when
# present; August 2023 bundles only ship HEK/K562, so the unscanned default is
# K562.
_PRIDICT2_HEADS = {
    "hek293t": ("HEK",),
    "k562": ("K562",),
    "k562mlh1dn": ("K562MLH1dn", "K562"),
}


def normalize_cell_line_key(value: str) -> str:
    return str(value).strip().lower().replace("-", "_")


def canonical_cell_line(value: str) -> str:
    """Return the canonical datasheet key (``hek`` and ``hek293t`` → ``hek293t``)."""
    key = normalize_cell_line_key(value)
    if not key:
        return key
    return _CANONICAL.get(key, key)


def cell_line_alias_group(value: str) -> tuple[str, ...]:
    """All DB/filename keys that should match a filter for this cell line."""
    key = normalize_cell_line_key(value)
    if not key:
        return ()
    canon = canonical_cell_line(key)
    members = {key, canon}
    members.update(alias for alias, target in _CANONICAL.items() if target == canon)
    return tuple(sorted(members))


def expand_cell_line_filter_values(values: Iterable[str]) -> list[str]:
    """Expand a cell-line filter list so ``hek293t`` also matches ``hek`` sheets."""
    expanded: list[str] = []
    seen: set[str] = set()
    for value in values:
        for key in cell_line_alias_group(str(value)):
            if key not in seen:
                seen.add(key)
                expanded.append(key)
    return expanded


def pridict2_head_for_cell_line(
    cell_line: Optional[str],
    available_heads: Optional[Iterable[str]] = None,
) -> str:
    """Pick a PRIDICT2 cell-type head for a datasheet cell line.

    HEK293T (``hek`` / ``hek293t``) → ``HEK``. K562 → ``K562``. K562-MLH1dn
    uses ``K562MLH1dn`` when that head exists, otherwise ``K562``. Other lines
    fall back to ``HEK``.
    """
    available = [str(head) for head in (available_heads or ())]
    key = canonical_cell_line(cell_line or "")
    wanted = list(_PRIDICT2_HEADS.get(key, ("HEK",)))
    if available:
        for head in wanted:
            if head in available:
                return head
        if "HEK" in available:
            return "HEK"
        return available[0]
    # No inventory (eval driver): skip heads that August 2023 bundles lack.
    return next((head for head in wanted if head != "K562MLH1dn"), wanted[0])
