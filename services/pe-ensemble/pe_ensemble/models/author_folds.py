"""Author original-fold predicates used by vendor provenance stamping.

DeepPrime and OptiPrime treat unlabeled/NaN folds as training (conservative
for leak checks). OPED treats unlabeled/NaN as not-train so missing labels
cannot inflate the published training-loci set. Those policies must stay
explicit — they are not interchangeable.
"""
from __future__ import annotations

from functools import partial

import pandas as pd


def is_author_train_fold(
    value: object,
    *,
    unlabeled_is_train: bool,
    parse_string_labels: bool = False,
) -> bool:
    """Return True when ``value`` is an author training fold.

    Author test is ``-1`` (and DeepPrime's string ``Test``). Other numeric
    labels are train. Unlabeled/NaN follows ``unlabeled_is_train``.
    """
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return unlabeled_is_train
    if parse_string_labels and isinstance(value, str):
        token = value.strip().lower()
        if token in {"test", "-1"}:
            return False
        try:
            value = float(token)
        except ValueError:
            return unlabeled_is_train
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return unlabeled_is_train
    if pd.isna(numeric):
        return unlabeled_is_train
    return numeric != -1.0


# DeepPrime published ``fold`` uses string ``Test`` plus numeric -1.
deepprime_is_author_train_fold = partial(
    is_author_train_fold,
    unlabeled_is_train=True,
    parse_string_labels=True,
)

# OptiPrime / mixed catalogs: unlabeled is train; no string ``Test`` token.
optiprime_is_author_train_fold = partial(
    is_author_train_fold,
    unlabeled_is_train=True,
)

# OPED: unlabeled must not count as published training loci.
oped_is_author_train_fold = partial(
    is_author_train_fold,
    unlabeled_is_train=False,
)
