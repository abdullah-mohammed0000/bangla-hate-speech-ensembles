"""Evaluation metrics (macro-F1 is the main metric)."""
from __future__ import annotations

import numpy as np
from sklearn.metrics import (accuracy_score, f1_score, precision_recall_fscore_support, precision_score,
                             recall_score)


def metrics(y, pred) -> dict:
    mp, mr, mf, _ = precision_recall_fscore_support(y, pred, average="macro", zero_division=0)
    return dict(
        Accuracy=accuracy_score(y, pred), Macro_P=mp, Macro_R=mr, Macro_F1=mf,
        Weighted_F1=f1_score(y, pred, average="weighted"),
        Hate_P=precision_score(y, pred, pos_label=1, zero_division=0),
        Hate_R=recall_score(y, pred, pos_label=1, zero_division=0),
        Hate_F1=f1_score(y, pred, pos_label=1, zero_division=0),
    )


def macro_f1(y, pred) -> float:
    """Fast macro-F1 for binary 0/1 arrays."""
    y, pred = np.asarray(y), np.asarray(pred)
    out = []
    for c in (0, 1):
        tp = ((pred == c) & (y == c)).sum()
        fp = ((pred == c) & (y != c)).sum()
        fn = ((pred != c) & (y == c)).sum()
        out.append(2 * tp / max(2 * tp + fp + fn, 1))
    return float(np.mean(out))
