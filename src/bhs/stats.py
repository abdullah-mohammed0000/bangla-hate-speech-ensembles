"""Statistical comparison of two systems on the same test set.

Rules used throughout the thesis:
  * McNemar test (continuity corrected) on per-comment correctness, one test per seed;
    Holm correction over the family of comparisons that are reported together.
  * Paired percentile bootstrap (2,000 resamples, seed 2024) of the macro-F1 difference (mean of the 3 seeds).
  * Verdict: "significant"     = Holm-McNemar significant in 3/3 seeds AND the bootstrap CI excludes 0,
             "not significant" = 0/3 seeds AND the CI contains 0, otherwise "mixed".
"""
from __future__ import annotations

import numpy as np
from scipy.stats import chi2

from . import config as C
from .metrics import macro_f1


def mcnemar(correct_a, correct_b) -> float:
    """Two-sided p-value of the McNemar test with continuity correction (1.0 if no discordant pairs)."""
    b = int((correct_a & ~correct_b).sum())
    c = int((~correct_a & correct_b).sum())
    return 1.0 if b + c == 0 else float(chi2.sf((abs(b - c) - 1) ** 2 / (b + c), 1))


def holm(p) -> np.ndarray:
    p = np.asarray(p, dtype=float)
    order = np.argsort(p)
    m, adj, run = len(p), np.empty(len(p)), 0.0
    for rank, i in enumerate(order):
        run = max(run, (m - rank) * p[i])
        adj[i] = min(1.0, run)
    return adj


class Bootstrap:
    """Paired bootstrap weights for one test set (the same resamples for all systems)."""

    def __init__(self, y, resamples=C.BOOT_RESAMPLES, seed=C.BOOT_SEED):
        self.y = np.asarray(y)
        n = len(self.y)
        rng = np.random.default_rng(seed)
        self.W = rng.multinomial(n, np.full(n, 1.0 / n), size=resamples).astype(np.float32)
        self.y1 = (self.y == 1).astype(np.float32)

    def macro_f1(self, pred) -> np.ndarray:
        p1 = (np.asarray(pred) == 1).astype(np.float32)
        tp = self.W @ (self.y1 * p1)
        fp = self.W @ ((1 - self.y1) * p1)
        fn = self.W @ (self.y1 * (1 - p1))
        tn = self.W @ ((1 - self.y1) * (1 - p1))
        return (2 * tp / np.maximum(2 * tp + fp + fn, 1e-9) + 2 * tn / np.maximum(2 * tn + fn + fp, 1e-9)) / 2


def verdict(seeds_significant: int, ci_excludes_zero: bool, n_seeds=3) -> str:
    if seeds_significant == n_seeds and ci_excludes_zero:
        return "significant"
    if seeds_significant == 0 and not ci_excludes_zero:
        return "not significant"
    return "mixed"


def ci_of_difference(boot: Bootstrap, preds_first: dict, preds_second: dict, seeds=C.SEEDS):
    """95% percentile CI of mean-over-seeds macro-F1 (first - second). preds_*: seed -> predicted classes."""
    d = (np.mean([boot.macro_f1(preds_first[s]) for s in seeds], 0)
         - np.mean([boot.macro_f1(preds_second[s]) for s in seeds], 0))
    lo, hi = np.percentile(d, [2.5, 97.5])
    return float(lo), float(hi)


def mean_macro_f1(y, preds: dict, seeds=C.SEEDS) -> float:
    return float(np.mean([macro_f1(y, preds[s]) for s in seeds]))
