"""Analysis functions used by the CLI scripts (all CPU, all work on a :class:`bhs.store.Store`)."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

from . import config as C
from .metrics import macro_f1
from .stats import Bootstrap, ci_of_difference, holm, mcnemar, mean_macro_f1, verdict

# systems used in the thesis' main tables
SYSTEMS = {
    "BanglaBERT + normalizer": "banglabert_norm",
    "XLM-RoBERTa": "xlmr",
    "BanglaBERT-large + normalizer": "banglabert_large_norm",
    "E2: BB-norm + XLM-R": "E2",
    "E6: BB-large-norm + BB-norm + XLM-R": "E6",
}


def _preds(store, system, regime, test, seeds=C.SEEDS):
    return {s: store.pred(system, regime, s, test) for s in seeds}


def compare_regimes(store, systems: dict, reg_a="A", reg_b="B", test="test_own", seeds=C.SEEDS) -> pd.DataFrame:
    """Change of macro-F1 when the SAME system is trained with reg_b instead of reg_a (B - A), Holm over `systems`."""
    y = store.labels(test)
    boot = Bootstrap(y)
    pa = {k: _preds(store, v, reg_a, test, seeds) for k, v in systems.items()}
    pb = {k: _preds(store, v, reg_b, test, seeds) for k, v in systems.items()}
    pvals = {s: [mcnemar(pa[k][s] == y, pb[k][s] == y) for k in systems] for s in seeds}
    adj = {s: holm(pvals[s]) for s in seeds}
    rows = []
    for i, k in enumerate(systems):
        lo, hi = ci_of_difference(boot, pb[k], pa[k], seeds)
        fa, fb = mean_macro_f1(y, pa[k], seeds), mean_macro_f1(y, pb[k], seeds)
        sig = sum(adj[s][i] < C.ALPHA for s in seeds)
        rows.append({"System": k, "Test": test, f"{reg_a}": round(fa, 4), f"{reg_b}": round(fb, 4),
                     "Diff": round(fb - fa, 4), "CI_low": round(lo, 4), "CI_high": round(hi, 4),
                     "Holm_p_per_seed": ", ".join(f"{adj[s][i]:.3g}" for s in seeds), "seeds_sig_of_3": sig,
                     "Verdict": verdict(sig, lo > 0 or hi < 0, len(seeds))})
    return pd.DataFrame(rows)


def study3_results(store, seeds=C.SEEDS) -> pd.DataFrame:
    """Accuracy, macro-F1, hate P/R/F1 (mean ± std over seeds) of the own-only systems on the own test set."""
    y = store.labels("test_own")
    rows = []
    for name, sysm in SYSTEMS.items():
        if not store.available(sysm, "O", "test_own", seeds):
            continue
        M = np.array([[accuracy_score(y, p), f1_score(y, p, average="macro"), f1_score(y, p, average="weighted"),
                       precision_score(y, p), recall_score(y, p), f1_score(y, p)]
                      for p in (store.pred(sysm, "O", s, "test_own") for s in seeds)])
        mu, sd = M.mean(0), M.std(0, ddof=1)
        rows.append(dict(System=name, **{n: f"{a:.4f} ± {b:.4f}" for n, a, b in
                                         zip(["Accuracy", "Macro_F1", "Weighted_F1", "Hate_P", "Hate_R", "Hate_F1"], mu, sd)}))
    return pd.DataFrame(rows)


def compare_systems(store, first, second, regime, test, seeds=C.SEEDS) -> dict:
    """E.g. E6 vs BanglaBERT+normalizer inside one regime (a single comparison, no Holm needed)."""
    y = store.labels(test)
    p1, p2 = _preds(store, first, regime, test, seeds), _preds(store, second, regime, test, seeds)
    pv = [mcnemar(p1[s] == y, p2[s] == y) for s in seeds]
    lo, hi = ci_of_difference(Bootstrap(y), p1, p2, seeds)
    sig = sum(p < C.ALPHA for p in pv)
    return dict(Diff=round(mean_macro_f1(y, p1, seeds) - mean_macro_f1(y, p2, seeds), 4), CI_low=round(lo, 4),
                CI_high=round(hi, 4), McNemar_p_per_seed=", ".join(f"{p:.3g}" for p in pv),
                seeds_sig_of_3=sig, Verdict=verdict(sig, lo > 0 or hi < 0, len(seeds)))


def result_grid(store, systems: dict, regimes, tests, seeds=C.SEEDS) -> pd.DataFrame:
    """Mean ± std macro-F1 for every system x regime x test set (the thesis result tables)."""
    rows = []
    for name, sysm in systems.items():
        for r in regimes:
            for t in tests:
                if not store.available(sysm, r, t, seeds):
                    continue
                y = store.labels(t)
                f = [macro_f1(y, store.pred(sysm, r, s, t)) for s in seeds]
                rows.append(dict(System=name, Regime=r, Test=t, Macro_F1=round(float(np.mean(f)), 4),
                                 Std=round(float(np.std(f, ddof=1)), 4)))
    return pd.DataFrame(rows)
