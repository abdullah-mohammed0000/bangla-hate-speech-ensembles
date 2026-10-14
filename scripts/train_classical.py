#!/usr/bin/env python
"""Train the classical baselines (TF-IDF word/char + SVM, BiLSTM) for the given regimes. CPU is enough
(the BiLSTM uses a GPU if there is one).

  python scripts/train_classical.py --regime A B
"""
import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from bhs import config as C  # noqa: E402
from bhs.classical import run_bilstm, run_svm  # noqa: E402
from bhs.data import load_splits  # noqa: E402
from bhs.store import Store  # noqa: E402

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--regime", nargs="+", default=["A", "B"], choices=["A", "B", "O"])
    ap.add_argument("--model", nargs="+", default=list(C.CLASSICAL), choices=list(C.CLASSICAL))
    ap.add_argument("--splits", default="data/splits")
    ap.add_argument("--out", default="runs/predictions")
    a = ap.parse_args()
    D, store, out = load_splits(a.splits), Store(a.out), Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    log = out / "results_classical.csv"
    for r in a.regime:
        for m in a.model:
            if m == "bilstm":
                for s in C.SEEDS:
                    if not store.has(m, r, s, "test_all"):
                        run_bilstm(r, s, D, out).to_csv(log, mode="a", header=not log.exists(), index=False)
            elif not all(store.has(m, r, s, "test_all") for s in C.SEEDS):
                run_svm(m, r, D, out).to_csv(log, mode="a", header=not log.exists(), index=False)
