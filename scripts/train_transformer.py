#!/usr/bin/env python
"""Fine-tune one transformer for one or more (model, regime, seed) combinations (GPU needed, T4 is enough).

Examples
  python scripts/train_transformer.py --model banglabert_norm --regime A B --seed 42 43 44
  python scripts/train_transformer.py --model banglabert_large_norm --regime O --seed 42
Finished runs are skipped, so the command can simply be run again after an interrupted session.
Outputs go to <out>/ : probs_<model>_<regime>_s<seed>_<test>.npy, valprobs_*, epochlog_*, results.csv
"""
import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from bhs import config as C  # noqa: E402
from bhs.data import load_bdshs, load_splits  # noqa: E402
from bhs.store import Store  # noqa: E402
from bhs.transformer import run_transformer  # noqa: E402

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", nargs="+", required=True, choices=list(C.MODELS))
    ap.add_argument("--regime", nargs="+", default=["A"], choices=["A", "B", "C", "O"])
    ap.add_argument("--seed", nargs="+", type=int, default=C.SEEDS)
    ap.add_argument("--splits", default="data/splits")
    ap.add_argument("--out", default="runs/predictions")
    ap.add_argument("--bdshs-only", metavar="RAW_DIR", help="Study 1 only: read just the public BD-SHS files from RAW_DIR "
                    "(no other corpus needed); implies --regime A and the BD-SHS test set")
    a = ap.parse_args()

    if a.bdshs_only:
        D, tests, a.regime = load_bdshs(a.bdshs_only), ["test_bdshs"], ["A"]
    else:
        D, tests = load_splits(a.splits), None
    store, out = Store(a.out), Path(a.out)
    for m in a.model:
        for r in a.regime:
            for s in a.seed:
                if store.has(m, r, s, "test_bdshs" if a.bdshs_only else "test_all"):
                    print(f"skip (done): {m} {r} s{s}")
                    continue
                rows = run_transformer(m, r, s, D, out, tests=tests)
                rows.to_csv(out / "results.csv", mode="a", header=not (out / "results.csv").exists(), index=False)
    log = out / "results.csv"
    if log.exists():
        rr = pd.read_csv(log)
        print(rr.groupby(["Model", "Regime", "Test"])["Macro_F1"].agg(["mean", "std"]).round(4))
