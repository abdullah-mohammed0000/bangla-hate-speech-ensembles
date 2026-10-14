#!/usr/bin/env python
"""Check that predictions reproduce the headline numbers of the thesis (tolerance 0.0001; missing runs are skipped).
For re-trained runs small differences are expected (GPU non-determinism).
Exit code 0 = everything reproduced.   python scripts/verify_release.py --predictions results/predictions"""
import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from bhs.metrics import macro_f1  # noqa: E402
from bhs.store import Store  # noqa: E402

SYS = {"BB-norm": "banglabert_norm", "XLM-R": "xlmr", "BB-large": "banglabert_large_norm", "E2": "E2", "E6": "E6"}
# (regime, system, (BD-SHS, merged, own)) macro-F1, mean of three seeds, as reported in the thesis
EXPECTED = [
    ("A", "BB-norm", (.9237, .7931, .7483)), ("A", "E2", (.9257, .7862, .7292)), ("A", "BB-large", (.9250, .7952, .7494)),
    ("A", "E6", (.9285, .7922, .7410)), ("B", "BB-norm", (.8837, .8802, .9387)), ("B", "E2", (.8912, .8845, .9399)),
    ("B", "BB-large", (.8840, .8820, .9422)), ("B", "E6", (.8901, .8860, .9430)),
]
EXPECTED_OWN_ONLY = {"BB-norm": .9540, "E6": .9575, "BB-large": .9578, "XLM-R": .9364}

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--predictions", default="results/predictions")
    st = Store(ap.parse_args().predictions)
    bad = checked = 0

    def f1(system, regime, test):
        y = st.labels(test)
        return float(np.mean([macro_f1(y, st.pred(SYS[system], regime, s, test)) for s in (42, 43, 44)]))

    for regime, system, exp in EXPECTED:
        try:
            got = tuple(round(f1(system, regime, t), 4) for t in ("test_bdshs", "test_all", "test_own"))
        except FileNotFoundError:
            print(f"SKIP Study 2, regime {regime}, {system}: not run / not in this folder")
            continue
        ok = all(abs(g - e) <= 1.1e-4 for g, e in zip(got, exp))
        bad += not ok
        checked += 1
        print(f"{'OK  ' if ok else 'DIFF'} Study 2, regime {regime}, {system:8s} got {got} thesis {exp}")
    for system, exp in EXPECTED_OWN_ONLY.items():
        try:
            g = round(f1(system, "O", "test_own"), 4)
        except FileNotFoundError:
            print(f"SKIP Study 3, own-only, {system}: not run / not in this folder")
            continue
        ok = abs(g - exp) <= 1.1e-4
        bad += not ok
        checked += 1
        print(f"{'OK  ' if ok else 'DIFF'} Study 3, own-only, {system:8s} got {g} thesis {exp}")
    try:
        g = round(f1("E6", "S", "test_bdshs"), 4)
        ok = abs(g - 0.9283) <= 1.1e-4
        bad += not ok
        checked += 1
        print(f"{'OK  ' if ok else 'DIFF'} Study 1, E6 on BD-SHS got {g} thesis 0.9283")
    except FileNotFoundError:
        print("SKIP Study 1 (original Study-1 runs, regime S, are only in the released predictions)")
    print(f"{checked} checks run:", "ALL REPRODUCED" if checked and not bad else (f"{bad} MISMATCH(ES)" if bad else "NOTHING TO CHECK"))
    sys.exit(1 if bad or not checked else 0)
