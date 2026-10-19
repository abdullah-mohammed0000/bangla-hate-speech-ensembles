#!/usr/bin/env python
"""Re-create the thesis result tables and statistics from a folder of predictions (CPU only, ~1 minute).

  python scripts/analyze.py --predictions results/predictions --out results/tables

Tables written (CSV):
  study1_bdshs.csv                         Study 1: ten systems on BD-SHS
  study2_single.csv / study2_combine.csv   macro-F1 (mean ± std, 3 seeds) of every system on the three test sets
  study2_single_vs_combine.csv             change when trained on the merged data (Combine - Single), with
                                           Holm-McNemar over the systems and bootstrap CI, one table per test set
  study2_size_control.csv                  size-matched control with mBERT (regimes A, C, B)
  study3_own_only.csv                      own-only systems on the own test set
  study3_e6_vs_bbnorm.csv                  E6 vs BanglaBERT + normalizer (own-only)
  study3_own_vs_combine.csv                own-only vs Combine on the own test set
"""
import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from bhs import analysis as A  # noqa: E402
from bhs.store import Store  # noqa: E402

TESTS = {"test_bdshs": "BD-SHS test", "test_all": "Merged test", "test_own": "Own-dataset test"}
# the ten systems of Study 2 (classical, recurrent, transformers, ensembles)
STUDY2 = {
    "TF-IDF char + SVM": "tfidf_char_svm", "BiLSTM": "bilstm", "mBERT": "mbert", "IndicBERTv2": "indicbertv2",
    "XLM-RoBERTa": "xlmr", "BanglaBERT": "banglabert", "BanglaBERT + normalizer": "banglabert_norm",
    "BanglaBERT-large + normalizer": "banglabert_large_norm", "E2: BB-norm + XLM-R": "E2",
    "E6: BB-large-norm + BB-norm + XLM-R": "E6",
}

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--predictions", default="results/predictions")
    ap.add_argument("--out", default="results/tables")
    a = ap.parse_args()
    store, out = Store(a.predictions), Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    pd.set_option("display.width", 250, "display.max_columns", 30)

    # ---- Study 1: trained and tested on BD-SHS ----
    # regime S = the original Study-1 runs (released predictions); for a new run Study 1 equals regime A on test_bdshs
    s1_reg = {k: ("S" if store.available(v, "S", "test_bdshs") else "A") for k, v in STUDY2.items()}
    s1 = {k: v for k, v in STUDY2.items() if store.available(v, s1_reg[k], "test_bdshs")}
    if s1:
        g = pd.concat([A.result_grid(store, {k: v}, [s1_reg[k]], ["test_bdshs"]) for k, v in s1.items()], ignore_index=True)
        g["Macro-F1 (mean ± std)"] = g.Macro_F1.map("{:.4f}".format) + " ± " + g.Std.map("{:.4f}".format)
        g[["System", "Regime", "Macro-F1 (mean ± std)"]].to_csv(out / "study1_bdshs.csv", index=False)
        print(f"\n=== Study 1: trained and tested on BD-SHS (test 5,029; regime S = original runs, A = new runs) ===")
        print(g[["System", "Regime", "Macro-F1 (mean ± std)"]].to_string(index=False))

    # Study 2 needs both regimes (A and B) of a system; systems that are not (yet) run are skipped
    systems = {k: v for k, v in STUDY2.items() if all(store.available(v, r, "test_own") for r in "AB")}
    if not systems:
        print("\nStudy 2: no system has both Single (A) and Combine (B) predictions yet - skipped.")
    else:
        for r, fname in (("A", "study2_single"), ("B", "study2_combine")):
            g = A.result_grid(store, systems, [r], list(TESTS))
            g["Cell"] = g.Macro_F1.map("{:.4f}".format) + " ± " + g.Std.map("{:.4f}".format)
            t = g.pivot(index="System", columns="Test", values="Cell").reindex(list(systems))[list(TESTS)]
            t.columns = [TESTS[c] for c in t.columns]
            t.to_csv(out / f"{fname}.csv")
            print(f"\n=== Study 2, trained on {'BD-SHS only (Single)' if r == 'A' else 'the merged data (Combine)'} ===")
            print(t.to_string())

        sv = pd.concat([A.compare_regimes(store, systems, "A", "B", test) for test in TESTS])
        sv = sv.rename(columns={"A": "Single", "B": "Combine"})
        sv.to_csv(out / "study2_single_vs_combine.csv", index=False)
        print("\n=== Combine - Single (Holm over the systems) ===")
        print(sv[["Test", "System", "Single", "Combine", "Diff", "CI_low", "CI_high", "seeds_sig_of_3", "Verdict"]].to_string(index=False))

    # ---- Study 2, size-matched control (mBERT): A = BD-SHS, C = BD-SHS resampled to the size of B, B = merged ----
    if all(store.available("mbert", r, "test_all") for r in "ACB"):
        sc = pd.concat([A.compare_regimes(store, {"mBERT": "mbert"}, a, b, t).rename(columns={a: "First", b: "Second"})
                        .assign(Comparison=f"{b} - {a}") for t in ("test_bdshs", "test_all") for a, b in (("A", "C"), ("C", "B"), ("A", "B"))])
        sc = sc[["Comparison", "Test", "First", "Second", "Diff", "CI_low", "CI_high", "Holm_p_per_seed", "seeds_sig_of_3", "Verdict"]]
        sc.to_csv(out / "study2_size_control.csv", index=False)
        print("\n=== Size-matched control (mBERT): C - A = size alone, B - C = merged data beyond size ===")
        print(sc.to_string(index=False))

    if store.available("E6", "O", "test_own"):  # needs the three own-only models of E6
        s3 = A.study3_results(store)
        s3.to_csv(out / "study3_own_only.csv", index=False)
        print("\n=== Study 3: own-only, own test set ===")
        print(s3.to_string(index=False))
        e6 = pd.DataFrame([dict(Comparison="E6 - BanglaBERT + normalizer (own-only)",
                                **A.compare_systems(store, "E6", "banglabert_norm", "O", "test_own"))])
        e6.to_csv(out / "study3_e6_vs_bbnorm.csv", index=False)
        print(e6.to_string(index=False))
        two = {k: v for k, v in {"BanglaBERT + normalizer": "banglabert_norm", "E6": "E6"}.items() if store.available(v, "B", "test_own")}
        if two:
            oc = A.compare_regimes(store, two, "B", "O", "test_own").rename(columns={"B": "Combine", "O": "Own_only"})
            oc.to_csv(out / "study3_own_vs_combine.csv", index=False)
            print("\n=== Own-only - Combine on the own test set ===")
            print(oc.to_string(index=False))
    else:
        print("\nStudy 3: own-only predictions of E6 (BanglaBERT-large, BanglaBERT + normalizer, XLM-R; regime O) not complete - skipped.")
