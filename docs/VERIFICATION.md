# Verification of the released results (14 October 2026)

Checks performed on the 70 archives of the final run:

1. **Headline numbers.** Macro-F1 of BanglaBERT + normalizer, XLM-R, BanglaBERT-large + normalizer, E2 and E6 on the three
   test sets (Single and Combine) and the own-only runs, recomputed from the stored probabilities: all 12 values and the
   Study 1 value (E6 0.9283) equal the thesis (tolerance 1e-4). Run `python scripts/verify_release.py`.
2. **Statistics.** `scripts/analyze.py` re-creates the Combine - Single changes (e.g. E6 -0.0383 / +0.0939 / +0.2020),
   the own-only vs Combine test (+0.0153 and +0.0146, 3/3 seeds, CI above 0) and E6 vs BanglaBERT + normalizer in
   Study 3 (+0.0035, CI [-0.0001, 0.0070], not significant).
3. **Data.** The nine split files rebuilt from the raw corpora have the same SHA-256 as the files used for training.
4. **Duplicates.** Archives uploaded twice are byte-identical, except the size-control archives `m3_mbert_C*`: four reruns
   of the same seeds differ slightly (non-deterministic GPU training). The thesis numbers (mean 0.8972 on BD-SHS) come
   from the two identical archives `m3_mbert_C (1)` and `(2)`; only these are in the release.
5. **Software.** All 96 job records with version information report Python 3.13.15, PyTorch 2.11.0+cu130,
   transformers 5.18.0, scikit-learn 1.6.1, Tesla T4.
6. **Refactored code.** The refactored TF-IDF char-SVM reproduces the stored probabilities (max difference 3e-8);
   the data pipeline reproduces all split hashes. The refactored GPU training code (`bhs.transformer`, BiLSTM) follows the
   legacy scripts line by line but was not re-run for this verification; run `scripts/train_transformer.py` for one
   seed and compare with `results/tables` to confirm in your environment.
