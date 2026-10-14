# Reproducing the experiments

## 1. What is run

| Study | Systems | Regimes | Seeds | Command |
|---|---|---|---|---|
| 1, 2 | mBERT, XLM-R, IndicBERTv2, BanglaBERT, BanglaBERT-unweighted (appendix), BanglaBERT + normalizer, BanglaBERT-large + normalizer | A (BD-SHS only), B (merged) | 42, 43, 44 | `train_transformer.py --model <m> --regime A B` |
| 1, 2 | TF-IDF word/char + SVM, BiLSTM | A, B | 42, 43, 44 | `train_classical.py --regime A B` |
| 2 (control) | mBERT | C (BD-SHS resampled to the size of B) | 42, 43, 44 | `train_transformer.py --model mbert --regime C` |
| 3 | BanglaBERT + normalizer, XLM-R, BanglaBERT-large + normalizer | O (own dataset only) | 42, 43, 44 | `train_transformer.py --model <m> --regime O` |

Ensembles (E1-E4, E6) are never trained: `bhs.store.Store` averages the stored probabilities of their members
(same seed). One transformer run = train on the regime's training set, keep the best validation epoch, predict all five
test sets, write `probs_<model>_<regime>_s<seed>_<test>.npy`.

Study 1 in the thesis was run before the Study 2 pipeline existed; its predictions are released as regime `S`. With the
current code, Study 1 equals regime `A` evaluated on `test_bdshs` (same data, same protocol).

## 2. Using several Colab accounts (GPU limits)

Every job is independent and resumable:

1. In each account run the cells of `notebooks/Main_Colab.ipynb` for the jobs you want (e.g. account 1: BanglaBERT-large, account 2: XLM-R ...). Each account clones the repo and uploads/prepares the same data; `prepare_data.py` prints OK, so all accounts are guaranteed to use identical splits.
2. At the end the notebook zips `runs/predictions` and downloads it.
3. In the main notebook, upload those zips in the "merge" cell: the `probs_*.npy`, `epochlog_*.csv` and `results*.csv` files are copied into one folder (file names are unique per model/regime/seed).
4. Run the analysis cell (`scripts/analyze.py`).

## 3. Checking the result

`python scripts/verify_release.py --predictions results/predictions` checks the released predictions against the thesis.
For a new run use `--predictions runs/predictions`; differences of a few thousandths are expected (GPU training is not
bit-wise deterministic), larger differences are not.

## 4. Legacy scripts

`legacy_colab_scripts/` holds the original step scripts (M1-M18, X1-X4, SHAP) exactly as they were run in Colab for the
thesis. They are kept as the record of how the numbers were produced; the `src/bhs` package is the cleaned-up version of
the same procedure. The SHAP analyses (thesis Chapters 5-6) are only available as legacy scripts.
