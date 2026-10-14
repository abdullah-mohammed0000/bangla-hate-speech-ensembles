# Bangla Hate Speech Detection with Transformer Ensembles

Code, predictions and result tables of the BSc thesis **"Bangla Hate Speech Detection with Transformer Ensembles:
Single-Corpus, Merged-Corpus and Own-Dataset Studies"** (Metropolitan University, Sylhet, 2026).
Authors: Abdullah Mohammed, Rayhan Ahmed Shurov. Supervisor: Nabila Zannat Rifa.

The thesis has three studies. Everything is run with three random seeds (42, 43, 44); differences are tested with
Holm-corrected McNemar tests and paired bootstrap confidence intervals.

| Study | Question | Main result (macro-F1, mean of 3 seeds) |
|---|---|---|
| **1** BD-SHS only | Which system is best on BD-SHS (train 40,224 / test 5,029)? | Ensemble **E6** (BanglaBERT-large + BanglaBERT + XLM-R, soft vote) **0.9283**; BanglaBERT + normalizer 0.9236 |
| **2** Single vs Combine | What happens when the same system is trained on a merge of four corpora (110,451 comments) instead of BD-SHS only? | Combine lowers BD-SHS macro-F1 by 0.025–0.041 and raises the merged test by 0.087–0.101 and the own-dataset test by 0.19–0.23 (all significant); E6: 0.9285 → 0.8901 / 0.7922 → 0.8860 / 0.7410 → 0.9430 |
| **3** Own dataset only | How high does a system get when trained and tested on the new own dataset (26,363 comments, test 2,636)? | **E6 0.9575**; own-only is significantly higher than Combine (+0.0146, CI [0.0074, 0.0221]) |

E6 vs BanglaBERT + normalizer is **not** significantly different in Study 1 or 3; the thesis does not claim that E6 is
significantly better than the best published result on BD-SHS.

## Repository layout

```
src/bhs/            the library (config, data, text, metrics, transformer, classical, store, stats, analysis)
scripts/            command-line entry points (see below)
notebooks/          Main_Colab.ipynb - run everything in Google Colab, resumable, multi-account friendly
results/tables/     final result tables (CSV) + analysis_log.txt
results/predictions.zip   predicted class probabilities of every run (+ test labels): re-run all statistics without a GPU
data/               README (where to get each corpus) - the data itself is not stored here
docs/               REPRODUCING.md, DATA.md, VERIFICATION.md
legacy_colab_scripts/    the original step-by-step Colab scripts that produced the thesis results (kept for the record)
tests/              unit tests and reproduction tests
```

## Quick start 1 - re-create the tables from the released predictions (no GPU, ~1 minute)

```bash
git clone https://github.com/abdullah-mohammed0000/bangla-hate-speech-ensembles.git && cd bangla-hate-speech-ensembles
pip install -r requirements.txt
unzip -q results/predictions.zip -d results/          # -> results/predictions/
python scripts/verify_release.py                      # ALL REPRODUCED = 13 headline checks against the thesis
python scripts/analyze.py                             # Study 1/2/3 tables + significance tests -> results/tables/
```

## Quick start 2 - retrain everything (Google Colab, T4 GPU)

Open `notebooks/Main_Colab.ipynb` in Colab and follow the cells. In short:

```bash
pip install -r requirements.txt -r requirements-train.txt
python scripts/prepare_data.py --raw data/raw --out data/splits          # prints OK if identical to the thesis files
python scripts/train_classical.py   --regime A B                          # SVMs + BiLSTM
python scripts/train_transformer.py --model banglabert_norm xlmr banglabert_large_norm --regime A B --seed 42 43 44
python scripts/train_transformer.py --model banglabert_norm xlmr banglabert_large_norm --regime O   --seed 42 43 44
python scripts/analyze.py --predictions runs/predictions --out results/tables_rerun
```

Finished runs are skipped, so an interrupted Colab session can simply be restarted; predictions from several Colab
accounts are merged by copying the `probs_*.npy` files into one folder (details in `docs/REPRODUCING.md`).

Study 1 only needs the public BD-SHS files: `python scripts/train_transformer.py --model banglabert_norm --seed 42 --bdshs-only data/raw`.

## Data

| Corpus | Used for | Source |
|---|---|---|
| BD-SHS (Romim et al.) | Studies 1-3 | original authors (see `data/README.md`) |
| BanglaHateBERT data (Jahan et al.) | merged corpus | original authors |
| BanHate | merged corpus | original authors |
| **Own dataset** (26,363 Facebook/YouTube comments, collected Aug-Oct 2026) | Studies 2-3 | **not public** (privacy); available from the authors on request |

`prepare_data.py` rebuilds the merged splits from the raw corpora and checks the SHA-256 of every output file
against the files used in the thesis (`src/bhs/config.py`). Without the own dataset you can still run Study 1 and
re-run all analyses from `results/predictions.zip`. The own-dataset test labels needed by the analyses are part of the
released predictions.

## Training protocol (all studies)

Fine-tuning with class-weighted cross-entropy (weights from the training set of the regime), AdamW, learning rate 2e-5,
weight decay 0.01, batch 32, 3 epochs, maximum length 128, fp16, no warm-up; the epoch with the best validation macro-F1
is kept. BanglaBERT "+ normalizer" models apply the csebuetnlp normalizer after cleaning. Ensembles average the class
probabilities of their members (same seed). Hyper-parameters are in `src/bhs/config.py`.

## Reproducibility notes

* All 96 job records that contain version information report a Colab Tesla T4 and the same software versions (Python 3.13.15, PyTorch 2.11.0+cu130,
  transformers 5.18.0, scikit-learn 1.6.1).
* GPU training is not bit-wise deterministic. A re-run of the same (model, regime, seed) can differ by a few thousandths
  of macro-F1 (observed for the size-matched control: seed 44 gave 0.8973, 0.8989 and 0.9001 in three runs).
  Differences of this size are not interpreted in the thesis. Re-training will therefore give numbers close to, not
  identical with, those in the tables; the released predictions reproduce the thesis numbers exactly.
* The SVMs are deterministic (the retrained char-SVM probabilities match the released ones to 3e-8).

## Citation

See `CITATION.cff`.

## License

Code: MIT (see `LICENSE`). The corpora keep the licences of their original authors; the own dataset is not released.
