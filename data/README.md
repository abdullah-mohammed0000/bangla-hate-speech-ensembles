# Data

This repository does **not** contain any corpus text. Put the raw files in `data/raw/` (zips are fine, they are
unpacked automatically) and run `python scripts/prepare_data.py`.

| Corpus | Files expected | Rows | Where from |
|---|---|---|---|
| BD-SHS | `train.csv`, `val.csv`, `test.csv` (columns `sentence`, `target`, `type`, `hate speech`) | 40,224 / 5,028 / 5,029 | Romim et al., LREC 2022 - obtain from the authors' official distribution |
| BanglaHateBERT data | `train_data.xlsx`, `valid_data.xlsx`, `test_data.xlsx` | official train/valid/test files | Jahan et al., ResTUP 2022 (the "BanglaHateBert" repository; its README points to the dataset) |
| BanHate | `Hate_Comment_Label.xlsx`, `Non_Hate_Comment_Label.xlsx` | 19,197 | Raquib et al., BLP-2025 |
| Own dataset | `Combined_Hate_Comments_Final_v5.xlsx`, `combine_nonhate_dataset_8.xlsx` | 13,062 hate + 13,301 non-hate = 26,363 | **not public**; available from the authors on request |

Please cite the original papers when you use these corpora; they keep their own licences.

## Integrity checks

* The BD-SHS files are checked against SHA-256 fingerprints (`BDSHS_SHA16` in `src/bhs/config.py`).
* `prepare_data.py` writes `composition.csv` (hate / non-hate counts per source and split), `audit_dropped_rows.csv`
  (every row that was removed and why) and `splits_info.json`, and prints **OK** only when all nine output files are
  byte-identical to the ones used in the thesis (`SPLIT_SHA16`).

## Without the own dataset

The own dataset is needed for Study 2 (merged corpus) and Study 3. Without it you can
* run Study 1 with the public BD-SHS files only (`--bdshs-only`), and
* re-create every table of all three studies from `results/predictions.zip` (the predictions and the test labels are included).

If you replace the own dataset with another corpus in the same two-file format, `prepare_data.py` will run, but the
hash check will (correctly) report that the files differ from the thesis files.
