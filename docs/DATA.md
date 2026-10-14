# Data used

Counts after cleaning and de-duplication (`data/splits/composition.csv`):

| File | BD-SHS | BanglaHateBERT | BanHate | Own | All |
|---|---|---|---|---|---|
| A_train / A_val | 40,224 / 5,028 | - | - | - | 40,224 / 5,028 |
| B_train | 40,122 | 10,476 | 15,234 | 20,990 | 86,822 |
| B_val | 5,025 | 1,495 | 1,907 | 2,632 | 11,059 |
| test_all (merged test) | 5,028 | 3,000 | 1,908 | 2,634 | 12,570 |
| single test sets | 5,029 | 3,001 | 1,908 | 2,636 | - |

Merged corpus before splitting: 110,451 comments. Regime O uses the own-dataset rows of `B_train` (20,990) and `B_val` (2,632).

Cleaning (`bhs.text.clean_text`): remove URLs, remove every character outside the Bangla block, word characters and
`. , ! ? ।`, collapse whitespace. BanHate and the own dataset are de-duplicated within the corpus and split 80/10/10
(stratified, seed 42); BD-SHS and BanglaHateBERT keep their official splits. Training rows whose text occurs in any
validation or test set are removed (no leakage), see `audit_dropped_rows.csv`.
