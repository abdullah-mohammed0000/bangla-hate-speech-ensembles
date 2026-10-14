# Legacy Colab scripts

The original, self-contained Colab scripts that produced the thesis results (paste a script into one Colab cell,
upload the input zips when asked). They are kept unchanged as the record of the experiments.

| Step | Purpose |
|---|---|
| `BDSHS_*`, `Study1_Final_Thesis.ipynb`, `PORTABLE_step7*`, `step8`, `step9_11`, `step12` | Study 1 (BD-SHS only), statistics, SHAP |
| `PORTABLE_stepM1_data_prep.py` | build the merged splits |
| `PORTABLE_stepM2*`, `M3*` | Study 2 classical and transformer jobs (regimes A, B) |
| `PORTABLE_stepM9*`, `M10` | ensembles and Single-vs-Combine statistics |
| `PORTABLE_stepM11*`, `M12*` | SHAP for the Combine models, size-matched control |
| `PORTABLE_stepM13`, `M14*`, `M15` | own-dataset test, BanglaBERT-large and E6 in Study 2 |
| `PORTABLE_stepM16*`, `M17`, `M18` | Study 3 (own dataset only) and own-only vs Combine |
| `PORTABLE_stepX1-X4` | large models and best-model selection of Study 1 |
| `PORTABLE_stepK2_kappa_check.py`, `stepE8_versions.py` | annotation agreement check, software versions |

The cleaned-up version of the same procedure is the `bhs` package in `src/` (see `docs/REPRODUCING.md`).
