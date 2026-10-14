"""Central configuration: models, hyper-parameters, test sets and the SHA-256 fingerprints of the data splits.

Everything that defines an experiment lives here, so that a run is fully described by
(model, regime, seed) plus this file.
"""
from __future__ import annotations

SEEDS = [42, 43, 44]
SPLIT_SEED = 42  # seed of the stratified 80/10/10 split (BanHate and own dataset)

# ---- training regimes ------------------------------------------------------------------------
# A  Single   : BD-SHS only                      (Study 1 training data, also "Single" of Study 2)
# B  Combine  : merged corpus (BD-SHS + BanglaHateBERT + BanHate + own dataset)
# C  Size-matched control: BD-SHS resampled with replacement to the size of B (mBERT only)
# O  Own-only : own dataset only                 (Study 3)
REGIMES = {"A": "Single (BD-SHS only)", "B": "Combine (merged)", "C": "Size-matched control", "O": "Own dataset only"}

# ---- models ----------------------------------------------------------------------------------
MODELS = {
    "mbert": dict(hf="bert-base-multilingual-cased", weighted=True, name="mBERT"),
    "xlmr": dict(hf="xlm-roberta-base", weighted=True, name="XLM-RoBERTa"),
    "banglabert": dict(hf="csebuetnlp/banglabert", weighted=True, name="BanglaBERT"),
    "banglabert_unw": dict(hf="csebuetnlp/banglabert", weighted=False, name="BanglaBERT (unweighted)"),
    "indicbertv2": dict(hf="ai4bharat/IndicBERTv2-MLM-only", weighted=True, name="IndicBERTv2"),
    "banglabert_norm": dict(hf="csebuetnlp/banglabert", weighted=True, norm=True, name="BanglaBERT + normalizer"),
    "banglabert_large_norm": dict(hf="csebuetnlp/banglabert_large", weighted=True, norm=True,
                                  name="BanglaBERT-large + normalizer"),
}
CLASSICAL = {"tfidf_word_svm": "TF-IDF word + SVM", "tfidf_char_svm": "TF-IDF char + SVM", "bilstm": "BiLSTM"}

# ---- ensembles: equal-weight soft voting (mean of class probabilities) of the members -------------
ENSEMBLES = {
    "E1": ["banglabert", "xlmr"],
    "E2": ["banglabert_norm", "xlmr"],
    "E3": ["banglabert", "xlmr", "mbert"],
    "E4": ["banglabert_norm", "xlmr", "mbert"],
    "E6": ["banglabert_large_norm", "banglabert_norm", "xlmr"],
}

# ---- fine-tuning hyper-parameters (identical in all studies) -----------------------------------
HP = dict(max_len=128, epochs=3, lr=2e-5, weight_decay=0.01, batch_train=32, batch_eval=64,
          grad_acc=1, warmup=0.0, fp16=True)
MAX_LEN = HP["max_len"]  # also used by the BiLSTM

# ---- data splits -----------------------------------------------------------------------------
TEST_SETS = ["test_bdshs", "test_banglahatebert", "test_banhate", "test_own", "test_all"]
SPLIT_FILES = ["A_train", "A_val", "B_train", "B_val"] + TEST_SETS
# SHA-256 (first 16 hex chars) and number of rows of the split files used in the thesis
SPLIT_SHA16 = {
    "A_train": "e2e765bc7028607e", "A_val": "be38f2bf02e901a3", "B_train": "5c3d0534bc182c8b",
    "B_val": "49a1c67ca0e7fcec", "test_all": "5c26e952999ca26c", "test_bdshs": "a49a667d635cc6bf",
    "test_banglahatebert": "360305cc8e870e4d", "test_banhate": "da22d1b1ca76af7d", "test_own": "873067fbbc95698d",
}
SPLIT_ROWS = {
    "A_train": 40224, "A_val": 5028, "B_train": 86822, "B_val": 11059, "test_all": 12570,
    "test_bdshs": 5029, "test_banglahatebert": 3001, "test_banhate": 1908, "test_own": 2636,
}
# raw BD-SHS files as distributed by the original authors
BDSHS_SHA16 = {"train": ("116c9e9eb7432dff", 40224), "val": ("67995e80777818a1", 5028),
               "test": ("4543a005555ee466", 5029)}

# ---- statistics ------------------------------------------------------------------------------
BOOT_RESAMPLES = 2000
BOOT_SEED = 2024
ALPHA = 0.05
