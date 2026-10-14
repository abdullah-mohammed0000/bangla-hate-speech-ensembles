"""Data preparation and loading.

``prepare_splits`` builds the four-corpus merged data from the raw files (it reproduces the thesis files
bit-for-bit: the SHA-256 of every output is checked against ``config.SPLIT_SHA16``).
``load_splits`` loads the prepared CSVs, verifies their hashes and returns the regimes A, B, C and O.
"""
from __future__ import annotations

import hashlib
import json
import platform
import re
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from . import config as C
from .text import clean_text

COLS = ["source", "orig_idx", "text", "label"]


def sha16(path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()[:16]


def _find(root: Path, pattern: str, ok=None) -> Path:
    hits = sorted(p for p in root.rglob("*") if p.is_file() and re.search(pattern, p.name, re.I) and (ok is None or ok(p)))
    if not hits:
        raise FileNotFoundError(f"file not found under {root}: {pattern}")
    return hits[0]


# ---------------------------------------------------------------------------------------------
def prepare_splits(raw_dir, out_dir, verify=True) -> dict:
    """Build ``A_*``, ``B_*`` and all test files from the raw corpora.

    ``raw_dir`` may contain the corpora as extracted folders or as .zip files (searched recursively):
      * BD-SHS:          train.csv, val.csv, test.csv
      * BanglaHateBERT:  train_data.xlsx, valid_data.xlsx, test_data.xlsx
      * BanHate:         Hate_Comment_Label.xlsx, Non_Hate_Comment_Label.xlsx
      * own dataset:     Combined_Hate_Comments_Final_v5.xlsx, combine_nonhate_dataset_8.xlsx
    """
    import zipfile

    raw_dir, out_dir = Path(raw_dir), Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for z in list(raw_dir.rglob("*.zip")):  # unpack zips next to themselves (nested zips included)
        target = z.with_suffix("")
        if not target.exists():
            with zipfile.ZipFile(z) as f:
                f.extractall(target)
    for z in list(raw_dir.rglob("*.zip")):
        target = z.with_suffix("")
        for inner in list(target.rglob("*.zip")):
            t2 = inner.with_suffix("")
            if not t2.exists():
                with zipfile.ZipFile(inner) as f:
                    f.extractall(t2)

    raw_files, audit = {}, []

    def reg(src, p):
        raw_files[f"{src}: {p.name}"] = dict(sha256_16=sha16(p))

    # 1. BD-SHS (as distributed; hash-checked)
    bd = {}
    for k, (h, n) in C.BDSHS_SHA16.items():
        p = _find(raw_dir, rf"^{k}\.csv$", lambda p: "sentence" in open(p, encoding="utf-8-sig").readline())
        d = pd.read_csv(p)
        if verify and not (sha16(p) == h and len(d) == n):
            raise ValueError(f"BD-SHS {k}: hash/row mismatch ({sha16(p)}, {len(d)} rows)")
        bd[k] = pd.DataFrame({"raw": d["sentence"].astype(str), "label": d["hate speech"].astype(int)})
        reg("BD-SHS", p)

    # 2. BanglaHateBERT: official train / valid / test files
    bhb = {}
    for k, fn in [("train", "train_data"), ("val", "valid_data"), ("test", "test_data")]:
        p = _find(raw_dir, rf"^{fn}\.xlsx$")
        d = pd.read_excel(p)
        bhb[k] = pd.DataFrame({"raw": d["post"].astype(str), "label": d["label"].astype(int)})
        reg("BanglaHateBERT", p)

    # 3. BanHate: label column H/NH; H -> 1, NH -> 0, the single sub-category code "per" -> 1
    pH, pN = _find(raw_dir, r"^Hate_Comment_Label\.xlsx$"), _find(raw_dir, r"^Non_Hate_Comment_Label\.xlsx$")
    bh_raw = pd.concat([pd.read_excel(pH), pd.read_excel(pN)], ignore_index=True)
    reg("BanHate", pH), reg("BanHate", pN)
    bh_map = {"H": 1, "NH": 0, "per": 1}
    assert set(bh_raw["H/NH"].unique()) <= set(bh_map), bh_raw["H/NH"].unique()
    bh_all = pd.DataFrame({"raw": bh_raw["Comment"].astype(str), "label": bh_raw["H/NH"].map(bh_map).astype(int)})

    # 4. own dataset: hate file (label 1) + non-hate file (label 0)
    pOH = _find(raw_dir, r"^Combined_Hate_Comments_Final_v5\.xlsx$")
    pON = _find(raw_dir, r"^combine_nonhate_dataset_8\.xlsx$")
    oh = pd.read_excel(pOH, sheet_name="Hate Comments")
    on = pd.read_excel(pON, sheet_name="BD-SHS")
    reg("Own", pOH), reg("Own", pON)
    assert set(oh.hate.unique()) == {1} and set(on.hate.unique()) == {0}
    own_all = pd.DataFrame({"raw": pd.concat([oh.text, on.text]).astype(str).values,
                            "label": np.r_[np.ones(len(oh), int), np.zeros(len(on), int)]})

    # clean + within-corpus de-duplication (BanHate and own dataset only)
    for d in list(bd.values()) + list(bhb.values()) + [bh_all, own_all]:
        d["text"] = d["raw"].map(clean_text)

    def dedup_within(d, src):
        n0 = len(d)
        e = (d.text == "").sum()
        d = d[d.text != ""].copy()
        conf = d.groupby("text").label.nunique()
        bad = set(conf[conf > 1].index)
        nconf = d.text.isin(bad).sum()
        d = d[~d.text.isin(bad)]
        nd = d.text.duplicated().sum()
        d = d.drop_duplicates("text")
        audit.append(dict(source=src, step="within-corpus clean-up before the 80/10/10 split", empty_after_cleaning=int(e),
                          rows_of_texts_with_conflicting_labels=int(nconf), exact_duplicates_collapsed=int(nd),
                          rows_in=n0, rows_out=len(d)))
        return d.reset_index(drop=True)

    def split_801010(d):
        tr, rest = train_test_split(d, test_size=0.2, stratify=d.label, random_state=C.SPLIT_SEED)
        va, te = train_test_split(rest, test_size=0.5, stratify=rest.label, random_state=C.SPLIT_SEED)
        return dict(train=tr.reset_index(drop=True), val=va.reset_index(drop=True), test=te.reset_index(drop=True))

    bh = split_801010(dedup_within(bh_all, "BanHate"))
    own = split_801010(dedup_within(own_all, "Own"))
    parts = {"BD-SHS": bd, "BanglaHateBERT": bhb, "BanHate": bh, "Own": own}  # priority order for duplicates
    order = list(parts)
    for s, p in parts.items():
        for k in p:
            p[k] = p[k].assign(source=s, orig_idx=np.arange(len(p[k])))

    # leakage rules: test and validation sets stay fixed
    test_texts = set().union(*[set(parts[s]["test"].text) for s in order])
    val_texts = set().union(*[set(parts[s]["val"].text) for s in order])

    b_val = pd.concat([parts[s]["val"] for s in order], ignore_index=True)
    n = len(b_val)
    b_val = b_val[~b_val.text.isin(test_texts)]
    audit.append(dict(source="all", step="merged validation: rows whose text occurs in a test set removed", rows_in=n, rows_out=len(b_val)))
    n = len(b_val)
    b_val = b_val[b_val.text != ""].drop_duplicates("text", keep="first").reset_index(drop=True)
    audit.append(dict(source="all", step="merged validation: empty and duplicate texts removed (first source kept)", rows_in=n, rows_out=len(b_val)))

    b_train = pd.concat([parts[s]["train"] for s in order], ignore_index=True)
    n = len(b_train)
    leak = b_train.text.isin(test_texts | set(parts["BD-SHS"]["val"].text) | set(b_val.text) | val_texts)
    leak_by_src = b_train[leak].source.value_counts().to_dict()
    b_train = b_train[~leak]
    audit.append(dict(source="all", step="merged train: rows whose text occurs in any validation/test set removed",
                      rows_in=n, rows_out=len(b_train), by_source=json.dumps(leak_by_src)))
    n = len(b_train)
    e = (b_train.text == "").sum()
    b_train = b_train[b_train.text != ""]
    conf = b_train.groupby("text").label.nunique()
    bad = set(conf[conf > 1].index)
    nconf = b_train.text.isin(bad).sum()
    b_train = b_train[~b_train.text.isin(bad)]
    nd = b_train.text.duplicated().sum()
    b_train = b_train.drop_duplicates("text", keep="first").reset_index(drop=True)
    audit.append(dict(source="all", step="merged train: empty / conflicting-label / duplicate texts removed (first source kept)",
                      empty_after_cleaning=int(e), rows_of_texts_with_conflicting_labels=int(nconf),
                      exact_duplicates_collapsed=int(nd), rows_in=n, rows_out=len(b_train)))

    tests = {s: parts[s]["test"].reset_index(drop=True) for s in order}
    t_all = pd.concat(tests.values(), ignore_index=True)
    n = len(t_all)
    rep_by_src = t_all[t_all.text.duplicated(keep="first")].source.value_counts().to_dict()
    t_all = t_all.drop_duplicates("text", keep="first").reset_index(drop=True)
    audit.append(dict(source="all", step="combined test: repeated texts across/inside test parts dropped (first source kept)",
                      rows_in=n, rows_out=len(t_all), by_source=json.dumps(rep_by_src)))

    a_train, a_val = parts["BD-SHS"]["train"].reset_index(drop=True), parts["BD-SHS"]["val"].reset_index(drop=True)

    tr_set, va_set = set(b_train.text), set(b_val.text)
    for s, t in tests.items():
        assert not (set(t.text) & tr_set), f"test/train leakage in {s}"
    assert not (va_set & tr_set) and not (set(t_all.text) & va_set), "validation leakage"
    assert (tests["BD-SHS"].raw.values == bd["test"].raw.values).all(), "BD-SHS test set changed"

    files_out = {"A_train": a_train, "A_val": a_val, "B_train": b_train, "B_val": b_val, "test_all": t_all}
    for s, t in tests.items():
        files_out[f"test_{s.lower().replace('-', '')}"] = t
    for name, d in files_out.items():
        d[COLS].to_csv(out_dir / f"{name}.csv", index=False, encoding="utf-8")

    comp = []
    for name, d in files_out.items():
        for s in order + ["ALL"]:
            x = d if s == "ALL" else d[d.source == s]
            if len(x):
                comp.append(dict(file=f"{name}.csv", source=s, rows=len(x), hate=int(x.label.sum()),
                                 non_hate=int((x.label == 0).sum()), hate_pct=round(100 * x.label.mean(), 2)))
    pd.DataFrame(comp).to_csv(out_dir / "composition.csv", index=False)
    pd.DataFrame(audit).to_csv(out_dir / "audit_dropped_rows.csv", index=False)
    hashes = {n_: sha16(out_dir / f"{n_}.csv") for n_ in files_out}
    json.dump(dict(seed=C.SPLIT_SEED, python=platform.python_version(), pandas=pd.__version__, numpy=np.__version__,
                   raw_inputs=raw_files, outputs=hashes), open(out_dir / "splits_info.json", "w"), indent=1, ensure_ascii=False)
    if verify:
        bad = {k: (hashes[k], C.SPLIT_SHA16[k]) for k in C.SPLIT_SHA16 if hashes.get(k) != C.SPLIT_SHA16[k]}
        if bad:
            print("WARNING: split hashes differ from the thesis files (different input files or library versions):", bad)
        else:
            print("OK: all split files are identical to the files used in the thesis.")
    return hashes


# ---------------------------------------------------------------------------------------------
def load_splits(split_dir, verify=True) -> dict:
    """Load the split CSVs, verify hash and row count, and add the regimes O and C.

    Returns a dict with keys A_train, A_val, B_train, B_val, O_train, O_val, the five test sets and
    (via :func:`regime_C_train`) the resampled control set.
    """
    split_dir = Path(split_dir)
    D = {}
    for k in C.SPLIT_FILES:
        p = split_dir / f"{k}.csv"
        if not p.exists():
            raise FileNotFoundError(f"{p} not found - run scripts/prepare_data.py (or download the splits)")
        raw = pd.read_csv(p)
        if verify:
            ok = sha16(p) == C.SPLIT_SHA16[k] and len(raw) == C.SPLIT_ROWS[k]
            if not ok:
                raise ValueError(f"{k}: hash/row count differs from the thesis file - STOP (got {sha16(p)}, {len(raw)} rows)")
        raw["text"] = raw["text"].fillna("").map(clean_text)  # idempotent
        raw["label"] = raw["label"].astype(int)
        D[k] = raw[COLS].reset_index(drop=True)
    D["O_train"] = D["B_train"][D["B_train"].source == "Own"].reset_index(drop=True)
    D["O_val"] = D["B_val"][D["B_val"].source == "Own"].reset_index(drop=True)
    assert len(D["O_train"]) == 20990 and len(D["O_val"]) == 2632, "unexpected own-dataset sizes"
    return D


def regime_C_train(D: dict, seed: int) -> pd.DataFrame:
    """Size-matched control: BD-SHS training set resampled with replacement to the size of B_train."""
    return D["A_train"].sample(n=len(D["B_train"]), replace=True, random_state=seed).reset_index(drop=True)


def get_regime(D: dict, regime: str, seed: int):
    """(train, val) for a regime letter."""
    if regime == "C":
        return regime_C_train(D, seed), D["A_val"]
    return D[f"{regime}_train"], D[f"{regime}_val"]


def load_bdshs(raw_dir) -> dict:
    """Study 1 only: BD-SHS as distributed (hash-checked). Returns A_train, A_val, test_bdshs.
    Needs only the public BD-SHS files; the results equal regime A / test_bdshs of the full pipeline."""
    raw_dir = Path(raw_dir)
    for z in list(raw_dir.rglob("*.zip")):
        import zipfile
        t = z.with_suffix("")
        if not t.exists():
            with zipfile.ZipFile(z) as f:
                f.extractall(t)
    D = {}
    for k, (h, n) in C.BDSHS_SHA16.items():
        p = _find(raw_dir, rf"^{k}\.csv$", lambda p: "sentence" in open(p, encoding="utf-8-sig").readline())
        d = pd.read_csv(p)
        if not (sha16(p) == h and len(d) == n):
            raise ValueError(f"BD-SHS {k}: hash/row mismatch")
        df = pd.DataFrame({"source": "BD-SHS", "orig_idx": np.arange(len(d)), "text": d["sentence"].astype(str).map(clean_text),
                           "label": d["hate speech"].astype(int)})
        D[{"train": "A_train", "val": "A_val", "test": "test_bdshs"}[k]] = df[COLS]
    return D
