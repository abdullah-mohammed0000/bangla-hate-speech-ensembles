"""The data preparation must reproduce the thesis split files exactly (needs the raw corpora in data/raw)."""
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from bhs import config as C  # noqa: E402
from bhs.data import prepare_splits  # noqa: E402

RAW = ROOT / "data" / "raw"


@pytest.mark.skipif(not RAW.exists() or not any(RAW.iterdir()), reason="raw corpora not present in data/raw")
def test_prepare_splits_is_bit_identical(tmp_path):
    hashes = prepare_splits(RAW, tmp_path)
    assert hashes == {k: C.SPLIT_SHA16[k] for k in hashes}
