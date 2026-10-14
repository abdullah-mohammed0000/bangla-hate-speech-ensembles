"""Unit tests of the statistics code and a reproduction test against the released predictions."""
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from bhs import config as C  # noqa: E402
from bhs.metrics import macro_f1  # noqa: E402
from bhs.stats import Bootstrap, holm, mcnemar, verdict  # noqa: E402
from bhs.text import clean_text  # noqa: E402


def test_holm_matches_manual():
    p = [0.01, 0.04, 0.03]
    assert np.allclose(holm(p), [0.03, 0.06, 0.06])


def test_mcnemar_symmetric_and_null():
    a = np.array([1, 1, 0, 0, 1, 0], bool)
    assert mcnemar(a, a) == 1.0
    b = np.array([1, 0, 0, 1, 1, 0], bool)
    assert mcnemar(a, b) == pytest.approx(mcnemar(b, a))


def test_verdict_rule():
    assert verdict(3, True) == "significant"
    assert verdict(0, False) == "not significant"
    assert verdict(2, True) == "mixed" and verdict(3, False) == "mixed"


def test_macro_f1_equals_sklearn():
    from sklearn.metrics import f1_score
    rng = np.random.default_rng(0)
    y, p = rng.integers(0, 2, 500), rng.integers(0, 2, 500)
    assert macro_f1(y, p) == pytest.approx(f1_score(y, p, average="macro"))


def test_bootstrap_point_estimate_close():
    rng = np.random.default_rng(1)
    y = rng.integers(0, 2, 800)
    p = np.where(rng.random(800) < 0.9, y, 1 - y)
    b = Bootstrap(y)
    assert abs(b.macro_f1(p).mean() - macro_f1(y, p)) < 0.01


def test_clean_text():
    assert clean_text("আমি http://x.com  ভালো!!! 😀") == "আমি ভালো!!!"


def test_config_ensembles():
    assert C.ENSEMBLES["E6"] == ["banglabert_large_norm", "banglabert_norm", "xlmr"]


@pytest.mark.skipif(not (ROOT / "results/predictions/labels_test_own.npy").exists(), reason="predictions not present")
def test_release_reproduces_thesis_numbers():
    from bhs.store import Store
    st = Store(ROOT / "results/predictions")
    y = st.labels("test_own")
    f = np.mean([macro_f1(y, st.pred("E6", "O", s, "test_own")) for s in C.SEEDS])
    assert f == pytest.approx(0.9575, abs=1.1e-4)
