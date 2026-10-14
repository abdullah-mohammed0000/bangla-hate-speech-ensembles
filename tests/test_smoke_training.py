"""Mechanics test of the training code with a tiny random BERT and synthetic text (CPU, ~10 s, no downloads).
It checks that training, best-epoch restore, probability files and the BiLSTM run end-to-end; it does NOT test accuracy."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("torch")
pytest.importorskip("transformers")
pytest.importorskip("datasets")
pytest.importorskip("accelerate")
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from bhs import config as C  # noqa: E402
from bhs.classical import run_bilstm  # noqa: E402
from bhs.store import Store  # noqa: E402
from bhs.transformer import run_transformer  # noqa: E402


def _frame(n, seed):
    rng = np.random.RandomState(seed)
    words = ["আমি", "তুমি", "ভালো", "খারাপ", "দেশ", "মানুষ", "কথা", "সুন্দর", "বাজে", "খবর"]
    lab = rng.randint(0, 2, n)
    text = [" ".join(rng.choice(words, 6)) + (" বাজে" if l else " ভালো") for l in lab]
    return pd.DataFrame({"source": "x", "orig_idx": np.arange(n), "text": text, "label": lab})


def test_training_and_store(tmp_path):
    from tokenizers import BertWordPieceTokenizer
    from transformers import BertConfig, BertForSequenceClassification, BertTokenizerFast

    D = {k: _frame(120, i) for i, k in enumerate(["A_train", "A_val", "B_train", "B_val"] + C.TEST_SETS)}
    corpus = tmp_path / "c.txt"
    corpus.write_text("\n".join(D["A_train"].text))
    wp = BertWordPieceTokenizer(lowercase=False)
    wp.train([str(corpus)], vocab_size=200, min_frequency=1)
    mdir = tmp_path / "m"
    mdir.mkdir()
    wp.save_model(str(mdir))
    tok = BertTokenizerFast(str(mdir / "vocab.txt"), do_lower_case=False)
    tok.save_pretrained(str(mdir))
    BertForSequenceClassification(BertConfig(vocab_size=tok.vocab_size, hidden_size=32, num_hidden_layers=2,
                                             num_attention_heads=2, intermediate_size=64, num_labels=2)).save_pretrained(str(mdir))
    out = tmp_path / "out"
    rows = run_transformer("tiny", "A", 42, D, out, hp=dict(epochs=2), model_cfg=dict(hf=str(mdir), weighted=True), use_fp16=False)
    assert set(rows.Test) == set(C.TEST_SETS)
    st = Store(out)
    assert st.probs("tiny", "A", 42, "test_own").shape == (120, 2)
    assert np.allclose(st.probs("tiny", "A", 42, "test_own").sum(1), 1, atol=1e-5)
    run_bilstm("A", 42, D, out, epochs=1)
    assert st.has("bilstm", "A", 42, "test_all")
