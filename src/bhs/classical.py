"""Classical baselines: TF-IDF + linear SVM (word and character n-grams) and a BiLSTM."""
from __future__ import annotations

import random
from collections import Counter

import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import f1_score
from sklearn.svm import LinearSVC

from . import config as C
from .data import get_regime
from .metrics import metrics
from .store import save_probs

VECTORIZERS = {
    "tfidf_word_svm": lambda: TfidfVectorizer(token_pattern=r"\S+", ngram_range=(1, 2), max_features=50000, sublinear_tf=True),
    "tfidf_char_svm": lambda: TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5), max_features=100000, sublinear_tf=True),
}
C_GRID = [0.1, 0.3, 1, 3]


def _log_rows(name, regime, seed, D, probs):
    rows = []
    for part in C.TEST_SETS:
        m = metrics(D[part].label.values, probs[part].argmax(1))
        m.update(Model=name, Regime=regime, Seed=seed, Test=part)
        rows.append(m)
    return rows


def run_svm(name: str, regime: str, D: dict, out_dir, seeds=C.SEEDS) -> pd.DataFrame:
    """The SVM is deterministic: it is fitted once per regime and the same probabilities are stored for every seed.
    The penalty C is chosen on the validation set of the regime; probabilities by 3-fold sigmoid calibration."""
    train, val = get_regime(D, regime, seeds[0])
    vec = VECTORIZERS[name]()
    Xtr, Xva = vec.fit_transform(train.text), vec.transform(val.text)
    best_c, best_f = None, -1.0
    for c in C_GRID:
        f = f1_score(val.label, LinearSVC(C=c, random_state=0).fit(Xtr, train.label).predict(Xva), average="macro")
        if f > best_f:
            best_c, best_f = c, f
    clf = CalibratedClassifierCV(LinearSVC(C=best_c, random_state=0), cv=3, method="sigmoid").fit(Xtr, train.label)
    probs = {part: clf.predict_proba(vec.transform(D[part].text)) for part in C.TEST_SETS}
    rows = []
    for s in seeds:
        for part in C.TEST_SETS:
            save_probs(out_dir, name, regime, s, part, probs[part])
        rows += _log_rows(name, regime, s, D, probs)
    print(f"{name} | {regime}: C={best_c}, val macro-F1 {best_f:.4f}")
    return pd.DataFrame(rows)


def run_bilstm(regime: str, seed: int, D: dict, out_dir, epochs: int = 8, device: str | None = None) -> pd.DataFrame:
    """Embedding(128) -> BiLSTM(128) -> mean pooling over real tokens -> dropout 0.4 -> linear; weighted CE; Adam 1e-3."""
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset

    dev = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    train, val = get_regime(D, regime, seed)
    cnt = Counter(w for t in train.text for w in t.split())
    vocab = {w: i + 2 for i, (w, c) in enumerate(cnt.most_common(50000)) if c >= 2}
    L = C.MAX_LEN

    def tens(d):
        return torch.tensor([([vocab.get(w, 1) for w in t.split()][:L] + [0] * L)[:L] for t in d.text])

    class BiLSTM(nn.Module):
        def __init__(s, V):
            super().__init__()
            s.e = nn.Embedding(V + 2, 128, padding_idx=0)
            s.l = nn.LSTM(128, 128, batch_first=True, bidirectional=True)
            s.d = nn.Dropout(0.4)
            s.f = nn.Linear(256, 2)

        def forward(s, x):
            o, _ = s.l(s.e(x))
            m = (x != 0).unsqueeze(-1).float()
            return s.f(s.d((o * m).sum(1) / m.sum(1).clamp(min=1)))

    Xtr, ytr, Xva = tens(train), torch.tensor(train.label.values), tens(val)
    cw = torch.tensor(len(train) / (2 * np.bincount(train.label.values)), dtype=torch.float).to(dev)
    m = BiLSTM(len(vocab)).to(dev)
    opt = torch.optim.Adam(m.parameters(), 1e-3)
    best, best_state, best_ep = -1.0, None, 0

    def predict(X):
        m.eval()
        with torch.no_grad():
            return torch.cat([torch.softmax(m(xb.to(dev)), 1).cpu() for xb in DataLoader(X, batch_size=256)]).numpy()

    for ep in range(epochs):
        m.train()
        for xb, yb in DataLoader(TensorDataset(Xtr, ytr), batch_size=64, shuffle=True):
            opt.zero_grad()
            nn.functional.cross_entropy(m(xb.to(dev)), yb.to(dev), weight=cw).backward()
            opt.step()
        f = f1_score(val.label.values, predict(Xva).argmax(1), average="macro")
        if f > best:
            best, best_state, best_ep = f, {k: v.clone() for k, v in m.state_dict().items()}, ep + 1
    m.load_state_dict(best_state)
    probs = {part: predict(tens(D[part])) for part in C.TEST_SETS}
    for part in C.TEST_SETS:
        save_probs(out_dir, "bilstm", regime, seed, part, probs[part])
    print(f"bilstm | {regime} | seed {seed}: vocab {len(vocab)}, best epoch {best_ep}, val macro-F1 {best:.4f}")
    return pd.DataFrame(_log_rows("bilstm", regime, seed, D, probs))
