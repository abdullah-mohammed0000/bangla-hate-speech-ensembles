# ============================================================================
# Study 2 (A = BD-SHS only, B = merged)  --  CLASSICAL MODELS JOB  [PORTABLE]
# TF-IDF word SVM, TF-IDF char SVM, BiLSTM.  Any Colab account, no Drive. Needs only merged_splits.zip (from M1).
#   GPU is NOT required (BiLSTM uses a GPU if present, else CPU - slower but works).
#   Paste the WHOLE file in ONE cell, run, upload merged_splits.zip. A zip  m2_classical.zip  downloads at the end.
# Same settings as Study 1; the SVM penalty C is re-chosen on the validation set of each regime.
# ============================================================================
JOB_MODELS = ['tfidf_word_svm', 'tfidf_char_svm', 'bilstm']
REGIMES    = ['B', 'A']
SEEDS      = [42, 43, 44]
QUICK      = False          # True = tiny smoke test only (do not use for real results)

import os, re, json, random, hashlib, time, glob, zipfile, warnings, shutil, gc, platform
import numpy as np, pandas as pd, torch, torch.nn as nn
warnings.filterwarnings('ignore')
dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu'); print('BiLSTM device:', dev)
OUT = '/content/job_results_m2' if os.path.exists('/content') else './job_results_m2'; os.makedirs(OUT, exist_ok=True)
MAX_LEN = 128
EXPECTED = {'A_train': 'e2e765bc7028607e', 'A_val': 'be38f2bf02e901a3', 'B_train': '5c3d0534bc182c8b', 'B_val': '49a1c67ca0e7fcec',
            'test_all': '5c26e952999ca26c', 'test_bdshs': 'a49a667d635cc6bf', 'test_banglahatebert': '360305cc8e870e4d',
            'test_banhate': 'da22d1b1ca76af7d', 'test_own': '873067fbbc95698d'}
ROWS = {'A_train': 40224, 'A_val': 5028, 'B_train': 86822, 'B_val': 11059, 'test_all': 12570,
        'test_bdshs': 5029, 'test_banglahatebert': 3001, 'test_banhate': 1908, 'test_own': 2636}
PARTS = ['test_bdshs', 'test_banglahatebert', 'test_banhate', 'test_own', 'test_all']
def set_seed(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed_all(s)

# ---------------- data ----------------
DATA = '/content/splits' if os.path.exists('/content') else './splits'; os.makedirs(DATA, exist_ok=True)
def have_all(): return all(os.path.exists(f'{DATA}/{k}.csv') for k in EXPECTED)
if not have_all():
    from google.colab import files
    print('Upload merged_splits.zip')
    for name, content in files.upload().items():
        p = f'/content/_up_{name}'; open(p, 'wb').write(content)
        with zipfile.ZipFile(p) as z: z.extractall('/content/_up_zip')
    for k in EXPECTED:
        hits = glob.glob(f'/content/_up_zip/**/{k}.csv', recursive=True)
        if hits: shutil.copy(hits[0], f'{DATA}/{k}.csv')
assert have_all(), 'some split CSV not found'
def sha16(p): return hashlib.sha256(open(p, 'rb').read()).hexdigest()[:16]
for k, h in EXPECTED.items():
    raw = pd.read_csv(f'{DATA}/{k}.csv'); ok = (sha16(f'{DATA}/{k}.csv') == h) and (len(raw) == ROWS[k])
    print(f'{k:20s} rows={len(raw):6d} sha256[:16]={sha16(f"{DATA}/{k}.csv")} ->', 'OK' if ok else '*** MISMATCH ***'); assert ok, 'STOP: split differs from M1'
def clean_text(text):
    text = str(text); text = re.sub(r'http\S+|www\S+', '', text)
    text = re.sub(r'[^ঀ-৿\s\w.,!?।]', '', text); return re.sub(r'\s+', ' ', text).strip()
def load(k):
    d = pd.read_csv(f'{DATA}/{k}.csv'); d['text'] = d['text'].fillna('').map(clean_text); d['label'] = d['label'].astype(int)
    return d[['source', 'orig_idx', 'text', 'label']].reset_index(drop=True)
D = {k: load(k) for k in EXPECTED}
if QUICK: D = {k: v.sample(min(len(v), 800), random_state=0).reset_index(drop=True) for k, v in D.items()}

from sklearn.metrics import accuracy_score, precision_recall_fscore_support, f1_score, precision_score, recall_score
def metrics(y, p):
    mp, mr, mf, _ = precision_recall_fscore_support(y, p, average='macro', zero_division=0)
    return dict(Accuracy=accuracy_score(y, p), Macro_P=mp, Macro_R=mr, Macro_F1=mf, Weighted_F1=f1_score(y, p, average='weighted'),
                Hate_P=precision_score(y, p, pos_label=1, zero_division=0), Hate_R=recall_score(y, p, pos_label=1, zero_division=0),
                Hate_F1=f1_score(y, p, pos_label=1, zero_division=0))
RUN_LOG = f'{OUT}/results_m2_classical.csv'; INFO = {}
def have(name, reg, seed): return os.path.exists(f'{OUT}/probs_{name}_{reg}_s{seed}_test_all.npy')
def save_and_log(name, reg, seed, probs):          # probs: dict part -> (n,2); 'val' included
    np.save(f'{OUT}/valprobs_{name}_{reg}_s{seed}.npy', probs['val']); rows = []
    for part in PARTS:
        np.save(f'{OUT}/probs_{name}_{reg}_s{seed}_{part}.npy', probs[part])
        m = metrics(D[part].label.values, probs[part].argmax(1)); m.update(Model=name, Regime=reg, Seed=seed, Test=part); rows.append(m)
        print(f'   {name:15s} {reg} s{seed} {part:20s} macro-F1 {m["Macro_F1"]:.4f}')
    r = pd.DataFrame(rows); r = r[['Model', 'Regime', 'Seed', 'Test'] + [c for c in r if c not in ('Model', 'Regime', 'Seed', 'Test')]]
    r.to_csv(RUN_LOG, mode='a', header=not os.path.exists(RUN_LOG), index=False)

# ---------------- SVMs (deterministic: fitted once per regime, the same probabilities are stored for each seed) ----------------
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
VECS = {'tfidf_word_svm': lambda: TfidfVectorizer(token_pattern=r'\S+', ngram_range=(1, 2), max_features=50000, sublinear_tf=True),
        'tfidf_char_svm': lambda: TfidfVectorizer(analyzer='char_wb', ngram_range=(2, 5), max_features=100000, sublinear_tf=True)}
def run_svm(name, reg):
    t0 = time.time(); tr, va = D[f'{reg}_train'], D[f'{reg}_val']
    vec = VECS[name](); Xtr = vec.fit_transform(tr.text); Xva = vec.transform(va.text)
    best_c, best_f = None, -1
    for C in [0.1, 0.3, 1, 3]:                               # C chosen on THIS regime's validation set
        f = f1_score(va.label, LinearSVC(C=C, random_state=0).fit(Xtr, tr.label).predict(Xva), average='macro')
        print(f'   {name} {reg} C={C}: val macro-F1 {f:.4f}')
        if f > best_f: best_c, best_f = C, f
    clf = CalibratedClassifierCV(LinearSVC(C=best_c, random_state=0), cv=3, method='sigmoid').fit(Xtr, tr.label)
    probs = {'val': clf.predict_proba(Xva)}
    for part in PARTS: probs[part] = clf.predict_proba(vec.transform(D[part].text))
    INFO[f'{name}_{reg}'] = dict(C=best_c, val_macro_f1=round(best_f, 4), features=len(vec.vocabulary_), minutes=round((time.time() - t0) / 60, 1))
    print(f'   {name} {reg}: chosen C={best_c} | features {len(vec.vocabulary_)} | {(time.time()-t0)/60:.1f} min')
    for s in SEEDS: save_and_log(name, reg, s, probs)

# ---------------- BiLSTM ----------------
from collections import Counter
from torch.utils.data import DataLoader, TensorDataset
class BiLSTM(nn.Module):
    def __init__(s, V):
        super().__init__(); s.e = nn.Embedding(V + 2, 128, padding_idx=0); s.l = nn.LSTM(128, 128, batch_first=True, bidirectional=True)
        s.d = nn.Dropout(0.4); s.f = nn.Linear(256, 2)
    def forward(s, x):
        o, _ = s.l(s.e(x)); m = (x != 0).unsqueeze(-1).float()
        return s.f(s.d((o * m).sum(1) / m.sum(1).clamp(min=1)))          # mean pooling over real tokens
def run_bilstm(reg, seed):
    t0 = time.time(); set_seed(seed); tr, va = D[f'{reg}_train'], D[f'{reg}_val']
    cnt = Counter(w for t in tr.text for w in t.split()); vocab = {w: i + 2 for i, (w, c) in enumerate(cnt.most_common(50000)) if c >= 2}
    def tens(d): return torch.tensor([([vocab.get(w, 1) for w in t.split()][:MAX_LEN] + [0] * MAX_LEN)[:MAX_LEN] for t in d.text])
    Xtr, ytr, Xva = tens(tr), torch.tensor(tr.label.values), tens(va)
    cw = torch.tensor(len(tr) / (2 * np.bincount(tr.label.values)), dtype=torch.float).to(dev)       # weighted CE, weights from THIS train set
    m = BiLSTM(len(vocab)).to(dev); opt = torch.optim.Adam(m.parameters(), 1e-3); best, best_state, best_ep = -1, None, 0
    def predict(X):
        m.eval()
        with torch.no_grad(): return torch.cat([torch.softmax(m(xb.to(dev)), 1).cpu() for xb in DataLoader(X, batch_size=256)]).numpy()
    for ep in range(8):
        m.train()
        for xb, yb in DataLoader(TensorDataset(Xtr, ytr), batch_size=64, shuffle=True):
            opt.zero_grad(); nn.functional.cross_entropy(m(xb.to(dev)), yb.to(dev), weight=cw).backward(); opt.step()
        f = f1_score(va.label.values, predict(Xva).argmax(1), average='macro')
        if f > best: best, best_state, best_ep = f, {k: v.clone() for k, v in m.state_dict().items()}, ep + 1
    m.load_state_dict(best_state); probs = {'val': predict(Xva)}
    assert abs(f1_score(va.label.values, probs['val'].argmax(1), average='macro') - best) < 1e-4, 'best-epoch restore mismatch'
    for part in PARTS: probs[part] = predict(tens(D[part]))
    INFO[f'bilstm_{reg}_s{seed}'] = dict(vocab=len(vocab), best_epoch=best_ep, val_macro_f1=round(best, 4), minutes=round((time.time() - t0) / 60, 1))
    print(f'   bilstm {reg} s{seed}: vocab {len(vocab)} | best epoch {best_ep} | val F1 {best:.4f} | {(time.time()-t0)/60:.1f} min')
    save_and_log('bilstm', reg, seed, probs)

def package():
    json.dump(dict(models=JOB_MODELS, regimes=REGIMES, seeds=SEEDS, device=str(dev), torch=torch.__version__, python=platform.python_version(),
                   date=time.strftime('%Y-%m-%d %H:%M:%S'), per_run=INFO, data_hashes={k: sha16(f'{DATA}/{k}.csv') for k in EXPECTED}),
              open(f'{OUT}/job_info_m2.json', 'w'), indent=2)
    zp = '/content/m2_classical.zip' if os.path.exists('/content') else './m2_classical.zip'
    with zipfile.ZipFile(zp, 'w', zipfile.ZIP_DEFLATED) as z:
        for f in glob.glob(f'{OUT}/*'): z.write(f, os.path.basename(f))
    print(f'[packaged] {zp}')
    try:
        from google.colab import files; files.download(zp)
    except Exception: pass

for reg in REGIMES:
    for name in JOB_MODELS:
        if name == 'bilstm':
            for s in SEEDS:
                if have(name, reg, s): print('skip', name, reg, s); continue
                print(f'\n=== {name} | {reg} | seed {s} ==='); run_bilstm(reg, s)
        else:
            if all(have(name, reg, s) for s in SEEDS): print('skip', name, reg); continue
            print(f'\n=== {name} | {reg} ==='); run_svm(name, reg)
        package()
rr = pd.read_csv(RUN_LOG); print('\nJOB DONE.')
print(rr.groupby(['Model', 'Regime', 'Test'])['Macro_F1'].agg(['mean', 'std']).round(4))
