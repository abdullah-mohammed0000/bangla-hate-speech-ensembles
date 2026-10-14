# ============================================================================
# STEP M17  --  STUDY 3: trained on the OWN dataset, tested on the OWN dataset   [PORTABLE, CPU only, no GPU]
#   Upload together (select several files in the dialog):
#       m9_merged.zip              (step M9: only the own-test labels are used, plus Single/Combine references)
#       m16_*_O_s42/43/44.zip      (step M16: 9 files - BB-norm, XLM-R, BB-large trained on the own dataset only)
#       [optional] m14_banglabert_large_norm_A/B_s*.zip   (step M14: for the E6 reference rows)
#   The two systems of Study 3:
#       BanglaBERT + normalizer
#       E6 = equal-weight soft voting of BanglaBERT-large + normalizer, BanglaBERT + normalizer and XLM-RoBERTa
#            (same seed) - exactly as in Study 1
#   Reports on the own-dataset test set (2,636 comments): accuracy, macro-F1, weighted F1, hate P/R/F1
#   (mean ± std over seeds 42/43/44), E6 vs BanglaBERT + normalizer (McNemar per seed, paired bootstrap 95% CI,
#   2,000 resamples, seed 2024, the usual 3-way verdict), and - for reference only - the same two systems
#   when trained on BD-SHS only (Single) or on the merged data (Combine).
#   Output: m17_study3.zip. Runtime ~1 min.
# ============================================================================
import os, glob, shutil, zipfile, warnings
import numpy as np, pandas as pd
from scipy.stats import chi2
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
warnings.filterwarnings('ignore')
SEEDS = [42, 43, 44]; T = 'test_own'
ROOT = '/content/m17' if os.path.exists('/content') else './m17'
shutil.rmtree(ROOT, ignore_errors=True); os.makedirs(f'{ROOT}/p'); os.makedirs(f'{ROOT}/tmp')
try:
    from google.colab import files
    print('Upload m9_merged.zip + the nine m16_*.zip (+ the six m14_*.zip if you have them), all at once')
    up = files.upload(); paths = []
    for n, c in up.items(): q = f'{ROOT}/tmp/{n}'; open(q, 'wb').write(c); paths.append(q)
except ImportError:
    paths = glob.glob('./inputs/*.zip')
for q in paths:
    if not q.lower().endswith('.zip'): continue
    d = f'{ROOT}/tmp/x_{os.path.basename(q)[:-4]}'; os.makedirs(d, exist_ok=True)
    with zipfile.ZipFile(q) as z: z.extractall(d)
    for f in glob.glob(f'{d}/**/*.npy', recursive=True):
        b = os.path.basename(f)
        if b.startswith(('probs_', 'labels_')): shutil.copy(f, f'{ROOT}/p/{b}')
y = np.load(f'{ROOT}/p/labels_{T}.npy'); print(f'own-dataset test set: {len(y)} comments, {int(y.sum())} hate')
def has(m, r): return all(os.path.exists(f'{ROOT}/p/probs_{m}_{r}_s{s}_{T}.npy') for s in SEEDS)
def P(m, r, s): return np.load(f'{ROOT}/p/probs_{m}_{r}_s{s}_{T}.npy')
for m in ['banglabert_norm', 'xlmr', 'banglabert_large_norm']:
    assert has(m, 'O'), f'm16 files for {m} missing (need seeds 42, 43, 44)'
SYS = {'BanglaBERT + normalizer': ['banglabert_norm'],
       'E6: BB-large-norm + BB-norm + XLM-R': ['banglabert_large_norm', 'banglabert_norm', 'xlmr']}
def avail(k, r): return all(has(m, r) for m in SYS[k])
def pred(k, r, s): return (sum(P(m, r, s) for m in SYS[k]) / len(SYS[k])).argmax(1)

# ---------- 1. Study 3 results ----------
rows = []
for k in SYS:
    M = np.array([[accuracy_score(y, p), f1_score(y, p, average='macro'), f1_score(y, p, average='weighted'),
                   precision_score(y, p), recall_score(y, p), f1_score(y, p)] for p in (pred(k, 'O', s) for s in SEEDS)])
    mu, sd = M.mean(0), M.std(0, ddof=1)
    rows.append(dict(System=k, **{n: f'{a:.4f} ± {b:.4f}' for n, a, b in zip(['Accuracy', 'Macro_F1', 'Weighted_F1', 'Hate_P', 'Hate_R', 'Hate_F1'], mu, sd)}))
res = pd.DataFrame(rows)
per_seed = pd.DataFrame([dict(System=k, Seed=s, Accuracy=round(accuracy_score(y, pred(k, 'O', s)), 4), Macro_F1=round(f1_score(y, pred(k, 'O', s), average='macro'), 4))
                         for k in SYS for s in SEEDS])

# ---------- 2. E6 vs BanglaBERT + normalizer (own-only models, own test) ----------
def mcnemar(ca, cb):
    b = int((ca & ~cb).sum()); c = int((~ca & cb).sum())
    return 1.0 if b + c == 0 else float(chi2.sf((abs(b - c) - 1) ** 2 / (b + c), 1))
def macro_f1(p):
    out = []
    for c in (0, 1):
        tp = ((p == c) & (y == c)).sum(); fp = ((p == c) & (y != c)).sum(); fn = ((p != c) & (y == c)).sum(); out.append(2 * tp / max(2 * tp + fp + fn, 1))
    return float(np.mean(out))
rng = np.random.default_rng(2024); W = rng.multinomial(len(y), np.full(len(y), 1 / len(y)), size=2000).astype(np.float32); y1 = (y == 1).astype(np.float32)
def fboot(p):
    p1 = (p == 1).astype(np.float32)
    tp = W @ (y1 * p1); fp = W @ ((1 - y1) * p1); fn = W @ (y1 * (1 - p1)); tn = W @ ((1 - y1) * (1 - p1))
    return (2 * tp / np.maximum(2 * tp + fp + fn, 1e-9) + 2 * tn / np.maximum(2 * tn + fn + fp, 1e-9)) / 2
a, b = 'E6: BB-large-norm + BB-norm + XLM-R', 'BanglaBERT + normalizer'
pv = [mcnemar(pred(a, 'O', s) == y, pred(b, 'O', s) == y) for s in SEEDS]       # one comparison -> Holm not needed
d = np.mean([fboot(pred(a, 'O', s)) for s in SEEDS], 0) - np.mean([fboot(pred(b, 'O', s)) for s in SEEDS], 0)
lo, hi = np.percentile(d, [2.5, 97.5]); sig = sum(p < 0.05 for p in pv); ex = lo > 0 or hi < 0
diff = np.mean([macro_f1(pred(a, 'O', s)) for s in SEEDS]) - np.mean([macro_f1(pred(b, 'O', s)) for s in SEEDS])
cmp_ = pd.DataFrame([dict(Comparison='E6 - BanglaBERT + normalizer', Diff_macroF1=round(diff, 4), CI_low=round(lo, 4), CI_high=round(hi, 4),
                          McNemar_p_per_seed=', '.join(f'{p:.3g}' for p in pv), seeds_sig_of_3=sig,
                          Verdict='significant' if sig == 3 and ex else 'not significant' if sig == 0 and not ex else 'mixed')])

# ---------- 3. reference: same systems trained on BD-SHS only (A) or merged data (B) ----------
ref = []
for k in SYS:
    r = dict(System=k)
    for reg, lab in (('O', 'Own-only (Study 3)'), ('A', 'Single: BD-SHS only'), ('B', 'Combine: merged')):
        if avail(k, reg):
            f = [macro_f1(pred(k, reg, s)) for s in SEEDS]; r[lab] = f'{np.mean(f):.4f} ± {np.std(f, ddof=1):.4f}'
        else: r[lab] = 'n/a (run M14 first)'
    ref.append(r)
ref = pd.DataFrame(ref)

pd.set_option('display.width', 250); pd.set_option('display.max_columns', 20)
print('\n========== STUDY 3: trained on own dataset, tested on own dataset (2,636) - mean ± std over 3 seeds ==========')
print(res.to_string(index=False))
print('\n---------- per seed ----------'); print(per_seed.to_string(index=False))
print('\n========== E6 against BanglaBERT + normalizer (Study 3) ==========')
print(cmp_.to_string(index=False))
print('\n========== Reference: macro-F1 on the own test set by training data ==========')
print(ref.to_string(index=False))
out = '/content/m17_study3.zip' if os.path.exists('/content') else './m17_study3.zip'
for n_, df_ in (('study3_results', res), ('study3_per_seed', per_seed), ('study3_e6_vs_bbnorm', cmp_), ('study3_reference', ref)): df_.to_csv(f'{ROOT}/{n_}.csv', index=False)
with zipfile.ZipFile(out, 'w') as z:
    for f in glob.glob(f'{ROOT}/*.csv'): z.write(f, os.path.basename(f))
print('[saved]', out)
try:
    from google.colab import files; files.download(out)
except Exception: pass
