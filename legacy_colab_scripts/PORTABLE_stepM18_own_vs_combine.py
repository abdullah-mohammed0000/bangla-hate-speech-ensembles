# ============================================================================
# STEP M18  --  Is own-only training (Study 3) significantly better than Combine on the own test set?   [PORTABLE, CPU only]
#   Upload together (select several files in the dialog):
#       m9_merged.zip                               (step M9: Combine probs of BB-norm and XLM-R + test labels)
#       m14_banglabert_large_norm_B_s42/43/44.zip   (step M14: Combine probs of BB-large)
#       m16_*_O_s42/43/44.zip                       (step M16: nine own-only runs)
#   For BanglaBERT + normalizer and E6, on the own-dataset test set (2,636 comments):
#     difference Own-only (O) - Combine (B) in macro-F1, McNemar per seed (Holm over the two systems),
#     paired percentile bootstrap 95% CI (2,000 resamples, seed 2024), the usual 3-way verdict.
#   Output: m18_own_vs_combine.zip. Runtime ~1 min.
# ============================================================================
import os, glob, shutil, zipfile, warnings
import numpy as np, pandas as pd
from scipy.stats import chi2
warnings.filterwarnings('ignore')
SEEDS = [42, 43, 44]; T = 'test_own'
ROOT = '/content/m18' if os.path.exists('/content') else './m18'
shutil.rmtree(ROOT, ignore_errors=True); os.makedirs(f'{ROOT}/p'); os.makedirs(f'{ROOT}/tmp')
try:
    from google.colab import files
    print('Upload m9_merged.zip + the three m14_..._B_s*.zip + the nine m16_*.zip, all at once')
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
def P(m, r, s):
    f = f'{ROOT}/p/probs_{m}_{r}_s{s}_{T}.npy'
    assert os.path.exists(f), f'missing {os.path.basename(f)} - check the uploads'
    return np.load(f)
SYS = {'BanglaBERT + normalizer': ['banglabert_norm'],
       'E6: BB-large-norm + BB-norm + XLM-R': ['banglabert_large_norm', 'banglabert_norm', 'xlmr']}
def pred(k, r, s): return (sum(P(m, r, s) for m in SYS[k]) / len(SYS[k])).argmax(1)
def macro_f1(p):
    out = []
    for c in (0, 1):
        tp = ((p == c) & (y == c)).sum(); fp = ((p == c) & (y != c)).sum(); fn = ((p != c) & (y == c)).sum(); out.append(2 * tp / max(2 * tp + fp + fn, 1))
    return float(np.mean(out))
def mcnemar(ca, cb):
    b = int((ca & ~cb).sum()); c = int((~ca & cb).sum())
    return 1.0 if b + c == 0 else float(chi2.sf((abs(b - c) - 1) ** 2 / (b + c), 1))
def holm(p):
    p = np.asarray(p); o = np.argsort(p); m = len(p); adj = np.empty(m); run = 0
    for k, i in enumerate(o): run = max(run, (m - k) * p[i]); adj[i] = min(1.0, run)
    return adj
rng = np.random.default_rng(2024); W = rng.multinomial(len(y), np.full(len(y), 1 / len(y)), size=2000).astype(np.float32); y1 = (y == 1).astype(np.float32)
def fboot(p):
    p1 = (p == 1).astype(np.float32)
    tp = W @ (y1 * p1); fp = W @ ((1 - y1) * p1); fn = W @ (y1 * (1 - p1)); tn = W @ ((1 - y1) * (1 - p1))
    return (2 * tp / np.maximum(2 * tp + fp + fn, 1e-9) + 2 * tn / np.maximum(2 * tn + fn + fp, 1e-9)) / 2
K = list(SYS)
pv = {s: [mcnemar(pred(k, 'O', s) == y, pred(k, 'B', s) == y) for k in K] for s in SEEDS}
adj = {s: holm(pv[s]) for s in SEEDS}
rows = []
for i, k in enumerate(K):
    fO = [macro_f1(pred(k, 'O', s)) for s in SEEDS]; fB = [macro_f1(pred(k, 'B', s)) for s in SEEDS]
    d = np.mean([fboot(pred(k, 'O', s)) for s in SEEDS], 0) - np.mean([fboot(pred(k, 'B', s)) for s in SEEDS], 0)
    lo, hi = np.percentile(d, [2.5, 97.5]); sig = sum(adj[s][i] < 0.05 for s in SEEDS); ex = lo > 0 or hi < 0
    rows.append(dict(System=k, Own_only=round(np.mean(fO), 4), Combine=round(np.mean(fB), 4), Diff_O_minus_B=round(np.mean(fO) - np.mean(fB), 4),
                     CI_low=round(lo, 4), CI_high=round(hi, 4), Holm_p_per_seed=', '.join(f'{adj[s][i]:.3g}' for s in SEEDS),
                     seeds_sig_of_3=sig, Verdict='significant' if sig == 3 and ex else 'not significant' if sig == 0 and not ex else 'mixed'))
df = pd.DataFrame(rows)
pd.set_option('display.width', 250); pd.set_option('display.max_columns', 20)
print('\n========== Own-only (Study 3) against Combine, own-dataset test set ==========')
print(df.to_string(index=False))
out = '/content/m18_own_vs_combine.zip' if os.path.exists('/content') else './m18_own_vs_combine.zip'
df.to_csv(f'{ROOT}/m18_own_vs_combine.csv', index=False)
with zipfile.ZipFile(out, 'w') as z: z.write(f'{ROOT}/m18_own_vs_combine.csv', 'm18_own_vs_combine.csv')
print('[saved]', out)
try:
    from google.colab import files; files.download(out)
except Exception: pass
