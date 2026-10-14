# ============================================================================
# STEP M15  --  Study 2 results for BanglaBERT-large + normalizer and E6   [PORTABLE, CPU only, no GPU]
#   Upload together (select several files in the dialog):
#       m9_merged.zip                                  (from step M9: probs of the 13 models + test labels)
#       m14_banglabert_large_norm_A_s42.zip / _s43 / _s44   (step M14, Single)
#       m14_banglabert_large_norm_B_s42.zip / _s43 / _s44   (step M14, Combine)
#   E6 = equal-weight soft voting of BanglaBERT-large + normalizer, BanglaBERT + normalizer and XLM-RoBERTa
#        (same seed, same regime) - exactly as in Study 1.
#   For each test set (BD-SHS, merged, OWN dataset):
#     * macro-F1 mean +- std over seeds 42/43/44, Single (A) and Combine (B); hate precision / recall
#     * change B - A: McNemar (continuity corrected) per seed, Holm over the two systems,
#       paired percentile bootstrap 95% CI (2,000 resamples, seed 2024); same 3-way verdict rule as M10/M13
#     * BanglaBERT + normalizer and E2 printed next to them for reference
#   Output: m15_results.zip (csv). Runtime ~1-2 min.
# ============================================================================
import os, re, glob, json, shutil, zipfile, warnings
import numpy as np, pandas as pd
from scipy.stats import chi2
warnings.filterwarnings('ignore')
SEEDS = [42, 43, 44]; TESTS = {'test_bdshs': 'BD-SHS test', 'test_all': 'Merged test', 'test_own': 'Own-dataset test'}
ROOT = '/content/m15' if os.path.exists('/content') else './m15'
shutil.rmtree(ROOT, ignore_errors=True); os.makedirs(f'{ROOT}/p'); os.makedirs(f'{ROOT}/tmp')

# ---------- 1. upload ----------
try:
    from google.colab import files
    print('Upload m9_merged.zip + the six m14_banglabert_large_norm_*.zip files (all at once)')
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
Y = {}
for t in TESTS:
    f = f'{ROOT}/p/labels_{t}.npy'; assert os.path.exists(f), f'labels_{t}.npy missing - upload m9_merged.zip'
    Y[t] = np.load(f)
print({TESTS[t]: (len(y), int(y.sum())) for t, y in Y.items()})
def P(m, r, s, t):
    f = f'{ROOT}/p/probs_{m}_{r}_s{s}_{t}.npy'
    assert os.path.exists(f), f'missing {os.path.basename(f)} - check the uploads'
    return np.load(f)
for r in 'AB':
    for s in SEEDS:
        for t in TESTS: assert len(P('banglabert_large_norm', r, s, t)) == len(Y[t])

SYSTEMS = {
    'BanglaBERT + normalizer (reference)': lambda r, s, t: P('banglabert_norm', r, s, t),
    'E2: BB-norm + XLM-R (reference)':     lambda r, s, t: P('E2', r, s, t),
    'BanglaBERT-large + normalizer':       lambda r, s, t: P('banglabert_large_norm', r, s, t),
    'E6: BB-large-norm + BB-norm + XLM-R': lambda r, s, t: (P('banglabert_large_norm', r, s, t) + P('banglabert_norm', r, s, t) + P('xlmr', r, s, t)) / 3,
}
NEW = ['BanglaBERT-large + normalizer', 'E6: BB-large-norm + BB-norm + XLM-R']
pred = {(k, r, s, t): f(r, s, t).argmax(1) for k, f in SYSTEMS.items() for r in 'AB' for s in SEEDS for t in TESTS}

# ---------- 2. helpers (same as M10 / M13) ----------
def macro_f1(y, p):
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
def boot_factory(y):
    rng = np.random.default_rng(2024); W = rng.multinomial(len(y), np.full(len(y), 1 / len(y)), size=2000).astype(np.float32)
    y1 = (y == 1).astype(np.float32)
    def f(p):
        p1 = (p == 1).astype(np.float32)
        tp = W @ (y1 * p1); fp = W @ ((1 - y1) * p1); fn = W @ (y1 * (1 - p1)); tn = W @ ((1 - y1) * (1 - p1))
        return (2 * tp / np.maximum(2 * tp + fp + fn, 1e-9) + 2 * tn / np.maximum(2 * tn + fn + fp, 1e-9)) / 2
    return f
def prec_rec(y, p):
    tp = ((p == 1) & (y == 1)).sum(); fp = ((p == 1) & (y == 0)).sum(); fn = ((p == 0) & (y == 1)).sum()
    return tp / max(tp + fp, 1), tp / max(tp + fn, 1)

# ---------- 3. results ----------
rows = []
for t, tn in TESTS.items():
    y = Y[t]; fb = boot_factory(y)
    pv = {s: [mcnemar(pred[(k, 'A', s, t)] == y, pred[(k, 'B', s, t)] == y) for k in NEW] for s in SEEDS}
    adj = {s: holm(pv[s]) for s in SEEDS}
    for k in SYSTEMS:
        fA = [macro_f1(y, pred[(k, 'A', s, t)]) for s in SEEDS]; fB = [macro_f1(y, pred[(k, 'B', s, t)]) for s in SEEDS]
        prA = np.mean([prec_rec(y, pred[(k, 'A', s, t)]) for s in SEEDS], 0); prB = np.mean([prec_rec(y, pred[(k, 'B', s, t)]) for s in SEEDS], 0)
        row = dict(Test=tn, System=k, Single_A=round(np.mean(fA), 4), Single_std=round(np.std(fA, ddof=1), 4),
                   Combine_B=round(np.mean(fB), 4), Combine_std=round(np.std(fB, ddof=1), 4), Diff_B_minus_A=round(np.mean(fB) - np.mean(fA), 4),
                   HateP_A=round(prA[0], 4), HateP_B=round(prB[0], 4), HateR_A=round(prA[1], 4), HateR_B=round(prB[1], 4))
        if k in NEW:
            i = NEW.index(k)
            d = np.mean([fb(pred[(k, 'B', s, t)]) for s in SEEDS], 0) - np.mean([fb(pred[(k, 'A', s, t)]) for s in SEEDS], 0)
            lo, hi = np.percentile(d, [2.5, 97.5]); sig = sum(adj[s][i] < 0.05 for s in SEEDS); ex = lo > 0 or hi < 0
            row.update(CI_low=round(lo, 4), CI_high=round(hi, 4), seeds_significant_of_3=sig,
                       Verdict='significant' if sig == 3 and ex else 'not significant' if sig == 0 and not ex else 'mixed')
        rows.append(row)
df = pd.DataFrame(rows)

# ---------- 4. print + save ----------
pd.set_option('display.width', 260); pd.set_option('display.max_columns', 30)
print('\n========== TABLE 1: COMBINE (B) - macro-F1, mean ± std over 3 seeds ==========')
print(df.pivot_table(index='System', columns='Test', values='Combine_B', sort=False).round(4).to_string())
print('\n========== TABLE 2: SINGLE (A) - macro-F1, mean ± std over 3 seeds ==========')
print(df.pivot_table(index='System', columns='Test', values='Single_A', sort=False).round(4).to_string())
print('\n========== STD over seeds (Single / Combine) ==========')
print(df[['Test', 'System', 'Single_std', 'Combine_std']].to_string(index=False))
print('\n========== CHANGE B - A for the two new systems (Holm-McNemar per seed + bootstrap CI) ==========')
print(df[df.System.isin(NEW)][['Test', 'System', 'Single_A', 'Combine_B', 'Diff_B_minus_A', 'CI_low', 'CI_high', 'seeds_significant_of_3', 'Verdict']].to_string(index=False))
print('\n========== HATE-CLASS PRECISION / RECALL ==========')
print(df[['Test', 'System', 'HateP_A', 'HateP_B', 'HateR_A', 'HateR_B']].to_string(index=False))
out = '/content/m15_results.zip' if os.path.exists('/content') else './m15_results.zip'
df.to_csv(f'{ROOT}/m15_results.csv', index=False)
with zipfile.ZipFile(out, 'w') as z: z.write(f'{ROOT}/m15_results.csv', 'm15_results.csv')
print('[saved]', out)
try:
    from google.colab import files; files.download(out)
except Exception: pass
