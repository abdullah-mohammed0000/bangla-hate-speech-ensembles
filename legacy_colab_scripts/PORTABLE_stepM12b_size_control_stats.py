# ============================================================================
# Study 2 -- STEP M12b: STATISTICS for the size-matched control (mBERT)      [PORTABLE, CPU only, NO GPU, ~1 min]
#   Upload together:  m9_merged.zip  (step M9)   and   m3_mbert_C.zip  (step M12)
#   Compares, for mBERT, on the SAME test set (BD-SHS test, then merged test), the three trainings:
#     A = Single (BD-SHS, 40,224)   C = BD-SHS resampled to 86,822 (size-matched control)   B = Combine (merged, 86,822)
#   Three comparisons: C vs A (does size alone change anything?), B vs C (does the merged data change anything beyond size?), B vs A (the Single-vs-Combine change).
#   Same rules as M10: McNemar (continuity corrected) per seed, Holm over the 3 comparisons, paired bootstrap 95% CI (2,000 resamples, seed 2024) of the macro-F1
#   difference (mean of 3 seeds). "significant" = Holm-McNemar significant in 3/3 seeds AND CI excludes 0.
# ============================================================================
import os, glob, zipfile, shutil, json
import numpy as np, pandas as pd
from scipy.stats import chi2
ROOT = '/content/m12b'; shutil.rmtree(ROOT, ignore_errors=True); os.makedirs(ROOT)
from google.colab import files
print('Upload m9_merged.zip and m3_mbert_C.zip (select both together)')
for name, content in files.upload().items():
    p = f'{ROOT}/{os.path.basename(name).replace(" ", "_")}'; open(p, 'wb').write(content)
    with zipfile.ZipFile(p) as z: z.extractall(f'{ROOT}/x_{os.path.basename(p)[:-4]}')
def find(pat):
    h = glob.glob(f'{ROOT}/**/{pat}', recursive=True); assert h, f'{pat} not found - upload the right zip(s)'; return h[0]
SEEDS = [42, 43, 44]; TESTS = {'test_bdshs': 'BD-SHS test', 'test_all': 'Merged (combined) test'}
labels = {k: np.load(find(f'labels_{k}.npy')) for k in TESTS}
def load(reg, s, t):
    pat = f'probs_mbert_{reg}_s{s}_{t}.npy'; return np.load(find(pat)).argmax(1)
pred = {(r, s, t): load(r, s, t) for r in 'ACB' for s in SEEDS for t in TESTS}
for t in TESTS: assert all(len(pred[(r, s, t)]) == len(labels[t]) for r in 'ACB' for s in SEEDS), 'length mismatch'
def mcnemar(ca, cb):
    b = int((ca & ~cb).sum()); c = int((~ca & cb).sum())
    if b + c == 0: return 1.0
    return float(chi2.sf((abs(b - c) - 1) ** 2 / (b + c), 1))
def holm(p):
    p = np.asarray(p); o = np.argsort(p); m = len(p); adj = np.empty(m); run = 0
    for rank, i in enumerate(o): run = max(run, (m - rank) * p[i]); adj[i] = min(1.0, run)
    return adj
R = 2000; rng = np.random.default_rng(2024)
def mf_boot(W, y, p):
    y1 = (y == 1).astype(np.float32); p1 = (p == 1).astype(np.float32)
    tp1 = W @ (y1 * p1); fp1 = W @ ((1 - y1) * p1); fn1 = W @ (y1 * (1 - p1)); tn1 = W @ ((1 - y1) * (1 - p1))
    return (2 * tp1 / np.maximum(2 * tp1 + fp1 + fn1, 1e-9) + 2 * tn1 / np.maximum(2 * tn1 + fn1 + fp1, 1e-9)) / 2
def mf(y, p):
    o = []
    for c in (0, 1):
        tp = ((p == c) & (y == c)).sum(); fp = ((p == c) & (y != c)).sum(); fn = ((p != c) & (y == c)).sum(); o.append(2 * tp / max(2 * tp + fp + fn, 1))
    return float(np.mean(o))
COMP = [('C', 'A', 'C (size-matched) vs A (Single): effect of size alone'), ('B', 'C', 'B (Combine) vs C: effect of the merged data beyond size'), ('B', 'A', 'B (Combine) vs A (Single): total Single-vs-Combine change')]
rows = []
for t, tn in TESTS.items():
    y = labels[t]; W = rng.multinomial(len(y), np.full(len(y), 1.0 / len(y)), size=R).astype(np.float32)
    boot = {r: np.mean([mf_boot(W, y, pred[(r, s, t)]) for s in SEEDS], axis=0) for r in 'ACB'}
    f = {r: np.mean([mf(y, pred[(r, s, t)]) for s in SEEDS]) for r in 'ACB'}
    pv = {s: [mcnemar(pred[(x, s, t)] == y, pred[(z, s, t)] == y) for x, z, _ in COMP] for s in SEEDS}
    adj = {s: holm(pv[s]) for s in SEEDS}
    for i, (x, z, name) in enumerate(COMP):
        d = boot[x] - boot[z]; lo, hi = np.percentile(d, [2.5, 97.5]); sig = sum(adj[s][i] < 0.05 for s in SEEDS); ex = bool(lo > 0 or hi < 0)
        rows.append(dict(Test=tn, Comparison=name, F1_first=round(f[x], 4), F1_second=round(f[z], 4), Diff=round(f[x] - f[z], 4), CI_low=round(lo, 4), CI_high=round(hi, 4),
                         seeds_significant_of_3=sig, Verdict='significant' if (sig == 3 and ex) else ('not significant' if sig == 0 and not ex else 'mixed')))
T = pd.DataFrame(rows); T.to_csv('/content/m12b_size_control_stats.csv', index=False)
pd.set_option('display.width', 250, 'display.max_colwidth', 70)
print('\n===== mBERT: macro-F1 of the three trainings (mean of 3 seeds) =====')
for t, tn in TESTS.items():
    y = labels[t]; print(f'{tn:24s} A (Single) {np.mean([mf(y, pred[("A", s, t)]) for s in SEEDS]):.4f} | C (size-matched) {np.mean([mf(y, pred[("C", s, t)]) for s in SEEDS]):.4f} | B (Combine) {np.mean([mf(y, pred[("B", s, t)]) for s in SEEDS]):.4f}')
print('\n===== statistics (Diff = first minus second) =====')
print(T.to_string(index=False))
files.download('/content/m12b_size_control_stats.csv')
