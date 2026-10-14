# ============================================================================
# STEP X2  --  LARGE MODEL: ensembles + statistics vs BanglaBERT + normalizer   [PORTABLE, CPU only, no GPU]
#   Upload together (select several files in the dialog):
#       archive__3_.zip                     (BD-SHS; only test.csv is used, for the true labels)
#       consolidated_results.zip            (from Step 8: probs of all 13 Study-1 models)
#       job_banglabert_large_norm_s*.zip    (from step X1; one or several zips, all seeds)
#       [optional] job_xlmr_large_s*.zip
#   What it does (rules fixed BEFORE looking at results):
#     * metrics of the large model(s): accuracy, macro-F1, weighted-F1, hate P/R, mean +- std over 3 seeds
#     * two pre-declared soft-voting ensembles (equal weights, same rule as E1-E4):
#         E5 = BanglaBERT-large(norm) + XLM-R           (E2 with the large model instead of the base)
#         E6 = BanglaBERT-large(norm) + BanglaBERT(norm) + XLM-R
#     * every new model vs BanglaBERT + normalizer AND vs E2:
#         McNemar (continuity corrected) per seed, Holm over all comparisons of that seed,
#         paired percentile bootstrap 95% CI of the macro-F1 difference (2,000 resamples, seed 2024),
#         verdict: significant = Holm-McNemar significant in 3/3 seeds AND CI excludes 0;
#                  not significant = 0/3 and CI contains 0; otherwise mixed.
#   Output: x2_large.zip (csv tables). Runs in ~1-2 min.
# ============================================================================
import os, re, glob, shutil, zipfile, hashlib, warnings
import numpy as np, pandas as pd
from scipy.stats import chi2
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
warnings.filterwarnings('ignore')
SEEDS = [42, 43, 44]; TEST_HASH, TEST_ROWS = '4543a005555ee466', 5029
ROOT = '/content/x2' if os.path.exists('/content') else './x2'
ALL, TMP = f'{ROOT}/all', f'{ROOT}/tmp'
shutil.rmtree(ROOT, ignore_errors=True); os.makedirs(ALL); os.makedirs(TMP)

# ---------- 1. upload ----------
try:
    from google.colab import files
    print('Upload archive__3_.zip + consolidated_results.zip + job_banglabert_large_norm_s*.zip (all at once)')
    up = files.upload(); paths = []
    for n, c in up.items(): p = f'{TMP}/{n}'; open(p, 'wb').write(c); paths.append(p)
except ImportError:
    paths = glob.glob('./inputs/*.zip')
test_csv = None
for p in paths:
    if not p.lower().endswith('.zip'): continue
    d = f'{TMP}/x_{os.path.basename(p)[:-4]}'; os.makedirs(d, exist_ok=True)
    with zipfile.ZipFile(p) as z: z.extractall(d)
    hit = glob.glob(f'{d}/**/test.csv', recursive=True)
    if hit: test_csv = hit[0]
    for f in glob.glob(f'{d}/**/probs_*_s*.npy', recursive=True): shutil.copy(f, f'{ALL}/{os.path.basename(f)}')
assert test_csv, 'archive__3_.zip (with test.csv) missing'
assert hashlib.sha256(open(test_csv, 'rb').read()).hexdigest()[:16] == TEST_HASH, 'test.csv differs - STOP'
y = pd.read_csv(test_csv)['hate speech'].astype(int).values; assert len(y) == TEST_ROWS
print('test.csv OK:', len(y), 'comments')

def P(name, s):
    f = f'{ALL}/probs_{name}_s{s}.npy'
    return np.load(f) if os.path.exists(f) else None
need = ['banglabert_norm', 'xlmr', 'banglabert_large_norm']
for n in need:
    miss = [s for s in SEEDS if P(n, s) is None]
    assert not miss, f'missing probs for {n}, seeds {miss} - upload the right zips'
# E2 exactly as in Study 1 (equal-weight soft voting)
probs = {}
for s in SEEDS:
    probs[('E2', s)] = (P('banglabert_norm', s) + P('xlmr', s)) / 2
    probs[('banglabert_norm', s)] = P('banglabert_norm', s)
    probs[('banglabert_large_norm', s)] = P('banglabert_large_norm', s)
    probs[('E5', s)] = (P('banglabert_large_norm', s) + P('xlmr', s)) / 2
    probs[('E6', s)] = (P('banglabert_large_norm', s) + P('banglabert_norm', s) + P('xlmr', s)) / 3
NEW = ['banglabert_large_norm', 'E5', 'E6']
if all(P('xlmr_large', s) is not None for s in SEEDS):
    for s in SEEDS: probs[('xlmr_large', s)] = P('xlmr_large', s)
    NEW.append('xlmr_large')
LABEL = {'banglabert_norm': 'BanglaBERT + normalizer (base)', 'E2': 'E2: BanglaBERT(norm) + XLM-R',
         'banglabert_large_norm': 'BanglaBERT-large + normalizer', 'xlmr_large': 'XLM-RoBERTa-large',
         'E5': 'E5: BanglaBERT-large(norm) + XLM-R', 'E6': 'E6: BanglaBERT-large(norm) + BanglaBERT(norm) + XLM-R'}
pred = {k: v.argmax(1) for k, v in probs.items()}

# ---------- 2. metrics ----------
rows = []
for m in ['banglabert_norm', 'E2'] + NEW:
    ms = []
    for s in SEEDS:
        p = pred[(m, s)]
        ms.append([accuracy_score(y, p), f1_score(y, p, average='macro'), f1_score(y, p, average='weighted'),
                   precision_score(y, p), recall_score(y, p)])
    ms = np.array(ms); mu, sd = ms.mean(0), ms.std(0, ddof=1)
    rows.append(dict(Model=LABEL[m], **{f'{k}': f'{a:.4f} ± {b:.4f}' for k, a, b in zip(['Accuracy', 'Macro_F1', 'Weighted_F1', 'Hate_P', 'Hate_R'], mu, sd)},
                     _acc=mu[0], _wf1=mu[2]))
tab = pd.DataFrame(rows)

# ---------- 3. statistics: each new model vs BB+norm and vs E2 ----------
def mcnemar(ca, cb):
    b = int((ca & ~cb).sum()); c = int((~ca & cb).sum())
    return 1.0 if b + c == 0 else float(chi2.sf((abs(b - c) - 1) ** 2 / (b + c), 1))
def holm(p):
    p = np.asarray(p); o = np.argsort(p); m = len(p); adj = np.empty(m); run = 0
    for r, i in enumerate(o): run = max(run, (m - r) * p[i]); adj[i] = min(1.0, run)
    return adj
R = 2000; rng = np.random.default_rng(2024)
W = rng.multinomial(len(y), np.full(len(y), 1 / len(y)), size=R).astype(np.float32)
def f1_boot(p):
    y1 = (y == 1).astype(np.float32); p1 = (p == 1).astype(np.float32)
    tp = W @ (y1 * p1); fp = W @ ((1 - y1) * p1); fn = W @ (y1 * (1 - p1)); tn = W @ ((1 - y1) * (1 - p1))
    return (2 * tp / np.maximum(2 * tp + fp + fn, 1e-9) + 2 * tn / np.maximum(2 * tn + fn + fp, 1e-9)) / 2
pairs = [(a, b) for a in NEW for b in ['banglabert_norm', 'E2']]
pv = {s: [mcnemar(pred[(a, s)] == y, pred[(b, s)] == y) for a, b in pairs] for s in SEEDS}
adj = {s: holm(pv[s]) for s in SEEDS}
st = []
for i, (a, b) in enumerate(pairs):
    fa = np.mean([f1_score(y, pred[(a, s)], average='macro') for s in SEEDS]); fb = np.mean([f1_score(y, pred[(b, s)], average='macro') for s in SEEDS])
    d = np.mean([f1_boot(pred[(a, s)]) for s in SEEDS], 0) - np.mean([f1_boot(pred[(b, s)]) for s in SEEDS], 0)
    lo, hi = np.percentile(d, [2.5, 97.5]); sig = sum(adj[s][i] < 0.05 for s in SEEDS); ex = lo > 0 or hi < 0
    st.append(dict(New=LABEL[a], Against=LABEL[b], Diff_macroF1=round(fa - fb, 4), CI_low=round(lo, 4), CI_high=round(hi, 4),
                   seeds_significant_of_3=sig, Verdict='significant' if sig == 3 and ex else 'not significant' if sig == 0 and not ex else 'mixed'))
st = pd.DataFrame(st)

# ---------- 4. print + save ----------
pd.set_option('display.width', 250); pd.set_option('display.max_columns', 20)
print('\n========== BD-SHS TEST (5,029) - mean ± std over seeds 42/43/44 ==========')
print(tab.drop(columns=['_acc', '_wf1']).to_string(index=False))
print('\n========== NEW MODELS vs BanglaBERT + normalizer and vs E2 ==========')
print(st.to_string(index=False))
print('\n========== PUBLISHED RESULTS ON BD-SHS (binary, whole dataset; different runs/splits, for orientation only) ==========')
print('Romim et al. 2022 (LREC)        BiLSTM + IFT     weighted F1 0.910  (70/15/15 split, accuracy not reported)')
print('Tarin et al. (preprint)         stacking ML      accuracy/F1 0.9149 (80:20 split, original class balance)')
print('Islam & Ameen 2025 (IEEE COMPAS) Llama-3.2-3B QLoRA accuracy 0.9248, weighted F1 0.9223 (80/10/10, 5,029 test, one run)')
for _, r in tab.iterrows():
    print(f"  ours: {r.Model:55s} accuracy {r._acc:.4f}  weighted F1 {r._wf1:.4f}")
out = '/content/x2_large.zip' if os.path.exists('/content') else './x2_large.zip'
tab.drop(columns=['_acc', '_wf1']).to_csv(f'{ROOT}/x2_metrics.csv', index=False); st.to_csv(f'{ROOT}/x2_stats.csv', index=False)
with zipfile.ZipFile(out, 'w') as z:
    for f in glob.glob(f'{ROOT}/*.csv'): z.write(f, os.path.basename(f))
print('[saved]', out)
try:
    from google.colab import files; files.download(out)
except Exception: pass
