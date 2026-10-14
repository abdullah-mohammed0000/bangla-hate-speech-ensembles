# ============================================================================
# Study 2  --  STEP M13: RESULTS ON THE OWN DATASET TEST SET   [PORTABLE, CPU only, NO GPU]
#   Input : m9_merged.zip  (made by step M9)   <- the ONLY file you upload.
#   Output: m13_own.zip (csv tables) + tables printed below.
# What it does, for all 13 models (2 SVMs, BiLSTM, 6 transformers, 4 ensembles):
#   * macro-F1 (mean +- std of 3 seeds) on the own-dataset test set (2,636 comments),
#     separately for Single (A, trained on BD-SHS only) and Combine (B, trained on merged data);
#   * change B - A with the same statistics as M10:
#       McNemar (continuity corrected) per seed + Holm over the 13 models,
#       paired percentile bootstrap 95% CI (2,000 resamples, seed 2024),
#       verdict: significant = Holm-McNemar significant in 3/3 seeds AND CI excludes 0;
#   * hate-class precision / recall on the own test set (Single vs Combine).
# Nothing is trained. Runs in about 1-2 minutes.
# ============================================================================
import os, json, glob, zipfile, shutil, warnings
import numpy as np, pandas as pd
from scipy.stats import chi2
warnings.filterwarnings('ignore')
ROOT = '/content/m13' if os.path.exists('/content') else './m13'
shutil.rmtree(ROOT, ignore_errors=True); os.makedirs(ROOT)
SEEDS = [42, 43, 44]; T = 'test_own'
NAMES = {'tfidf_word_svm': 'TF-IDF word + SVM', 'tfidf_char_svm': 'TF-IDF char + SVM', 'bilstm': 'BiLSTM', 'mbert': 'mBERT', 'xlmr': 'XLM-RoBERTa',
         'banglabert': 'BanglaBERT', 'banglabert_unw': 'BanglaBERT (unweighted)', 'banglabert_norm': 'BanglaBERT + normalizer', 'indicbertv2': 'IndicBERTv2',
         'E1': 'Ens E1: BanglaBERT + XLM-R', 'E2': 'Ens E2: BanglaBERT(norm) + XLM-R', 'E3': 'Ens E3: BanglaBERT + XLM-R + mBERT', 'E4': 'Ens E4: BanglaBERT(norm) + XLM-R + mBERT'}
ORDER = list(NAMES)

# ---------------- 1. load m9_merged.zip ----------------
def get_zip():
    try:
        from google.colab import files
        print('Upload m9_merged.zip'); up = files.upload(); name = list(up)[0]; open(f'{ROOT}/in.zip', 'wb').write(up[name]); return f'{ROOT}/in.zip'
    except ImportError:
        return './inputs/m9_merged.zip'
zp = get_zip()
with zipfile.ZipFile(zp) as z: z.extractall(f'{ROOT}/m9')
y = np.load(f'{ROOT}/m9/labels_{T}.npy')
info = json.load(open(f'{ROOT}/m9/m9_info.json')); models = sorted(info['models'], key=ORDER.index)
assert len(models) == 13, 'expected 13 models'
print(f'own test set: {len(y)} comments, {int(y.sum())} hate')
pred = {(m, r, s): np.load(f'{ROOT}/m9/probs/probs_{m}_{r}_s{s}_{T}.npy').argmax(1) for m in models for r in 'AB' for s in SEEDS}

# ---------------- 2. helpers (same as M10) ----------------
def mcnemar(ca, cb):
    b = int((ca & ~cb).sum()); c = int((~ca & cb).sum())
    if b + c == 0: return 1.0
    return float(chi2.sf((abs(b - c) - 1) ** 2 / (b + c), 1))
def holm(p):
    p = np.asarray(p); o = np.argsort(p); m = len(p); adj = np.empty(m); run = 0
    for rank, i in enumerate(o): run = max(run, (m - rank) * p[i]); adj[i] = min(1.0, run)
    return adj
def macro_f1(y, p):
    out = []
    for c in (0, 1):
        tp = ((p == c) & (y == c)).sum(); fp = ((p == c) & (y != c)).sum(); fn = ((p != c) & (y == c)).sum(); out.append(2 * tp / max(2 * tp + fp + fn, 1))
    return float(np.mean(out))
R = 2000; rng = np.random.default_rng(2024)
W = rng.multinomial(len(y), np.full(len(y), 1.0 / len(y)), size=R).astype(np.float32)
def macro_f1_boot(p):
    y1 = (y == 1).astype(np.float32); p1 = (p == 1).astype(np.float32)
    tp = W @ (y1 * p1); fp = W @ ((1 - y1) * p1); fn = W @ (y1 * (1 - p1)); tn = W @ ((1 - y1) * (1 - p1))
    return (2 * tp / np.maximum(2 * tp + fp + fn, 1e-9) + 2 * tn / np.maximum(2 * tn + fn + fp, 1e-9)) / 2

# ---------------- 3. scores + Single vs Combine statistics ----------------
rows, pv = [], {s: [] for s in SEEDS}
for m in models:
    fA = [macro_f1(y, pred[(m, 'A', s)]) for s in SEEDS]; fB = [macro_f1(y, pred[(m, 'B', s)]) for s in SEEDS]
    for s in SEEDS: pv[s].append(mcnemar(pred[(m, 'A', s)] == y, pred[(m, 'B', s)] == y))
    d = np.mean([macro_f1_boot(pred[(m, 'B', s)]) for s in SEEDS], axis=0) - np.mean([macro_f1_boot(pred[(m, 'A', s)]) for s in SEEDS], axis=0)
    lo, hi = np.percentile(d, [2.5, 97.5])
    pr = {}
    for r in 'AB':
        P_, R_ = [], []
        for s in SEEDS:
            p = pred[(m, r, s)]; tp = ((p == 1) & (y == 1)).sum(); fp = ((p == 1) & (y == 0)).sum(); fn = ((p == 0) & (y == 1)).sum()
            P_.append(tp / max(tp + fp, 1)); R_.append(tp / max(tp + fn, 1))
        pr[r] = (np.mean(P_), np.mean(R_))
    rows.append(dict(Model=NAMES[m], Single_A=round(np.mean(fA), 4), Single_std=round(np.std(fA, ddof=1), 4), Combine_B=round(np.mean(fB), 4), Combine_std=round(np.std(fB, ddof=1), 4),
                     Diff_B_minus_A=round(np.mean(fB) - np.mean(fA), 4), CI_low=round(lo, 4), CI_high=round(hi, 4), CI_excludes_0=bool(lo > 0 or hi < 0),
                     HateP_A=round(pr['A'][0], 4), HateR_A=round(pr['A'][1], 4), HateP_B=round(pr['B'][0], 4), HateR_B=round(pr['B'][1], 4)))
adj = {s: holm(pv[s]) for s in SEEDS}
for i, r in enumerate(rows):
    sig = sum(adj[s][i] < 0.05 for s in SEEDS); r['seeds_significant_of_3'] = sig
    r['Verdict'] = 'significant' if sig == 3 and r['CI_excludes_0'] else 'not significant' if sig == 0 and not r['CI_excludes_0'] else 'mixed'
df = pd.DataFrame(rows); df.to_csv(f'{ROOT}/own_test_single_vs_combine.csv', index=False)

# ---------------- 4. save + print ----------------
out = '/content/m13_own.zip' if os.path.exists('/content') else './m13_own.zip'
with zipfile.ZipFile(out, 'w', zipfile.ZIP_DEFLATED) as z:
    for f in glob.glob(f'{ROOT}/*.csv'): z.write(f, os.path.basename(f))
pd.set_option('display.width', 250); pd.set_option('display.max_columns', 30)
print('\n========== TABLE 1: COMBINE (B) on the OWN test set - macro-F1, mean +- std of 3 seeds ==========')
print(df[['Model', 'Combine_B', 'Combine_std']].to_string(index=False))
print('\n========== TABLE 2: SINGLE (A) on the OWN test set - macro-F1, mean +- std of 3 seeds ==========')
print(df[['Model', 'Single_A', 'Single_std']].to_string(index=False))
print('\n========== CHANGE B - A on the OWN test set (Holm-McNemar per seed + bootstrap CI) ==========')
print(df[['Model', 'Single_A', 'Combine_B', 'Diff_B_minus_A', 'CI_low', 'CI_high', 'seeds_significant_of_3', 'Verdict']].to_string(index=False))
print('\n========== HATE-CLASS PRECISION / RECALL on the OWN test set ==========')
print(df[['Model', 'HateP_A', 'HateP_B', 'HateR_A', 'HateR_B']].to_string(index=False))
print(f'\n-> {int((df.Verdict == "significant").sum())} significant, {int((df.Verdict == "mixed").sum())} mixed, {int((df.Verdict == "not significant").sum())} not significant (of 13)')
print(f'[saved] {out}')
try:
    from google.colab import files; files.download(out)
except Exception: pass
