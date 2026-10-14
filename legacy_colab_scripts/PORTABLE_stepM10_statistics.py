# ============================================================================
# Study 2  --  STEP M10: STATISTICS (Single vs Combine)   [PORTABLE, CPU only, NO GPU]
#   Input : m9_merged.zip  (made by step M9)   <- the ONLY file you upload.
#   Output: m10_stats.zip  (all tables as csv) + the main tables printed below.
# Question answered for EVERY model (13 models: SVMs, BiLSTM, 6 transformers, 4 ensembles):
#   when the SAME model is trained on merged data (Combine, B) instead of BD-SHS only (Single, A), is the change on the SAME test set
#   (a) on the BD-SHS test set, (b) on the merged (combined) test set     real, or could it be chance?
# Tests (same rules as Study 1):
#   * McNemar test with continuity correction on per-comment correctness (A vs B), one test per seed, then Holm correction over the 13 models.
#   * Paired percentile-bootstrap 95% CI (2,000 resamples, seed 2024) of the macro-F1 difference B - A (mean of the 3 seeds).
#   * Reporting rule: a change is called "significant" only if Holm-McNemar is significant in 3 of 3 seeds AND the bootstrap CI excludes 0.
#     Otherwise it is called "mixed" (some seeds only / CI includes 0) or "not significant".
# Per-source numbers are NOT part of this step.
# ============================================================================
import os, re, io, json, glob, zipfile, shutil, warnings
import numpy as np, pandas as pd
from scipy.stats import chi2
warnings.filterwarnings('ignore')
ROOT = '/content/m10' if os.path.exists('/content') else './m10'
shutil.rmtree(ROOT, ignore_errors=True); os.makedirs(ROOT)
SEEDS = [42, 43, 44]; TESTS = {'test_bdshs': 'BD-SHS test', 'test_all': 'Merged (combined) test'}
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
labels = {k: np.load(f'{ROOT}/m9/labels_{k}.npy') for k in TESTS}
info = json.load(open(f'{ROOT}/m9/m9_info.json')); models = sorted(info['models'], key=ORDER.index)
print('models:', [NAMES[m] for m in models]); assert len(models) == 13, 'expected 13 models (9 + 4 ensembles) - re-run M9 with all zips'
def P(m, r, s, part): return np.load(f'{ROOT}/m9/probs/probs_{m}_{r}_s{s}_{part}.npy')
pred = {(m, r, s, t): P(m, r, s, t).argmax(1) for m in models for r in 'AB' for s in SEEDS for t in TESTS}

# ---------------- 2. McNemar + Holm ----------------
def mcnemar(ca, cb):                         # ca, cb: boolean "correct" vectors of the two models on the same comments
    b = int((ca & ~cb).sum()); c = int((~ca & cb).sum())
    if b + c == 0: return 0.0, 1.0, b, c
    stat = (abs(b - c) - 1) ** 2 / (b + c); return stat, float(chi2.sf(stat, 1)), b, c
def holm(p):
    p = np.asarray(p); o = np.argsort(p); m = len(p); adj = np.empty(m); run = 0
    for rank, i in enumerate(o): run = max(run, (m - rank) * p[i]); adj[i] = min(1.0, run)
    return adj

# ---------------- 3. paired bootstrap of macro-F1 ----------------
R = 2000; rng = np.random.default_rng(2024)
def boot_weights(n): return rng.multinomial(n, np.full(n, 1.0 / n), size=R).astype(np.float32)   # R x n resample counts (paired across models)
def macro_f1_boot(W, y, p):                  # macro-F1 for every resample, p = predictions (0/1)
    y1 = (y == 1).astype(np.float32); p1 = (p == 1).astype(np.float32)
    tp1 = W @ (y1 * p1); fp1 = W @ ((1 - y1) * p1); fn1 = W @ (y1 * (1 - p1)); tn1 = W @ ((1 - y1) * (1 - p1))
    f_hate = 2 * tp1 / np.maximum(2 * tp1 + fp1 + fn1, 1e-9); f_non = 2 * tn1 / np.maximum(2 * tn1 + fn1 + fp1, 1e-9)
    return (f_hate + f_non) / 2
def macro_f1(y, p):
    out = []
    for c in (0, 1):
        tp = ((p == c) & (y == c)).sum(); fp = ((p == c) & (y != c)).sum(); fn = ((p != c) & (y == c)).sum(); out.append(2 * tp / max(2 * tp + fp + fn, 1))
    return float(np.mean(out))

rows = []
for t, tname in TESTS.items():
    y = labels[t]; W = boot_weights(len(y)); pvals = {s: [] for s in SEEDS}; tmp = []
    for m in models:
        accs = {}
        for s in SEEDS:
            ca = pred[(m, 'A', s, t)] == y; cb = pred[(m, 'B', s, t)] == y; stat, p, b, c = mcnemar(ca, cb); pvals[s].append(p); accs[s] = (b, c)
        fa = np.mean([macro_f1(y, pred[(m, 'A', s, t)]) for s in SEEDS]); fb = np.mean([macro_f1(y, pred[(m, 'B', s, t)]) for s in SEEDS])
        ba = np.mean([macro_f1_boot(W, y, pred[(m, 'A', s, t)]) for s in SEEDS], axis=0); bb = np.mean([macro_f1_boot(W, y, pred[(m, 'B', s, t)]) for s in SEEDS], axis=0)
        d = bb - ba; lo, hi = np.percentile(d, [2.5, 97.5])
        tmp.append(dict(Test=tname, Model=NAMES[m], Single_A=round(fa, 4), Combine_B=round(fb, 4), Diff_B_minus_A=round(fb - fa, 4), CI_low=round(lo, 4), CI_high=round(hi, 4),
                        CI_excludes_0=bool(lo > 0 or hi < 0), accs=accs))
    adj = {s: holm(pvals[s]) for s in SEEDS}
    for i, r in enumerate(tmp):
        sig = sum(adj[s][i] < 0.05 for s in SEEDS); r['Holm_p_s42'], r['Holm_p_s43'], r['Holm_p_s44'] = [float(f'{adj[s][i]:.3g}') for s in SEEDS]
        r['seeds_significant_of_3'] = sig
        r['Verdict'] = ('significant' if sig == 3 and r['CI_excludes_0'] else 'not significant' if sig == 0 and not r['CI_excludes_0'] else 'mixed')
        r['Direction'] = 'Combine LOWER' if r['Diff_B_minus_A'] < 0 else 'Combine HIGHER'
        r.pop('accs'); rows.append(r)
A_vs_B = pd.DataFrame(rows); A_vs_B.to_csv(f'{ROOT}/stat_single_vs_combine.csv', index=False)

# ---------------- 4. hate-class precision / recall shift on the BD-SHS test set ----------------
pr = []
for m in models:
    d = {'Model': NAMES[m]}
    for r, tag in (('A', 'Single'), ('B', 'Combine')):
        pp, rr = [], []
        for s in SEEDS:
            y = labels['test_bdshs']; p = pred[(m, r, s, 'test_bdshs')]
            tp = ((p == 1) & (y == 1)).sum(); fp = ((p == 1) & (y == 0)).sum(); fn = ((p == 0) & (y == 1)).sum()
            pp.append(tp / max(tp + fp, 1)); rr.append(tp / max(tp + fn, 1))
        d[f'Hate precision {tag}'] = round(float(np.mean(pp)), 4); d[f'Hate recall {tag}'] = round(float(np.mean(rr)), 4)
    d['Recall change'] = round(d['Hate recall Combine'] - d['Hate recall Single'], 4); d['Precision change'] = round(d['Hate precision Combine'] - d['Hate precision Single'], 4)
    pr.append(d)
PR = pd.DataFrame(pr); PR.to_csv(f'{ROOT}/hate_precision_recall_shift_bdshs.csv', index=False)

# ---------------- 5. key within-regime comparisons (is one model better than another?) ----------------
KEY = [('E2', 'banglabert_norm', 'ensemble E2 vs its best member BanglaBERT+normalizer'), ('E4', 'E2', 'E4 (adds mBERT) vs E2'),
       ('banglabert_norm', 'banglabert', 'normalizer effect: BanglaBERT+normalizer vs BanglaBERT'), ('banglabert', 'banglabert_unw', 'weighted vs unweighted loss (BanglaBERT)'),
       ('xlmr', 'banglabert', 'XLM-R vs BanglaBERT'), ('tfidf_char_svm', 'xlmr', 'char SVM vs XLM-R'), ('tfidf_char_svm', 'bilstm', 'char SVM vs BiLSTM')]
kr = []
for r, tag in (('A', 'Single (A)'), ('B', 'Combine (B)')):
    for t, tname in TESTS.items():
        y = labels[t]; W = boot_weights(len(y)); tmp = []
        for a, b, desc in KEY:
            ps = [mcnemar(pred[(a, r, s, t)] == y, pred[(b, r, s, t)] == y)[1] for s in SEEDS]
            fa = np.mean([macro_f1(y, pred[(a, r, s, t)]) for s in SEEDS]); fb = np.mean([macro_f1(y, pred[(b, r, s, t)]) for s in SEEDS])
            d = np.mean([macro_f1_boot(W, y, pred[(a, r, s, t)]) for s in SEEDS], axis=0) - np.mean([macro_f1_boot(W, y, pred[(b, r, s, t)]) for s in SEEDS], axis=0)
            lo, hi = np.percentile(d, [2.5, 97.5]); tmp.append(dict(Trained_on=tag, Test=tname, Comparison=desc, F1_first=round(fa, 4), F1_second=round(fb, 4), Diff=round(fa - fb, 4),
                                                                   CI_low=round(lo, 4), CI_high=round(hi, 4), CI_excludes_0=bool(lo > 0 or hi < 0), ps=ps))
        adjs = {s: holm([x['ps'][i] for x in tmp]) for i, s in enumerate(SEEDS)}   # Holm over the 7 comparisons, per seed
        for i, x in enumerate(tmp):
            sig = sum(adjs[s][i] < 0.05 for s in SEEDS); x['seeds_significant_of_3'] = sig
            x['Verdict'] = 'significant' if sig == 3 and x['CI_excludes_0'] else 'not significant' if sig == 0 and not x['CI_excludes_0'] else 'mixed'; x.pop('ps'); kr.append(x)
KR = pd.DataFrame(kr); KR.to_csv(f'{ROOT}/stat_key_comparisons.csv', index=False)

# ---------------- 6. confusion matrices (seed 42) ----------------
cm = []
for m in models:
    for r, tag in (('A', 'Single'), ('B', 'Combine')):
        for t, tname in TESTS.items():
            y = labels[t]; p = pred[(m, r, 42, t)]
            cm.append(dict(Model=NAMES[m], Trained_on=tag, Test=tname, TN=int(((y == 0) & (p == 0)).sum()), FP=int(((y == 0) & (p == 1)).sum()), FN=int(((y == 1) & (p == 0)).sum()), TP=int(((y == 1) & (p == 1)).sum())))
pd.DataFrame(cm).to_csv(f'{ROOT}/confusion_matrices_seed42.csv', index=False)

# ---------------- 7. save + print ----------------
out = '/content/m10_stats.zip' if os.path.exists('/content') else './m10_stats.zip'
with zipfile.ZipFile(out, 'w', zipfile.ZIP_DEFLATED) as z:
    for f in glob.glob(f'{ROOT}/*.csv'): z.write(f, os.path.basename(f))
pd.set_option('display.width', 250); pd.set_option('display.max_columns', 30); pd.set_option('display.max_colwidth', 60)
cols = ['Model', 'Single_A', 'Combine_B', 'Diff_B_minus_A', 'CI_low', 'CI_high', 'seeds_significant_of_3', 'Verdict']
for tname in TESTS.values():
    print(f'\n========== SINGLE (A) vs COMBINE (B) on the {tname.upper()} - same model, same test set; macro-F1, mean of 3 seeds ==========')
    print(A_vs_B[A_vs_B.Test == tname][cols].to_string(index=False))
    v = A_vs_B[A_vs_B.Test == tname]
    print(f'  -> {int((v.Verdict == "significant").sum())} significant, {int((v.Verdict == "mixed").sum())} mixed, {int((v.Verdict == "not significant").sum())} not significant (of 13 models); '
          f'Combine lower in {int((v.Diff_B_minus_A < 0).sum())}, higher in {int((v.Diff_B_minus_A > 0).sum())}')
print('\n========== HATE-CLASS PRECISION / RECALL on the BD-SHS test set (mean of 3 seeds) ==========')
print(PR.to_string(index=False))
print('\n========== KEY COMPARISONS INSIDE one training regime ==========')
print(KR[['Trained_on', 'Test', 'Comparison', 'Diff', 'CI_low', 'CI_high', 'seeds_significant_of_3', 'Verdict']].to_string(index=False))
print(f'\n[saved] {out}')
try:
    from google.colab import files; files.download(out)
except Exception: pass
