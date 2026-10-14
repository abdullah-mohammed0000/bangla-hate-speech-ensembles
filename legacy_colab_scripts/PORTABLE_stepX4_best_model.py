# ============================================================================
# STEP X4  --  BEST BD-SHS MODEL: large models + ensembles chosen on the VALIDATION set   [PORTABLE, CPU only]
#   Upload together (select several files in the dialog):
#       archive__3_.zip                    (BD-SHS: val.csv and test.csv are used for the true labels)
#       consolidated_results.zip           (Step 8: probs + valprobs of the Study-1 models)
#       job_banglabert_large_norm_s42/43/44.zip   (step X1)
#       job_xlmr_large_s42/43/44.zip              (step X3; optional - if missing, XLM-R-large is skipped)
#   RULE: every choice (ensemble members, weights, decision threshold) is made on the VALIDATION set only.
#         The test set is used once, at the end, for reporting. Nothing is chosen by looking at test scores.
#   Systems reported (all with seeds 42/43/44, mean +- std):
#     BB-large           BanglaBERT-large + normalizer, default rule (argmax)
#     XLMR-large         XLM-RoBERTa-large, default rule                              (if uploaded)
#     E5                 BB-large(norm) + XLM-R base, equal weights        (pre-declared)
#     E6                 BB-large(norm) + BB(norm) + XLM-R base, equal     (pre-declared)
#     E7                 BB-large(norm) + XLMR-large, equal weights        (pre-declared, if uploaded)
#     E8 (val-selected)  greedy ensemble selection with replacement (Caruana et al. 2004) over all
#                        transformers, max 10 picks, by validation macro-F1; then the decision threshold
#                        on P(hate) is tuned on validation (0.30-0.70, step 0.01)
#   Statistics vs BanglaBERT + normalizer and vs E2 (Study-1 best): McNemar (continuity corrected) per seed,
#   Holm over all comparisons, paired bootstrap 95% CI (2,000, seed 2024); verdict rule as in the thesis.
#   Output: x4_best.zip. Runtime: ~2-3 min.
# ============================================================================
import os, re, glob, shutil, zipfile, hashlib, warnings
import numpy as np, pandas as pd
from scipy.stats import chi2
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
warnings.filterwarnings('ignore')
SEEDS = [42, 43, 44]
HASH = {'val': ('67995e80777818a1', 5028), 'test': ('4543a005555ee466', 5029)}
ROOT = '/content/x4' if os.path.exists('/content') else './x4'
ALL, TMP = f'{ROOT}/all', f'{ROOT}/tmp'
shutil.rmtree(ROOT, ignore_errors=True); os.makedirs(ALL); os.makedirs(TMP)

# ---------- 1. upload ----------
try:
    from google.colab import files
    print('Upload archive__3_.zip + consolidated_results.zip + job_banglabert_large_norm_s*.zip + job_xlmr_large_s*.zip')
    up = files.upload(); paths = []
    for n, c in up.items(): p = f'{TMP}/{n}'; open(p, 'wb').write(c); paths.append(p)
except ImportError:
    paths = glob.glob('./inputs/*.zip')
csv = {}
for p in paths:
    if not p.lower().endswith('.zip'): continue
    d = f'{TMP}/x_{os.path.basename(p)[:-4]}'; os.makedirs(d, exist_ok=True)
    with zipfile.ZipFile(p) as z: z.extractall(d)
    for k in HASH:
        hit = glob.glob(f'{d}/**/{k}.csv', recursive=True)
        if hit: csv[k] = hit[0]
    for f in glob.glob(f'{d}/**/*probs_*_s*.npy', recursive=True): shutil.copy(f, f'{ALL}/{os.path.basename(f)}')
Y = {}
for k, (h, n) in HASH.items():
    assert k in csv, f'{k}.csv not found - upload archive__3_.zip'
    assert hashlib.sha256(open(csv[k], 'rb').read()).hexdigest()[:16] == h, f'{k}.csv differs - STOP'
    Y[k] = pd.read_csv(csv[k])['hate speech'].astype(int).values; assert len(Y[k]) == n
print('val/test OK:', len(Y['val']), len(Y['test']))
yv, yt = Y['val'], Y['test']

def load(name, split):
    f = f'{ALL}/{"valprobs" if split == "val" else "probs"}_{name}_s{{}}.npy'
    out = {s: np.load(f.format(s)) for s in SEEDS if os.path.exists(f.format(s))}
    return out if len(out) == 3 else None
POOL_NAMES = ['mbert', 'xlmr', 'banglabert', 'banglabert_unw', 'banglabert_norm', 'indicbertv2', 'banglabert_large_norm', 'xlmr_large']
V, T = {}, {}
for n in POOL_NAMES:
    v, t = load(n, 'val'), load(n, 'test')
    if v and t: V[n], T[n] = v, t
    else: print(f'  (not available, skipped: {n})')
for n in ['banglabert_norm', 'xlmr', 'banglabert_large_norm']:
    assert n in T, f'{n} probs/valprobs missing for some seed - check the uploads'
print('pool:', list(T))

# ---------- 2. build systems (choices on validation only) ----------
def mf1(y, p): return f1_score(y, p, average='macro')
def avg(names, D, s, w=None):
    w = np.ones(len(names)) if w is None else np.asarray(w, float)
    return sum(wi * D[n][s] for wi, n in zip(w, names)) / w.sum()
def greedy(s, max_picks=10):
    picks, best = [], -1
    for _ in range(max_picks):
        cand = [(mf1(yv, avg(picks + [n], V, s).argmax(1)), n) for n in V]
        f, n = max(cand)
        if f <= best + 1e-6 and picks: break
        picks.append(n); best = f
    return picks, best
def tune_thr(pv):
    grid = np.round(np.arange(0.30, 0.701, 0.01), 2)
    sc = [mf1(yv, (pv[:, 1] >= t).astype(int)) for t in grid]
    return float(grid[int(np.argmax(sc))])

SYS = {}   # name -> {seed: test predictions}
def add(label, fn):
    SYS[label] = {s: fn(s) for s in SEEDS}
add('BanglaBERT + normalizer (base)', lambda s: T['banglabert_norm'][s].argmax(1))
add('E2: BB(norm) + XLM-R  [Study 1]', lambda s: avg(['banglabert_norm', 'xlmr'], T, s).argmax(1))
add('BB-large + normalizer', lambda s: T['banglabert_large_norm'][s].argmax(1))
if 'xlmr_large' in T: add('XLM-R-large', lambda s: T['xlmr_large'][s].argmax(1))
add('E5: BB-large(norm) + XLM-R', lambda s: avg(['banglabert_large_norm', 'xlmr'], T, s).argmax(1))
add('E6: BB-large(norm) + BB(norm) + XLM-R', lambda s: avg(['banglabert_large_norm', 'banglabert_norm', 'xlmr'], T, s).argmax(1))
if 'xlmr_large' in T: add('E7: BB-large(norm) + XLM-R-large', lambda s: avg(['banglabert_large_norm', 'xlmr_large'], T, s).argmax(1))
sel_log, e8 = [], {}
for s in SEEDS:
    picks, vbest = greedy(s); names = sorted(set(picks)); w = [picks.count(n) for n in names]
    thr = tune_thr(avg(names, V, s, w)); vthr = mf1(yv, (avg(names, V, s, w)[:, 1] >= thr).astype(int))
    e8[s] = (avg(names, T, s, w)[:, 1] >= thr).astype(int)
    sel_log.append(dict(Seed=s, Members_weights=', '.join(f'{n} x{c}' for n, c in zip(names, w)), Val_macroF1_argmax=round(vbest, 4),
                        Threshold=thr, Val_macroF1_with_threshold=round(vthr, 4)))
SYS['E8: validation-selected ensemble + threshold'] = e8
sel_log = pd.DataFrame(sel_log)

# ---------- 3. metrics ----------
rows = []
for lab, P in SYS.items():
    m = np.array([[accuracy_score(yt, P[s]), mf1(yt, P[s]), f1_score(yt, P[s], average='weighted'), precision_score(yt, P[s]), recall_score(yt, P[s])] for s in SEEDS])
    mu, sd = m.mean(0), m.std(0, ddof=1)
    rows.append(dict(System=lab, **{k: f'{a:.4f} ± {b:.4f}' for k, a, b in zip(['Accuracy', 'Macro_F1', 'Weighted_F1', 'Hate_P', 'Hate_R'], mu, sd)},
                     acc=mu[0], mf=mu[1], wf=mu[2]))
tab = pd.DataFrame(rows)

# ---------- 4. statistics ----------
def mcnemar(ca, cb):
    b = int((ca & ~cb).sum()); c = int((~ca & cb).sum())
    return 1.0 if b + c == 0 else float(chi2.sf((abs(b - c) - 1) ** 2 / (b + c), 1))
def holm(p):
    p = np.asarray(p); o = np.argsort(p); m = len(p); adj = np.empty(m); run = 0
    for r, i in enumerate(o): run = max(run, (m - r) * p[i]); adj[i] = min(1.0, run)
    return adj
R = 2000; rng = np.random.default_rng(2024)
W = rng.multinomial(len(yt), np.full(len(yt), 1 / len(yt)), size=R).astype(np.float32)
def fboot(p):
    y1 = (yt == 1).astype(np.float32); p1 = (p == 1).astype(np.float32)
    tp = W @ (y1 * p1); fp = W @ ((1 - y1) * p1); fn = W @ (y1 * (1 - p1)); tn = W @ ((1 - y1) * (1 - p1))
    return (2 * tp / np.maximum(2 * tp + fp + fn, 1e-9) + 2 * tn / np.maximum(2 * tn + fn + fp, 1e-9)) / 2
base = ['BanglaBERT + normalizer (base)', 'E2: BB(norm) + XLM-R  [Study 1]']
new = [k for k in SYS if k not in base]
pairs = [(a, b) for a in new for b in base]
pv = {s: [mcnemar(SYS[a][s] == yt, SYS[b][s] == yt) for a, b in pairs] for s in SEEDS}
adj = {s: holm(pv[s]) for s in SEEDS}
st = []
for i, (a, b) in enumerate(pairs):
    d = np.mean([fboot(SYS[a][s]) for s in SEEDS], 0) - np.mean([fboot(SYS[b][s]) for s in SEEDS], 0)
    lo, hi = np.percentile(d, [2.5, 97.5]); k = sum(adj[s][i] < 0.05 for s in SEEDS); ex = lo > 0 or hi < 0
    diff = np.mean([mf1(yt, SYS[a][s]) for s in SEEDS]) - np.mean([mf1(yt, SYS[b][s]) for s in SEEDS])
    st.append(dict(New=a, Against=b, Diff_macroF1=round(diff, 4), CI_low=round(lo, 4), CI_high=round(hi, 4), seeds_sig_of_3=k,
                   Verdict='significant' if k == 3 and ex else 'not significant' if k == 0 and not ex else 'mixed'))
st = pd.DataFrame(st)

# ---------- 5. print + save ----------
pd.set_option('display.width', 260); pd.set_option('display.max_columns', 20); pd.set_option('display.max_colwidth', 80)
print('\n========== E8 selection on the VALIDATION set (per seed) ==========')
print(sel_log.to_string(index=False))
print('\n========== BD-SHS TEST (5,029) - mean ± std over seeds 42/43/44 ==========')
print(tab[['System', 'Accuracy', 'Macro_F1', 'Weighted_F1', 'Hate_P', 'Hate_R']].to_string(index=False))
print('\n========== NEW SYSTEMS vs BanglaBERT + normalizer and vs E2 ==========')
print(st.to_string(index=False))
print('\n========== PUBLISHED (binary, whole BD-SHS) ==========')
print('Romim et al. 2022          BiLSTM + IFT            weighted F1 0.9100              (70/15/15 split)')
print('Tarin et al. (preprint)    stacking ML             accuracy 0.9149, F1 0.9149      (80:20 split)')
print('Islam & Ameen 2025         Llama-3.2-3B QLoRA      accuracy 0.9248, weighted F1 0.9223 (same 80/10/10 files, one run)')
for _, r in tab.iterrows():
    print(f'ours  {r.System:48s} accuracy {r.acc:.4f}  weighted F1 {r.wf:.4f}   ->  vs Islam & Ameen: {100*(r.acc-0.9248):+.2f} / {100*(r.wf-0.9223):+.2f} points')
out = '/content/x4_best.zip' if os.path.exists('/content') else './x4_best.zip'
tab.drop(columns=['acc', 'mf', 'wf']).to_csv(f'{ROOT}/x4_metrics.csv', index=False); st.to_csv(f'{ROOT}/x4_stats.csv', index=False); sel_log.to_csv(f'{ROOT}/x4_selection.csv', index=False)
with zipfile.ZipFile(out, 'w') as z:
    for f in glob.glob(f'{ROOT}/*.csv'): z.write(f, os.path.basename(f))
print('[saved]', out)
try:
    from google.colab import files; files.download(out)
except Exception: pass
