# ============================================================================
# STEPS 9-10-11  --  MAIN TABLE + STATISTICS + PER-CLASS / ROBUSTNESS      CPU only, any Colab account
#   Runtime: any (CPU is fine, ~2-4 min). Paste this WHOLE file in ONE cell and run.
#   Upload together:  archive__3_.zip   and   consolidated_results.zip   (the file Step 8 produced)
#   (if Step 8 was run in the SAME Colab session, nothing needs to be uploaded)
#   Output: analysis_results.zip downloads automatically (all tables as CSV).
#   STEP 9  main results table (mean +- std over 3 seeds, 4 decimals, weighted-F1 too)
#   STEP 10 McNemar (continuity-corrected) all pairs + Holm, bootstrap 95% CI, paired bootstrap of F1 differences
#   STEP 11 per-class report, confusion matrices, robustness check (test rows that also occur in train removed)
# ============================================================================
import os, re, glob, shutil, zipfile, hashlib, warnings, itertools
import numpy as np, pandas as pd
from scipy.stats import chi2
from google.colab import files
from IPython.display import display
warnings.filterwarnings('ignore')
pd.set_option('display.width', 250); pd.set_option('display.max_columns', 40)

SEEDS, N_BOOT, ALPHA, BOOT_SEED = [42, 43, 44], 2000, 0.05, 2024
TEST_HASH, TEST_ROWS = '4543a005555ee466', 5029
MODELS = [('tfidf_word_svm', 'TF-IDF word + SVM'), ('tfidf_char_svm', 'TF-IDF char + SVM'), ('bilstm', 'BiLSTM'),
          ('mbert', 'mBERT'), ('xlmr', 'XLM-RoBERTa'), ('banglabert', 'BanglaBERT'), ('banglabert_unw', 'BanglaBERT (unweighted)'),
          ('banglabert_norm', 'BanglaBERT + normalizer'), ('indicbertv2', 'IndicBERTv2'),
          ('ens_banglabert_xlmr', 'Ens: BanglaBERT+XLM-R'), ('ens_banglabert_norm_xlmr', 'Ens: BanglaBERT(norm)+XLM-R'),
          ('ens_banglabert_xlmr_mbert', 'Ens: BanglaBERT+XLM-R+mBERT'), ('ens_banglabert_norm_xlmr_mbert', 'Ens: BanglaBERT(norm)+XLM-R+mBERT')]
KEYS = [k for k, _ in MODELS]; LABEL = dict(MODELS)
TMP, OUT = '/content/_up9', '/content/analysis_out'
for d in (TMP, OUT): shutil.rmtree(d, ignore_errors=True); os.makedirs(d)

# ---------------- load ----------------
# if Step 8 was run in THIS same Colab session, its files are still here -> no need to upload again
SAME = {'consolidated_results.zip': '/content/consolidated_results.zip', 'archive__3_.zip': '/content/_up8/archive__3_.zip'}
if all(os.path.exists(v) for v in SAME.values()):
    print('Step 8 files found in this session -> using them (no upload needed)'); up = {n: open(v, 'rb').read() for n, v in SAME.items()}
else:
    print('Upload archive__3_.zip + consolidated_results.zip (select both)'); up = files.upload()
for n, c in up.items():
    open(f'{TMP}/{n}', 'wb').write(c)
    if n.lower().endswith('.zip'):
        d = f'{TMP}/x_{os.path.splitext(n)[0]}'; os.makedirs(d, exist_ok=True)
        with zipfile.ZipFile(f'{TMP}/{n}') as z: z.extractall(d)
def find(name): 
    h = glob.glob(f'{TMP}/x_*/**/{name}', recursive=True); return h[0] if h else None
test_csv, train_csv = find('test.csv'), find('train.csv'); assert test_csv and train_csv, 'archive__3_.zip not uploaded'
assert find('y_test.npy'), 'consolidated_results.zip not uploaded'
h = hashlib.sha256(open(test_csv, 'rb').read()).hexdigest()[:16]
test, train = pd.read_csv(test_csv), pd.read_csv(train_csv)
assert h == TEST_HASH and len(test) == TEST_ROWS, 'test.csv differs from the thesis dataset - STOP'
y = test['hate speech'].astype(int).values
assert (np.load(find('y_test.npy')) == y).all(), 'y_test in the zip does not match test.csv'
print(f'data OK (test rows={len(test)}, sha256[:16]={h})')
PROB = {}
for k in KEYS:
    for s in SEEDS:
        f = find(f'probs_{k}_s{s}.npy'); assert f, f'missing probabilities: {k} seed {s}'
        P = np.load(f); assert P.shape == (len(y), 2), f'bad shape {k} {s}'
        PROB[(k, s)] = P
PRED = {key: P.argmax(1) for key, P in PROB.items()}
print(f'loaded {len(PRED)} prediction sets ({len(KEYS)} models x {len(SEEDS)} seeds)')

def metrics(yt, p):
    out = {}
    tp = ((yt == 1) & (p == 1)).sum(); fp = ((yt == 0) & (p == 1)).sum(); fn = ((yt == 1) & (p == 0)).sum(); tn = ((yt == 0) & (p == 0)).sum()
    P1, R1 = tp / max(tp + fp, 1), tp / max(tp + fn, 1); P0, R0 = tn / max(tn + fn, 1), tn / max(tn + fp, 1)
    F1, F0 = 2 * P1 * R1 / max(P1 + R1, 1e-12), 2 * P0 * R0 / max(P0 + R0, 1e-12)
    n1, n0 = (yt == 1).sum(), (yt == 0).sum()
    return dict(Accuracy=(tp + tn) / len(yt), Macro_F1=(F1 + F0) / 2, Weighted_F1=(F1 * n1 + F0 * n0) / len(yt),
                Hate_P=P1, Hate_R=R1, Hate_F1=F1, NonHate_P=P0, NonHate_R=R0, NonHate_F1=F0)
def meanstd(df, cols):
    g = df.groupby('Model', sort=False)[cols]; m, s = g.mean(), g.std(ddof=1).fillna(0)
    return pd.DataFrame({c: [f'{a:.4f} ± {b:.4f}' for a, b in zip(m[c], s[c])] for c in cols}, index=m.index)

# ======================= STEP 9: main results table =======================
rows = []
for k in KEYS:
    for s in SEEDS: r = metrics(y, PRED[(k, s)]); r.update(Model=k, Seed=s); rows.append(r)
R = pd.DataFrame(rows); R.to_csv(f'{OUT}/step9_per_seed_metrics.csv', index=False)
cols9 = ['Accuracy', 'Macro_F1', 'Weighted_F1', 'Hate_P', 'Hate_R', 'Hate_F1']
T9 = meanstd(R, cols9); T9.index = [LABEL[k] for k in T9.index]; T9.to_csv(f'{OUT}/step9_main_results_mean_std.csv')
print('\n' + '=' * 20, 'STEP 9: MAIN RESULTS (test set, mean ± std over 3 seeds)', '=' * 20); display(T9)
print('Note: SVMs are deterministic (identical for the 3 seeds, std = 0).')

# ======================= STEP 10: statistics =======================
# --- McNemar: for every pair of models, in each seed; Holm adjustment over all pairs (per seed) ---
pairs = list(itertools.combinations(KEYS, 2))
def mcnemar(a, b):
    ca, cb = (a == y), (b == y); n01 = int((ca & ~cb).sum()); n10 = int((~ca & cb).sum())
    if n01 + n10 == 0: return n01, n10, 1.0
    stat = (abs(n01 - n10) - 1) ** 2 / (n01 + n10); return n01, n10, float(chi2.sf(stat, 1))
def holm(p):
    p = np.asarray(p); o = np.argsort(p); m = len(p); adj = np.empty(m); run = 0
    for rank, i in enumerate(o): run = max(run, (m - rank) * p[i]); adj[i] = min(1.0, run)
    return adj
res = {}
for s in SEEDS:
    raw = [mcnemar(PRED[(a, s)], PRED[(b, s)]) for a, b in pairs]
    adj = holm([r[2] for r in raw]); res[s] = (raw, adj)
rowsM = []
for i, (a, b) in enumerate(pairs):
    r42 = res[42][0][i]
    rowsM.append(dict(A=LABEL[a], B=LABEL[b], A_right_B_wrong=r42[0], A_wrong_B_right=r42[1], p_raw_seed42=r42[2], p_holm_seed42=res[42][1][i],
                      significant_seed42=res[42][1][i] < ALPHA, seeds_significant_of_3=sum(res[s][1][i] < ALPHA for s in SEEDS)))
M = pd.DataFrame(rowsM); M.to_csv(f'{OUT}/step10_mcnemar_all_pairs.csv', index=False)
print('\n' + '=' * 20, f'STEP 10a: McNEMAR (continuity-corrected), Holm over {len(pairs)} pairs, alpha={ALPHA}', '=' * 20)
print('Primary run = seed 42; "seeds_significant_of_3" repeats the whole test for seeds 42/43/44.')
Mshow = M.copy(); Mshow['p_raw_seed42'] = Mshow.p_raw_seed42.map('{:.2e}'.format); Mshow['p_holm_seed42'] = Mshow.p_holm_seed42.map('{:.2e}'.format)
display(Mshow.sort_values(['significant_seed42', 'seeds_significant_of_3']).reset_index(drop=True))
print(f'Significant pairs (seed 42, Holm): {int(M.significant_seed42.sum())} of {len(pairs)}')
print('NOT significant pairs (seed 42, Holm):'); display(M[~M.significant_seed42][['A', 'B', 'p_holm_seed42', 'seeds_significant_of_3']])

# --- bootstrap: same resampled test comments for every model (paired); each model = mean of its 3 seeds ---
rng = np.random.default_rng(BOOT_SEED); n = len(y)
idx = rng.integers(0, n, size=(N_BOOT, n)); Yb = (y[idx] == 1)
def boot_macro(p):
    Pb = (p[idx] == 1); tp = (Yb & Pb).sum(1); fp = (~Yb & Pb).sum(1); fn = (Yb & ~Pb).sum(1); tn = (~Yb & ~Pb).sum(1)
    f1 = 2 * tp / np.maximum(2 * tp + fp + fn, 1); f0 = 2 * tn / np.maximum(2 * tn + fn + fp, 1); return (f1 + f0) / 2
BM = {k: np.mean([boot_macro(PRED[(k, s)]) for s in SEEDS], axis=0) for k in KEYS}
OBS = {k: R[R.Model == k].Macro_F1.mean() for k in KEYS}
ci = pd.DataFrame([dict(Model=LABEL[k], Macro_F1_mean_of_3_seeds=OBS[k], CI95_low=np.percentile(BM[k], 2.5), CI95_high=np.percentile(BM[k], 97.5)) for k in KEYS])
ci.to_csv(f'{OUT}/step10_bootstrap_ci.csv', index=False)
print('\n' + '=' * 20, f'STEP 10b: BOOTSTRAP 95% CI of macro-F1 ({N_BOOT} resamples, percentile; mean of 3 seeds)', '=' * 20)
display(ci.round(4))
dif = []
for a, b in pairs:
    d = BM[a] - BM[b]; lo, hi = np.percentile(d, [2.5, 97.5])
    dif.append(dict(A=LABEL[a], B=LABEL[b], Diff_A_minus_B=OBS[a] - OBS[b], CI95_low=lo, CI95_high=hi, CI_excludes_0=bool(lo > 0 or hi < 0)))
D = pd.DataFrame(dif); D.to_csv(f'{OUT}/step10_paired_bootstrap_diffs.csv', index=False)
KEYPAIRS = [('banglabert_norm', 'banglabert'), ('banglabert', 'banglabert_unw'), ('xlmr', 'banglabert'), ('indicbertv2', 'xlmr'),
            ('mbert', 'xlmr'), ('tfidf_char_svm', 'bilstm'), ('bilstm', 'tfidf_word_svm'), ('ens_banglabert_norm_xlmr', 'banglabert_norm'),
            ('ens_banglabert_norm_xlmr', 'xlmr'), ('ens_banglabert_xlmr', 'banglabert')]
print('\nSTEP 10c: PAIRED BOOTSTRAP of the macro-F1 difference (A - B), selected pairs (all 78 pairs are in the CSV)')
sel = D[[(a, b) in [(LABEL[x], LABEL[z]) for x, z in KEYPAIRS] for a, b in zip(D.A, D.B)]]
if len(sel) < len(KEYPAIRS):
    rev = D[[(b, a) in [(LABEL[x], LABEL[z]) for x, z in KEYPAIRS] for a, b in zip(D.A, D.B)]].copy()
    rev[['A', 'B']] = rev[['B', 'A']].values; rev['Diff_A_minus_B'] *= -1; rev[['CI95_low', 'CI95_high']] = -rev[['CI95_high', 'CI95_low']].values; sel = pd.concat([sel, rev])
display(sel.round(4))

# ======================= STEP 11: per-class, confusion, robustness =======================
rows = []
for k in KEYS:
    for s in SEEDS:
        r = metrics(y, PRED[(k, s)]); rows.append(dict(Model=k, Seed=s, **{c: r[c] for c in ['NonHate_P', 'NonHate_R', 'NonHate_F1', 'Hate_P', 'Hate_R', 'Hate_F1']}))
PC = pd.DataFrame(rows); PCs = meanstd(PC, ['NonHate_P', 'NonHate_R', 'NonHate_F1', 'Hate_P', 'Hate_R', 'Hate_F1'])
PCs.index = [LABEL[k] for k in PCs.index]; PCs.to_csv(f'{OUT}/step11_per_class_mean_std.csv')
print('\n' + '=' * 20, 'STEP 11a: PER-CLASS precision / recall / F1 (mean ± std over 3 seeds)', '=' * 20); display(PCs)
cm_rows = []
for k in KEYS:
    for s in SEEDS:
        p = PRED[(k, s)]; cm_rows.append(dict(Model=LABEL[k], Seed=s, TN=int(((y == 0) & (p == 0)).sum()), FP=int(((y == 0) & (p == 1)).sum()),
                                              FN=int(((y == 1) & (p == 0)).sum()), TP=int(((y == 1) & (p == 1)).sum())))
CM = pd.DataFrame(cm_rows); CM.to_csv(f'{OUT}/step11_confusion_counts_all_seeds.csv', index=False)
print('\nSTEP 11b: CONFUSION COUNTS (seed 42; rows = true class, TN/FP = non-hate, FN/TP = hate)'); display(CM[CM.Seed == 42].drop(columns='Seed').reset_index(drop=True))
# --- robustness: remove test comments whose cleaned text also occurs in train ---
def clean_text(t):
    t = str(t); t = re.sub(r'http\S+|www\S+', '', t); t = re.sub(r'[^ঀ-৿\s\w.,!?।]', '', t); return re.sub(r'\s+', ' ', t).strip()
trset = set(train['sentence'].map(clean_text)); mask = ~test['sentence'].map(clean_text).isin(trset).values
print(f'\nSTEP 11c: ROBUSTNESS - test comments whose text also occurs in train: {int((~mask).sum())} (expected 18); remaining {int(mask.sum())}')
rows = []
for k in KEYS:
    for s in SEEDS:
        a = metrics(y, PRED[(k, s)])['Macro_F1']; b = metrics(y[mask], PRED[(k, s)][mask])['Macro_F1']; rows.append(dict(Model=k, Seed=s, Macro_F1_full=a, Macro_F1_without_overlap=b))
RB = pd.DataFrame(rows); g = RB.groupby('Model', sort=False)
rb = pd.DataFrame({'Macro-F1 full test': [f'{m:.4f} ± {s:.4f}' for m, s in zip(g.Macro_F1_full.mean(), g.Macro_F1_full.std(ddof=1).fillna(0))],
                   'Macro-F1 without the overlapping rows': [f'{m:.4f} ± {s:.4f}' for m, s in zip(g.Macro_F1_without_overlap.mean(), g.Macro_F1_without_overlap.std(ddof=1).fillna(0))],
                   'Change': (g.Macro_F1_without_overlap.mean() - g.Macro_F1_full.mean()).map('{:+.4f}'.format).values}, index=[LABEL[k] for k in g.groups.keys()])
rb.to_csv(f'{OUT}/step11_robustness_without_train_overlap.csv'); display(rb)
print('Largest absolute change in macro-F1:', f"{(g.Macro_F1_without_overlap.mean() - g.Macro_F1_full.mean()).abs().max():.4f}")

# ---------------- save + download ----------------
zp = '/content/analysis_results.zip'
with zipfile.ZipFile(zp, 'w', zipfile.ZIP_DEFLATED) as z:
    for f in sorted(glob.glob(f'{OUT}/*')): z.write(f, os.path.basename(f))
print(f'\n[saved] {len(os.listdir(OUT))} tables -> {zp}'); files.download(zp)
