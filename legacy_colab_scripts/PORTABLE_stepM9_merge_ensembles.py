# ============================================================================
# Study 2  --  STEP M9: MERGE all job zips + build the 4 ensembles + main tables   [PORTABLE, CPU only, NO GPU]
#   Nothing is trained here. Ensembles = equal-weight average of the saved probabilities of their members.
#   1. Any Colab (CPU is enough). Paste the WHOLE file in ONE cell, run.
#   2. Upload, when asked: merged_splits.zip  +  m2_classical.zip  +  every m3_*.zip you have (you may upload them in 2-3 rounds;
#      the script asks again until you press Cancel / upload nothing).
#   3. Output: m9_merged.zip (all probabilities incl. ensembles, labels, result tables)  -> this ONE zip is the input of step M10 (statistics).
# Missing models are skipped (the script says which); re-run later with more zips and it simply builds the fuller version.
# ============================================================================
import os, re, io, json, glob, zipfile, hashlib, shutil, warnings, itertools
import numpy as np, pandas as pd
warnings.filterwarnings('ignore')
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, f1_score, precision_score, recall_score

ROOT = '/content/m9' if os.path.exists('/content') else './m9'
UP, PR, OUT = f'{ROOT}/up', f'{ROOT}/probs', f'{ROOT}/out'
shutil.rmtree(PR, ignore_errors=True); shutil.rmtree(OUT, ignore_errors=True)   # a re-run in the same session starts clean
for d in (UP, PR, OUT): os.makedirs(d, exist_ok=True)
SEEDS = [42, 43, 44]
PARTS = ['test_bdshs', 'test_banglahatebert', 'test_banhate', 'test_own', 'test_all']
LABELS_EXPECTED = {'test_all': '5c26e952999ca26c', 'test_bdshs': 'a49a667d635cc6bf', 'test_banglahatebert': '360305cc8e870e4d',
                   'test_banhate': 'da22d1b1ca76af7d', 'test_own': '873067fbbc95698d'}
ROWS = {'test_all': 12570, 'test_bdshs': 5029, 'test_banglahatebert': 3001, 'test_banhate': 1908, 'test_own': 2636}
NAMES = {'tfidf_word_svm': 'TF-IDF word + SVM', 'tfidf_char_svm': 'TF-IDF char + SVM', 'bilstm': 'BiLSTM', 'mbert': 'mBERT', 'xlmr': 'XLM-RoBERTa',
         'banglabert': 'BanglaBERT', 'banglabert_unw': 'BanglaBERT (unweighted)', 'banglabert_norm': 'BanglaBERT + normalizer',
         'indicbertv2': 'IndicBERTv2', 'E1': 'Ens E1: BanglaBERT + XLM-R', 'E2': 'Ens E2: BanglaBERT(norm) + XLM-R',
         'E3': 'Ens E3: BanglaBERT + XLM-R + mBERT', 'E4': 'Ens E4: BanglaBERT(norm) + XLM-R + mBERT'}
ORDER = list(NAMES)
ENS = {'E1': ['banglabert', 'xlmr'], 'E2': ['banglabert_norm', 'xlmr'],
       'E3': ['banglabert', 'xlmr', 'mbert'], 'E4': ['banglabert_norm', 'xlmr', 'mbert']}
def sha16(b): return hashlib.sha256(b).hexdigest()[:16]

# ---------------- 1. upload ----------------
def get_zips():
    try:
        from google.colab import files
    except Exception:
        return glob.glob('./inputs/*.zip')
    got = []
    while True:
        print('\nUpload zip files (merged_splits.zip, m2_classical.zip, m3_*.zip). Cancel / upload nothing when finished.')
        try: up = files.upload()
        except Exception: up = {}
        if not up: break
        for name, content in up.items():
            p = f'{UP}/{name}'; open(p, 'wb').write(content); got.append(p)
    return got
zips = get_zips() or glob.glob(f'{UP}/*.zip')
print('zips:', [os.path.basename(z) for z in zips])

# ---------------- 2. labels (merged_splits.zip) with hash check ----------------
labels = {}
for z in zips:
    with zipfile.ZipFile(z) as zf:
        names = [n for n in zf.namelist() if n.endswith('.csv')]
        if any(os.path.basename(n) == 'test_all.csv' for n in names):
            for n in names:
                k = os.path.basename(n)[:-4]
                if k in LABELS_EXPECTED:
                    b = zf.read(n); d = pd.read_csv(io.BytesIO(b))
                    ok = sha16(b) == LABELS_EXPECTED[k] and len(d) == ROWS[k]
                    print(f'  labels {k:20s} rows={len(d)} hash {sha16(b)} ->', 'OK' if ok else '*** MISMATCH ***'); assert ok
                    labels[k] = d['label'].astype(int).values
assert set(labels) == set(PARTS), 'merged_splits.zip not uploaded (labels missing)'

# ---------------- 3. collect probabilities ----------------
pat = re.compile(r'^probs_(.+)_([AB])_s(\d+)_(test_\w+)\.npy$')
for z in zips:
    with zipfile.ZipFile(z) as zf:
        for n in zf.namelist():
            mm = pat.match(os.path.basename(n))
            if mm and mm.group(1) not in ENS:      # ensembles are always rebuilt here, never read from an older m9 zip
                with zf.open(n) as f, open(f'{PR}/{os.path.basename(n)}', 'wb') as g: shutil.copyfileobj(f, g)
have = {}
for p in glob.glob(f'{PR}/probs_*.npy'):
    m = pat.match(os.path.basename(p))
    if m: have.setdefault((m.group(1), m.group(2), int(m.group(3))), set()).add(m.group(4))
models = sorted({k[0] for k in have}, key=lambda x: ORDER.index(x) if x in ORDER else 99)
print('\nmodels found:', models)
complete = {}
for m in models:
    ok = all(have.get((m, r, s), set()) >= set(PARTS) for r in 'AB' for s in SEEDS)
    complete[m] = ok
    print(f'  {NAMES.get(m, m):38s}', 'COMPLETE (A,B x 3 seeds x 5 test sets)' if ok else 'INCOMPLETE -> skipped')
usable = [m for m in models if complete[m]]
for m in ['tfidf_word_svm','tfidf_char_svm','bilstm','mbert','xlmr','banglabert','banglabert_unw','banglabert_norm','indicbertv2']:
    if m not in models: print('  MISSING model:', NAMES[m])

# ---------------- 4. ensembles (equal-weight soft voting) ----------------
built = []
for e, mem in ENS.items():
    if not all(m in usable for m in mem): print(f'  {e} not built (needs {mem})'); continue
    for r in 'AB':
        for s in SEEDS:
            for part in PARTS:
                P = np.mean([np.load(f'{PR}/probs_{m}_{r}_s{s}_{part}.npy') for m in mem], axis=0)
                np.save(f'{PR}/probs_{e}_{r}_s{s}_{part}.npy', P.astype(np.float32))
    built.append(e); print(f'  built {e}: {" + ".join(NAMES[m] for m in mem)}')
usable += built

# ---------------- 5. metrics ----------------
def metrics(y, p):
    mp, mr, mf, _ = precision_recall_fscore_support(y, p, average='macro', zero_division=0)
    return dict(Accuracy=accuracy_score(y, p), Macro_P=mp, Macro_R=mr, Macro_F1=mf,
                Hate_P=precision_score(y, p, pos_label=1, zero_division=0), Hate_R=recall_score(y, p, pos_label=1, zero_division=0),
                Hate_F1=f1_score(y, p, pos_label=1, zero_division=0))
rows = []
for m in usable:
    for r in 'AB':
        for s in SEEDS:
            for part in PARTS:
                P = np.load(f'{PR}/probs_{m}_{r}_s{s}_{part}.npy'); assert P.shape == (ROWS[part], 2), (m, r, s, part, P.shape)
                d = metrics(labels[part], P.argmax(1)); d.update(Model=m, Regime=r, Seed=s, Test=part); rows.append(d)
R = pd.DataFrame(rows); R['Name'] = R.Model.map(NAMES)
R.to_csv(f'{OUT}/results_long.csv', index=False)

def ms(x): return f'{x.mean():.4f} ± {x.std(ddof=1):.4f}' if len(x) > 1 and x.std(ddof=1) > 0 else f'{x.mean():.4f} ± 0.0000'
g = R.groupby(['Model', 'Regime', 'Test'])
mean = g[['Accuracy', 'Macro_F1', 'Hate_P', 'Hate_R']].mean(); std = g[['Accuracy', 'Macro_F1']].std(ddof=1).fillna(0)
mean.round(4).reset_index().to_csv(f'{OUT}/summary_means.csv', index=False)

# MAIN TABLE: Single (A) vs Combine (B), on BD-SHS test and on combined test
main = []
for m in sorted(usable, key=ORDER.index):
    row = {'Model': NAMES[m]}
    for part, tag in [('test_bdshs', 'BD-SHS test'), ('test_all', 'Combined test')]:
        a = mean.loc[(m, 'A', part), 'Macro_F1']; b = mean.loc[(m, 'B', part), 'Macro_F1']
        row[f'{tag}: Single (A)'] = round(a, 4); row[f'{tag}: Combine (B)'] = round(b, 4); row[f'{tag}: B - A'] = round(b - a, 4)
    for r, tag in [('A', 'Single (A)'), ('B', 'Combine (B)')]:
        row[f'Stability gap {tag} (BD-SHS minus combined)'] = round(mean.loc[(m, r, 'test_bdshs'), 'Macro_F1'] - mean.loc[(m, r, 'test_all'), 'Macro_F1'], 4)
    main.append(row)
MAIN = pd.DataFrame(main); MAIN.to_csv(f'{OUT}/main_single_vs_combine.csv', index=False)

# accuracy and mean±std table (all models, all test sets; per-source numbers are stored, not interpreted)
pretty = []
for m in sorted(usable, key=ORDER.index):
    for r in 'AB':
        row = {'Model': NAMES[m], 'Regime': 'Single (A)' if r == 'A' else 'Combine (B)'}
        for part in PARTS:
            x = R[(R.Model == m) & (R.Regime == r) & (R.Test == part)]
            row[f'{part} macro-F1'] = ms(x.Macro_F1); row[f'{part} acc'] = ms(x.Accuracy)
        pretty.append(row)
PRETTY = pd.DataFrame(pretty); PRETTY.to_csv(f'{OUT}/all_results_mean_std.csv', index=False)

# ---------------- 6. save the package for M10 ----------------
for k, v in labels.items(): np.save(f'{OUT}/labels_{k}.npy', v)
json.dump(dict(models=usable, ensembles=built, seeds=SEEDS, parts=PARTS, regimes={'A': 'Single (BD-SHS only)', 'B': 'Combine (merged)'},
               skipped=[m for m in models if not complete[m]]), open(f'{OUT}/m9_info.json', 'w'), indent=2)
zp = '/content/m9_merged.zip' if os.path.exists('/content') else './m9_merged.zip'
with zipfile.ZipFile(zp, 'w', zipfile.ZIP_DEFLATED) as z:
    for f in glob.glob(f'{PR}/probs_*.npy'): z.write(f, 'probs/' + os.path.basename(f))
    for f in glob.glob(f'{OUT}/*'): z.write(f, os.path.basename(f))
print(f'\n[saved] {zp}  ({os.path.getsize(zp)/1e6:.1f} MB)')

pd.set_option('display.width', 250); pd.set_option('display.max_columns', 30)
# two separate tables (as agreed): Table 1 = Combine model (trained on merged), Table 2 = Single model (trained on BD-SHS only)
def table(regime):
    out = []
    for m in sorted(usable, key=ORDER.index):
        out.append({'Model': NAMES[m], 'BD-SHS test': round(mean.loc[(m, regime, 'test_bdshs'), 'Macro_F1'], 4),
                    'Merged (combined) test': round(mean.loc[(m, regime, 'test_all'), 'Macro_F1'], 4)})
    return pd.DataFrame(out)
T1, T2 = table('B'), table('A')
T1.to_csv(f'{OUT}/table1_combine_model.csv', index=False); T2.to_csv(f'{OUT}/table2_single_model.csv', index=False)
with zipfile.ZipFile(zp, 'a') as z:
    for f in ('table1_combine_model.csv', 'table2_single_model.csv'): z.write(f'{OUT}/{f}', f)
print('\n=========== TABLE 1: COMBINE model (trained on merged data) - macro-F1, mean of 3 seeds ===========')
print(T1.to_string(index=False))
print('\n=========== TABLE 2: SINGLE model (trained on BD-SHS only) - macro-F1, mean of 3 seeds ===========')
print(T2.to_string(index=False))
print('\n=========== Single (A) vs Combine (B): change when going from Single to Combine (B - A) ===========')
print(MAIN[['Model', 'BD-SHS test: B - A', 'Combined test: B - A']].to_string(index=False))
try:
    from google.colab import files; files.download(zp)
except Exception: pass
