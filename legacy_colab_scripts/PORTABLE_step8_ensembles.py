# ============================================================================
# STEP 8  --  ENSEMBLES (soft voting)        CPU only: any Colab account, MAIN or PORTABLE, no GPU, no Drive
#   Runtime: any (CPU is fine). Paste this WHOLE file in ONE cell and run.
#   When asked, upload ALL of these together (select several files in the dialog):
#       archive__3_.zip                      (BD-SHS, only test.csv is used, for the true labels)
#       progress_after_7b3_banglabert.zip    (results of the main run: mBERT, XLM-R, BanglaBERT, SVM, BiLSTM)
#       job_banglabert_unw.zip, job_indicbertv2.zip, job_banglabert_norm.zip   (portable runs)
#   Output: consolidated_results.zip downloads automatically. KEEP IT: Steps 9-11 need only
#           archive__3_.zip + consolidated_results.zip.
# ============================================================================
import os, re, json, glob, shutil, zipfile, hashlib, warnings
import numpy as np, pandas as pd
from google.colab import files
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, f1_score, precision_score, recall_score
warnings.filterwarnings('ignore')

SEEDS = [42, 43, 44]
TEST_HASH, TEST_ROWS = '4543a005555ee466', 5029
ALL, OUT, TMP = '/content/consolidated', '/content/step8_out', '/content/_up8'
for d in (ALL, OUT, TMP): shutil.rmtree(d, ignore_errors=True); os.makedirs(d)

# ---------- 1. upload and unpack everything ----------
print('Upload archive__3_.zip + progress zip + job_*.zip files (select all at once)')
up = files.upload()
test_csv = None
for n, c in up.items():
    p = f'{TMP}/{n}'; open(p, 'wb').write(c)
    if not n.lower().endswith('.zip'): continue
    d = f'{TMP}/x_{os.path.splitext(n)[0]}'; os.makedirs(d, exist_ok=True)
    with zipfile.ZipFile(p) as z: z.extractall(d)
    hit = glob.glob(f'{d}/**/test.csv', recursive=True)
    if hit: test_csv = hit[0]; print('  data zip      :', n)
    k = 0
    for f in glob.glob(f'{d}/**/*', recursive=True):      # works for flat zips and for a zipped Drive folder
        b = os.path.basename(f)
        if os.path.isfile(f) and re.match(r'(probs_|valprobs_|epochlog_|results_per_run|results_job_|job_info_|run_info|hyperparameters|svm_best_C)', b):
            shutil.copy(f, f'{ALL}/{b}'); k += 1
    if k: print(f'  results zip   : {n}  ({k} files)')
assert test_csv, 'archive__3_.zip (with test.csv) was not uploaded'

# ---------- 2. true test labels (hash checked) ----------
h = hashlib.sha256(open(test_csv, 'rb').read()).hexdigest()[:16]
test = pd.read_csv(test_csv); y_test = test['hate speech'].astype(int).values
assert h == TEST_HASH and len(test) == TEST_ROWS, 'test.csv differs from the thesis dataset - STOP'
print(f'test.csv OK  (rows={len(test)}, sha256[:16]={h})')
np.save(f'{ALL}/y_test.npy', y_test)

def metrics(y, p):
    mp, mr, mf, _ = precision_recall_fscore_support(y, p, average='macro', zero_division=0)
    return dict(Accuracy=accuracy_score(y, p), Macro_P=mp, Macro_R=mr, Macro_F1=mf, Weighted_F1=f1_score(y, p, average='weighted'),
                Hate_P=precision_score(y, p, pos_label=1, zero_division=0), Hate_R=recall_score(y, p, pos_label=1, zero_division=0),
                Hate_F1=f1_score(y, p, pos_label=1, zero_division=0))

# ---------- 3. merge the per-run result tables + integrity check ----------
tabs = [pd.read_csv(f) for f in glob.glob(f'{ALL}/results_per_run*.csv') + glob.glob(f'{ALL}/results_job_*.csv')]
runs = pd.concat(tabs, ignore_index=True).drop_duplicates(subset=['Model', 'Seed'], keep='last').sort_values(['Model', 'Seed']).reset_index(drop=True)
bad, checked = 0, 0
for _, r in runs.iterrows():
    f = f'{ALL}/probs_{r.Model}_s{int(r.Seed)}.npy'
    if os.path.exists(f):
        checked += 1
        if abs(metrics(y_test, np.load(f).argmax(1))['Macro_F1'] - r.Macro_F1) > 1e-6: bad += 1; print('  MISMATCH', r.Model, r.Seed)
print(f'Integrity: macro-F1 recomputed from saved probabilities for {checked} runs ->', 'ALL MATCH' if bad == 0 else f'{bad} MISMATCH(ES)')
runs.to_csv(f'{ALL}/merged_results_per_run.csv', index=False)
print('\nRuns per model (merged):'); print(runs.groupby('Model').size().rename('n_seeds').to_string())

# ---------- 4. soft-voting ensembles (equal weights, nothing tuned on test) ----------
ENS = {
    'ens_banglabert_xlmr':            ['banglabert', 'xlmr'],
    'ens_banglabert_norm_xlmr':       ['banglabert_norm', 'xlmr'],
    'ens_banglabert_xlmr_mbert':      ['banglabert', 'xlmr', 'mbert'],
    'ens_banglabert_norm_xlmr_mbert': ['banglabert_norm', 'xlmr', 'mbert'],
}
need = sorted({m for v in ENS.values() for m in v})
missing = [f'{m}_s{s}' for m in need for s in SEEDS if not os.path.exists(f'{ALL}/probs_{m}_s{s}.npy')]
assert not missing, f'missing probability files: {missing}'
rows = []
for name, members in ENS.items():
    for s in SEEDS:
        P = np.mean([np.load(f'{ALL}/probs_{m}_s{s}.npy') for m in members], axis=0)
        np.save(f'{ALL}/probs_{name}_s{s}.npy', P)
        m_ = metrics(y_test, P.argmax(1)); m_.update(Model=name, Seed=s); rows.append(m_)
ens = pd.DataFrame(rows)[['Model', 'Seed', 'Accuracy', 'Macro_P', 'Macro_R', 'Macro_F1', 'Weighted_F1', 'Hate_P', 'Hate_R', 'Hate_F1']]
ens.to_csv(f'{ALL}/results_step8_ensembles.csv', index=False)
print('\nEnsemble results per seed:'); print(ens.round(4).to_string(index=False))
cols = ['Accuracy', 'Macro_F1', 'Weighted_F1', 'Hate_P', 'Hate_R', 'Hate_F1']
single = runs[runs.Model.isin(need)]
print('\nMEAN +- STD over 3 seeds (single models first, then ensembles):')
for df_ in (single, ens):
    g = df_.groupby('Model')[cols].agg(['mean', 'std']).round(4); print(g.to_string()); print()

# ---------- 5. save + download ----------
zp = '/content/consolidated_results.zip'
with zipfile.ZipFile(zp, 'w', zipfile.ZIP_DEFLATED) as z:
    for f in sorted(glob.glob(f'{ALL}/*')): z.write(f, os.path.basename(f))
print(f'[saved] {len(os.listdir(ALL))} files -> {zp}'); files.download(zp)
