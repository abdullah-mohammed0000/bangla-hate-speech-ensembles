# =====================================================================
# STEP M1  -  DATA PREPARATION FOR STUDY 2 (merged corpus)      [PORTABLE]
# Runs on CPU, in ANY Colab account (no Drive needed). Takes about 1-2 minutes.
#
# UPLOAD these 4 zip files when asked (select all four at once):
#   archive__3_.zip        (BD-SHS: train.csv, val.csv, test.csv)
#   Banhate.zip            (Hate_Comment_Label.xlsx, Non_Hate_Comment_Label.xlsx)
#   BanglaHateBert-main.zip(train_data.xlsx, valid_data.xlsx, test_data.xlsx, ...)
#   Own_Data.zip           (Combined_Hate_Comments_Final_v5.xlsx, combine_nonhate_dataset_8.xlsx)
#
# OUTPUT: merged_splits.zip  (downloads automatically). Keep it: every later job
# (M2-M8) uploads this same zip, and checks its hash, so all accounts use identical data.
# =====================================================================
import os, re, json, glob, hashlib, zipfile, shutil, sys, platform
import numpy as np, pandas as pd
from sklearn.model_selection import train_test_split

SEED = 42            # seed for the 80/10/10 split of BanHate and Own
IN, OUT = '/content/_m1_in', '/content/merged_splits'
for d in (IN, OUT):
    shutil.rmtree(d, ignore_errors=True); os.makedirs(d)

# ---------------- 1. get the four zips ----------------
try:
    from google.colab import files
    print('Upload the 4 zip files (archive__3_.zip, Banhate.zip, BanglaHateBert-main.zip, Own_Data.zip)')
    up = files.upload(); ZIPS = []
    for name, data in up.items():
        p = f'{IN}/{name}'; open(p, 'wb').write(data); ZIPS.append(p)
except ImportError:                      # running outside Colab: use the zips found in IN_DIR
    ZIPS = glob.glob(os.environ.get('IN_DIR', '.') + '/*.zip')
for i, z in enumerate(ZIPS):
    with zipfile.ZipFile(z) as f: f.extractall(f'{IN}/x{i}')

def find(pattern, must_have=None):
    hits = [p for p in glob.glob(f'{IN}/**/*', recursive=True) if re.search(pattern, os.path.basename(p), re.I)]
    if must_have: hits = [p for p in hits if must_have(p)]
    assert len(hits) >= 1, f'file not found: {pattern}'
    return sorted(hits)[0]
def sha16(p): return hashlib.sha256(open(p, 'rb').read()).hexdigest()[:16]

# ---------------- 2. cleaning (identical to Study 1) ----------------
def clean_text(text):
    text = str(text)
    text = re.sub(r'http\S+|www\S+', '', text)
    text = re.sub(r'[^ঀ-৿\s\w.,!?।]', '', text)
    return re.sub(r'\s+', ' ', text).strip()

# ---------------- 3. load the four corpora ----------------
RAW_FILES = {}
def reg(src, p): RAW_FILES[f'{src}: {os.path.basename(p)}'] = dict(rows=None, sha256_16=sha16(p))

# 3a. BD-SHS (distributed files, unchanged; hash-checked against Study 1)
BD_REF = {'train': ('116c9e9eb7432dff', 40224), 'val': ('67995e80777818a1', 5028), 'test': ('4543a005555ee466', 5029)}
bd = {}
for k, (h, n) in BD_REF.items():
    p = find(rf'^{k}\.csv$', lambda p: 'sentence' in open(p, encoding='utf-8-sig').readline())
    d = pd.read_csv(p); ok = (sha16(p) == h and len(d) == n)
    print(f'BD-SHS {k:5s} rows={len(d):6d} sha256[:16]={sha16(p)} ->', 'OK' if ok else '*** MISMATCH ***'); assert ok
    bd[k] = pd.DataFrame({'raw': d['sentence'].astype(str), 'label': d['hate speech'].astype(int)}); reg('BD-SHS', p)

# 3b. BanglaHateBERT: its own official train / valid / test files
bhb = {}
for k, fn in [('train', 'train_data'), ('val', 'valid_data'), ('test', 'test_data')]:
    p = find(rf'^{fn}\.xlsx$'); d = pd.read_excel(p); reg('BanglaHateBERT', p)
    bhb[k] = pd.DataFrame({'raw': d['post'].astype(str), 'label': d['label'].astype(int)})
    print(f'BanglaHateBERT {k:5s} rows={len(d)}')

# 3c. BanHate: two files (hate / non-hate), column "H/NH". Label rule:
#     H -> 1, NH -> 0 (the label column is followed even for the 16 NH rows found in the hate file);
#     the single row labelled "per" (a hate sub-category code, found in the hate file) -> 1.
pH, pN = find(r'^Hate_Comment_Label\.xlsx$'), find(r'^Non_Hate_Comment_Label\.xlsx$')
bh_raw = pd.concat([pd.read_excel(pH), pd.read_excel(pN)], ignore_index=True); reg('BanHate', pH); reg('BanHate', pN)
bh_map = {'H': 1, 'NH': 0, 'per': 1}
assert set(bh_raw['H/NH'].unique()) <= set(bh_map), bh_raw['H/NH'].unique()
bh_all = pd.DataFrame({'raw': bh_raw['Comment'].astype(str), 'label': bh_raw['H/NH'].map(bh_map).astype(int)})
print(f'BanHate rows={len(bh_all)}  labels: {bh_raw["H/NH"].value_counts().to_dict()}')

# 3d. Own dataset: hate file (label 1) + non-hate file (label 0)
pOH, pON = find(r'^Combined_Hate_Comments_Final_v5\.xlsx$'), find(r'^combine_nonhate_dataset_8\.xlsx$')
oh, on = pd.read_excel(pOH, sheet_name='Hate Comments'), pd.read_excel(pON, sheet_name='BD-SHS'); reg('Own', pOH); reg('Own', pON)
assert set(oh.hate.unique()) == {1} and set(on.hate.unique()) == {0}
own_all = pd.DataFrame({'raw': pd.concat([oh.text, on.text]).astype(str).values,
                        'label': np.r_[np.ones(len(oh), int), np.zeros(len(on), int)]})
print(f'Own rows={len(own_all)}  (hate {len(oh)}, non-hate {len(on)})')

# ---------------- 4. clean + within-corpus de-duplication (BanHate, Own only) ----------------
for d in list(bd.values()) + list(bhb.values()) + [bh_all, own_all]:
    d['text'] = d['raw'].map(clean_text)

AUDIT = []   # every row that is dropped is counted here
def dedup_within(d, src):
    n0 = len(d); e = (d.text == '').sum(); d = d[d.text != ''].copy()
    conf = d.groupby('text').label.nunique(); bad = set(conf[conf > 1].index)
    nconf = d.text.isin(bad).sum(); d = d[~d.text.isin(bad)]
    nd = d.text.duplicated().sum(); d = d.drop_duplicates('text')
    AUDIT.append(dict(source=src, step='within-corpus clean-up before the 80/10/10 split', empty_after_cleaning=int(e),
                      rows_of_texts_with_conflicting_labels=int(nconf), exact_duplicates_collapsed=int(nd), rows_in=n0, rows_out=len(d)))
    return d.reset_index(drop=True)

def split_801010(d):
    tr, rest = train_test_split(d, test_size=0.2, stratify=d.label, random_state=SEED)
    va, te = train_test_split(rest, test_size=0.5, stratify=rest.label, random_state=SEED)
    return dict(train=tr.reset_index(drop=True), val=va.reset_index(drop=True), test=te.reset_index(drop=True))
bh = split_801010(dedup_within(bh_all, 'BanHate'))
own = split_801010(dedup_within(own_all, 'Own'))

PARTS = {'BD-SHS': bd, 'BanglaHateBERT': bhb, 'BanHate': bh, 'Own': own}   # priority order for duplicates
ORDER = list(PARTS)
for s, p in PARTS.items():
    for k in p: p[k] = p[k].assign(source=s, orig_idx=np.arange(len(p[k])))

# ---------------- 5. leakage rules (test and validation sets stay fixed) ----------------
test_texts = set().union(*[set(PARTS[s]['test'].text) for s in ORDER])
val_texts = set().union(*[set(PARTS[s]['val'].text) for s in ORDER])

# 5a. validation rows that also occur in any test set are removed from validation
B_val = pd.concat([PARTS[s]['val'] for s in ORDER], ignore_index=True)
n = len(B_val); in_test = B_val.text.isin(test_texts); B_val = B_val[~in_test]
AUDIT.append(dict(source='all', step='merged validation: rows whose text occurs in a test set removed', rows_in=n, rows_out=len(B_val)))
n = len(B_val); B_val = B_val[B_val.text != '']; B_val = B_val.drop_duplicates('text', keep='first')
AUDIT.append(dict(source='all', step='merged validation: empty and duplicate texts removed (first source kept)', rows_in=n, rows_out=len(B_val)))
B_val = B_val.reset_index(drop=True)

# 5b. training rows that occur in ANY validation or test set are removed; then merged-train duplicates
B_train = pd.concat([PARTS[s]['train'] for s in ORDER], ignore_index=True)
n = len(B_train); leak = B_train.text.isin(test_texts | set(PARTS['BD-SHS']['val'].text) | set(B_val.text) | val_texts)
leak_by_src = B_train[leak].source.value_counts().to_dict(); B_train = B_train[~leak]
AUDIT.append(dict(source='all', step='merged train: rows whose text occurs in any validation/test set removed', rows_in=n, rows_out=len(B_train), by_source=json.dumps(leak_by_src)))
n = len(B_train); e = (B_train.text == '').sum(); B_train = B_train[B_train.text != '']
conf = B_train.groupby('text').label.nunique(); bad = set(conf[conf > 1].index); nconf = B_train.text.isin(bad).sum(); B_train = B_train[~B_train.text.isin(bad)]
nd = B_train.text.duplicated().sum(); B_train = B_train.drop_duplicates('text', keep='first').reset_index(drop=True)
AUDIT.append(dict(source='all', step='merged train: empty / conflicting-label / duplicate texts removed (first source kept)',
                  empty_after_cleaning=int(e), rows_of_texts_with_conflicting_labels=int(nconf), exact_duplicates_collapsed=int(nd), rows_in=n, rows_out=len(B_train)))

# 5c. test sets: kept exactly as they are per source; the combined test set drops repeated texts
tests = {s: PARTS[s]['test'].reset_index(drop=True) for s in ORDER}
T_all = pd.concat(tests.values(), ignore_index=True); n = len(T_all)
rep_by_src = T_all[T_all.text.duplicated(keep='first')].source.value_counts().to_dict()
T_all = T_all.drop_duplicates('text', keep='first').reset_index(drop=True)
AUDIT.append(dict(source='all', step='combined test: repeated texts across/inside test parts dropped (first source kept)', rows_in=n, rows_out=len(T_all), by_source=json.dumps(rep_by_src)))

# study-1 style A-side files (BD-SHS only, exactly as distributed)
A_train, A_val = PARTS['BD-SHS']['train'].reset_index(drop=True), PARTS['BD-SHS']['val'].reset_index(drop=True)

# ---------------- 6. final checks ----------------
Tr, Va = set(B_train.text), set(B_val.text)
for s, t in tests.items():
    assert not (set(t.text) & Tr), f'test/train leakage in {s}'
assert not (Va & Tr) and not (set(T_all.text) & Va) , 'validation leakage'
assert len(tests['BD-SHS']) == 5029 and sha16(find(r'^test\.csv$', lambda p: 'sentence' in open(p, encoding='utf-8-sig').readline())) == BD_REF['test'][0]
assert (tests['BD-SHS'].raw.values == bd['test'].raw.values).all(), 'BD-SHS test set changed!'
print('\nFINAL CHECKS PASSED: no text of any test or validation set occurs in the merged training set; BD-SHS test unchanged (5,029 rows).')

# ---------------- 7. save ----------------
COLS = ['source', 'orig_idx', 'text', 'label']
files_out = {'A_train.csv': A_train, 'A_val.csv': A_val, 'B_train.csv': B_train, 'B_val.csv': B_val, 'test_all.csv': T_all}
for s, t in tests.items(): files_out[f'test_{s.lower().replace("-", "")}.csv'] = t
for n_, d in files_out.items(): d[COLS].to_csv(f'{OUT}/{n_}', index=False, encoding='utf-8')

comp = []
for n_, d in files_out.items():
    for s in ORDER:
        x = d[d.source == s]
        if len(x): comp.append(dict(file=n_, source=s, rows=len(x), hate=int(x.label.sum()), non_hate=int((x.label == 0).sum()), hate_pct=round(100 * x.label.mean(), 2)))
    comp.append(dict(file=n_, source='ALL', rows=len(d), hate=int(d.label.sum()), non_hate=int((d.label == 0).sum()), hate_pct=round(100 * d.label.mean(), 2)))
comp = pd.DataFrame(comp); comp.to_csv(f'{OUT}/composition.csv', index=False)
pd.DataFrame(AUDIT).to_csv(f'{OUT}/audit_dropped_rows.csv', index=False)
info = dict(seed=SEED, split_rule='BanHate and Own: after within-corpus de-duplication, stratified 80/10/10 (random_state=42); BD-SHS and BanglaHateBERT: files as distributed',
            python=platform.python_version(), pandas=pd.__version__, numpy=np.__version__,
            raw_inputs=RAW_FILES, outputs={n_: sha16(f'{OUT}/{n_}') for n_ in files_out})
json.dump(info, open(f'{OUT}/splits_info.json', 'w'), indent=1, ensure_ascii=False)

print('\n' + '=' * 30, 'COMPOSITION', '=' * 30)
print(comp.to_string(index=False))
print('\n' + '=' * 30, 'DROPPED ROWS (audit)', '=' * 30)
pd.set_option('display.width', 250, 'display.max_colwidth', 80); print(pd.DataFrame(AUDIT).fillna('').to_string(index=False))
print('\nfile hashes:', json.dumps(info['outputs'], indent=1))

zp = shutil.make_archive('/content/merged_splits', 'zip', OUT)
print('\n[saved]', zp, '| sha256[:16] =', sha16(zp))
try: files.download(zp)
except Exception: pass
