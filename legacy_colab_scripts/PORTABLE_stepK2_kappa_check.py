# ============================================================================
# STEP K2 -- check the kappa sheet against the own dataset      [CPU, ~1 min, no GPU, no Drive]
#   Upload together:  for_kaap_test_result_-_Annotation.csv  (the 800-comment sheet)   and   merged_splits.zip
#   (optional, better if you have it: your raw own-dataset csv - then upload that too; any csv with a text column and a 0/1 label column)
#   Checks: (1) how many of the 800 comments are found in the own dataset, (2) for those, does the majority vote of the 3 annotators equal the label stored in the dataset?
#   Nothing is changed; it only prints and saves kappa_check.csv.
# ============================================================================
import os, re, glob, zipfile, shutil
import numpy as np, pandas as pd
from google.colab import files
UP = '/content/_k2'; shutil.rmtree(UP, ignore_errors=True); os.makedirs(UP)
print('Upload the annotation sheet csv + merged_splits.zip (+ optional raw own csv)')
for name, content in files.upload().items():
    p = f'{UP}/{name.replace(" ", "_")}'; open(p, 'wb').write(content)
    if name.lower().endswith('.zip'):
        with zipfile.ZipFile(p) as z: z.extractall(f'{UP}/zip')
def clean_text(t):
    t = str(t); t = re.sub(r'http\S+|www\S+', '', t); t = re.sub(r'[^ঀ-৿\s\w.,!?।]', '', t); return re.sub(r'\s+', ' ', t).strip()
sheet = [f for f in glob.glob(f'{UP}/*.csv') if 'Annotation' in f or 'kaap' in f.lower()]; assert sheet, 'annotation sheet csv not found'
S = pd.read_csv(sheet[0]); ann = list(S.columns[2:]); S['maj'] = (S[ann].sum(1) >= 2).astype(int); S['key'] = S.text.map(clean_text)
print('sheet:', len(S), 'comments | annotators:', ann)
own = []
for k in ['B_train', 'B_val', 'test_own']:
    for f in glob.glob(f'{UP}/zip/**/{k}.csv', recursive=True):
        d = pd.read_csv(f); d = d[d.source.astype(str).str.lower().str.contains('own')]; d['split'] = k; own.append(d[['split', 'orig_idx', 'text', 'label']])
src = 'merged_splits.zip (cleaned, de-duplicated own rows from B_train, B_val, test_own)'
raw = [f for f in glob.glob(f'{UP}/*.csv') if f not in sheet]
if raw:
    r = pd.read_csv(raw[0]); tc = [c for c in r.columns if 'text' in c.lower() or 'sentence' in c.lower() or 'comment' in c.lower()][0]
    lc = [c for c in r.columns if 'label' in c.lower() or 'hate' in c.lower()][0]; own = [pd.DataFrame({'split': 'raw', 'orig_idx': r.index, 'text': r[tc], 'label': r[lc]})]; src = f'raw file {os.path.basename(raw[0])} (columns {tc}, {lc})'
O = pd.concat(own); O['key'] = O.text.map(clean_text); O = O.drop_duplicates('key')
print('own dataset source used:', src, '|', len(O), 'rows')
M = S.merge(O[['key', 'split', 'label']], on='key', how='left')
found = M.label.notna(); print(f'\nfound in the own dataset: {int(found.sum())} of {len(S)}; not found: {int((~found).sum())}')
if found.any():
    q = M[found]; agree = (q.maj == q.label.astype(int)); print(f'majority vote == stored label: {int(agree.sum())} of {len(q)} ({agree.mean():.4f})')
    print('found per split:', q.split.value_counts().to_dict())
    if (~agree).any(): print('\nDISAGREEMENTS (row, majority vote, stored label, split):'); print(q[~agree][['row', 'maj', 'label', 'split']].to_string(index=False))
if (~found).any(): print('\nNOT FOUND rows (first 20):', M[~found].row.tolist()[:20], '  (can be de-duplication or cleaning removed them; check with the raw file)')
M.to_csv('/content/kappa_check.csv', index=False); files.download('/content/kappa_check.csv')
