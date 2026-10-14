# ============================================================================
# STEP M11b  --  Single (Study 1) vs Combine SHAP comparison on the SAME 20 comments     CPU only, ~1 min, no GPU, no Drive
#   Upload both:  shap_results.zip (Study 1 step 12)  and  shap_combine_results.zip (step M11)
#   Output: shap_single_vs_combine_20.csv (downloads automatically)
# ============================================================================
import os, glob, zipfile, shutil, re, json
import numpy as np, pandas as pd
from scipy.stats import spearmanr
from google.colab import files
UP = '/content/_cmp'; shutil.rmtree(UP, ignore_errors=True); os.makedirs(UP)
print('Upload shap_results.zip and shap_combine_results.zip (select both together)')
for name, content in files.upload().items():
    d = f'{UP}/{re.sub(r"[^A-Za-z0-9_]", "_", name)}'; os.makedirs(d, exist_ok=True)
    p = f'{d}/{name}'; open(p, 'wb').write(content)
    if name.lower().endswith('.zip'):
        with zipfile.ZipFile(p) as z: z.extractall(d)
def find(n):
    h = glob.glob(f'{UP}/**/{n}', recursive=True); assert h, f'{n} not found - upload the right zip'; return h[0]
S = pd.read_csv(find('shap_sample_20.csv')); SW = pd.read_csv(find('shap_word_attributions.csv'))
C = pd.read_csv(find('shap_combine_sample_20.csv')); CW = pd.read_csv(find('shap_combine_word_attributions.csv'))
assert list(S.test_row) == list(C.test_row), 'The two runs explain different comments - STOP'
print('same 20 test rows in both runs: OK |', sorted(S.test_row))
rows = []
for i in S.test_row:
    s = S[S.test_row == i].iloc[0]; c = C[C.test_row == i].iloc[0]
    a = SW[SW.test_row == i].sort_values('position'); b = CW[CW.test_row == i].sort_values('position')
    same = list(a.word.astype(str)) == list(b.word.astype(str))
    rho = spearmanr(a.shap_value_hate.values, b.shap_value_hate.values)[0] if same and len(a) > 2 else np.nan
    ta = set(a.nlargest(3, 'shap_value_hate').word.astype(str)); tb = set(b.nlargest(3, 'shap_value_hate').word.astype(str))
    rows.append(dict(test_row=i, true_label=int(c.true_label), pred_single=int(s.predicted), pred_combine=int(c.predicted),
                     p_hate_single=s.p_hate, p_hate_combine=c.p_hate, same_prediction=bool(s.predicted == c.predicted), n_words=len(a), same_tokenisation=same,
                     spearman_word_values=None if np.isnan(rho) else round(float(rho), 3), top3_overlap=len(ta & tb), top3_single=sorted(ta), top3_combine=sorted(tb)))
R = pd.DataFrame(rows); R.to_csv('/content/shap_single_vs_combine_20.csv', index=False)
pd.set_option('display.width', 250, 'display.max_colwidth', 60)
print('\n=== SINGLE (Study 1) vs COMBINE, the same 20 comments ===')
print(R[['test_row', 'true_label', 'pred_single', 'pred_combine', 'p_hate_single', 'p_hate_combine', 'spearman_word_values', 'top3_overlap']].to_string(index=False))
print(f'\nsame prediction in {int(R.same_prediction.sum())}/20 | Single correct {int((R.pred_single==R.true_label).sum())}/20, Combine correct {int((R.pred_combine==R.true_label).sum())}/20')
for lab, nm in ((1, 'hate'), (0, 'non-hate')):
    q = R[R.true_label == lab]; print(f'mean p(hate) on the 10 {nm} comments: Single {q.p_hate_single.mean():.3f} -> Combine {q.p_hate_combine.mean():.3f}')
print(f'median Spearman of word values = {R.spearman_word_values.dropna().astype(float).median():.3f} | mean top-3 word overlap = {R.top3_overlap.mean():.2f} of 3 | same tokenisation in {int(R.same_tokenisation.sum())}/20')
print('\nDisagreements:'); print(R[~R.same_prediction][['test_row', 'true_label', 'pred_single', 'pred_combine', 'p_hate_single', 'p_hate_combine']].to_string(index=False) if (~R.same_prediction).any() else '  none')
files.download('/content/shap_single_vs_combine_20.csv')
