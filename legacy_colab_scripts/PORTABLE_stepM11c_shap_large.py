# ============================================================================
# STEP M11c  --  LARGE-SAMPLE SHAP: Single (A) vs Combine (B), BanglaBERT + normalizer, seed 42, on ~220 comments
#   GPU (T4), any Colab account, no Drive. About 55-60 min (set N_PER_CLASS lower for a faster run).
#   1. Runtime > Change runtime type > T4 GPU     2. paste this WHOLE file in ONE cell and run
#   3. Upload together:  merged_splits.zip  +  archive__3_.zip
#   4. At the end shap_large_results.zip downloads automatically.
#   What it does: trains BOTH models (A = BD-SHS only, B = merged; same recipe as M3), picks N_PER_CLASS hate + N_PER_CLASS non-hate BD-SHS test comments
#   (random, seed 2025) PLUS the 20 comments used before (seed 2024), explains every comment with word-level SHAP for BOTH models, and summarises:
#   p(hate) shift, agreement of the two explanations, and the words that push hardest toward 'hate' in each model.
# ============================================================================
!pip -q install shap sentencepiece ftfy regex
!rm -rf /content/nz && git clone -q https://github.com/csebuetnlp/normalizer /content/nz
import sys, types, re
_m = types.ModuleType('emoji'); _m.get_emoji_regexp = lambda: re.compile('(?!x)x'); sys.modules['emoji'] = _m
sys.path.insert(0, '/content/nz')
from normalizer import normalize as bn_normalize
import os, json, random, hashlib, time, glob, zipfile, warnings, shutil, gc, platform
import numpy as np, pandas as pd, torch, torch.nn as nn
warnings.filterwarnings('ignore')
assert torch.cuda.is_available(), 'No GPU! Runtime > Change runtime type > T4 GPU, then run again.'
print('GPU:', torch.cuda.get_device_name(0), '| torch', torch.__version__)

SEED, HF, MAX_LEN, EPOCHS, LR, WD, BATCH_TR, BATCH_EV = 42, 'csebuetnlp/banglabert', 128, 3, 2e-5, 0.01, 32, 64
SAMPLE_SEED, N_PER_CLASS, MAX_EVALS = 2024, 10, 2000
SEED2, N_PER_CLASS = 2025, 100   # N_PER_CLASS: hate AND non-hate comments EACH (100 -> 200 new + the old 20)
OUT = '/content/shap_large_out'; os.makedirs(OUT, exist_ok=True)   # NOT wiped: a finished model is skipped when you re-run in the same session
EXP_S2 = {'A_train': ('e2e765bc7028607e', 40224), 'A_val': ('be38f2bf02e901a3', 5028), 'B_train': ('5c3d0534bc182c8b', 86822), 'B_val': ('49a1c67ca0e7fcec', 11059), 'test_all': ('5c26e952999ca26c', 12570), 'test_bdshs': ('a49a667d635cc6bf', 5029)}
EXP_S1 = {'test': ('4543a005555ee466', 5029)}

# ---------------- uploads (hash-checked) ----------------
from google.colab import files
D2, D1, UP = '/content/splits', '/content/bdshs', '/content/_up_zip'
for d in (D2, D1, UP): os.makedirs(d, exist_ok=True)
def ready(): return all(os.path.exists(f'{D2}/{k}.csv') for k in EXP_S2) and os.path.exists(f'{D1}/test.csv')
if not ready():
    print('Upload merged_splits.zip + archive__3_.zip (+ optional shap_results.zip), select them together')
    for name, content in files.upload().items():
        p = f'/content/_up_{re.sub(r"[^A-Za-z0-9_.]", "_", name)}'; open(p, 'wb').write(content)
        if name.lower().endswith('.zip'):
            with zipfile.ZipFile(p) as z: z.extractall(f'{UP}/{os.path.basename(p)[:-4]}')
    for k in EXP_S2:
        h = glob.glob(f'{UP}/**/{k}.csv', recursive=True)
        if h: shutil.copy(h[0], f'{D2}/{k}.csv')
    h = [x for x in glob.glob(f'{UP}/**/test.csv', recursive=True) if 'splits' not in x]
    if h: shutil.copy(h[0], f'{D1}/test.csv')
assert ready(), 'merged_splits.zip or archive__3_.zip (test.csv) missing'
def sha16(p): return hashlib.sha256(open(p, 'rb').read()).hexdigest()[:16]
for dd, EX in ((D2, EXP_S2), (D1, EXP_S1)):
    for k, (h, n) in EX.items():
        raw = pd.read_csv(f'{dd}/{k}.csv'); ok = (sha16(f'{dd}/{k}.csv') == h) and (len(raw) == n)
        print(f'{k:10s} rows={len(raw):6d} sha256[:16]={sha16(f"{dd}/{k}.csv")} ->', 'OK' if ok else '*** MISMATCH ***'); assert ok, 'Dataset differs - STOP'

def clean_text(text):
    text = str(text)
    text = re.sub(r'http\S+|www\S+', '', text)
    text = re.sub(r'[^ঀ-৿\s\w.,!?।]', '', text)
    return re.sub(r'\s+', ' ', text).strip()
def load2(k):
    d = pd.read_csv(f'{D2}/{k}.csv'); d['text'] = d['text'].fillna('').map(clean_text); d['label'] = d['label'].astype(int)
    return d[['source', 'orig_idx', 'text', 'label']].reset_index(drop=True)
D = {k: load2(k) for k in ['A_train', 'A_val', 'B_train', 'B_val', 'test_all', 'test_bdshs']}
t1 = pd.read_csv(f'{D1}/test.csv').rename(columns={'sentence': 'text_raw', 'hate speech': 'label'})
t1['text'] = t1['text_raw'].map(clean_text); t1['label'] = t1['label'].astype(int)
t1['text_norm'] = [bn_normalize(t) for t in t1.text]; y1 = t1.label.values

# sample: the 20 old rows (seed 2024, 10+10) + new random rows (seed 2025, N_PER_CLASS+N_PER_CLASS), no overlap
rng = np.random.default_rng(SAMPLE_SEED)
OLD = sorted(int(i) for i in list(rng.choice(np.where(y1 == 1)[0], 10, replace=False)) + list(rng.choice(np.where(y1 == 0)[0], 10, replace=False)))
rng2 = np.random.default_rng(SEED2)
okrow = lambda i: len(str(t1.text_norm[i]).split()) >= 2      # SHAP needs at least 2 words (an empty comment crashed the first run)
pool1 = np.array([i for i in np.where(y1 == 1)[0] if i not in OLD and okrow(i)]); pool0 = np.array([i for i in np.where(y1 == 0)[0] if i not in OLD and okrow(i)])
NEW = sorted(int(i) for i in list(rng2.choice(pool1, N_PER_CLASS, replace=False)) + list(rng2.choice(pool0, N_PER_CLASS, replace=False)))
IDX = sorted(set(OLD) | set(NEW)); print(f'{len(OLD)} old + {len(NEW)} new = {len(IDX)} comments')
json.dump(dict(old_rows=OLD, new_rows=NEW, rule=f'old: seed {SAMPLE_SEED} 10+10; new: seed {SEED2} {N_PER_CLASS}+{N_PER_CLASS} from the rest'), open(f'{OUT}/sample_rows.json', 'w'))

from sklearn.metrics import f1_score
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer, TrainerCallback
from datasets import Dataset
import shap
from shap.maskers._text import SimpleTokenizer
tk = AutoTokenizer.from_pretrained(HF)
class KeepBest(TrainerCallback):
    def __init__(self): self.best, self.state, self.epoch = -1, None, 0
    def on_evaluate(self, args, state, control, metrics=None, model=None, **kw):
        f = (metrics or {}).get('eval_f1', -1)
        if f > self.best and model is not None:
            self.best, self.epoch = f, int(round(state.epoch)); self.state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
def mk(texts, labels):
    ds = Dataset.from_dict({'text': texts, 'labels': labels})
    return ds.map(lambda b: tk(b['text'], truncation=True, max_length=MAX_LEN), batched=True, remove_columns=['text'])
nz = lambda d: [bn_normalize(t) for t in d.text]

def train_and_explain(reg):
    fc, fw = f'{OUT}/shap_{reg}_comments.csv', f'{OUT}/shap_{reg}_words.csv'
    if os.path.exists(fc) and os.path.exists(fw) and sorted(pd.read_csv(fc).test_row) == sorted(IDX):
        print(f'[{reg}] already finished in this session -> skipped'); return pd.read_csv(fc), pd.read_csv(fw)
    t0 = time.time(); train, val = D[f'{reg}_train'], D[f'{reg}_val']
    CW = torch.tensor(len(train) / (2 * np.bincount(train.label.values)), dtype=torch.float)
    class WT(Trainer):
        def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
            labels = inputs.pop('labels'); out = model(**inputs)
            loss = nn.CrossEntropyLoss(weight=CW.to(out.logits.device))(out.logits.float(), labels)
            return (loss, out) if return_outputs else loss
    random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED); torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.deterministic = True; torch.backends.cudnn.benchmark = False
    dtr, dva = mk(nz(train), train.label.tolist()), mk(nz(val), val.label.tolist())
    model = AutoModelForSequenceClassification.from_pretrained(HF, num_labels=2)
    args = TrainingArguments(output_dir=f'/content/ckpt_{reg}', num_train_epochs=EPOCHS, learning_rate=LR, weight_decay=WD, per_device_train_batch_size=BATCH_TR,
        per_device_eval_batch_size=BATCH_EV, eval_strategy='epoch', save_strategy='no', load_best_model_at_end=False, seed=SEED, data_seed=SEED,
        fp16=True, report_to='none', disable_tqdm=True, logging_strategy='epoch')
    def cm(ep):
        p = ep.predictions[0] if isinstance(ep.predictions, tuple) else ep.predictions
        return {'f1': f1_score(ep.label_ids, p.argmax(-1), average='macro')}
    kb = KeepBest(); tr = WT(model=model, args=args, train_dataset=dtr, eval_dataset=dva, processing_class=tk, compute_metrics=cm, callbacks=[kb])
    tr.train(); tr.model.load_state_dict(kb.state)
    def P(texts, labels):
        p = tr.predict(mk(texts, labels)).predictions; p = p[0] if isinstance(p, tuple) else p
        return torch.softmax(torch.tensor(p).float(), 1).numpy()
    pv = P(nz(val), val.label.tolist()); vf = f1_score(val.label.values, pv.argmax(1), average='macro'); assert abs(vf - kb.best) < 1e-4, 'best-epoch restore mismatch!'
    P1 = P(t1.text_norm.tolist(), t1.label.tolist())
    print(f'[{reg}] best epoch {kb.epoch} | val F1 {vf:.4f} | BD-SHS test macro-F1 {f1_score(y1, P1.argmax(1), average="macro"):.4f} | train {(time.time()-t0)/60:.1f} min')
    mdl = tr.model.eval()
    def predict_fn(texts):
        texts = [str(t) for t in texts]; out = []
        for i in range(0, len(texts), 128):
            enc = tk(texts[i:i + 128], truncation=True, max_length=MAX_LEN, padding=True, return_tensors='pt').to('cuda')
            with torch.no_grad(), torch.autocast('cuda', dtype=torch.float16): lg = mdl(**enc).logits.float()
            out.append(torch.softmax(lg, 1).cpu().numpy())
        return np.concatenate(out)
    texts = [t1.text_norm[i] for i in IDX]
    masker = shap.maskers.Text(SimpleTokenizer(r'\s+'), mask_token='[MASK] ', collapse_mask_token=False)
    ex = shap.Explainer(predict_fn, masker, algorithm='permutation', output_names=['non-hate', 'hate'], seed=0)
    t1s = time.time(); rows, longrows, failed = [], [], []
    for j, i in enumerate(IDX):
        try:
            sv = ex([texts[j]], max_evals=MAX_EVALS, batch_size=64)
            words = [str(w).strip() for w in sv.data[0]]; vals = sv.values[0][:, 1]
        except Exception as e:
            failed.append((i, repr(e)[:80])); continue
        rows.append(dict(test_row=i, true_label=int(y1[i]), predicted=int(P1[i].argmax()), p_hate=round(float(P1[i][1]), 4), n_words=len(words)))
        for k, (w, v) in enumerate(zip(words, vals)): longrows.append(dict(test_row=i, position=k, word=w, shap=float(v)))
        if (j + 1) % 40 == 0: print(f'   [{reg}] {j + 1}/{len(IDX)} comments explained, {(time.time()-t1s)/60:.1f} min')
    if failed: print(f'[{reg}] SHAP failed for {len(failed)} comment(s), skipped:', failed)
    print(f'[{reg}] SHAP on {len(rows)} comments: {(time.time()-t1s)/60:.1f} min')
    np.save(f'{OUT}/probs_{reg}_s42_bdshs_study1test.npy', P1)
    pd.DataFrame(rows).to_csv(f'{OUT}/shap_{reg}_comments.csv', index=False); W = pd.DataFrame(longrows); W.to_csv(f'{OUT}/shap_{reg}_words.csv', index=False)
    del tr, model, mdl, ex; gc.collect(); torch.cuda.empty_cache()
    return pd.DataFrame(rows), W

RA, WA = train_and_explain('A')     # Single
RB, WB = train_and_explain('B')     # Combine

# ---------------- summary: Single vs Combine ----------------
from scipy.stats import spearmanr, wilcoxon
M = RA.merge(RB, on=['test_row', 'true_label'], suffixes=('_single', '_combine')); M['sample'] = np.where(M.test_row.isin(OLD), 'old20', 'new'); M = M.reset_index(drop=True)
rho, ov = [], []
for i in M.test_row:
    a = WA[WA.test_row == i].sort_values('position'); b = WB[WB.test_row == i].sort_values('position')
    assert list(a.word) == list(b.word)
    rho.append(spearmanr(a.shap, b.shap)[0] if len(a) > 2 and a.shap.nunique() > 1 and b.shap.nunique() > 1 else np.nan)
    ov.append(len(set(a.nlargest(3, 'shap').word) & set(b.nlargest(3, 'shap').word)))
M['spearman'] = rho; M['top3_overlap'] = ov; M.to_csv(f'{OUT}/shap_large_single_vs_combine.csv', index=False)
print(f'\n===== SINGLE (A) vs COMBINE (B) on {len(M)} BD-SHS test comments ({int((M.true_label==1).sum())} hate, {int((M.true_label==0).sum())} non-hate) =====')
for nm, q in (('all', M), ('hate comments', M[M.true_label == 1]), ('non-hate comments', M[M.true_label == 0])):
    ca, cb = int((q.predicted_single == q.true_label).sum()), int((q.predicted_combine == q.true_label).sum())
    try: wp = wilcoxon(q.p_hate_combine, q.p_hate_single).pvalue
    except Exception: wp = np.nan
    print(f'{nm:18s} n={len(q):3d} | correct Single {ca}, Combine {cb} | mean p(hate) Single {q.p_hate_single.mean():.3f} -> Combine {q.p_hate_combine.mean():.3f} (paired Wilcoxon p={wp:.2g}) | same prediction {int((q.predicted_single==q.predicted_combine).sum())}')
print(f'explanation agreement: median Spearman of word values = {np.nanmedian(M.spearman):.3f} (n={int(M.spearman.notna().sum())}) | mean top-3 overlap = {M.top3_overlap.mean():.2f} of 3')
# which words push hardest toward 'hate' (mean SHAP, words seen >= 3 times) in each model
def topwords(W, n=15):
    g = W.groupby('word').shap.agg(['mean', 'count']); g = g[g['count'] >= 3].sort_values('mean', ascending=False); return g.head(n)
TA, TB = topwords(WA), topwords(WB)
print('\nTop words toward HATE (mean SHAP, >=3 occurrences)'); print(pd.concat([TA.reset_index().rename(columns={'word': 'Single_word', 'mean': 'Single_meanSHAP', 'count': 'n'}),
      TB.reset_index().rename(columns={'word': 'Combine_word', 'mean': 'Combine_meanSHAP', 'count': 'n2'})], axis=1).round(3).to_string(index=False))
sa, sb = set(topwords(WA, 30).index), set(topwords(WB, 30).index); print(f'overlap of the top-30 hate words: {len(sa & sb)} of 30')
topwords(WA, 50).to_csv(f'{OUT}/top_hate_words_single.csv'); topwords(WB, 50).to_csv(f'{OUT}/top_hate_words_combine.csv')
# the comments where predictions differ
dd = M[M.predicted_single != M.predicted_combine]; print('\nDisagreements (Single vs Combine):', len(dd)); print(dd[['test_row', 'true_label', 'predicted_single', 'predicted_combine', 'p_hate_single', 'p_hate_combine']].to_string(index=False) if len(dd) else '  none')

import transformers, shap as _s
json.dump(dict(gpu=torch.cuda.get_device_name(0), torch=torch.__version__, transformers=transformers.__version__, shap=_s.__version__, python=platform.python_version(),
               date=time.strftime('%Y-%m-%d %H:%M:%S'), n_comments=len(IDX), data_hashes={k: sha16(f'{D2}/{k}.csv') for k in EXP_S2}, study1_test_hash=sha16(f'{D1}/test.csv')), open(f'{OUT}/shap_large_info.json', 'w'), indent=2)
zp = '/content/shap_large_results.zip'
with zipfile.ZipFile(zp, 'w', zipfile.ZIP_DEFLATED) as z:
    for f in sorted(glob.glob(f'{OUT}/*')): z.write(f, os.path.basename(f))
print(f'\n[saved] {len(os.listdir(OUT))} files -> {zp}'); files.download(zp)
