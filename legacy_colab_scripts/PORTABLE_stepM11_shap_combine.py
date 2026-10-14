# ============================================================================
# STEP M11  --  SHAP on the COMBINE model (BanglaBERT + normalizer, trained on merged data, seed 42)
#   GPU (T4), any Colab account, no Drive.   Takes about 30-35 min (training ~18 min + SHAP ~10 min).
#   1. Runtime > Change runtime type > T4 GPU     2. paste this WHOLE file in ONE cell and run
#   3. Upload these files (select all together in one dialog):
#        merged_splits.zip      (from step M1)
#        archive__3_.zip        (the original BD-SHS train/val/test of Study 1)
#        shap_results.zip       (OPTIONAL, from Study 1 step 12 - if present, the script also builds the Single-vs-Combine comparison table)
#   4. At the end shap_combine_results.zip downloads automatically.
#   What it does: re-trains the Combine model (regime B, same settings as M3), takes the SAME 20 comments as Study 1's SHAP
#   (10 hate + 10 non-hate from the BD-SHS test, numpy seed 2024 -> same row numbers), explains them with word-level SHAP,
#   and (if shap_results.zip is given) compares word by word with the Single model's explanation of the same 20 comments.
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
OUT = '/content/shap_combine_out'; shutil.rmtree(OUT, ignore_errors=True); os.makedirs(OUT)
EXP_S2 = {'B_train': ('5c3d0534bc182c8b', 86822), 'B_val': ('49a1c67ca0e7fcec', 11059), 'test_all': ('5c26e952999ca26c', 12570), 'test_bdshs': ('a49a667d635cc6bf', 5029)}
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
S1_SAMPLE = glob.glob(f'{UP}/**/shap_sample_20.csv', recursive=True); S1_WORDS = glob.glob(f'{UP}/**/shap_word_attributions.csv', recursive=True)
S1_IDX = glob.glob(f'{UP}/**/sample_indices.json', recursive=True)
print('Study-1 SHAP files found (optional):', bool(S1_SAMPLE and S1_WORDS))

def clean_text(text):
    text = str(text)
    text = re.sub(r'http\S+|www\S+', '', text)
    text = re.sub(r'[^ঀ-৿\s\w.,!?।]', '', text)
    return re.sub(r'\s+', ' ', text).strip()
def load2(k):
    d = pd.read_csv(f'{D2}/{k}.csv'); d['text'] = d['text'].fillna('').map(clean_text); d['label'] = d['label'].astype(int)
    return d[['source', 'orig_idx', 'text', 'label']].reset_index(drop=True)
train, val, test_all, test_bd = load2('B_train'), load2('B_val'), load2('test_all'), load2('test_bdshs')
t1 = pd.read_csv(f'{D1}/test.csv').rename(columns={'sentence': 'text_raw', 'hate speech': 'label'})
t1['text'] = t1['text_raw'].map(clean_text); t1['label'] = t1['label'].astype(int)
t1['text_norm'] = [bn_normalize(t) for t in t1.text]; y1 = t1.label.values       # normalizer AFTER clean_text, as in the final run
print({'B_train': len(train), 'B_val': len(val), 'test_all': len(test_all), 'test_bdshs': len(test_bd), 'study1_test': len(t1)})

# ---------------- same 20 comments as Study 1 ----------------
rng = np.random.default_rng(SAMPLE_SEED)
IDX = sorted(int(i) for i in list(rng.choice(np.where(y1 == 1)[0], N_PER_CLASS, replace=False)) + list(rng.choice(np.where(y1 == 0)[0], N_PER_CLASS, replace=False)))
if S1_IDX:
    ref = json.load(open(S1_IDX[0]))['test_row_indices']; assert ref == IDX, 'Row numbers differ from the Study 1 SHAP sample - STOP'
    print('sample row numbers identical to Study 1 sample_indices.json: OK')
else: print('(sample_indices.json not uploaded: row numbers regenerated with the same rule, seed', SAMPLE_SEED, ')')
print('20 rows:', IDX)

# ---------------- training: Combine model = M3 recipe (BanglaBERT + normalizer, weighted loss, regime B, seed 42) ----------------
from sklearn.metrics import f1_score
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer, TrainerCallback
from datasets import Dataset
CW = torch.tensor(len(train) / (2 * np.bincount(train.label.values)), dtype=torch.float)
class WeightedTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.pop('labels'); out = model(**inputs)
        loss = nn.CrossEntropyLoss(weight=CW.to(out.logits.device))(out.logits.float(), labels)
        return (loss, out) if return_outputs else loss
class KeepBest(TrainerCallback):
    def __init__(self): self.best, self.state, self.epoch = -1, None, 0
    def on_evaluate(self, args, state, control, metrics=None, model=None, **kw):
        f = (metrics or {}).get('eval_f1', -1)
        if f > self.best and model is not None:
            self.best, self.epoch = f, int(round(state.epoch)); self.state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED); torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True; torch.backends.cudnn.benchmark = False
t0 = time.time(); tk = AutoTokenizer.from_pretrained(HF)
def mk(texts, labels):
    ds = Dataset.from_dict({'text': texts, 'labels': labels})
    return ds.map(lambda b: tk(b['text'], truncation=True, max_length=MAX_LEN), batched=True, remove_columns=['text'])
nz = lambda d: [bn_normalize(t) for t in d.text]
dtr, dva = mk(nz(train), train.label.tolist()), mk(nz(val), val.label.tolist())
model = AutoModelForSequenceClassification.from_pretrained(HF, num_labels=2)
args = TrainingArguments(output_dir='/content/ckpt', num_train_epochs=EPOCHS, learning_rate=LR, weight_decay=WD, per_device_train_batch_size=BATCH_TR,
    per_device_eval_batch_size=BATCH_EV, eval_strategy='epoch', save_strategy='no', load_best_model_at_end=False, seed=SEED, data_seed=SEED,
    fp16=True, report_to='none', disable_tqdm=True, logging_strategy='epoch')
def cm(ep):
    p = ep.predictions[0] if isinstance(ep.predictions, tuple) else ep.predictions
    return {'f1': f1_score(ep.label_ids, p.argmax(-1), average='macro')}
kb = KeepBest()
tr = WeightedTrainer(model=model, args=args, train_dataset=dtr, eval_dataset=dva, processing_class=tk, compute_metrics=cm, callbacks=[kb])
tr.train(); tr.model.load_state_dict(kb.state)
def probs_of_texts(texts, labels):
    p = tr.predict(mk(texts, labels)).predictions; p = p[0] if isinstance(p, tuple) else p
    return torch.softmax(torch.tensor(p).float(), 1).numpy()
P_val = probs_of_texts(nz(val), val.label.tolist()); vf = f1_score(val.label.values, P_val.argmax(1), average='macro')
assert abs(vf - kb.best) < 1e-4, 'best-epoch restore mismatch!'
P_bd = probs_of_texts(nz(test_bd), test_bd.label.tolist()); P_all = probs_of_texts(nz(test_all), test_all.label.tolist())
print(f'   best epoch = {kb.epoch} | logged val F1 {kb.best:.4f} | re-computed {vf:.4f} | time {(time.time()-t0)/60:.1f} min')
print(f'   THIS Combine model, seed 42:  BD-SHS test macro-F1 = {f1_score(test_bd.label.values, P_bd.argmax(1), average="macro"):.4f} | merged test macro-F1 = {f1_score(test_all.label.values, P_all.argmax(1), average="macro"):.4f}')
print('   (3-seed means in M9: BD-SHS 0.8837, merged 0.8802; one seed differs a little)')
np.save(f'{OUT}/probs_combine_model_s42_test_bdshs.npy', P_bd); np.save(f'{OUT}/probs_combine_model_s42_test_all.npy', P_all)
P1 = probs_of_texts(t1.text_norm.tolist(), t1.label.tolist())          # Combine model on Study-1 test rows (all 5,029)
print(f'   Combine model on the Study-1 test file (5,029 rows): macro-F1 = {f1_score(y1, P1.argmax(1), average="macro"):.4f}')

# ---------------- SHAP (same method as Study 1) ----------------
import shap
from shap.maskers._text import SimpleTokenizer
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
explainer = shap.Explainer(predict_fn, masker, algorithm='permutation', output_names=['non-hate', 'hate'], seed=0)
sv = explainer(texts, max_evals=MAX_EVALS, batch_size=64)
rows, longrows = [], []
for j, i in enumerate(IDX):
    words = [str(w).strip() for w in sv.data[j]]; vals = sv.values[j][:, 1]; base = float(sv.base_values[j][1])
    fx = float(predict_fn([texts[j]])[0][1]); pred = int(P1[i].argmax()); add_err = abs(base + vals.sum() - fx)
    order = np.argsort(-vals); pos = [(words[k], round(float(vals[k]), 4)) for k in order if vals[k] > 0][:5]
    neg = [(words[k], round(float(vals[k]), 4)) for k in order[::-1] if vals[k] < 0][:5]
    rows.append(dict(test_row=i, true_label=int(y1[i]), predicted=pred, correct=bool(pred == y1[i]), p_hate=round(float(P1[i][1]), 4), base_value_p_hate=round(base, 4),
                     shap_additivity_error=round(add_err, 4), n_words=len(words), top_words_toward_hate=pos, top_words_toward_non_hate=neg,
                     text_raw=t1.text_raw[i], text_model_input=texts[j]))
    for k, (w, v) in enumerate(zip(words, vals)): longrows.append(dict(test_row=i, position=k, word=w, shap_value_hate=round(float(v), 5)))
R = pd.DataFrame(rows); R.to_csv(f'{OUT}/shap_combine_sample_20.csv', index=False); W = pd.DataFrame(longrows); W.to_csv(f'{OUT}/shap_combine_word_attributions.csv', index=False)
try: open(f'{OUT}/shap_combine_text_plots.html', 'w', encoding='utf-8').write(shap.plots.text(sv[:, :, 1], display=False))
except Exception as e: print('   (HTML plot skipped:', repr(e)[:120], ')')
print('\nCombine-model SHAP finished. Prediction correct for', int(R.correct.sum()), 'of 20 | max SHAP additivity error', R.shap_additivity_error.max())
pd.set_option('display.max_colwidth', 80, 'display.width', 250)
print(R[['test_row', 'true_label', 'predicted', 'correct', 'p_hate', 'top_words_toward_hate', 'top_words_toward_non_hate']].to_string(index=False))

# ---------------- Single (Study 1) vs Combine, same 20 comments ----------------
if S1_SAMPLE and S1_WORDS:
    from scipy.stats import spearmanr
    S = pd.read_csv(S1_SAMPLE[0]); SW = pd.read_csv(S1_WORDS[0])
    assert list(S.test_row) == IDX, 'Study 1 sample rows differ'
    comp = []
    for i in IDX:
        s = S[S.test_row == i].iloc[0]; c = R[R.test_row == i].iloc[0]
        a = SW[SW.test_row == i].sort_values('position'); b = W[W.test_row == i].sort_values('position')
        same_words = list(a.word.astype(str)) == list(b.word.astype(str))
        rho = spearmanr(a.shap_value_hate.values, b.shap_value_hate.values)[0] if same_words and len(a) > 2 else np.nan
        ta = set(a.nlargest(3, 'shap_value_hate').word.astype(str)); tb = set(b.nlargest(3, 'shap_value_hate').word.astype(str))
        comp.append(dict(test_row=i, true_label=int(c.true_label), pred_single=int(s.predicted), pred_combine=int(c.predicted), p_hate_single=s.p_hate, p_hate_combine=c.p_hate,
                         same_prediction=bool(s.predicted == c.predicted), n_words=len(a), same_tokenisation=same_words, spearman_word_values=None if np.isnan(rho) else round(float(rho), 3),
                         top3_overlap=len(ta & tb), top3_single=sorted(ta), top3_combine=sorted(tb)))
    C = pd.DataFrame(comp); C.to_csv(f'{OUT}/shap_single_vs_combine_20.csv', index=False)
    print('\n=== SINGLE (Study 1) vs COMBINE, the same 20 comments ===')
    print(C[['test_row', 'true_label', 'pred_single', 'pred_combine', 'p_hate_single', 'p_hate_combine', 'spearman_word_values', 'top3_overlap']].to_string(index=False))
    print(f'same prediction in {int(C.same_prediction.sum())}/20 | Single correct {int((C.pred_single==C.true_label).sum())}/20, Combine correct {int((C.pred_combine==C.true_label).sum())}/20')
    print(f'mean p(hate) on the 10 hate comments: Single {C[C.true_label==1].p_hate_single.mean():.3f} -> Combine {C[C.true_label==1].p_hate_combine.mean():.3f}')
    print(f'mean p(hate) on the 10 non-hate comments: Single {C[C.true_label==0].p_hate_single.mean():.3f} -> Combine {C[C.true_label==0].p_hate_combine.mean():.3f}')
    print(f'median Spearman of word values (Single vs Combine) = {C.spearman_word_values.dropna().astype(float).median():.3f} | mean top-3 word overlap = {C.top3_overlap.mean():.2f} of 3')
else: print('\n(shap_results.zip of Study 1 not uploaded -> comparison table skipped; run again with it, or send both result zips)')

# ---------------- package ----------------
import transformers, shap as _s
json.dump(dict(model='Combine model: banglabert_norm, regime B (merged), seed 42, re-trained for SHAP', gpu=torch.cuda.get_device_name(0), torch=torch.__version__,
               transformers=transformers.__version__, shap=_s.__version__, python=platform.python_version(), date=time.strftime('%Y-%m-%d %H:%M:%S'),
               best_epoch=kb.epoch, sample_rows=IDX, data_hashes={k: sha16(f'{D2}/{k}.csv') for k in EXP_S2}, study1_test_hash=sha16(f'{D1}/test.csv')),
          open(f'{OUT}/shap_combine_info.json', 'w'), indent=2)
zp = '/content/shap_combine_results.zip'
with zipfile.ZipFile(zp, 'w', zipfile.ZIP_DEFLATED) as z:
    for f in sorted(glob.glob(f'{OUT}/*')): z.write(f, os.path.basename(f))
print(f'\n[saved] {len(os.listdir(OUT))} files -> {zp}'); files.download(zp)
