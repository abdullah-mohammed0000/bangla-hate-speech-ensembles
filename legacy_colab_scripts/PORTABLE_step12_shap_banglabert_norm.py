# ============================================================================
# STEP 12  --  SHAP on BanglaBERT + normalizer (seed 42)        GPU (T4), any Colab account, no Drive
#   1. Runtime > Change runtime type > T4 GPU        2. paste this WHOLE file in ONE cell and run
#   3. upload archive__3_.zip when asked              4. at the end shap_results.zip downloads automatically
#   What it does: re-trains BanglaBERT+normalizer (seed 42, identical settings to the final run, ~10 min),
#   checks that its test score matches the run already in the results (0.9246), picks 20 test comments
#   (10 hate + 10 non-hate, random, fixed seed), and explains each one with SHAP (word level).
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
REF_MACRO_F1 = 0.9246          # banglabert_norm seed 42 in the final run (for the comparison printout only)
OUT = '/content/shap_out'; shutil.rmtree(OUT, ignore_errors=True); os.makedirs(OUT)
EXPECTED = {'train': ('116c9e9eb7432dff', 40224), 'val': ('67995e80777818a1', 5028), 'test': ('4543a005555ee466', 5029)}

# ---------------- data (upload, hash-checked) ----------------
from google.colab import files
DATA = '/content/bdshs'; os.makedirs(DATA, exist_ok=True)
def have_all(): return all(os.path.exists(f'{DATA}/{k}.csv') for k in EXPECTED)
if not have_all():
    print('Upload archive__3_.zip')
    for name, content in files.upload().items():
        p = f'/content/_up_{name}'; open(p, 'wb').write(content)
        if name.lower().endswith('.zip'):
            with zipfile.ZipFile(p) as z: z.extractall('/content/_up_zip')
    for k in EXPECTED:
        hits = glob.glob(f'/content/_up_zip/**/{k}.csv', recursive=True)
        if hits: shutil.copy(hits[0], f'{DATA}/{k}.csv')
assert have_all(), 'train/val/test CSV not found'
def sha16(p): return hashlib.sha256(open(p, 'rb').read()).hexdigest()[:16]
for k, (h, n) in EXPECTED.items():
    raw = pd.read_csv(f'{DATA}/{k}.csv'); ok = (sha16(f'{DATA}/{k}.csv') == h) and (len(raw) == n)
    print(f'{k:5s} rows={len(raw):6d} sha256[:16]={sha16(f"{DATA}/{k}.csv")} ->', 'OK' if ok else '*** MISMATCH ***'); assert ok, 'Dataset differs - STOP'
def clean_text(text):
    text = str(text)
    text = re.sub(r'http\S+|www\S+', '', text)
    text = re.sub(r'[^ঀ-৿\s\w.,!?।]', '', text)
    return re.sub(r'\s+', ' ', text).strip()
def load(k):
    d = pd.read_csv(f'{DATA}/{k}.csv').rename(columns={'sentence': 'text_raw', 'hate speech': 'label'})
    d['text'] = d['text_raw'].map(clean_text); d['label'] = d['label'].astype(int)
    d['text_norm'] = [bn_normalize(t) for t in d.text]            # normalizer applied AFTER clean_text (as in the final run)
    return d[['text_raw', 'text', 'text_norm', 'label']].reset_index(drop=True)
train, val, test = load('train'), load('val'), load('test'); y_test = test.label.values
print({'train': len(train), 'val': len(val), 'test': len(test)})

# ---------------- training: identical to the final run (BanglaBERT + normalizer, weighted loss, seed 42) ----------------
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
def mk(d):
    ds = Dataset.from_dict({'text': d.text_norm.tolist(), 'labels': d.label.tolist()})
    return ds.map(lambda b: tk(b['text'], truncation=True, max_length=MAX_LEN), batched=True, remove_columns=['text'])
dtr, dva, dte = mk(train), mk(val), mk(test)
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
def probs_of(ds):
    p = tr.predict(ds).predictions; p = p[0] if isinstance(p, tuple) else p
    return torch.softmax(torch.tensor(p).float(), 1).numpy()
P_test, P_val = probs_of(dte), probs_of(dva)
vf = f1_score(val.label.values, P_val.argmax(1), average='macro'); assert abs(vf - kb.best) < 1e-4, 'best-epoch restore mismatch!'
mf = f1_score(y_test, P_test.argmax(1), average='macro')
print(f'   best epoch = {kb.epoch} | logged val F1 {kb.best:.4f} | re-computed val F1 {vf:.4f} | time {(time.time()-t0)/60:.1f} min')
print(f'   test macro-F1 of THIS re-trained model = {mf:.4f}   (run already in the results: {REF_MACRO_F1:.4f}; GPU non-determinism can move it by about 0.001-0.003)')
np.save(f'{OUT}/probs_shapmodel_banglabert_norm_s42.npy', P_test)

# ---------------- SHAP ----------------
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

def explain_and_save(predict_fn, test, y_test, P_test, OUT):
    # 20 test comments: N_PER_CLASS hate + N_PER_CLASS non-hate, random with a fixed seed (stratified by TRUE label)
    rng = np.random.default_rng(SAMPLE_SEED)
    idx = sorted(int(i) for i in list(rng.choice(np.where(y_test == 1)[0], N_PER_CLASS, replace=False)) + list(rng.choice(np.where(y_test == 0)[0], N_PER_CLASS, replace=False)))
    json.dump(dict(test_row_indices=idx, rule=f'{N_PER_CLASS} hate + {N_PER_CLASS} non-hate test comments, random (numpy default_rng seed {SAMPLE_SEED}), stratified by true label',
                   note='row index = 0-based position in test.csv; reuse the same indices for any later SHAP run'), open(f'{OUT}/sample_indices.json', 'w'), indent=2)
    texts = [test.text_norm[i] for i in idx]
    # word-level SHAP (whitespace words). The mask token carries a trailing space so that a masked word does not glue to the next word.
    masker = shap.maskers.Text(SimpleTokenizer(r'\s+'), mask_token='[MASK] ', collapse_mask_token=False)
    explainer = shap.Explainer(predict_fn, masker, algorithm='permutation', output_names=['non-hate', 'hate'], seed=0)
    sv = explainer(texts, max_evals=MAX_EVALS, batch_size=64)
    rows, longrows = [], []
    for j, i in enumerate(idx):
        words = [str(w).strip() for w in sv.data[j]]; vals = sv.values[j][:, 1]; base = float(sv.base_values[j][1])
        fx = float(predict_fn([texts[j]])[0][1]); pred = int(P_test[i].argmax()); add_err = abs(base + vals.sum() - fx)
        order = np.argsort(-vals); pos = [(words[k], round(float(vals[k]), 4)) for k in order if vals[k] > 0][:5]
        neg = [(words[k], round(float(vals[k]), 4)) for k in order[::-1] if vals[k] < 0][:5]
        rows.append(dict(test_row=i, true_label=int(y_test[i]), predicted=pred, correct=bool(pred == y_test[i]), p_hate=round(float(P_test[i][1]), 4),
                         base_value_p_hate=round(base, 4), shap_additivity_error=round(add_err, 4), n_words=len(words),
                         top_words_toward_hate=pos, top_words_toward_non_hate=neg, text_raw=test.text_raw[i], text_model_input=texts[j]))
        for k, (w, v) in enumerate(zip(words, vals)): longrows.append(dict(test_row=i, position=k, word=w, shap_value_hate=round(float(v), 5)))
    R = pd.DataFrame(rows); R.to_csv(f'{OUT}/shap_sample_20.csv', index=False); pd.DataFrame(longrows).to_csv(f'{OUT}/shap_word_attributions.csv', index=False)
    try:
        html = shap.plots.text(sv[:, :, 1], display=False); open(f'{OUT}/shap_text_plots.html', 'w', encoding='utf-8').write(html)
    except Exception as e: print('   (HTML plot skipped:', repr(e)[:120], ')')
    return R
R = explain_and_save(predict_fn, test, y_test, P_test, OUT)
print('\nSHAP finished. Sample = 10 hate + 10 non-hate test comments; prediction correct for', int(R.correct.sum()), 'of', len(R))
print('max SHAP additivity error (base + sum of word values vs model output):', R.shap_additivity_error.max())
pd.set_option('display.max_colwidth', 90, 'display.width', 250)
print(R[['test_row', 'true_label', 'predicted', 'correct', 'p_hate', 'top_words_toward_hate', 'top_words_toward_non_hate']].to_string(index=False))

# ---------------- package ----------------
import transformers, shap as _s
json.dump(dict(model='banglabert_norm seed 42 (re-trained for SHAP)', gpu=torch.cuda.get_device_name(0), torch=torch.__version__, transformers=transformers.__version__,
               shap=_s.__version__, python=platform.python_version(), date=time.strftime('%Y-%m-%d %H:%M:%S'), retrained_test_macro_F1=round(float(mf), 4),
               reference_macro_F1=REF_MACRO_F1, best_epoch=kb.epoch, data_hashes={k: sha16(f'{DATA}/{k}.csv') for k in EXPECTED}), open(f'{OUT}/shap_info.json', 'w'), indent=2)
zp = '/content/shap_results.zip'
with zipfile.ZipFile(zp, 'w', zipfile.ZIP_DEFLATED) as z:
    for f in sorted(glob.glob(f'{OUT}/*')): z.write(f, os.path.basename(f))
print(f'\n[saved] {len(os.listdir(OUT))} files -> {zp}'); files.download(zp)
