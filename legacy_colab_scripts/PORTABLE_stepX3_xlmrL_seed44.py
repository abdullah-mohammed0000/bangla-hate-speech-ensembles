# ============================================================================
# STEP X3 (seed 44)  --  XLM-R-LARGE on BD-SHS single corpus   [PORTABLE, T4 GPU, any Colab account, no Drive]
#   Goal: a stronger single model than BanglaBERT + normalizer (base), same data / cleaning / protocol.
#   1. Runtime > Change runtime type > T4 GPU
#   2. Paste this WHOLE file in ONE cell, run. Upload archive__3_.zip (BD-SHS) when asked.
#   3. After EVERY seed a zip  job_xlmr_large_s44.zip downloads -> keep it (needed by step X4).
#   Time on a free T4: about 45-60 min for xlmr_large (one seed).
#   If Colab limits are a problem: run ONE seed per account, e.g. SEEDS = [42] here, [43] in a 2nd account, [44] in a 3rd.
#   Differences to the base runs (because the models are 3x larger): LR 1e-5 (instead of 2e-5), 10% warm-up,
#   batch 16 x 2 gradient-accumulation steps (= 32, same effective batch). Everything else identical.
# ============================================================================
JOB_MODELS = ['xlmr_large']   # XLM-RoBERTa-large (560M parameters)
SEEDS      = [44]                    # THIS FILE = seed 44 only (other seeds: separate files)

!pip -q install shap sentencepiece ftfy regex
# csebuetnlp normalizer (BanglaBERT authors' text normalizer). Its pinned old deps fail to build on new Python,
# so: clone the repo, use modern ftfy, and stub the unused 'emoji' module (emoji handling is OFF by default).
!rm -rf /content/nz && git clone -q https://github.com/csebuetnlp/normalizer /content/nz
import sys, types, re
_m = types.ModuleType('emoji'); _m.get_emoji_regexp = lambda: re.compile('(?!x)x'); sys.modules['emoji'] = _m
sys.path.insert(0, '/content/nz')
from normalizer import normalize as bn_normalize
print('normalizer test:', repr(bn_normalize('আমি  বাংলায় গান গাই!!')))
import os, re, json, random, hashlib, time, glob, zipfile, warnings, shutil, gc, platform
import numpy as np, pandas as pd, torch, torch.nn as nn
warnings.filterwarnings('ignore')
assert torch.cuda.is_available(), 'No GPU! Runtime > Change runtime type > T4 GPU, then run again.'
print('GPU:', torch.cuda.get_device_name(0), '| torch', torch.__version__)

MAX_LEN, EPOCHS, LR, WD, BATCH_TR, BATCH_EV, GRAD_ACC, WARMUP = 128, 3, 1e-5, 0.01, 16, 64, 2, 0.1
OUT = '/content/job_results'; os.makedirs(OUT, exist_ok=True)
MODELS = {
    'mbert':          dict(hf='bert-base-multilingual-cased', weighted=True,  label='mBERT'),
    'xlmr':           dict(hf='xlm-roberta-base',             weighted=True,  label='XLM-RoBERTa'),
    'banglabert':     dict(hf='csebuetnlp/banglabert',        weighted=True,  label='BanglaBERT'),
    'banglabert_unw': dict(hf='csebuetnlp/banglabert',        weighted=False, label='BanglaBERT (unweighted)'),
    'indicbertv2':    dict(hf='ai4bharat/IndicBERTv2-MLM-only', weighted=True, label='IndicBERTv2'),
    'banglabert_norm': dict(hf='csebuetnlp/banglabert',       weighted=True,  label='BanglaBERT (+normalizer)', norm=True),
    'banglabert_large_norm': dict(hf='csebuetnlp/banglabert_large', weighted=True, label='BanglaBERT-large (+normalizer)', norm=True),
    'xlmr_large':     dict(hf='xlm-roberta-large',            weighted=True,  label='XLM-RoBERTa-large'),
}
EXPECTED = {'train': ('116c9e9eb7432dff', 40224), 'val': ('67995e80777818a1', 5028), 'test': ('4543a005555ee466', 5029)}
JOB_TAG = '_'.join(JOB_MODELS)

def set_seed(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed_all(s)
    torch.backends.cudnn.deterministic = True; torch.backends.cudnn.benchmark = False

# ---------------- data (upload) ----------------
from google.colab import files
DATA = '/content/bdshs'; os.makedirs(DATA, exist_ok=True)
def have_all(): return all(os.path.exists(f'{DATA}/{k}.csv') for k in EXPECTED)
if not have_all():
    print('Upload archive__3_.zip  (or train.csv, val.csv, test.csv)')
    for name, content in files.upload().items():
        p = f'/content/_up_{name}'; open(p, 'wb').write(content)
        if name.lower().endswith('.zip'):
            with zipfile.ZipFile(p) as z: z.extractall('/content/_up_zip')
    for k in EXPECTED:
        hits = glob.glob(f'/content/_up_{k}.csv') + glob.glob(f'/content/_up_zip/**/{k}.csv', recursive=True)
        if hits: shutil.copy(hits[0], f'{DATA}/{k}.csv')
assert have_all(), 'train/val/test CSV not found'
def sha16(p): return hashlib.sha256(open(p, 'rb').read()).hexdigest()[:16]
for k, (h, n) in EXPECTED.items():
    raw = pd.read_csv(f'{DATA}/{k}.csv'); ok = (sha16(f'{DATA}/{k}.csv') == h) and (len(raw) == n)
    print(f'{k:5s} rows={len(raw):6d} sha256[:16]={sha16(f"{DATA}/{k}.csv")} ->', 'OK' if ok else '*** MISMATCH ***')
    assert ok, 'Dataset differs from the thesis dataset - STOP'
def clean_text(text):
    text = str(text)
    text = re.sub(r'http\S+|www\S+', '', text)
    text = re.sub(r'[^\u0980-\u09FF\s\w.,!?\u0964]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text
def load(k):
    d = pd.read_csv(f'{DATA}/{k}.csv').rename(columns={'sentence': 'text_raw', 'hate speech': 'label'})
    d['text'] = d['text_raw'].map(clean_text); d['label'] = d['label'].astype(int)
    return d[['text_raw', 'text', 'label']].reset_index(drop=True)
train, val, test = load('train'), load('val'), load('test')
y_test = test.label.values
print({'train': len(train), 'val': len(val), 'test': len(test)})

# ---------------- metrics / helpers ----------------
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, f1_score, precision_score, recall_score
def metrics(y, p):
    mp, mr, mf, _ = precision_recall_fscore_support(y, p, average='macro', zero_division=0)
    return dict(Accuracy=accuracy_score(y, p), Macro_P=mp, Macro_R=mr, Macro_F1=mf, Weighted_F1=f1_score(y, p, average='weighted'),
                Hate_P=precision_score(y, p, pos_label=1, zero_division=0), Hate_R=recall_score(y, p, pos_label=1, zero_division=0),
                Hate_F1=f1_score(y, p, pos_label=1, zero_division=0))
def save_probs(name, seed, probs): np.save(f'{OUT}/probs_{name}_s{seed}.npy', probs)
def have(name, seed): return os.path.exists(f'{OUT}/probs_{name}_s{seed}.npy')
RUN_LOG = f'{OUT}/results_job_{JOB_TAG}.csv'
def log_run(name, seed, probs):
    m = metrics(y_test, probs.argmax(1)); m.update(Model=name, Seed=seed)
    row = pd.DataFrame([m])[['Model', 'Seed'] + [c for c in m if c not in ('Model', 'Seed')]]
    row.to_csv(RUN_LOG, mode='a', header=not os.path.exists(RUN_LOG), index=False)
    print(name, 'seed', seed, {k: round(v, 4) for k, v in m.items() if k not in ('Model', 'Seed')})

# ---------------- training (identical to the main run) ----------------
from transformers import (AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer, TrainerCallback)
from datasets import Dataset
CW = torch.tensor(len(train) / (2 * np.bincount(train.label.values)), dtype=torch.float)
class WeightedTrainer(Trainer):
    use_w = True
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.pop('labels'); out = model(**inputs)
        w = CW.to(out.logits.device) if self.use_w else None
        loss = nn.CrossEntropyLoss(weight=w)(out.logits.float(), labels)
        return (loss, out) if return_outputs else loss
class KeepBest(TrainerCallback):
    def __init__(self): self.best, self.state, self.epoch = -1, None, 0
    def on_evaluate(self, args, state, control, metrics=None, model=None, **kw):
        f = (metrics or {}).get('eval_f1', -1)
        if f > self.best and model is not None:
            self.best, self.epoch = f, int(round(state.epoch))
            self.state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
def run_transformer(name, seed):
    cfg = MODELS[name]; hf = cfg['hf']; set_seed(seed); t0 = time.time()
    tk = AutoTokenizer.from_pretrained(hf)
    def mk(d):
        texts = [bn_normalize(t) for t in d.text] if cfg.get('norm') else d.text.tolist()   # normalizer applied AFTER clean_text
        ds = Dataset.from_dict({'text': texts, 'labels': d.label.tolist()})
        return ds.map(lambda b: tk(b['text'], truncation=True, max_length=MAX_LEN), batched=True, remove_columns=['text'])
    dtr, dva, dte = mk(train), mk(val), mk(test)
    model = AutoModelForSequenceClassification.from_pretrained(hf, num_labels=2)
    WARM_STEPS = int(WARMUP * EPOCHS * -(-len(dtr) // (BATCH_TR * GRAD_ACC)))   # 10% of all optimizer steps (warmup_ratio was removed in transformers 5)
    args = TrainingArguments(output_dir=f'/content/ckpt/{name}_s{seed}', num_train_epochs=EPOCHS, learning_rate=LR, weight_decay=WD,
        per_device_train_batch_size=BATCH_TR, per_device_eval_batch_size=BATCH_EV, gradient_accumulation_steps=GRAD_ACC, warmup_steps=WARM_STEPS, eval_strategy='epoch', save_strategy='no',
        load_best_model_at_end=False, seed=seed, data_seed=seed, fp16=True, report_to='none', disable_tqdm=True, logging_strategy='epoch')
    def cm(ep):
        p = ep.predictions[0] if isinstance(ep.predictions, tuple) else ep.predictions
        return {'f1': f1_score(ep.label_ids, p.argmax(-1), average='macro')}
    kb = KeepBest()
    tr = WeightedTrainer(model=model, args=args, train_dataset=dtr, eval_dataset=dva, processing_class=tk, compute_metrics=cm, callbacks=[kb])
    tr.use_w = cfg['weighted']; tr.train(); tr.model.load_state_dict(kb.state)
    ep_log = [dict(Model=name, Seed=seed, Epoch=int(round(h['epoch'])), Val_Macro_F1=round(h['eval_f1'], 4)) for h in tr.state.log_history if 'eval_f1' in h]
    pd.DataFrame(ep_log).to_csv(f'{OUT}/epochlog_{name}_s{seed}.csv', index=False)
    def probs(ds):
        p = tr.predict(ds).predictions; p = p[0] if isinstance(p, tuple) else p
        return torch.softmax(torch.tensor(p).float(), 1).numpy()
    pr, pv = probs(dte), probs(dva)
    np.save(f'{OUT}/valprobs_{name}_s{seed}.npy', pv)
    vf = f1_score(val.label.values, pv.argmax(1), average='macro')
    print(f'   best epoch = {kb.epoch} | logged val F1 {kb.best:.4f} | re-computed val F1 {vf:.4f} | time {(time.time()-t0)/60:.1f} min')
    assert abs(vf - kb.best) < 1e-4, 'best-epoch restore mismatch!'
    save_probs(name, seed, pr); log_run(name, seed, pr)
    del tr, model, kb; gc.collect(); torch.cuda.empty_cache()

def package(tag):
    import transformers, sklearn
    json.dump(dict(job=JOB_MODELS, seeds=SEEDS, max_len=MAX_LEN, epochs=EPOCHS, lr=LR, wd=WD, batch=BATCH_TR, grad_acc=GRAD_ACC, warmup=WARMUP, gpu=torch.cuda.get_device_name(0),
                   torch=torch.__version__, transformers=transformers.__version__, sklearn=sklearn.__version__, python=platform.python_version(),
                   date=time.strftime('%Y-%m-%d %H:%M:%S'), data_hashes={k: sha16(f'{DATA}/{k}.csv') for k in EXPECTED}),
              open(f'{OUT}/job_info_{JOB_TAG}.json', 'w'), indent=2)
    zp = f'/content/job_{JOB_TAG}{tag}.zip'
    with zipfile.ZipFile(zp, 'w', zipfile.ZIP_DEFLATED) as z:
        for f in glob.glob(f'{OUT}/*'): z.write(f, os.path.basename(f))
    print(f'[packaged] {zp}  ({len(os.listdir(OUT))} files)'); files.download(zp)

# ---------------- run the job; package after EVERY model so nothing is lost ----------------
for name in JOB_MODELS:
    for s in SEEDS:
        if have(name, s): print('skip (already done):', name, s); continue
        run_transformer(name, s)
        package(f'_s{s}')    # downloads job_<name>_s<seed>.zip after EVERY seed, so nothing is lost
rr = pd.read_csv(RUN_LOG)
print('\nJOB DONE.'); display(rr.round(4))
print(rr.groupby('Model')[['Accuracy', 'Macro_F1', 'Weighted_F1', 'Hate_P', 'Hate_R']].agg(['mean', 'std']).round(4))
print('Reference (same test set): BanglaBERT + normalizer base = Accuracy 0.9236, Macro-F1 0.9236, Weighted-F1 0.9237')
