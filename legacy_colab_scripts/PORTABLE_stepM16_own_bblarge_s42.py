# ============================================================================
# STEP M16 (Own-only, BanglaBERT-large + normalizer, seed 42)   [PORTABLE, T4 GPU]
#   NEW regime O = trained on the OWN dataset only (its part of the merged training set, 20,990 comments;
#   validation = its part of the merged validation set, 2,632 comments). Tested on ALL test sets
#   (own-dataset test 2,636 = main result; also BD-SHS, BanglaHateBERT, BanHate and merged test).
#   Needs only merged_splits.zip (from step M1). Any Colab account, no Drive.
#   1. Runtime > Change runtime type > T4 GPU
#   2. Paste this WHOLE file in ONE cell, run. Upload merged_splits.zip when asked.
#   3. At the end  m16_banglabert_large_norm_O_s42.zip  downloads -> keep it (needed by step M17).
#   Time on a free T4: about ~15 min.
#   Settings: large-model settings (LR 1e-5, 10% warm-up, batch 16 x 2).
# ============================================================================
JOB_MODELS = ['banglabert_large_norm']   # PRESET - do not change
REGIMES    = ['O']                     # O = own dataset only
SEEDS      = [42]                      # PRESET for this file

!pip -q install sentencepiece ftfy regex
import os, re, sys, types, json, random, hashlib, time, glob, zipfile, warnings, shutil, gc, platform
import numpy as np, pandas as pd, torch, torch.nn as nn
warnings.filterwarnings('ignore')
assert torch.cuda.is_available(), 'No GPU! Runtime > Change runtime type > T4 GPU, then run again.'
print('GPU:', torch.cuda.get_device_name(0), '| torch', torch.__version__)

MODELS = {
    'mbert':          dict(hf='bert-base-multilingual-cased', weighted=True),
    'xlmr':           dict(hf='xlm-roberta-base',             weighted=True),
    'banglabert':     dict(hf='csebuetnlp/banglabert',        weighted=True),
    'banglabert_unw': dict(hf='csebuetnlp/banglabert',        weighted=False),
    'indicbertv2':    dict(hf='ai4bharat/IndicBERTv2-MLM-only', weighted=True),
    'banglabert_norm': dict(hf='csebuetnlp/banglabert',       weighted=True, norm=True),
    'banglabert_large_norm': dict(hf='csebuetnlp/banglabert_large', weighted=True, norm=True),
}
bn_normalize = None
if any(MODELS[m].get('norm') for m in JOB_MODELS):
    # csebuetnlp normalizer: stub the unused 'emoji' module (emoji handling is OFF by default), use modern ftfy
    !rm -rf /content/nz && git clone -q https://github.com/csebuetnlp/normalizer /content/nz
    _m = types.ModuleType('emoji'); _m.get_emoji_regexp = lambda: re.compile('(?!x)x'); sys.modules['emoji'] = _m
    sys.path.insert(0, '/content/nz')
    from normalizer import normalize as bn_normalize
    print('normalizer test:', repr(bn_normalize('আমি  বাংলায় গান গাই!!')))

MAX_LEN, EPOCHS, LR, WD, BATCH_TR, BATCH_EV, GRAD_ACC, WARMUP = 128, 3, 1e-5, 0.01, 16, 64, 2, 0.1
OUT = '/content/job_results'; os.makedirs(OUT, exist_ok=True)
JOB_TAG = '_'.join(JOB_MODELS) + '_' + ''.join(REGIMES) + '_s' + '_'.join(map(str, SEEDS))
EXPECTED = {'A_train': 'e2e765bc7028607e', 'A_val': 'be38f2bf02e901a3', 'B_train': '5c3d0534bc182c8b', 'B_val': '49a1c67ca0e7fcec',
            'test_all': '5c26e952999ca26c', 'test_bdshs': 'a49a667d635cc6bf', 'test_banglahatebert': '360305cc8e870e4d',
            'test_banhate': 'da22d1b1ca76af7d', 'test_own': '873067fbbc95698d'}
ROWS = {'A_train': 40224, 'A_val': 5028, 'B_train': 86822, 'B_val': 11059, 'test_all': 12570,
        'test_bdshs': 5029, 'test_banglahatebert': 3001, 'test_banhate': 1908, 'test_own': 2636}
PARTS = ['test_bdshs', 'test_banglahatebert', 'test_banhate', 'test_own', 'test_all']

def set_seed(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed_all(s)
    torch.backends.cudnn.deterministic = True; torch.backends.cudnn.benchmark = False

# ---------------- data (upload merged_splits.zip) ----------------
from google.colab import files
DATA = '/content/splits'; os.makedirs(DATA, exist_ok=True)
def have_all(): return all(os.path.exists(f'{DATA}/{k}.csv') for k in EXPECTED)
if not have_all():
    print('Upload merged_splits.zip')
    for name, content in files.upload().items():
        p = f'/content/_up_{name}'; open(p, 'wb').write(content)
        with zipfile.ZipFile(p) as z: z.extractall('/content/_up_zip')
    for k in EXPECTED:
        hits = glob.glob(f'/content/_up_zip/**/{k}.csv', recursive=True)
        if hits: shutil.copy(hits[0], f'{DATA}/{k}.csv')
assert have_all(), 'some split CSV not found in the zip'
def sha16(p): return hashlib.sha256(open(p, 'rb').read()).hexdigest()[:16]
for k, h in EXPECTED.items():
    raw = pd.read_csv(f'{DATA}/{k}.csv'); ok = (sha16(f'{DATA}/{k}.csv') == h) and (len(raw) == ROWS[k])
    print(f'{k:20s} rows={len(raw):6d} sha256[:16]={sha16(f"{DATA}/{k}.csv")} ->', 'OK' if ok else '*** MISMATCH ***')
    assert ok, 'Split file differs from the one made in step M1 - STOP'
def clean_text(text):          # identical to Study 1 (already applied in M1 for dedup; re-applied here, it is idempotent)
    text = str(text)
    text = re.sub(r'http\S+|www\S+', '', text)
    text = re.sub(r'[^ঀ-৿\s\w.,!?।]', '', text)
    return re.sub(r'\s+', ' ', text).strip()
def load(k):
    d = pd.read_csv(f'{DATA}/{k}.csv'); d['text'] = d['text'].fillna('').map(clean_text); d['label'] = d['label'].astype(int)
    return d[['source', 'orig_idx', 'text', 'label']].reset_index(drop=True)
D = {k: load(k) for k in EXPECTED}
# regime O: the OWN-dataset part of the merged training / validation sets (already de-duplicated and leak-free in step M1)
D['O_train'] = D['B_train'][D['B_train'].source == 'Own'].reset_index(drop=True)
D['O_val']   = D['B_val'][D['B_val'].source == 'Own'].reset_index(drop=True)
print('regime O (own only): train', len(D['O_train']), '| val', len(D['O_val']))
assert len(D['O_train']) == 20990 and len(D['O_val']) == 2632, 'unexpected own-dataset sizes - STOP'

from sklearn.metrics import accuracy_score, precision_recall_fscore_support, f1_score, precision_score, recall_score
def metrics(y, p):
    mp, mr, mf, _ = precision_recall_fscore_support(y, p, average='macro', zero_division=0)
    return dict(Accuracy=accuracy_score(y, p), Macro_P=mp, Macro_R=mr, Macro_F1=mf, Weighted_F1=f1_score(y, p, average='weighted'),
                Hate_P=precision_score(y, p, pos_label=1, zero_division=0), Hate_R=recall_score(y, p, pos_label=1, zero_division=0),
                Hate_F1=f1_score(y, p, pos_label=1, zero_division=0))
def have(name, reg, seed): return os.path.exists(f'{OUT}/probs_{name}_{reg}_s{seed}_test_all.npy')
RUN_LOG = f'{OUT}/results_{JOB_TAG}.csv'

from transformers import (AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer, TrainerCallback)
from datasets import Dataset
class WeightedTrainer(Trainer):
    use_w = True; CW = None
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.pop('labels'); out = model(**inputs)
        w = self.CW.to(out.logits.device) if self.use_w else None
        loss = nn.CrossEntropyLoss(weight=w)(out.logits.float(), labels)
        return (loss, out) if return_outputs else loss
class KeepBest(TrainerCallback):
    def __init__(self): self.best, self.state, self.epoch = -1, None, 0
    def on_evaluate(self, args, state, control, metrics=None, model=None, **kw):
        f = (metrics or {}).get('eval_f1', -1)
        if f > self.best and model is not None:
            self.best, self.epoch = f, int(round(state.epoch))
            self.state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

def run_transformer(name, reg, seed):
    cfg = MODELS[name]; hf = cfg['hf']; set_seed(seed); t0 = time.time()
    train, val = D[f'{reg}_train'], D[f'{reg}_val']
    tk = AutoTokenizer.from_pretrained(hf)
    def mk(d):
        texts = [bn_normalize(t) for t in d.text] if cfg.get('norm') else d.text.tolist()   # normalizer AFTER clean_text
        ds = Dataset.from_dict({'text': texts, 'labels': d.label.tolist()})
        return ds.map(lambda b: tk(b['text'], truncation=True, max_length=MAX_LEN), batched=True, remove_columns=['text'])
    dtr, dva = mk(train), mk(val)
    model = AutoModelForSequenceClassification.from_pretrained(hf, num_labels=2)
    WARM_STEPS = int(WARMUP * EPOCHS * -(-len(dtr) // (BATCH_TR * GRAD_ACC)))   # 10% of all optimizer steps
    args = TrainingArguments(output_dir=f'/content/ckpt/{name}_{reg}_s{seed}', num_train_epochs=EPOCHS, learning_rate=LR, weight_decay=WD,
        per_device_train_batch_size=BATCH_TR, per_device_eval_batch_size=BATCH_EV, gradient_accumulation_steps=GRAD_ACC, warmup_steps=WARM_STEPS, eval_strategy='epoch', save_strategy='no',
        load_best_model_at_end=False, seed=seed, data_seed=seed, fp16=True, report_to='none', disable_tqdm=True, logging_strategy='epoch')
    def cm(ep):
        p = ep.predictions[0] if isinstance(ep.predictions, tuple) else ep.predictions
        return {'f1': f1_score(ep.label_ids, p.argmax(-1), average='macro')}
    kb = KeepBest()
    tr = WeightedTrainer(model=model, args=args, train_dataset=dtr, eval_dataset=dva, processing_class=tk, compute_metrics=cm, callbacks=[kb])
    tr.CW = torch.tensor(len(train) / (2 * np.bincount(train.label.values)), dtype=torch.float)   # class weights from THIS training set
    tr.use_w = cfg['weighted']; tr.train(); tr.model.load_state_dict(kb.state)
    pd.DataFrame([dict(Model=name, Regime=reg, Seed=seed, Epoch=int(round(h['epoch'])), Val_Macro_F1=round(h['eval_f1'], 4))
                  for h in tr.state.log_history if 'eval_f1' in h]).to_csv(f'{OUT}/epochlog_{name}_{reg}_s{seed}.csv', index=False)
    def probs(ds):
        p = tr.predict(ds).predictions; p = p[0] if isinstance(p, tuple) else p
        return torch.softmax(torch.tensor(p).float(), 1).numpy()
    pv = probs(dva); vf = f1_score(val.label.values, pv.argmax(1), average='macro')
    print(f'   best epoch = {kb.epoch} | logged val F1 {kb.best:.4f} | re-computed {vf:.4f} | train+val {(time.time()-t0)/60:.1f} min')
    assert abs(vf - kb.best) < 1e-4, 'best-epoch restore mismatch!'
    np.save(f'{OUT}/valprobs_{name}_{reg}_s{seed}.npy', pv)
    rows = []
    for part in PARTS:
        pr = probs(mk(D[part])); np.save(f'{OUT}/probs_{name}_{reg}_s{seed}_{part}.npy', pr)
        m = metrics(D[part].label.values, pr.argmax(1)); m.update(Model=name, Regime=reg, Seed=seed, Test=part); rows.append(m)
        print(f'   {part:20s} macro-F1 {m["Macro_F1"]:.4f}  acc {m["Accuracy"]:.4f}')
    r = pd.DataFrame(rows); r = r[['Model', 'Regime', 'Seed', 'Test'] + [c for c in r if c not in ('Model', 'Regime', 'Seed', 'Test')]]
    r.to_csv(RUN_LOG, mode='a', header=not os.path.exists(RUN_LOG), index=False)
    del tr, model, kb; gc.collect(); torch.cuda.empty_cache()

def package():
    import transformers, sklearn
    json.dump(dict(models=JOB_MODELS, regimes=REGIMES, seeds=SEEDS, max_len=MAX_LEN, epochs=EPOCHS, lr=LR, wd=WD, batch=BATCH_TR, grad_acc=GRAD_ACC, warmup=WARMUP, gpu=torch.cuda.get_device_name(0),
                   torch=torch.__version__, transformers=transformers.__version__, sklearn=sklearn.__version__, python=platform.python_version(),
                   date=time.strftime('%Y-%m-%d %H:%M:%S'), data_hashes={k: sha16(f'{DATA}/{k}.csv') for k in EXPECTED}),
              open(f'{OUT}/job_info_{JOB_TAG}.json', 'w'), indent=2)
    zp = f'/content/m16_{JOB_TAG}.zip'
    with zipfile.ZipFile(zp, 'w', zipfile.ZIP_DEFLATED) as z:
        for f in glob.glob(f'{OUT}/*'): z.write(f, os.path.basename(f))
    print(f'[packaged] {zp}  ({len(os.listdir(OUT))} files)'); files.download(zp)

# ---------------- run ----------------
for name in JOB_MODELS:
    for reg in REGIMES:
        for s in SEEDS:
            if have(name, reg, s): print('skip (already done):', name, reg, s); continue
            print(f'\n=== {name} | regime {reg} | seed {s} ==='); run_transformer(name, reg, s)
        package()
rr = pd.read_csv(RUN_LOG)
print('\nJOB DONE.')
print(rr.groupby(['Model', 'Regime', 'Test'])['Macro_F1'].agg(['mean', 'std']).round(4))
