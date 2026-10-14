# ============================================================================
# Study 2  --  SIZE-MATCHED CONTROL EXPERIMENT (regime C)  [PORTABLE]
#   Question: is the drop under merged training caused by MORE DATA or by DIFFERENT (heterogeneous) data?
#   C = BD-SHS only, bootstrap-resampled to 86,822 rows (same size as the merged set, no new source material).
#   Compare C with A (BD-SHS, 40,224) and B (merged, 86,822) on the SAME BD-SHS test set.
# Works in ANY Colab account, no Drive. Needs only merged_splits.zip (from step M1).
#   1. Runtime > Change runtime type > T4 GPU
#   2. Edit JOB_MODELS / REGIMES below, paste this WHOLE file in ONE cell, run.
#   3. Upload merged_splits.zip when asked.
#   4. After every (model, regime) a zip  m3_<models>_<regimes>.zip  downloads automatically (nothing is lost if Colab stops).
# Same cleaning, same hyper-parameters, same training code as Study 1. Only the data differ.
# ============================================================================
JOB_MODELS = ['mbert']               # control experiment; you may also run ['banglabert_norm'] in another account
REGIMES    = ['C']                   # C = SIZE-MATCHED CONTROL: BD-SHS train set re-sampled WITH replacement to the size of the merged train set (86,822)
SEEDS      = [42, 43, 44]

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
}
bn_normalize = None
if any(MODELS[m].get('norm') for m in JOB_MODELS):
    # csebuetnlp normalizer: stub the unused 'emoji' module (emoji handling is OFF by default), use modern ftfy
    !rm -rf /content/nz && git clone -q https://github.com/csebuetnlp/normalizer /content/nz
    _m = types.ModuleType('emoji'); _m.get_emoji_regexp = lambda: re.compile('(?!x)x'); sys.modules['emoji'] = _m
    sys.path.insert(0, '/content/nz')
    from normalizer import normalize as bn_normalize
    print('normalizer test:', repr(bn_normalize('আমি  বাংলায় গান গাই!!')))

MAX_LEN, EPOCHS, LR, WD, BATCH_TR, BATCH_EV = 128, 3, 2e-5, 0.01, 32, 64
OUT = '/content/job_results'; os.makedirs(OUT, exist_ok=True)
JOB_TAG = '_'.join(JOB_MODELS) + '_' + ''.join(REGIMES)
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
    if reg == 'C':   # size-matched control: seed-specific bootstrap of A_train to the size of B_train; validation = A_val
        train = D['A_train'].sample(n=ROWS['B_train'], replace=True, random_state=seed).reset_index(drop=True); val = D['A_val']
    else: train, val = D[f'{reg}_train'], D[f'{reg}_val']
    tk = AutoTokenizer.from_pretrained(hf)
    def mk(d):
        texts = [bn_normalize(t) for t in d.text] if cfg.get('norm') else d.text.tolist()   # normalizer AFTER clean_text
        ds = Dataset.from_dict({'text': texts, 'labels': d.label.tolist()})
        return ds.map(lambda b: tk(b['text'], truncation=True, max_length=MAX_LEN), batched=True, remove_columns=['text'])
    dtr, dva = mk(train), mk(val)
    model = AutoModelForSequenceClassification.from_pretrained(hf, num_labels=2)
    args = TrainingArguments(output_dir=f'/content/ckpt/{name}_{reg}_s{seed}', num_train_epochs=EPOCHS, learning_rate=LR, weight_decay=WD,
        per_device_train_batch_size=BATCH_TR, per_device_eval_batch_size=BATCH_EV, eval_strategy='epoch', save_strategy='no',
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
    json.dump(dict(models=JOB_MODELS, regimes=REGIMES, seeds=SEEDS, max_len=MAX_LEN, epochs=EPOCHS, lr=LR, wd=WD, gpu=torch.cuda.get_device_name(0),
                   torch=torch.__version__, transformers=transformers.__version__, sklearn=sklearn.__version__, python=platform.python_version(),
                   date=time.strftime('%Y-%m-%d %H:%M:%S'), data_hashes={k: sha16(f'{DATA}/{k}.csv') for k in EXPECTED}),
              open(f'{OUT}/job_info_{JOB_TAG}.json', 'w'), indent=2)
    zp = f'/content/m3_{JOB_TAG}.zip'
    with zipfile.ZipFile(zp, 'w', zipfile.ZIP_DEFLATED) as z:
        for f in glob.glob(f'{OUT}/*'): z.write(f, os.path.basename(f))
    print(f'[packaged] {zp}  ({len(os.listdir(OUT))} files)'); files.download(zp)

# ---------------- run: B first, then A; package after every (model, regime) ----------------
for name in JOB_MODELS:
    for reg in REGIMES:
        for s in SEEDS:
            if have(name, reg, s): print('skip (already done):', name, reg, s); continue
            print(f'\n=== {name} | regime {reg} | seed {s} ==='); run_transformer(name, reg, s)
        package()
rr = pd.read_csv(RUN_LOG)
print('\nJOB DONE.')
print(rr.groupby(['Model', 'Regime', 'Test'])['Macro_F1'].agg(['mean', 'std']).round(4))
