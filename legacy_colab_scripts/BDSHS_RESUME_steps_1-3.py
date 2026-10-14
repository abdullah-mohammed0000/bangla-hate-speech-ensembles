# ============================================================================
# BD-SHS single-corpus FINAL run  --  RESUME script (Steps 1, 2, 3 in one go)
# Use this after Colab disconnects / you come back later.
#   1. Runtime > Change runtime type > T4 GPU
#   2. Paste each block below into its own cell, run top to bottom.
#   3. In BLOCK C upload your saved backup zip (optional) and archive__3_.zip
# MAX_LEN is 128 here (decision after truncation check, Step 3c).
# ============================================================================

# ===================== BLOCK A : install + GPU + versions =====================
!pip -q install shap sentencepiece
import torch, sys, platform, importlib
print('Python      :', sys.version.split()[0]); print('PyTorch     :', torch.__version__)
assert torch.cuda.is_available(), 'No GPU! Runtime > Change runtime type > T4 GPU, then run again.'
print('GPU         :', torch.cuda.get_device_name(0))
for name in ['transformers','tokenizers','accelerate','datasets','sklearn','numpy','pandas','scipy','statsmodels','shap','sentencepiece']:
    try:
        m = importlib.import_module(name); print(f'{name:13s}:', getattr(m, '__version__', '?'))
    except Exception as e: print(f'{name:13s}: NOT INSTALLED ({e})')

# ===================== BLOCK B : config (Step 2, MAX_LEN = 128) =====================
import os, re, json, random, hashlib, time, glob, zipfile, warnings, shutil, gc
import numpy as np, pandas as pd, torch
warnings.filterwarnings('ignore')

SEEDS    = [42, 43, 44]
MAX_LEN  = 128
EPOCHS   = 3
LR       = 2e-5
WD       = 0.01
BATCH_TR = 32
BATCH_EV = 64
N_BOOT   = 2000
ALPHA    = 0.05
OUT = '/content/results'; os.makedirs(OUT, exist_ok=True)

MODELS = {
    'mbert':          dict(hf='bert-base-multilingual-cased', weighted=True,  label='mBERT'),
    'xlmr':           dict(hf='xlm-roberta-base',             weighted=True,  label='XLM-RoBERTa'),
    'banglabert':     dict(hf='csebuetnlp/banglabert',        weighted=True,  label='BanglaBERT'),
    'banglabert_unw': dict(hf='csebuetnlp/banglabert',        weighted=False, label='BanglaBERT (unweighted)'),
    'indicbertv2':    dict(hf='ai4bharat/IndicBERTv2-MLM-only', weighted=True, label='IndicBERTv2'),
}
EXPECTED = {'train': ('116c9e9eb7432dff', 40224), 'val': ('67995e80777818a1', 5028), 'test': ('4543a005555ee466', 5029)}

def set_seed(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed_all(s)
    torch.backends.cudnn.deterministic = True; torch.backends.cudnn.benchmark = False

HP = pd.DataFrame([
    ['TF-IDF word + SVM', 'word 1-2 gram, max 50,000 feat., sublinear TF; LinearSVC C=1.0 (probability-calibrated, 3-fold)'],
    ['TF-IDF char + SVM', 'char_wb 2-5 gram, max 100,000 feat., sublinear TF; LinearSVC C=1.0 (probability-calibrated, 3-fold)'],
    ['BiLSTM', f'emb 128, BiLSTM 128, dropout 0.4, Adam lr 1e-3, batch 64, up to 8 epochs, best val macro-F1, max_len {MAX_LEN}'],
    ['Transformers', f'lr {LR}, weight decay {WD}, batch {BATCH_TR}, {EPOCHS} epochs, max_len {MAX_LEN}, fp16, best epoch by val macro-F1'],
    ['Loss', 'weighted cross-entropy (inverse class frequency); unweighted ablation for BanglaBERT'],
    ['Seeds', str(SEEDS)],
    ['Bootstrap', f'{N_BOOT} resamples, 95% percentile CI'],
    ['McNemar', 'continuity-corrected, Holm adjustment over all pairs'],
], columns=['Component', 'Setting'])
pd.set_option('display.max_colwidth', 200); display(HP); HP.to_csv(f'{OUT}/hyperparameters.csv', index=False)

import transformers, sklearn, scipy, statsmodels, platform
RUN_INFO = dict(date=time.strftime('%Y-%m-%d %H:%M:%S'), seeds=SEEDS, max_len=MAX_LEN, epochs=EPOCHS, lr=LR, wd=WD,
    batch_train=BATCH_TR, n_boot=N_BOOT, checkpoints={k: v['hf'] for k, v in MODELS.items()},
    python=platform.python_version(), torch=torch.__version__, gpu=torch.cuda.get_device_name(0),
    transformers=transformers.__version__, sklearn=sklearn.__version__, numpy=np.__version__,
    pandas=pd.__version__, scipy=scipy.__version__, statsmodels=statsmodels.__version__, expected_data=EXPECTED)
json.dump(RUN_INFO, open(f'{OUT}/run_info.json', 'w'), indent=2)

def backup(name='backup'):
    zp = f'/content/{name}.zip'
    with zipfile.ZipFile(zp, 'w', zipfile.ZIP_DEFLATED) as z:
        for f in glob.glob(f'{OUT}/*'): z.write(f, os.path.basename(f))
    from google.colab import files; files.download(zp)

# ===================== BLOCK C : restore backup (optional) + load BD-SHS =====================
# Optional: if you have a saved backup zip, upload it first; it restores all earlier result files.
from google.colab import files
print('OPTIONAL: upload your backup zip(s) (e.g. progress_*.zip). Press Cancel / skip if none.')
try:
    for n, c in files.upload().items():
        if n.lower().endswith('.zip') and 'archive' not in n.lower():
            open(f'/content/_bk_{n}', 'wb').write(c)
            with zipfile.ZipFile(f'/content/_bk_{n}') as z: z.extractall(OUT)
            print('restored', n)
except Exception as e:
    print('no backup restored:', repr(e))

DATA = '/content/bdshs'; os.makedirs(DATA, exist_ok=True)
BASE_URL = ''   # later: raw GitHub folder URL with train.csv / val.csv / test.csv
def have_all(): return all(os.path.exists(f'{DATA}/{k}.csv') for k in EXPECTED)
if not have_all() and BASE_URL:
    import urllib.request
    for k in EXPECTED: urllib.request.urlretrieve(f'{BASE_URL.rstrip("/")}/{k}.csv', f'{DATA}/{k}.csv')
if not have_all():
    print('Upload archive__3_.zip  (or train.csv, val.csv, test.csv)')
    for name, content in files.upload().items():
        p = f'/content/_up_{name}'; open(p, 'wb').write(content)
        if name.lower().endswith('.zip'):
            with zipfile.ZipFile(p) as z: z.extractall('/content/_up_zip')
    for k in EXPECTED:
        hits = glob.glob(f'/content/_up_{k}.csv') + glob.glob(f'/content/_up_zip/**/{k}.csv', recursive=True)
        if hits: shutil.copy(hits[0], f'{DATA}/{k}.csv')
assert have_all(), 'train/val/test CSV not found - upload again'

def sha16(p): return hashlib.sha256(open(p, 'rb').read()).hexdigest()[:16]
all_ok = True
for k, (h, n) in EXPECTED.items():
    raw = pd.read_csv(f'{DATA}/{k}.csv'); ok = (sha16(f'{DATA}/{k}.csv') == h) and (len(raw) == n); all_ok &= ok
    print(f'{k:5s} rows={len(raw):6d}  sha256[:16]={sha16(f"{DATA}/{k}.csv")}  ->', 'OK' if ok else '*** MISMATCH ***')
print('DATASET INTEGRITY:', 'ALL OK' if all_ok else 'MISMATCH - do not use for thesis')

def clean_text(text):
    text = str(text)
    text = re.sub(r'http\S+|www\S+', '', text)
    text = re.sub(r'[^ঀ-৿\s\w.,!?।]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text
def load(k):
    d = pd.read_csv(f'{DATA}/{k}.csv').rename(columns={'sentence': 'text_raw', 'hate speech': 'label'})
    d['text'] = d['text_raw'].map(clean_text); d['label'] = d['label'].astype(int)
    return d[['text_raw', 'text', 'label']].reset_index(drop=True)
train, val, test = load('train'), load('val'), load('test')
y_test = test.label.values; np.save(f'{OUT}/y_test.npy', y_test)
all_df = pd.concat([train.assign(split='train'), val.assign(split='val'), test.assign(split='test')], ignore_index=True)
train.to_pickle('/content/train.pkl'); val.to_pickle('/content/val.pkl'); test.to_pickle('/content/test.pkl')
print({'train': len(train), 'val': len(val), 'test': len(test)}, '| train hate:', int(train.label.sum()))
print('Variables ready: train, val, test, all_df, y_test, MODELS, SEEDS, MAX_LEN, OUT, backup()')
print('Already-finished result files in', OUT, ':', sorted(os.listdir(OUT)))

# ===================== BLOCK D : Step 4 helpers + reload saved settings (needed for Step 7 onwards) =====================
# Run this AFTER block C (and after restoring your backup zip there).
from sklearn.metrics import (accuracy_score, precision_recall_fscore_support, f1_score, precision_score, recall_score)

def metrics(y, p):
    mp, mr, mf, _ = precision_recall_fscore_support(y, p, average='macro', zero_division=0)
    return dict(Accuracy=accuracy_score(y, p), Macro_P=mp, Macro_R=mr, Macro_F1=mf,
                Weighted_F1=f1_score(y, p, average='weighted'),
                Hate_P=precision_score(y, p, pos_label=1, zero_division=0),
                Hate_R=recall_score(y, p, pos_label=1, zero_division=0),
                Hate_F1=f1_score(y, p, pos_label=1, zero_division=0))
def save_probs(name, seed, probs):  np.save(f'{OUT}/probs_{name}_s{seed}.npy', probs)
def have(name, seed):               return os.path.exists(f'{OUT}/probs_{name}_s{seed}.npy')
def get(name, seed):                return np.load(f'{OUT}/probs_{name}_s{seed}.npy')
RUN_LOG = f'{OUT}/results_per_run.csv'
def log_run(name, seed, probs):
    m = metrics(y_test, probs.argmax(1)); m.update(Model=name, Seed=seed)
    row = pd.DataFrame([m])[['Model', 'Seed'] + [c for c in m if c not in ('Model', 'Seed')]]
    row.to_csv(RUN_LOG, mode='a', header=not os.path.exists(RUN_LOG), index=False)
    print(name, 'seed', seed, {k: round(v, 4) for k, v in m.items() if k not in ('Model', 'Seed')})

# reload the settings that were updated during Steps 2-patch, 5 and 6 (SVM C, BiLSTM details, MAX_LEN=128)
if os.path.exists(f'{OUT}/hyperparameters.csv'): HP = pd.read_csv(f'{OUT}/hyperparameters.csv')
if os.path.exists(f'{OUT}/run_info.json'):       RUN_INFO = json.load(open(f'{OUT}/run_info.json'))
assert MAX_LEN == 128 and RUN_INFO.get('max_len') == 128, 'MAX_LEN should be 128'
print('Restored. Finished runs so far:')
if os.path.exists(RUN_LOG): display(pd.read_csv(RUN_LOG).groupby('Model').size().rename('n_seeds_done'))
print('Next step: Step 7a (run_transformer) -> then 7b.')

# ===================== BLOCK E : Step 7 definitions (run AFTER block D) =====================
import torch.nn as nn
from transformers import (AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer, TrainerCallback)
from datasets import Dataset

assert MAX_LEN == 128
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
    cfg = MODELS[name]; hf = cfg['hf']
    set_seed(seed); t0 = time.time()
    tk = AutoTokenizer.from_pretrained(hf)
    def mk(d):
        ds = Dataset.from_dict({'text': d.text.tolist(), 'labels': d.label.tolist()})
        return ds.map(lambda b: tk(b['text'], truncation=True, max_length=MAX_LEN), batched=True, remove_columns=['text'])
    dtr, dva, dte = mk(train), mk(val), mk(test)
    model = AutoModelForSequenceClassification.from_pretrained(hf, num_labels=2)
    args = TrainingArguments(output_dir=f'/content/ckpt/{name}_s{seed}', num_train_epochs=EPOCHS, learning_rate=LR,
        weight_decay=WD, per_device_train_batch_size=BATCH_TR, per_device_eval_batch_size=BATCH_EV,
        eval_strategy='epoch', save_strategy='no', load_best_model_at_end=False, seed=seed, data_seed=seed,
        fp16=True, report_to='none', disable_tqdm=True, logging_strategy='epoch')
    def cm(ep):
        p = ep.predictions[0] if isinstance(ep.predictions, tuple) else ep.predictions
        return {'f1': f1_score(ep.label_ids, p.argmax(-1), average='macro')}
    kb = KeepBest()
    tr = WeightedTrainer(model=model, args=args, train_dataset=dtr, eval_dataset=dva,
                         processing_class=tk, compute_metrics=cm, callbacks=[kb])
    tr.use_w = cfg['weighted']
    tr.train()
    tr.model.load_state_dict(kb.state)
    ep_log = [dict(Model=name, Seed=seed, Epoch=int(round(h['epoch'])), Val_Macro_F1=round(h['eval_f1'], 4))
              for h in tr.state.log_history if 'eval_f1' in h]
    pd.DataFrame(ep_log).to_csv(f'{OUT}/epochlog_{name}_s{seed}.csv', index=False)
    def probs(ds):
        p = tr.predict(ds).predictions; p = p[0] if isinstance(p, tuple) else p
        return torch.softmax(torch.tensor(p).float(), 1).numpy()
    pr, pv = probs(dte), probs(dva)
    np.save(f'{OUT}/valprobs_{name}_s{seed}.npy', pv)
    vf = f1_score(val.label.values, pv.argmax(1), average='macro')
    print(f'   best epoch = {kb.epoch} | logged val F1 {kb.best:.4f} | re-computed val F1 {vf:.4f} | time {(time.time()-t0)/60:.1f} min')
    save_probs(name, seed, pr); log_run(name, seed, pr)
    if name == 'banglabert' and seed == SEEDS[0]:
        tr.save_model('/content/models/banglabert'); tk.save_pretrained('/content/models/banglabert')
    del tr, model, kb; gc.collect(); torch.cuda.empty_cache()

def save_progress(tag):
    rep  = f'BD-SHS single-corpus run - progress "{tag}"  ({time.strftime("%Y-%m-%d %H:%M:%S")})\n'
    rep += '\n===== Settings =====\n' + json.dumps(json.load(open(f'{OUT}/run_info.json')), indent=2)
    if os.path.exists(RUN_LOG):
        rr = pd.read_csv(RUN_LOG)
        rep += '\n\n===== Results per run (test set) =====\n' + rr.round(4).to_string(index=False)
        rep += '\n\n===== Mean and std over seeds =====\n' + \
               rr.groupby('Model')[['Accuracy', 'Macro_F1', 'Weighted_F1', 'Hate_P', 'Hate_R', 'Hate_F1']].agg(['mean', 'std']).round(4).to_string()
    for f in sorted(glob.glob(f'{OUT}/epochlog_*.csv')):
        rep += f'\n\n===== {os.path.basename(f)} =====\n' + pd.read_csv(f).to_string(index=False)
    open(f'{OUT}/PROGRESS_REPORT.txt', 'w', encoding='utf-8').write(rep)
    print(f'[saved] {len(os.listdir(OUT))} files in {OUT}  -> downloading progress_{tag}.zip')
    backup(f'progress_{tag}')
print('Block E ready: run_transformer, save_progress, WeightedTrainer defined.')
