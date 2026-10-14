# BD-SHS single-corpus FINAL code. Colab e protita '# ===== CELL =====' alada cell e paste koro, upor theke nich porjonto cell by cell run koro.
# Runtime > Change runtime type > T4 GPU age set koro.

# ===================== CELL 1 =====================
# 1. Setup -----------------------------------------------------------------
!pip -q install transformers==4.44.2 accelerate datasets shap statsmodels scikit-learn sentencepiece
import torch, sys
print('Python', sys.version.split()[0], '| torch', torch.__version__)
print('CUDA available:', torch.cuda.is_available())
assert torch.cuda.is_available(), 'No GPU! Runtime > Change runtime type > T4 GPU, then Run all again.'
print('GPU:', torch.cuda.get_device_name(0))

# ===================== CELL 2 =====================
# 2. Config ----------------------------------------------------------------
import os, re, json, random, hashlib, time, glob, zipfile, warnings
import numpy as np, pandas as pd
warnings.filterwarnings('ignore')

SEEDS      = [42]          # add e.g. [42, 43, 44] for multi-seed runs (3x longer)
MAX_LEN    = 64
EPOCHS     = 3
LR         = 2e-5
WD         = 0.01
BATCH_TR   = 32
BATCH_EV   = 64
N_BOOT     = 2000
OUT        = '/content/results'
os.makedirs(OUT, exist_ok=True)

# Optional: if the three CSVs are hosted at a public URL (GitHub raw / Hugging Face), put the base here.
# Files expected: train.csv, val.csv, test.csv.  Leave '' to upload them by hand.
BASE_URL   = ''

# Integrity check of the dataset used in the thesis (first 16 hex of SHA-256) and row counts
EXPECTED = {'train': ('116c9e9eb7432dff', 40224),
            'val':   ('67995e80777818a1', 5028),
            'test':  ('4543a005555ee466', 5029)}

def set_seed(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed_all(s)

# ===================== CELL 3 =====================
# 3. Load BD-SHS (no Drive) -------------------------------------------------
import urllib.request, shutil
DATA = '/content/bdshs'; os.makedirs(DATA, exist_ok=True)

def have_all():
    return all(os.path.exists(f'{DATA}/{k}.csv') for k in EXPECTED)

if not have_all() and BASE_URL:
    for k in EXPECTED:
        urllib.request.urlretrieve(f'{BASE_URL.rstrip("/")}/{k}.csv', f'{DATA}/{k}.csv')

if not have_all():
    from google.colab import files
    print('Upload train.csv, val.csv, test.csv  (or the zip containing them)')
    up = files.upload()
    for name, content in up.items():
        p = f'/content/_up_{name}'
        open(p, 'wb').write(content)
        if name.lower().endswith('.zip'):
            with zipfile.ZipFile(p) as z: z.extractall('/content/_up_zip')
    for k in EXPECTED:
        hits = glob.glob(f'/content/_up_{k}.csv') + glob.glob(f'/content/_up_zip/**/{k}.csv', recursive=True)
        if hits: shutil.copy(hits[0], f'{DATA}/{k}.csv')
assert have_all(), 'train/val/test CSVs not found'

def sha16(p): return hashlib.sha256(open(p,'rb').read()).hexdigest()[:16]
ok = True
for k,(h,n) in EXPECTED.items():
    df_ = pd.read_csv(f'{DATA}/{k}.csv')
    good = (sha16(f'{DATA}/{k}.csv') == h) and len(df_) == n
    ok &= good
    print(f'{k:5s} rows={len(df_):6d} sha={sha16(f"{DATA}/{k}.csv")} ->', 'OK' if good else 'MISMATCH')
if not ok:
    print('WARNING: files differ from the dataset used in the thesis. Results will not match the reported numbers.')

def clean_text(text):
    text = str(text)
    text = re.sub(r'http\S+|www\S+', '', text)
    text = re.sub(r'[^\u0980-\u09FF\s\w.,!?।]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def load(k):
    d = pd.read_csv(f'{DATA}/{k}.csv')
    d = d.rename(columns={'sentence':'text','hate speech':'label'})
    d['text'] = d['text'].map(clean_text)
    d['label'] = d['label'].astype(int)
    return d[['text','label']].reset_index(drop=True)
train, val, test = load('train'), load('val'), load('test')
print({k: len(v) for k,v in dict(train=train,val=val,test=test).items()})
print('train label counts:', train.label.value_counts().to_dict())
y_test = test.label.values
np.save(f'{OUT}/y_test.npy', y_test)

# ===================== CELL 4 =====================
# 4. Metrics + result store -------------------------------------------------
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, f1_score
def metrics(y, p):
    pr, rc, f, _ = precision_recall_fscore_support(y, p, average='macro', zero_division=0)
    return dict(Accuracy=accuracy_score(y,p), Precision=pr, Recall=rc, F1=f,
                F1_hate=f1_score(y,p,pos_label=1), F1_weighted=f1_score(y,p,average='weighted'))

def save_probs(name, seed, probs):
    np.save(f'{OUT}/probs_{name}_s{seed}.npy', probs)
def have(name, seed): return os.path.exists(f'{OUT}/probs_{name}_s{seed}.npy')
def get(name, seed): return np.load(f'{OUT}/probs_{name}_s{seed}.npy')
print('ready')

# ===================== CELL 5 =====================
# 5. TF-IDF + SVM (word and char) --------------------------------------------
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
for name, vec in [('tfidf_word_svm', TfidfVectorizer(token_pattern=r'\S+', ngram_range=(1,2), max_features=50000, sublinear_tf=True)),
                  ('tfidf_char_svm', TfidfVectorizer(analyzer='char_wb', ngram_range=(2,5), max_features=100000, sublinear_tf=True))]:
    for s in SEEDS:
        if have(name, s): continue
        set_seed(s)
        Xtr = vec.fit_transform(train.text); Xte = vec.transform(test.text)
        clf = CalibratedClassifierCV(LinearSVC(C=1.0, random_state=s), cv=3).fit(Xtr, train.label)
        pr = clf.predict_proba(Xte); save_probs(name, s, pr)
        print(name, s, {k: round(v,4) for k,v in metrics(y_test, pr.argmax(1)).items()})

# ===================== CELL 6 =====================
# 6. BiLSTM -----------------------------------------------------------------
import torch.nn as nn
from collections import Counter
from torch.utils.data import DataLoader, TensorDataset
dev = torch.device('cuda')
tok = lambda t: t.split()
cnt = Counter(w for t in train.text for w in tok(t))
vocab = {w:i+2 for i,(w,c) in enumerate(cnt.most_common(50000)) if c>=2}
def enc(t, L=MAX_LEN):
    ids = [vocab.get(w,1) for w in tok(t)][:L]; return ids + [0]*(L-len(ids))
def tens(d): return torch.tensor([enc(t) for t in d.text]), torch.tensor(d.label.values)
class BiLSTM(nn.Module):
    def __init__(s, V):
        super().__init__(); s.e=nn.Embedding(V+2,128,padding_idx=0); s.l=nn.LSTM(128,128,batch_first=True,bidirectional=True)
        s.d=nn.Dropout(0.4); s.f=nn.Linear(256,2)
    def forward(s,x):
        o,_=s.l(s.e(x)); m=(x!=0).unsqueeze(-1).float()
        return s.f(s.d((o*m).sum(1)/m.sum(1).clamp(min=1)))
for s in SEEDS:
    if have('bilstm', s): continue
    set_seed(s)
    Xtr,ytr = tens(train); Xva,yva = tens(val); Xte,_ = tens(test)
    m = BiLSTM(len(vocab)).to(dev); opt = torch.optim.Adam(m.parameters(), 1e-3)
    best, best_state = -1, None
    for ep in range(8):
        m.train()
        for xb,yb in DataLoader(TensorDataset(Xtr,ytr), batch_size=64, shuffle=True):
            opt.zero_grad(); nn.functional.cross_entropy(m(xb.to(dev)), yb.to(dev)).backward(); opt.step()
        m.eval()
        with torch.no_grad(): pv = torch.cat([m(xb.to(dev)).argmax(1).cpu() for xb in DataLoader(Xva,batch_size=256)])
        f = f1_score(yva, pv, average='macro')
        if f>best: best, best_state = f, {k:v.clone() for k,v in m.state_dict().items()}
    m.load_state_dict(best_state); m.eval()
    with torch.no_grad(): pr = torch.cat([torch.softmax(m(xb.to(dev)),1).cpu() for xb in DataLoader(Xte,batch_size=256)]).numpy()
    save_probs('bilstm', s, pr); print('bilstm', s, {k: round(v,4) for k,v in metrics(y_test, pr.argmax(1)).items()})

# ===================== CELL 7 =====================
# 7. Transformer fine-tuning ------------------------------------------------
from transformers import (AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer)
from datasets import Dataset
MODELS = {  # name: (hf id, weighted loss?)
    'mbert':          ('bert-base-multilingual-cased', True),
    'xlmr':           ('xlm-roberta-base', True),
    'banglabert':     ('csebuetnlp/banglabert', True),
    'banglabert_unw': ('csebuetnlp/banglabert', False),   # ablation: no class weights
    'indicbertv2':    ('ai4bharat/IndicBERTv2-MLM-only', True),   # supplementary
}
cw = len(train)/(2*np.bincount(train.label.values)); CW = torch.tensor(cw, dtype=torch.float)
print('class weights', cw)

class WeightedTrainer(Trainer):
    use_w = True
    def compute_loss(self, model, inputs, return_outputs=False, **kw):
        labels = inputs.pop('labels'); out = model(**inputs)
        w = CW.to(out.logits.device) if self.use_w else None
        loss = nn.CrossEntropyLoss(weight=w)(out.logits, labels)
        return (loss, out) if return_outputs else loss

def run_transformer(name, seed):
    hf, weighted = MODELS[name]
    set_seed(seed)
    tk = AutoTokenizer.from_pretrained(hf)
    mk = lambda d: Dataset.from_pandas(d).map(lambda b: tk(b['text'], truncation=True, max_length=MAX_LEN), batched=True)
    dtr, dva, dte = mk(train), mk(val), mk(test)
    model = AutoModelForSequenceClassification.from_pretrained(hf, num_labels=2)
    args = TrainingArguments(output_dir=f'/content/ckpt/{name}_{seed}', num_train_epochs=EPOCHS, learning_rate=LR,
        weight_decay=WD, per_device_train_batch_size=BATCH_TR, per_device_eval_batch_size=BATCH_EV, eval_strategy='epoch',
        save_strategy='epoch', save_total_limit=1, load_best_model_at_end=True, metric_for_best_model='f1', seed=seed,
        fp16=True, report_to='none', logging_steps=500)
    cm = lambda ep: {'f1': f1_score(ep.label_ids, ep.predictions.argmax(1), average='macro')}
    tr = WeightedTrainer(model=model, args=args, train_dataset=dtr, eval_dataset=dva, tokenizer=tk, compute_metrics=cm)
    tr.use_w = weighted; tr.train()
    pr = torch.softmax(torch.tensor(tr.predict(dte).predictions), 1).numpy()
    save_probs(name, seed, pr)
    if name == 'banglabert' and seed == SEEDS[0]:
        tr.save_model('/content/models/banglabert'); tk.save_pretrained('/content/models/banglabert')
    print(name, seed, {k: round(v,4) for k,v in metrics(y_test, pr.argmax(1)).items()})
    import gc; del tr, model; gc.collect(); torch.cuda.empty_cache()

for name in MODELS:
    for s in SEEDS:
        if have(name, s): print('skip (done)', name, s); continue
        t0=time.time(); run_transformer(name, s); print(f'  {(time.time()-t0)/60:.1f} min')

# ===================== CELL 8 =====================
# 8. Ensemble (BanglaBERT + XLM-R, mean of softmax) --------------------------
for s in SEEDS:
    if not have('ensemble', s):
        save_probs('ensemble', s, (get('banglabert', s) + get('xlmr', s))/2)
print('ensemble done')

# ===================== CELL 9 =====================
# 9. Main results table -----------------------------------------------------
ORDER = [('tfidf_word_svm','TF-IDF (word) + SVM'),('tfidf_char_svm','TF-IDF (char) + SVM'),('bilstm','BiLSTM'),
         ('mbert','mBERT'),('xlmr','XLM-RoBERTa'),('banglabert','BanglaBERT'),('banglabert_unw','BanglaBERT (unweighted)'),
         ('ensemble','Ensemble (BanglaBERT+XLM-R)'),('indicbertv2','IndicBERTv2 (suppl.)')]
rows=[]
for key,label in ORDER:
    ms=[metrics(y_test, get(key,s).argmax(1)) for s in SEEDS if have(key,s)]
    if not ms: continue
    d={k: np.mean([m[k] for m in ms]) for k in ms[0]}; d['Model']=label; d['n_seeds']=len(ms); rows.append(d)
res=pd.DataFrame(rows).set_index('Model').round(4); res.to_csv(f'{OUT}/results_main.csv'); res

# ===================== CELL 10 =====================
# 10. Bootstrap 95% CI (macro-F1) and McNemar vs best primary model (first seed) -
from statsmodels.stats.contingency_tables import mcnemar
rng=np.random.RandomState(0); s0=SEEDS[0]; n=len(y_test)
idx=[rng.randint(0,n,n) for _ in range(N_BOOT)]
ci=[]
preds={k:get(k,s0).argmax(1) for k,_ in ORDER if have(k,s0)}
for k,l in ORDER:
    if k not in preds: continue
    fs=[f1_score(y_test[i], preds[k][i], average='macro') for i in idx]
    ci.append(dict(Model=l, F1=f1_score(y_test,preds[k],average='macro'), CI_low=np.percentile(fs,2.5), CI_high=np.percentile(fs,97.5)))
ci=pd.DataFrame(ci).set_index('Model').round(4); ci.to_csv(f'{OUT}/bootstrap_ci.csv'); display(ci)

primary=[k for k,_ in ORDER if k in preds and k not in ('indicbertv2','banglabert_unw','ensemble')]
best=max(primary, key=lambda k: f1_score(y_test,preds[k],average='macro')); print('Best primary single model:', best)
lab=dict(ORDER); mc=[]
for k in preds:
    if k==best: continue
    a=(preds[best]==y_test); b=(preds[k]==y_test)
    t=[[np.sum(a&b),np.sum(a&~b)],[np.sum(~a&b),np.sum(~a&~b)]]
    r=mcnemar(t, exact=False, correction=True)
    mc.append(dict(Comparison=f'{lab[best]} vs {lab[k]}', b_only_best_correct=t[0][1], other_only_correct=t[1][0], chi2=r.statistic, p_value=r.pvalue))
mc=pd.DataFrame(mc).set_index('Comparison'); mc['significant(p<0.05)']=mc.p_value<0.05
mc.round(6).to_csv(f'{OUT}/mcnemar.csv'); display(mc.round(6))

# ===================== CELL 11 =====================
# 11. Per-class report and confusion matrices -------------------------------
from sklearn.metrics import classification_report, confusion_matrix
rep={}
for k,l in ORDER:
    if k in preds:
        print('=====',l); print(classification_report(y_test,preds[k],target_names=['Non-hate','Hate'],digits=4))
        rep[l]=confusion_matrix(y_test,preds[k]).tolist()
json.dump(rep, open(f'{OUT}/confusion_matrices.json','w'))

# ===================== CELL 12 =====================
# 12. SHAP (BanglaBERT) -----------------------------------------------------
try:
    import shap
    from transformers import pipeline
    pipe=pipeline('text-classification', model='/content/models/banglabert', tokenizer='/content/models/banglabert',
                  device=0, top_k=None, truncation=True, max_length=MAX_LEN)
    explainer=shap.Explainer(pipe, shap.maskers.Text(pipe.tokenizer), output_names=['Non-hate','Hate'])
    hate_idx=test.index[test.label==1][:5].tolist(); non_idx=test.index[test.label==0][:3].tolist()
    sv=explainer(test.text[hate_idx+non_idx].tolist())
    html=shap.plots.text(sv, display=False); open(f'{OUT}/shap_banglabert.html','w').write(html)
    shap.plots.text(sv)
except Exception as e:
    print('SHAP skipped:', repr(e))

# ===================== CELL 13 =====================
# 13. Package everything and download (no Drive) -------------------------------
info = dict(seeds=SEEDS, max_len=MAX_LEN, epochs=EPOCHS, lr=LR, batch=BATCH_TR, n_boot=N_BOOT,
            data_hashes={k: sha16(f'{DATA}/{k}.csv') for k in EXPECTED}, torch=torch.__version__)
json.dump(info, open(f'{OUT}/run_info.json','w'), indent=2)
zp='/content/bdshs_single_corpus_results.zip'
with zipfile.ZipFile(zp,'w',zipfile.ZIP_DEFLATED) as z:
    for f in glob.glob(f'{OUT}/*'): z.write(f, os.path.basename(f))
from google.colab import files
files.download(zp)
