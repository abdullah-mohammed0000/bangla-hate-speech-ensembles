# ============================================================================
# MERGE script  --  run in the MAIN Colab (after the ONE-CELL resume has finished: "ALL READY")
# Upload one or more job_*.zip files that came from other accounts; they are merged into /content/results.
# ============================================================================
from google.colab import files
import shutil, zipfile, glob, os, json
JOBTMP = '/content/_jobs'; shutil.rmtree(JOBTMP, ignore_errors=True); os.makedirs(JOBTMP)
print('Upload job_*.zip file(s)')
for n, c in files.upload().items():
    if n.lower().endswith('.zip'):
        d = f'{JOBTMP}/{os.path.splitext(n)[0]}'; os.makedirs(d, exist_ok=True)
        open(f'{d}.zip', 'wb').write(c)
        with zipfile.ZipFile(f'{d}.zip') as z: z.extractall(d)
        print('extracted', n)

new_rows = []
for d in sorted(glob.glob(f'{JOBTMP}/*/')):
    info = glob.glob(f'{d}/job_info_*.json')
    if info:
        ji = json.load(open(info[0]))
        # same dataset, same setting, same library version as the main run?
        assert ji['data_hashes'] == {k: v[0] for k, v in EXPECTED.items()}, 'job used a DIFFERENT dataset!'
        assert ji['max_len'] == MAX_LEN and ji['epochs'] == EPOCHS and ji['lr'] == LR, 'job used different hyper-parameters!'
        print(f'job {ji["job"]}: GPU={ji["gpu"]}, transformers={ji["transformers"]} (main: {RUN_INFO["transformers"]}), date={ji["date"]}')
        shutil.copy(info[0], f'{OUT}/{os.path.basename(info[0])}')
    for f in glob.glob(f'{d}/probs_*.npy') + glob.glob(f'{d}/valprobs_*.npy') + glob.glob(f'{d}/epochlog_*.csv'):
        shutil.copy(f, f'{OUT}/{os.path.basename(f)}')
    for f in glob.glob(f'{d}/results_job_*.csv'):
        new_rows.append(pd.read_csv(f))

assert new_rows, 'no results_job_*.csv found in the uploaded zip(s)'
new = pd.concat(new_rows, ignore_index=True)
old = pd.read_csv(RUN_LOG) if os.path.exists(RUN_LOG) else pd.DataFrame(columns=new.columns)
allr = pd.concat([old, new], ignore_index=True).drop_duplicates(subset=['Model', 'Seed'], keep='last')
allr.to_csv(RUN_LOG, index=False)

# integrity: recompute metrics from the saved probabilities and compare with the logged rows
bad = 0
for _, r in new.iterrows():
    p = np.load(f'{OUT}/probs_{r.Model}_s{int(r.Seed)}.npy')
    assert p.shape == (len(y_test), 2), f'bad shape for {r.Model} {r.Seed}'
    m = metrics(y_test, p.argmax(1))
    if abs(m['Macro_F1'] - r.Macro_F1) > 1e-6: bad += 1; print('MISMATCH', r.Model, r.Seed)
print('Integrity check:', 'ALL MATCH' if bad == 0 else f'{bad} MISMATCH(ES)')
print('\nMerged. Runs per model now:'); display(allr.groupby('Model').size().rename('n_seeds'))
backup('after_merge')
