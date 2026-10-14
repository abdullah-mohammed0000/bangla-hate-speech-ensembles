# ============================================================================
# STEP E8 -- Software versions for the thesis (Appendix Table A.2)   [PORTABLE, no upload, ~10 s]
#   Run this in the SAME Colab environment you trained in (GPU runtime if possible).
#   Output: print the text and paste it back.
# ============================================================================
import sys, platform, subprocess, importlib
print('Python      :', sys.version.split()[0])
print('Platform    :', platform.platform())
for name in ['torch', 'transformers', 'sklearn', 'numpy', 'pandas', 'scipy', 'datasets', 'accelerate', 'shap', 'matplotlib', 'normalizer', 'bnlp']:
    try:
        m = importlib.import_module(name); print(f'{name:12s}:', getattr(m, '__version__', 'installed (no version attr)'))
    except Exception:
        print(f'{name:12s}: not installed')
try:
    import torch
    print('CUDA        :', torch.version.cuda, '| GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'none')
except Exception: pass
print('\n--- pip freeze (selected) ---')
out = subprocess.run([sys.executable, '-m', 'pip', 'freeze'], capture_output=True, text=True).stdout.splitlines()
keys = ('torch', 'transformers', 'scikit-learn', 'numpy', 'pandas', 'scipy', 'shap', 'accelerate', 'datasets', 'normalizer', 'bnlp', 'tokenizers', 'sentencepiece')
print('\n'.join(l for l in out if l.lower().startswith(keys)))
