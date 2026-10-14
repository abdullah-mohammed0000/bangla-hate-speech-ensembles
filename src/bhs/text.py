"""Text cleaning and the BanglaBERT normalizer."""
from __future__ import annotations

import re
import subprocess
import sys
import types
from pathlib import Path

_URL = re.compile(r"http\S+|www\S+")
_NON_BANGLA = re.compile(r"[^ঀ-৿\s\w.,!?।]")  # keep Bangla block, word chars, . , ! ? and the danda
_SPACES = re.compile(r"\s+")


def clean_text(text) -> str:
    """Remove URLs and every character outside the Bangla block / word characters / basic punctuation."""
    text = str(text)
    text = _URL.sub("", text)
    text = _NON_BANGLA.sub("", text)
    return _SPACES.sub(" ", text).strip()


def get_bn_normalizer(cache_dir: str | Path = "~/.cache/bhs/normalizer"):
    """Return csebuetnlp's ``normalize`` function (cloned from GitHub; the unused ``emoji`` module is stubbed).

    Used for the "+ normalizer" models. It is applied AFTER :func:`clean_text`.
    """
    cache = Path(cache_dir).expanduser()
    if not (cache / "normalizer").exists():
        cache.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(["git", "clone", "-q", "https://github.com/csebuetnlp/normalizer", str(cache)], check=True)
    stub = types.ModuleType("emoji")
    stub.get_emoji_regexp = lambda: re.compile("(?!x)x")  # emoji handling is off by default
    sys.modules.setdefault("emoji", stub)
    sys.path.insert(0, str(cache))
    from normalizer import normalize  # noqa: E402

    return normalize
