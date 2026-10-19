"""Prediction store: a folder of ``probs_<model>_<regime>_s<seed>_<testset>.npy`` files plus ``labels_<testset>.npy``.

Training scripts write into such a folder; every analysis script reads from it, so the statistics can be
re-run on a laptop (no GPU, no data) from the released predictions.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from . import config as C


class Store:
    def __init__(self, root):
        self.root = Path(root)
        self._cache = {}

    def path(self, model, regime, seed, test) -> Path:
        return self.root / f"probs_{model}_{regime}_s{seed}_{test}.npy"

    def has(self, model, regime, seed, test) -> bool:
        return self.path(model, regime, seed, test).exists()

    def labels(self, test) -> np.ndarray:
        return np.load(self.root / f"labels_{test}.npy")

    def probs(self, model, regime, seed, test) -> np.ndarray:
        key = (model, regime, seed, test)
        if key not in self._cache:
            if model in C.ENSEMBLES and not self.has(*key):
                self._cache[key] = np.mean([self.probs(m, regime, seed, test) for m in C.ENSEMBLES[model]], axis=0)
            else:
                self._cache[key] = np.load(self.path(*key))
        return self._cache[key]

    def pred(self, system, regime, seed, test) -> np.ndarray:
        """Predicted classes of a system: a model name, an ensemble name, or a list of model names (soft vote)."""
        if isinstance(system, (list, tuple)):
            return np.mean([self.probs(m, regime, seed, test) for m in system], axis=0).argmax(1)
        return self.probs(system, regime, seed, test).argmax(1)

    def available(self, system, regime, test, seeds=C.SEEDS) -> bool:
        members = system if isinstance(system, (list, tuple)) else C.ENSEMBLES.get(system, [system])
        if not isinstance(system, (list, tuple)) and self.has(system, regime, seeds[0], test):
            members = [system]
        return all(self.has(m, regime, s, test) for m in members for s in seeds)


def save_probs(root, model, regime, seed, test, probs) -> None:
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    np.save(root / f"probs_{model}_{regime}_s{seed}_{test}.npy", np.asarray(probs, dtype=np.float32))


def save_labels(root, D: dict) -> None:
    """Write labels_<test>.npy for every test set in D, so that a prediction folder can be analysed on its own."""
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    for t in C.TEST_SETS:
        if t in D:
            np.save(root / f"labels_{t}.npy", D[t].label.values.astype(np.int64))
