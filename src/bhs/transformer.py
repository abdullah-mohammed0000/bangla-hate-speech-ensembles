"""Fine-tuning of one transformer for one (model, regime, seed) and prediction on all test sets.

Protocol (identical in every study): class-weighted cross-entropy (weights from the training set of the regime),
3 epochs, AdamW lr 2e-5, weight decay 0.01, batch 32, max length 128, fp16, no warm-up; the epoch with the highest
validation macro-F1 is kept (weights held in memory, no checkpoint reload).
"""
from __future__ import annotations

import gc
import random
import time
from pathlib import Path

import numpy as np
import pandas as pd

from . import config as C
from .data import get_regime
from .metrics import metrics
from .store import save_probs


def set_seed(seed: int) -> None:
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def run_transformer(name: str, regime: str, seed: int, D: dict, out_dir, hp: dict | None = None,
                    model_cfg: dict | None = None, normalizer=None, use_fp16: bool | None = None,
                    tests: list | None = None) -> pd.DataFrame:
    """Train and evaluate. Writes into ``out_dir``:
    probs_<name>_<regime>_s<seed>_<test>.npy (5 test sets), valprobs_*.npy, epochlog_*.csv. Returns the metrics rows.
    ``model_cfg`` / ``hp`` may be given to override the registry (used by the smoke test)."""
    import torch
    import torch.nn as nn
    from datasets import Dataset
    from sklearn.metrics import f1_score
    from transformers import (AutoModelForSequenceClassification, AutoTokenizer, Trainer, TrainerCallback,
                              TrainingArguments)

    hp = {**C.HP, **(hp or {})}
    tests = tests or C.TEST_SETS
    cfg = model_cfg or C.MODELS[name]
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    set_seed(seed)
    t0 = time.time()
    train, val = get_regime(D, regime, seed)
    if cfg.get("norm") and normalizer is None:
        from .text import get_bn_normalizer

        normalizer = get_bn_normalizer()
    tk = AutoTokenizer.from_pretrained(cfg["hf"])

    def encode(df):
        texts = [normalizer(t) for t in df.text] if cfg.get("norm") else df.text.tolist()  # normalizer AFTER clean_text
        ds = Dataset.from_dict({"text": texts, "labels": df.label.tolist()})
        return ds.map(lambda b: tk(b["text"], truncation=True, max_length=hp["max_len"]), batched=True, remove_columns=["text"])

    dtr, dva = encode(train), encode(val)
    model = AutoModelForSequenceClassification.from_pretrained(cfg["hf"], num_labels=2)
    steps_per_epoch = -(-len(dtr) // (hp["batch_train"] * hp["grad_acc"]))
    args = TrainingArguments(
        output_dir=str(out_dir / "_ckpt"), num_train_epochs=hp["epochs"], learning_rate=hp["lr"],
        weight_decay=hp["weight_decay"], per_device_train_batch_size=hp["batch_train"],
        per_device_eval_batch_size=hp["batch_eval"], gradient_accumulation_steps=hp["grad_acc"],
        warmup_steps=int(hp["warmup"] * hp["epochs"] * steps_per_epoch), eval_strategy="epoch", save_strategy="no",
        load_best_model_at_end=False, seed=seed, data_seed=seed,
        fp16=torch.cuda.is_available() if use_fp16 is None else use_fp16,
        report_to="none", disable_tqdm=True, logging_strategy="epoch")

    class WeightedTrainer(Trainer):
        class_weights = None
        weighted = True

        def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
            labels = inputs.pop("labels")
            out = model(**inputs)
            w = self.class_weights.to(out.logits.device) if self.weighted else None
            loss = nn.CrossEntropyLoss(weight=w)(out.logits.float(), labels)
            return (loss, out) if return_outputs else loss

    class KeepBest(TrainerCallback):
        def __init__(self):
            self.best, self.state, self.epoch = -1, None, 0

        def on_evaluate(self, args, state, control, metrics=None, model=None, **kw):
            f = (metrics or {}).get("eval_f1", -1)
            if f > self.best and model is not None:
                self.best, self.epoch = f, int(round(state.epoch))
                self.state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    def compute_metrics(ep):
        p = ep.predictions[0] if isinstance(ep.predictions, tuple) else ep.predictions
        return {"f1": f1_score(ep.label_ids, p.argmax(-1), average="macro")}

    kb = KeepBest()
    tr = WeightedTrainer(model=model, args=args, train_dataset=dtr, eval_dataset=dva, processing_class=tk,
                         compute_metrics=compute_metrics, callbacks=[kb])
    tr.class_weights = torch.tensor(len(train) / (2 * np.bincount(train.label.values)), dtype=torch.float)
    tr.weighted = cfg["weighted"]
    tr.train()
    tr.model.load_state_dict(kb.state)

    pd.DataFrame([dict(Model=name, Regime=regime, Seed=seed, Epoch=int(round(h["epoch"])), Val_Macro_F1=round(h["eval_f1"], 4))
                  for h in tr.state.log_history if "eval_f1" in h]).to_csv(out_dir / f"epochlog_{name}_{regime}_s{seed}.csv", index=False)

    def probs(ds):
        p = tr.predict(ds).predictions
        p = p[0] if isinstance(p, tuple) else p
        return torch.softmax(torch.tensor(p).float(), 1).numpy()

    pv = probs(dva)
    vf = f1_score(val.label.values, pv.argmax(1), average="macro")
    assert abs(vf - kb.best) < 1e-4, "best-epoch restore mismatch"
    np.save(out_dir / f"valprobs_{name}_{regime}_s{seed}.npy", pv)
    rows = []
    for part in tests:
        pr = probs(encode(D[part]))
        save_probs(out_dir, name, regime, seed, part, pr)
        m = metrics(D[part].label.values, pr.argmax(1))
        m.update(Model=name, Regime=regime, Seed=seed, Test=part)
        rows.append(m)
    print(f"{name} | {regime} | seed {seed}: best epoch {kb.epoch}, val macro-F1 {kb.best:.4f}, {(time.time() - t0) / 60:.1f} min")
    del tr, model, kb
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    df = pd.DataFrame(rows)
    return df[["Model", "Regime", "Seed", "Test"] + [c for c in df if c not in ("Model", "Regime", "Seed", "Test")]]
