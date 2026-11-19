"""Baselines, all on per-window summary statistics and the same split as the model.

Unsupervised (benign-only training, like ZeroDayShield):
  iforest   Isolation Forest
  pca       PCA reconstruction error
  ocsvm     One-class SVM (RBF) on a training subsample
  ae        dense autoencoder reconstruction error
Supervised closed-set:
  rf        Random Forest trained on benign + known attack categories; run once per
            held-out category so that category is unseen at test time

Every method gets its alert threshold from benign validation windows at the same
target false-positive rate, so the comparison uses the same alert budget.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.svm import OneClassSVM

from .data.events import Windows, summarise
from .eval.metrics import detection_metrics, evaluate
from .pipeline import RunConfig, Split


def _ae_scores(train, others, cfg: RunConfig, epochs: int = 40):
    torch.manual_seed(cfg.seed)
    d = train.shape[1]
    net = nn.Sequential(nn.Linear(d, 128), nn.GELU(), nn.Linear(128, 32), nn.GELU(),
                        nn.Linear(32, 128), nn.GELU(), nn.Linear(128, d)).to(cfg.device)
    opt = torch.optim.AdamW(net.parameters(), 1e-3, weight_decay=1e-4)
    x = torch.as_tensor(train, device=cfg.device)
    gen = torch.Generator().manual_seed(cfg.seed)
    for _ in range(epochs):
        for idx in torch.randperm(len(x), generator=gen).split(256):
            b = x[idx.to(cfg.device)]
            loss = ((net(b) - b) ** 2).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
    net.eval()
    with torch.no_grad():
        return [((net(t) - t) ** 2).mean(1).cpu().numpy() for t in (torch.as_tensor(o, device=cfg.device) for o in others)]


def run_unsupervised(split: Split, cfg: RunConfig) -> dict:
    """`split` must already be scaled. Returns {method: metrics}."""
    sc = StandardScaler().fit(summarise(split.train))
    tr, va, te = (np.clip(sc.transform(summarise(w)), -20, 20).astype(np.float32)
                  for w in (split.train, split.val, split.test))
    rng = np.random.default_rng(cfg.seed)
    scores = {}
    forest = IsolationForest(n_estimators=200, random_state=cfg.seed, n_jobs=-1).fit(tr)
    scores["iforest"] = (-forest.score_samples(va), -forest.score_samples(te))
    pca = PCA(n_components=0.95, random_state=cfg.seed).fit(tr)
    err = lambda x: ((x - pca.inverse_transform(pca.transform(x))) ** 2).mean(1)
    scores["pca"] = (err(va), err(te))
    sub = tr[rng.choice(len(tr), min(len(tr), 10000), replace=False)]
    svm = OneClassSVM(kernel="rbf", gamma="scale", nu=0.01).fit(sub)
    scores["ocsvm"] = (-svm.score_samples(va), -svm.score_samples(te))
    scores["ae"] = tuple(_ae_scores(tr, (va, te), cfg))
    out = {}
    for name, (val_score, test_score) in scores.items():
        threshold = float(np.quantile(val_score, 1 - cfg.target_fpr))
        out[name] = evaluate(test_score, split.test, threshold, cfg.window_s, cfg.persist)
    return out


def run_supervised_loco(split: Split, cfg: RunConfig) -> dict:
    """Leave-one-category-out Random Forest. For each category C the classifier is
    trained on benign + every known category except C, then scored on the shared
    test set. Returns per-category recall when unseen vs when seen in training."""
    if split.known is None or not len(split.known):
        return {}
    xb, xk, xv, xt = (summarise(w) for w in (split.train, split.known, split.val, split.test))
    known_cat = np.asarray(split.known.attack).astype(str)
    test_cat = np.asarray(split.test.attack).astype(str)
    label = split.test.label
    rng = np.random.default_rng(cfg.seed)
    if len(xb) > 4 * max(len(xk), 5000):  # keep the classes within the same order of magnitude
        xb = xb[rng.choice(len(xb), 4 * max(len(xk), 5000), replace=False)]

    def fit_score(exclude):
        keep = known_cat != exclude
        if not keep.any():
            return None
        x = np.concatenate([xb, xk[keep]])
        y = np.concatenate([np.zeros(len(xb)), np.ones(int(keep.sum()))])
        rf = RandomForestClassifier(300, n_jobs=-1, random_state=cfg.seed, class_weight="balanced").fit(x, y)
        return rf.predict_proba(xv)[:, 1], rf.predict_proba(xt)[:, 1]

    def metrics_for(scores, name):
        val_score, test_score = scores
        threshold = float(np.quantile(val_score, 1 - cfg.target_fpr))
        pick = (label == 0) | (test_cat == name)
        return detection_metrics(test_score[pick], label[pick], threshold)

    full = fit_score(exclude=None)
    out = {"all_seen": evaluate(full[1], split.test, float(np.quantile(full[0], 1 - cfg.target_fpr)),
                                cfg.window_s, cfg.persist), "by_attack": {}}
    for name in sorted(set(test_cat[label == 1])):
        row = {"in_known": bool((known_cat == name).any()), "seen": metrics_for(full, name)}
        held = fit_score(exclude=name)
        row["unseen"] = metrics_for(held, name) if held is not None else None
        out["by_attack"][name] = row
    return out
