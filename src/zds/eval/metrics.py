"""Detection metrics, per-category breakdown, and early (partial-trace) detection."""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score

from ..data.events import BENIGN

PREFIXES = (0.05, 0.10, 0.20, 0.30, 1.0)


def detection_metrics(score: np.ndarray, label: np.ndarray, threshold: float) -> dict:
    pred = score > threshold
    tp = int((pred & (label == 1)).sum())
    fp = int((pred & (label == 0)).sum())
    fn = int((~pred & (label == 1)).sum())
    tn = int((~pred & (label == 0)).sum())
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    both = len(np.unique(label)) == 2
    return {
        "precision": precision,
        "recall": recall,
        "f1": 2 * precision * recall / (precision + recall) if precision + recall else 0.0,
        "fpr": fp / (fp + tn) if fp + tn else 0.0,
        "auprc": float(average_precision_score(label, score)) if both else float("nan"),
        "auroc": float(roc_auc_score(label, score)) if both else float("nan"),
        "n_attack": int((label == 1).sum()),
        "n_benign": int((label == 0).sum()),
    }


def per_category(score, label, attack, threshold) -> dict:
    """Each attack category against all benign test windows."""
    out = {}
    benign = label == 0
    for name in sorted(set(attack[label == 1].astype(str))):
        pick = benign | ((label == 1) & (attack == name))
        out[name] = detection_metrics(score[pick], label[pick], threshold)
    return out


@dataclass
class Episode:
    attack: str
    idx: np.ndarray  # window indices, time-ordered


def find_episodes(start, label, attack, window_s: float, max_gap: int = 2) -> list[Episode]:
    """Group attack windows of one category into episodes; a gap of more than
    `max_gap` windows starts a new episode."""
    episodes = []
    for name in sorted(set(attack[label == 1].astype(str))):
        idx = np.flatnonzero((label == 1) & (attack == name))
        idx = idx[np.argsort(start[idx], kind="stable")]
        breaks = np.flatnonzero(np.diff(start[idx]) > (max_gap + 1) * window_s + 1e-6) + 1
        episodes += [Episode(name, part) for part in np.split(idx, breaks)]
    return episodes


def first_alert(above: np.ndarray, persist: int) -> int | None:
    """Index at which `persist` consecutive windows are above threshold."""
    run = 0
    for i, hit in enumerate(above):
        run = run + 1 if hit else 0
        if run >= persist:
            return i
    return None


def early_detection(score, start, label, attack, threshold, window_s, persist: int = 1,
                    prefixes=PREFIXES, max_gap: int = 2) -> dict:
    """Truncate each attack episode to its first p% and ask whether a stable alert
    (`persist` consecutive windows over threshold) has fired by then."""
    episodes = find_episodes(start, label, attack, window_s, max_gap)
    rows = []
    for ep in episodes:
        hit = first_alert(score[ep.idx] > threshold, persist)
        n = len(ep.idx)
        rows.append({
            "attack": ep.attack, "n_windows": n, "alert_at": hit,
            "ttd_s": None if hit is None else float(start[ep.idx[hit]] - start[ep.idx[0]] + window_s),
            "fraction_observed": None if hit is None else (hit + 1) / n,
        })

    def summarise(sel):
        if not sel:
            return {}
        out = {"n_episodes": len(sel)}
        for p in prefixes:
            seen = [r["alert_at"] is not None and r["alert_at"] < math.ceil(p * r["n_windows"]) for r in sel]
            out[f"detected@{int(p * 100)}%"] = float(np.mean(seen))
        found = [r for r in sel if r["alert_at"] is not None]
        out["median_ttd_s"] = float(np.median([r["ttd_s"] for r in found])) if found else None
        out["median_fraction_observed"] = float(np.median([r["fraction_observed"] for r in found])) if found else None
        return out

    return {
        "overall": summarise(rows),
        "by_attack": {a: summarise([r for r in rows if r["attack"] == a]) for a in sorted({r["attack"] for r in rows})},
        "episodes": rows,
    }


def evaluate(score, windows, threshold, window_s, persist: int = 1) -> dict:
    label, attack = np.asarray(windows.label), np.asarray(windows.attack).astype(str)
    return {
        "threshold": float(threshold),
        "overall": detection_metrics(score, label, threshold),
        "by_attack": per_category(score, label, attack, threshold),
        "early": early_detection(score, windows.start, label, attack, threshold, window_s, persist),
    }


__all__ = ["BENIGN", "PREFIXES", "detection_metrics", "per_category", "find_episodes", "early_detection", "evaluate"]
