"""Sanity run on synthetic telemetry: full model and its main ablations.

Usage: python scripts/run_synthetic.py [epochs]
"""
import sys

from zds.data.events import build_windows
from zds.data.synthetic import make_synthetic
from zds.models.zds import ModelConfig
from zds.pipeline import RunConfig, run

VARIANTS = {
    "full": ({}, ()),
    "full_mean": ({}, (), "mean"),
    "full_max": ({}, (), "max"),
    "no_fusion": ({"fusion": False}, ()),
    "no_contrast": ({"contrastive": False}, ()),
    "no_temporal": ({"temporal": False}, ()),
    "host_only": ({}, ("net",)),
    "net_only": ({}, ("host",)),
}

if __name__ == "__main__":
    epochs = int(sys.argv[1]) if len(sys.argv) > 1 else 20
    host, net = make_synthetic(hours=48, n_episodes=40, seed=0)
    windows = build_windows(host, net, window_s=60, max_host=12, max_net=48)
    print(f"{len(windows)} windows, {int(windows.label.sum())} attack")
    rows = {}
    for name, (overrides, drop, *agg) in VARIANTS.items():
        cfg = RunConfig(epochs=epochs, aggregate=agg[0] if agg else "top2", model=ModelConfig(**overrides))
        res = run(windows, cfg, drop=drop, out_dir=f"runs/synthetic/{name}", log=lambda *_: None)
        rows[name] = res
        o = res["overall"]
        cats = "  ".join(f"{k}={v['recall']:.2f}" for k, v in res["by_attack"].items())
        print(f"{name:12s} AUPRC {o['auprc']:.3f}  F1 {o['f1']:.3f}  FPR {o['fpr']:.3f}  recall: {cats}  [{res['seconds']}s]")
    e = rows["full"]["early"]["overall"]
    print("full early detection:", {k: v for k, v in e.items()})
