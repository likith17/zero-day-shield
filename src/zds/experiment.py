"""Run the study on one dataset: model, ablations, baselines, held-out categories.

    python -m zds.experiment --dataset synthetic --out runs/synthetic
    python -m zds.experiment --dataset cicids2018 --out runs/cicids2018 --seeds 0 1 2
"""
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

import numpy as np

from .baselines import run_supervised_loco, run_unsupervised
from .data.events import Windows, build_windows
from .models.zds import ModelConfig
from .pipeline import RunConfig, run_split, scale_split, time_split

# name -> (model overrides, layers to drop). Variants needing both layers are
# skipped automatically on network-only datasets.
VARIANTS = {
    "zds_full": ({}, ()),
    "no_fusion": ({"fusion": False}, ()),
    "no_contrast": ({"contrastive": False}, ()),
    "no_temporal": ({"temporal": False}, ()),
    "no_order": ({"order_task": False}, ()),
    "no_stats": ({"_stats_view": False}, ()),
    "host_only": ({}, ("net",)),
    "net_only": ({}, ("host",)),
}
PAIRED_ONLY = {"no_fusion", "no_contrast", "host_only", "net_only"}


def load(dataset: str, args) -> tuple[Windows, float]:
    if dataset == "synthetic":
        from .data.synthetic import make_synthetic
        host, net = make_synthetic(hours=48, n_episodes=40, seed=0)
        return build_windows(host, net, 60.0, max_host=12, max_net=48), 60.0
    if dataset == "cicids2018":
        from .data.cicids import load_windows
        return load_windows(args.raw or "data/raw/cicids2018", args.window_s or 10.0, args.max_net), args.window_s or 10.0
    raise SystemExit(f"dataset '{dataset}' has no loader yet (see docs/DATA.md)")


def _mean_std(values):
    v = np.array([x for x in values if x is not None and not np.isnan(x)], dtype=float)
    return (float(v.mean()), float(v.std())) if len(v) else (float("nan"), 0.0)


def summary_table(results: dict) -> str:
    """Markdown table: mean +- std over seeds for each method."""
    cols = [("AUPRC", ("overall", "auprc")), ("AUROC", ("overall", "auroc")), ("F1", ("overall", "f1")),
            ("Recall", ("overall", "recall")), ("FPR", ("overall", "fpr")),
            ("Det@10%", ("early", "overall", "detected@10%")), ("Det@30%", ("early", "overall", "detected@30%")),
            ("Det@100%", ("early", "overall", "detected@100%"))]
    lines = ["| method | " + " | ".join(c for c, _ in cols) + " |", "|---|" + "---|" * len(cols)]
    for name, runs in results.items():
        cells = []
        for _, path in cols:
            vals = []
            for r in runs:
                for key in path:
                    r = r.get(key) if isinstance(r, dict) else None
                vals.append(r)
            m, s = _mean_std(vals)
            cells.append(f"{m:.3f} ± {s:.3f}")
        lines.append(f"| {name} | " + " | ".join(cells) + " |")
    return "\n".join(lines)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--raw", default=None, help="raw data directory (default data/raw/<dataset>)")
    ap.add_argument("--seeds", type=int, nargs="+", default=[0])
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--window-s", type=float, default=None)
    ap.add_argument("--max-net", type=int, default=96)
    ap.add_argument("--variants", nargs="+", default=list(VARIANTS))
    ap.add_argument("--no-baselines", action="store_true")
    ap.add_argument("--cross-day", action="store_true",
                    help="split benign once over the whole timeline instead of inside each day")
    args = ap.parse_args(argv)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    windows, window_s = load(args.dataset, args)
    paired = windows.host_x is not None and windows.net_x is not None
    print(f"{args.dataset}: {len(windows)} windows, {int(windows.label.sum())} attack, paired={paired}")
    results: dict[str, list] = {}
    loco = []
    for seed in args.seeds:
        base = RunConfig(window_s=window_s, epochs=args.epochs, seed=seed, per_group_split=not args.cross_day)
        raw_split = time_split(windows, base.train_frac, base.val_frac, window_s, seed, base.per_group_split)
        for name in args.variants:
            overrides, drop = VARIANTS[name]
            if not paired and name in PAIRED_ONLY:
                continue
            cfg = copy.deepcopy(base)
            overrides = dict(overrides)
            cfg.stats_view = overrides.pop("_stats_view", True)
            cfg.model = ModelConfig(**overrides)
            split, _ = scale_split(raw_split, drop)
            res = run_split(split, cfg, out / f"seed{seed}" / name, log=lambda *_: None)
            results.setdefault(name, []).append(res)
            o = res["overall"]
            print(f"seed {seed} {name:12s} AUPRC {o['auprc']:.3f} F1 {o['f1']:.3f} recall {o['recall']:.3f} "
                  f"FPR {o['fpr']:.3f} [{res['seconds']}s]", flush=True)
        if not args.no_baselines:
            split, _ = scale_split(raw_split)
            for name, res in run_unsupervised(split, base).items():
                results.setdefault(name, []).append(res)
                o = res["overall"]
                print(f"seed {seed} {name:12s} AUPRC {o['auprc']:.3f} F1 {o['f1']:.3f} recall {o['recall']:.3f} "
                      f"FPR {o['fpr']:.3f}", flush=True)
            sup = run_supervised_loco(split, base)
            if sup:
                loco.append(sup)
                results.setdefault("rf_all_seen", []).append(sup["all_seen"])
    table = summary_table(results)
    (out / "results.json").write_text(json.dumps({"results": results, "supervised_loco": loco}, indent=2, default=str))
    (out / "summary.md").write_text(table + "\n", encoding="utf8")
    print("\n" + table)


if __name__ == "__main__":
    main()
