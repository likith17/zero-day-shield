"""Split -> scale -> self-supervised training -> calibrated scoring -> evaluation."""
from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import torch

from .data.events import Scaler, Windows, summarise
from .eval.metrics import detection_metrics, evaluate, find_episodes
from .models.zds import ModelConfig, ZeroDayShield

LAYERS = ("host", "net")


@dataclass
class RunConfig:
    window_s: float = 60.0
    train_frac: float = 0.5  # of the timeline; benign windows only
    val_frac: float = 0.1  # benign windows only, used for calibration + threshold
    target_fpr: float = 0.01
    persist: int = 1  # consecutive windows over threshold needed for an alert
    aggregate: str = "top2"  # how component z-scores combine: mean | max | top2
    epochs: int = 30
    batch_size: int = 256
    lr: float = 3e-4
    weight_decay: float = 0.01
    knn_k: int = 5
    knn_bank: int = 20000
    seed: int = 0
    per_group_split: bool = True  # split benign inside each recording (day)
    stats_view: bool = True  # also score distance to benign windows in summary-statistic space
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    model: ModelConfig = field(default_factory=ModelConfig)


@dataclass
class Split:
    train: Windows  # benign only
    val: Windows  # benign only
    test: Windows  # benign after the validation period + held-back attack episodes
    known: Windows | None = None  # attack episodes a supervised baseline may train on


def time_split(w: Windows, train_frac: float, val_frac: float, window_s: float = 60.0, seed: int = 0,
               per_group: bool = True) -> Split:
    """Benign windows are split chronologically (train / val / test).

    With `per_group` the split is made inside every recording (capture day), so
    each day contributes to all three parts and day-to-day drift does not
    masquerade as attack signal. `per_group=False` splits the whole timeline once,
    which is the harder cross-day setting.

    Attack windows never reach the self-supervised model. They are grouped into
    episodes and, per category, half the episodes (rounded up) go to `test` and the
    rest to `known`. `known` exists only so that supervised baselines can be trained
    with whole categories left out; every method is evaluated on the same `test`.
    """
    benign = w.label == 0
    group = np.zeros(len(w), int) if w.group is None or not per_group else w.group
    # Position of each window inside its own recording, 0..1.
    pos = np.zeros(len(w))
    for g in np.unique(group):
        m = group == g
        t0, t1 = w.start[m].min(), w.start[m].max()
        pos[m] = (w.start[m] - t0) / max(t1 - t0, 1e-9)
    a, b = train_frac, train_frac + val_frac
    rng = np.random.default_rng(seed)
    attack = np.asarray(w.attack).astype(str)
    by_cat: dict[str, list] = {}
    for ep in find_episodes(w.start, w.label, attack, window_s):
        by_cat.setdefault(ep.attack, []).append(ep.idx)
    test_idx, known_idx = [], []
    for name in sorted(by_cat):
        eps = by_cat[name]
        order = rng.permutation(len(eps))
        n_test = -(-len(eps) // 2)
        test_idx += [eps[i] for i in order[:n_test]]
        known_idx += [eps[i] for i in order[n_test:]]
    cat = lambda parts: np.concatenate(parts) if parts else np.array([], dtype=int)
    test = np.sort(np.concatenate([np.flatnonzero((pos >= b) & benign), cat(test_idx)]))
    return Split(
        w.subset(np.flatnonzero((pos < a) & benign)),
        w.subset(np.flatnonzero((pos >= a) & (pos < b) & benign)),
        w.subset(test),
        w.subset(np.sort(cat(known_idx))),
    )


def scale_split(split: Split, drop: tuple[str, ...] = ()) -> tuple[Split, dict]:
    """Standardise each layer with benign-train statistics. `drop` removes layers."""
    scalers = {}
    parts = {k: getattr(split, k) for k in ("train", "val", "test", "known") if getattr(split, k) is not None}
    out = {k: Windows(v.start, v.label, v.attack, group=v.group) for k, v in parts.items()}
    for n in LAYERS:
        if n in drop or getattr(split.train, f"{n}_x") is None:
            continue
        sc = Scaler.fit(getattr(split.train, f"{n}_x"), getattr(split.train, f"{n}_mask"))
        scalers[n] = sc
        for k, v in parts.items():
            mask = getattr(v, f"{n}_mask")
            setattr(out[k], f"{n}_x", sc.transform(getattr(v, f"{n}_x"), mask))
            setattr(out[k], f"{n}_dt", getattr(v, f"{n}_dt"))
            setattr(out[k], f"{n}_mask", mask)
            setattr(out[k], f"{n}_n", np.log1p(getattr(v, f"{n}_n")).astype(np.float32))
    return Split(**out), scalers


def _tensors(w: Windows, device) -> dict:
    out = {}
    for n in LAYERS:
        if getattr(w, f"{n}_x") is not None:
            for part in ("x", "dt", "mask", "n"):
                out[f"{n}_{part}"] = torch.as_tensor(getattr(w, f"{n}_{part}"), device=device)
    return out


def _batches(data: dict, size: int, shuffle: bool, gen=None):
    n = len(next(iter(data.values())))
    order = torch.randperm(n, generator=gen, device="cpu") if shuffle else torch.arange(n)
    for i in range(0, n, size):
        idx = order[i:i + size].to(next(iter(data.values())).device)
        yield {k: v[idx] for k, v in data.items()}


def train_model(split: Split, cfg: RunConfig, log=print) -> ZeroDayShield:
    torch.manual_seed(cfg.seed)
    mc = cfg.model
    mc.host_dim = 0 if split.train.host_x is None else split.train.host_x.shape[-1]
    mc.net_dim = 0 if split.train.net_x is None else split.train.net_x.shape[-1]
    model = ZeroDayShield(mc).to(cfg.device)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    train, val = _tensors(split.train, cfg.device), _tensors(split.val, cfg.device)
    steps = cfg.epochs * max(1, -(-len(split.train) // cfg.batch_size))
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, cfg.lr, total_steps=steps, pct_start=0.1)
    gen = torch.Generator().manual_seed(cfg.seed)
    best, best_state = float("inf"), None
    for epoch in range(cfg.epochs):
        model.train()
        for batch in _batches(train, cfg.batch_size, True, gen):
            if len(next(iter(batch.values()))) < 2:
                continue
            loss, _ = model.loss(batch)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
        # Model selection on benign validation loss (no attack labels involved).
        model.eval()
        torch.manual_seed(1234)
        with torch.no_grad():
            losses = [float(model.loss(b)[0]) * len(next(iter(b.values()))) for b in _batches(val, cfg.batch_size, False)
                      if len(next(iter(b.values()))) >= 2]
        val_loss = sum(losses) / max(1, len(split.val))
        torch.manual_seed(cfg.seed + epoch + 1)
        if val_loss < best:
            best, best_state = val_loss, {k: v.detach().clone() for k, v in model.state_dict().items()}
        if epoch % max(1, cfg.epochs // 10) == 0 or epoch == cfg.epochs - 1:
            log(f"epoch {epoch + 1:3d}/{cfg.epochs}  val_loss {val_loss:.4f}  best {best:.4f}")
    model.load_state_dict(best_state)
    return model.eval()


def describe(model: ZeroDayShield, w: Windows, cfg: RunConfig) -> dict[str, np.ndarray]:
    chunks: dict[str, list] = {}
    for batch in _batches(_tensors(w, cfg.device), 1024, False):
        for k, v in model.describe(batch).items():
            chunks.setdefault(k, []).append(v.float().cpu().numpy())
    out = {k: np.concatenate(v) for k, v in chunks.items()}
    if cfg.stats_view:
        out["embed_stats"] = np.clip(summarise(w), -20, 20)
    return out


@dataclass
class Scorer:
    """Turns raw evidence into one anomaly score using benign windows only.

    Evidence = the model's raw components plus, for every embedding view, the mean
    distance to its k nearest benign-training neighbours. Each component is shifted
    by its benign-validation median and divided by (99th percentile - median), so
    1.0 means "as extreme as the benign 1% tail" for every component.
    Reconstruction errors are scored two-sided: a window that is far *easier* to
    reconstruct than benign traffic (e.g. a flood of identical flows) is as
    abnormal as one that is far harder. `top2` aggregation averages the two
    largest components, so one strong signal is not averaged away.
    """

    banks: dict  # view -> (bank tensor, mean, std)
    k: int
    center: dict
    spread: dict  # component -> (upper scale, lower scale or None)
    aggregate: str = "top2"
    threshold: float = 0.0

    def components(self, desc: dict) -> dict[str, np.ndarray]:
        comp = {k: v for k, v in desc.items() if not k.startswith("embed_")}
        for view, (bank, mu, sd) in self.banks.items():
            z = torch.as_tensor((desc[f"embed_{view}"] - mu) / sd, device=bank.device)
            dists = [torch.cdist(z[i:i + 2048], bank).topk(self.k, largest=False).values.mean(1).cpu().numpy()
                     for i in range(0, len(z), 2048)]
            comp[f"{view}_knn"] = np.concatenate(dists)
        return comp

    def zscores(self, desc: dict) -> dict[str, np.ndarray]:
        comp = self.components(desc)
        out = {}
        for k in sorted(self.center):
            hi, lo = self.spread[k]
            d = comp[k] - self.center[k]
            out[k] = d / hi if lo is None else np.maximum(d / hi, -d / lo)
        return out

    def score(self, desc: dict) -> np.ndarray:
        z = np.stack(list(self.zscores(desc).values()))
        if self.aggregate == "mean":
            return z.mean(0)
        if self.aggregate == "max" or len(z) < 2:
            return z.max(0)
        return np.sort(z, axis=0)[-2:].mean(0)


def fit_scorer(train_desc: dict, val_desc: dict, cfg: RunConfig) -> Scorer:
    rng = np.random.default_rng(cfg.seed)
    n = len(next(iter(train_desc.values())))
    pick = rng.choice(n, cfg.knn_bank, replace=False) if n > cfg.knn_bank else np.arange(n)
    banks = {}
    for key, emb in train_desc.items():
        if key.startswith("embed_"):
            emb = emb[pick]
            mu, sd = emb.mean(0), emb.std(0) + 1e-6
            banks[key[6:]] = (torch.as_tensor((emb - mu) / sd, device=cfg.device), mu, sd)
    sc = Scorer(banks, cfg.knn_k, {}, {}, cfg.aggregate)
    for k, v in sc.components(val_desc).items():
        lo, med, hi = np.percentile(v, [1, 50, 99])
        two_sided = k.endswith(("_recon", "_fused"))
        sc.center[k] = float(med)
        sc.spread[k] = (float(max(hi - med, 1e-6)), float(max(med - lo, 1e-6)) if two_sided else None)
    sc.threshold = float(np.quantile(sc.score(val_desc), 1 - cfg.target_fpr))
    return sc


def run(windows: Windows, cfg: RunConfig, drop: tuple[str, ...] = (), out_dir: str | Path | None = None,
        log=print) -> dict:
    """Full experiment on prepared windows. Returns the metrics dictionary."""
    split, _ = scale_split(time_split(windows, cfg.train_frac, cfg.val_frac, cfg.window_s, cfg.seed, cfg.per_group_split), drop)
    return run_split(split, cfg, out_dir, log)


def run_split(split: Split, cfg: RunConfig, out_dir: str | Path | None = None, log=print) -> dict:
    """Train, calibrate and evaluate on an already scaled split."""
    t0 = time.time()
    log(f"windows: train {len(split.train)}  val {len(split.val)}  test {len(split.test)} "
        f"({int(split.test.label.sum())} attack)")
    model = train_model(split, cfg, log)
    scorer = fit_scorer(describe(model, split.train, cfg), describe(model, split.val, cfg), cfg)
    test_desc = describe(model, split.test, cfg)
    score = scorer.score(test_desc)
    result = evaluate(score, split.test, scorer.threshold, cfg.window_s, cfg.persist)
    result["component_auprc"] = {k: detection_metrics(z, split.test.label, np.inf)["auprc"]
                                 for k, z in scorer.zscores(test_desc).items()}
    result["config"] = asdict(cfg)
    result["seconds"] = round(time.time() - t0, 1)
    if out_dir:
        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        (out / "metrics.json").write_text(json.dumps(result, indent=2, default=str))
        np.savez_compressed(out / "scores.npz", score=score, label=split.test.label,
                            attack=split.test.attack.astype(str), start=split.test.start)
        torch.save(model.state_dict(), out / "model.pt")
    return result
