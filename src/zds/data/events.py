"""Canonical event tables and time-window pairing of host and network telemetry.

Every dataset loader produces one `EventTable` per modality. `build_windows` then
cuts the timeline into fixed windows and, for each window, emits the host and the
network event sequence observed in it. That shared window is the only link between
the two layers (the datasets carry no per-event join key).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

BENIGN = "normal"


@dataclass
class EventTable:
    """Time-ordered events of one modality. `x` is numeric, not yet standardised."""

    ts: np.ndarray  # [N] seconds, float64, sorted ascending
    x: np.ndarray  # [N, F] float32
    label: np.ndarray  # [N] 0 benign / 1 attack
    attack: np.ndarray  # [N] attack category, BENIGN for benign rows
    features: list[str] = field(default_factory=list)

    def __post_init__(self):
        order = np.argsort(self.ts, kind="stable")
        self.ts = np.asarray(self.ts, dtype=np.float64)[order]
        self.x = np.asarray(self.x, dtype=np.float32)[order]
        self.label = np.asarray(self.label, dtype=np.int8)[order]
        self.attack = np.asarray(self.attack, dtype=object)[order]

    def __len__(self):
        return len(self.ts)


@dataclass
class Windows:
    """Paired fixed-length windows. A modality that is absent has `*_x is None`."""

    start: np.ndarray  # [W] window start time (s)
    label: np.ndarray  # [W] 1 if any attack event falls in the window
    attack: np.ndarray  # [W] dominant attack category, BENIGN otherwise
    host_x: np.ndarray | None = None  # [W, Lh, Fh]
    host_dt: np.ndarray | None = None  # [W, Lh] seconds since window start
    host_mask: np.ndarray | None = None  # [W, Lh] True where an event exists
    host_n: np.ndarray | None = None  # [W] events in the window before thinning
    net_x: np.ndarray | None = None
    net_dt: np.ndarray | None = None
    net_mask: np.ndarray | None = None
    net_n: np.ndarray | None = None
    group: np.ndarray | None = None  # [W] recording id (e.g. capture day)

    def __len__(self):
        return len(self.start)

    def subset(self, idx) -> "Windows":
        return Windows(**{k: None if v is None else v[idx] for k, v in vars(self).items()})

    def save(self, path) -> None:
        arrays = {k: v for k, v in vars(self).items() if v is not None}
        arrays["attack"] = self.attack.astype(str)
        np.savez_compressed(path, **arrays)

    @classmethod
    def load(cls, path) -> "Windows":
        with np.load(path) as z:
            data = {k: z[k] for k in z.files}
        data["attack"] = data["attack"].astype(object)
        return cls(**data)


def concat_windows(parts: list["Windows"]) -> "Windows":
    """Stack windows from several recordings; each part becomes its own group."""
    data = {}
    for name in vars(parts[0]):
        vals = [getattr(p, name) for p in parts]
        data[name] = None if vals[0] is None else np.concatenate(vals)
    data["group"] = np.concatenate([np.full(len(p), i, dtype=np.int32) for i, p in enumerate(parts)])
    out = Windows(**data)
    return out.subset(np.argsort(out.start, kind="stable"))


def _pack(table: EventTable, edges: np.ndarray, max_len: int):
    """Slice a table into windows; windows with more than `max_len` events are
    thinned to evenly spaced events so the whole window stays covered in order."""
    n_win = len(edges) - 1
    lo = np.searchsorted(table.ts, edges[:-1], side="left")
    hi = np.searchsorted(table.ts, edges[1:], side="left")
    x = np.zeros((n_win, max_len, table.x.shape[1]), dtype=np.float32)
    dt = np.zeros((n_win, max_len), dtype=np.float32)
    mask = np.zeros((n_win, max_len), dtype=bool)
    n_attack = np.zeros(n_win, dtype=np.int32)
    attack = np.full(n_win, BENIGN, dtype=object)
    for w in range(n_win):
        a, b = lo[w], hi[w]
        if b <= a:
            continue
        idx = np.arange(a, b) if b - a <= max_len else np.linspace(a, b - 1, max_len).round().astype(int)
        k = len(idx)
        x[w, :k] = table.x[idx]
        dt[w, :k] = table.ts[idx] - edges[w]
        mask[w, :k] = True
        bad = table.label[a:b] == 1
        n_attack[w] = int(bad.sum())
        if n_attack[w]:
            names, counts = np.unique(table.attack[a:b][bad].astype(str), return_counts=True)
            attack[w] = names[counts.argmax()]
    return x, dt, mask, n_attack, attack, (hi - lo).astype(np.float32)


def build_windows(
    host: EventTable | None,
    net: EventTable | None,
    window_s: float = 60.0,
    max_host: int = 32,
    max_net: int = 64,
    require_both: bool = True,
) -> Windows:
    """Pair host and network events that fall in the same `window_s` window."""
    tables = [t for t in (host, net) if t is not None and len(t)]
    if not tables:
        raise ValueError("need at least one non-empty modality")
    # Paired mode only covers the period where both layers were recorded.
    t0 = max(t.ts[0] for t in tables) if require_both else min(t.ts[0] for t in tables)
    t1 = min(t.ts[-1] for t in tables) if require_both else max(t.ts[-1] for t in tables)
    if t1 <= t0:
        raise ValueError("host and network telemetry do not overlap in time")
    if (t1 - t0) / window_s > 5e6:
        raise ValueError(f"time span {t1 - t0:.0f}s gives too many windows; check for stray timestamps")
    edges = np.arange(t0, t1 + window_s, window_s)
    n_win = len(edges) - 1
    out = Windows(start=edges[:-1].copy(), label=np.zeros(n_win, np.int8),
                  attack=np.full(n_win, BENIGN, dtype=object), group=np.zeros(n_win, np.int32))
    counts = np.zeros(n_win, np.int32)
    present = []
    for name, table, max_len in (("host", host, max_host), ("net", net, max_net)):
        if table is None or not len(table):
            continue
        x, dt, mask, n_attack, attack, n_events = _pack(table, edges, max_len)
        setattr(out, f"{name}_n", n_events)
        setattr(out, f"{name}_x", x)
        setattr(out, f"{name}_dt", dt)
        setattr(out, f"{name}_mask", mask)
        present.append(mask.any(1))
        better = n_attack > counts  # the layer with more attack events names the window
        out.attack[better] = attack[better]
        counts = np.maximum(counts, n_attack)
    out.label = (counts > 0).astype(np.int8)
    keep = np.logical_and.reduce(present) if require_both else np.logical_or.reduce(present)
    return out.subset(np.flatnonzero(keep))


@dataclass
class Scaler:
    """Robust per-feature scaling fitted on benign training events only."""

    center: np.ndarray
    scale: np.ndarray

    @classmethod
    def fit(cls, x: np.ndarray, mask: np.ndarray) -> "Scaler":
        ev = signed_log(x[mask])
        center = np.median(ev, axis=0)
        q1, q3 = np.percentile(ev, [25, 75], axis=0)
        scale = np.where(q3 - q1 > 1e-6, q3 - q1, np.maximum(ev.std(0), 1.0))
        return cls(center.astype(np.float32), scale.astype(np.float32))

    def transform(self, x: np.ndarray, mask: np.ndarray) -> np.ndarray:
        z = np.clip((signed_log(x) - self.center) / self.scale, -10, 10)
        return (z * mask[..., None]).astype(np.float32)


def signed_log(x: np.ndarray) -> np.ndarray:
    """Compress heavy-tailed counters (bytes, packets, durations)."""
    return np.sign(x) * np.log1p(np.abs(x))


def summarise(w: Windows) -> np.ndarray:
    """[W, D] summary of scaled windows: per layer the (log) event count plus the
    mean / std / min / max of every feature."""
    parts = []
    for n in ("host", "net"):
        x, mask = getattr(w, f"{n}_x"), getattr(w, f"{n}_mask")
        if x is None:
            continue
        m = mask[..., None].astype(np.float32)
        count = np.maximum(mask.sum(1, keepdims=True), 1).astype(np.float32)
        mean = (x * m).sum(1) / count
        std = np.sqrt((((x - mean[:, None]) ** 2) * m).sum(1) / count)
        hi = np.where(m > 0, x, -np.inf).max(1)
        lo = np.where(m > 0, x, np.inf).min(1)
        hi, lo = np.where(np.isfinite(hi), hi, 0), np.where(np.isfinite(lo), lo, 0)
        rate = getattr(w, f"{n}_n")  # already log1p-scaled by scale_split
        parts += [rate[:, None].astype(np.float32), mean, std, hi, lo]
    return np.concatenate(parts, axis=1).astype(np.float32)
