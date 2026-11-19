"""Synthetic host + network telemetry for pipeline tests and sanity checks.

NOT a benchmark. A hidden "activity" level drives both layers, so benign host and
network behaviour are correlated. Four attack types each break something different:

  scan        network-only: bursts of tiny short flows
  miner       host-only: CPU pinned high
  mismatch    cross-layer only: busy network with an idle host (each layer on its
              own stays inside its benign range)
  reorder     temporal only: the request -> response ordering of flows is reversed
"""
from __future__ import annotations

import numpy as np

from .events import BENIGN, EventTable

ATTACKS = ("scan", "miner", "mismatch", "reorder")
HOST_FEATURES = ["cpu", "mem", "procs", "disk_io", "ctx_switch", "net_io"]
NET_FEATURES = ["duration", "src_bytes", "dst_bytes", "src_pkts", "dst_pkts", "is_response"]


def _activity(t: np.ndarray, rng) -> np.ndarray:
    """Slow daily-like cycle in [0, 1] with a little drift."""
    return np.clip(0.5 + 0.35 * np.sin(2 * np.pi * t / 7200.0) + 0.1 * np.sin(2 * np.pi * t / 613.0)
                   + rng.normal(0, 0.03, len(t)), 0.02, 1.0)


def make_synthetic(hours: float = 24.0, n_episodes: int = 24, episode_s: float = 900.0, seed: int = 0):
    """Return (host, net) event tables. The first 60% of the timeline is attack-free."""
    rng = np.random.default_rng(seed)
    total = hours * 3600.0
    # Attack episodes live in the last 40% of the timeline, round-robin over types.
    starts = np.sort(rng.uniform(0.6 * total, total - episode_s, n_episodes))
    episodes = [(s, s + episode_s, ATTACKS[i % len(ATTACKS)]) for i, s in enumerate(starts)]

    def active(t):
        name = np.full(len(t), BENIGN, dtype=object)
        for s, e, kind in episodes:
            name[(t >= s) & (t < e)] = kind
        return name

    # ---- host: one sample every 5 s
    ht = np.arange(0, total, 5.0)
    ha = _activity(ht, rng)
    hk = active(ht)
    ha_eff = np.where(hk == "mismatch", 0.08, ha)  # idle host during mismatch
    cpu = 5 + 70 * ha_eff + rng.normal(0, 3, len(ht))
    cpu = np.where(hk == "miner", 97 + rng.normal(0, 1, len(ht)), cpu)
    host = np.stack([
        cpu,
        30 + 25 * ha_eff + rng.normal(0, 2, len(ht)),
        80 + 60 * ha_eff + rng.normal(0, 4, len(ht)),
        np.exp(2 + 3 * ha_eff + rng.normal(0, 0.3, len(ht))),
        np.exp(5 + 2 * ha_eff + rng.normal(0, 0.2, len(ht))),
        np.exp(4 + 4 * ha_eff + rng.normal(0, 0.3, len(ht))),
    ], axis=1)
    host_label = np.isin(hk, ("miner", "mismatch")).astype(np.int8)
    host_tab = EventTable(ht, host, host_label, np.where(host_label == 1, hk, BENIGN), HOST_FEATURES)

    # ---- network: request/response flow pairs, rate follows activity
    grid = np.arange(0, total, 1.0)
    ga = _activity(grid, rng)
    gk = active(grid)
    rate = 0.05 + 0.5 * ga
    rate = np.where(gk == "mismatch", 0.5, rate)  # looks like a busy period
    rate = np.where(gk == "scan", 3.0, rate)
    n_pairs = rng.poisson(rate)
    pt = np.repeat(grid, n_pairs) + rng.uniform(0, 1, n_pairs.sum())
    pk = np.repeat(gk, n_pairs)
    pa = np.where(pk == "mismatch", 0.9, np.repeat(ga, n_pairs))
    m = len(pt)
    req = np.stack([
        np.exp(rng.normal(-2, 0.4, m)), np.exp(5 + pa + rng.normal(0, 0.3, m)),
        np.exp(4 + rng.normal(0, 0.3, m)), rng.poisson(4 + 4 * pa) + 1.0, rng.poisson(3, m) + 1.0, np.zeros(m),
    ], axis=1)
    resp = np.stack([
        np.exp(rng.normal(-1, 0.4, m)), np.exp(4 + rng.normal(0, 0.3, m)),
        np.exp(8 + 2 * pa + rng.normal(0, 0.4, m)), rng.poisson(3, m) + 1.0, rng.poisson(8 + 10 * pa) + 1.0, np.ones(m),
    ], axis=1)
    scan = pk == "scan"
    req[scan, :5] = np.stack([np.full(scan.sum(), 0.001), np.full(scan.sum(), 40.0), np.zeros(scan.sum()),
                              np.ones(scan.sum()), np.zeros(scan.sum())], axis=1)
    gap = np.where(pk == "reorder", -0.2, 0.2)  # response before request when reordered
    nt = np.concatenate([pt, pt + gap])
    nx = np.concatenate([req, resp])[:, :]
    keep = np.concatenate([np.ones(m, bool), ~scan])  # scans get no response
    nk = np.concatenate([pk, pk])
    net_label = np.isin(nk, ("scan", "mismatch", "reorder")).astype(np.int8)
    net_tab = EventTable(nt[keep], nx[keep], net_label[keep],
                         np.where(net_label == 1, nk, BENIGN)[keep], NET_FEATURES)
    return host_tab, net_tab
