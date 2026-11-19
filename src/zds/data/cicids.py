"""CSE-CIC-IDS2018 processed flow CSVs -> network-only windows.

Quirks handled here:
  * timestamps are 12-hour without AM/PM; the capture ran roughly 08:00-19:00, so
    hours 1-7 are afternoon and get +12 h
  * some files repeat the header row mid-file, contain inf/NaN rates, and have a
    handful of rows stamped 1970
  * the 20-02-2018 file has four extra identifier columns (Flow ID, IPs, Src Port)
The processed files carry no IP addresses (except that one day), so windows are
testbed-wide rather than per host.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .events import BENIGN, EventTable, Windows, build_windows, concat_windows

DROP = ["Flow ID", "Src IP", "Src Port", "Dst IP", "Timestamp", "Label", "Dst Port", "Protocol"]


def _features(df: pd.DataFrame) -> tuple[np.ndarray, list[str]]:
    num = df.drop(columns=[c for c in DROP if c in df.columns]).apply(pd.to_numeric, errors="coerce")
    port = pd.to_numeric(df["Dst Port"], errors="coerce").fillna(0)
    proto = pd.to_numeric(df["Protocol"], errors="coerce").fillna(-1)
    extra = pd.DataFrame({
        "dst_port_log": np.log1p(port), "dst_port_wellknown": (port < 1024).astype(np.float32),
        "proto_tcp": (proto == 6).astype(np.float32), "proto_udp": (proto == 17).astype(np.float32),
    }, index=df.index)
    out = pd.concat([num, extra], axis=1)
    x = out.to_numpy(dtype=np.float32, na_value=0.0)
    np.nan_to_num(x, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
    return x, list(out.columns)


def load_day(path: str | Path, chunksize: int = 500_000) -> EventTable:
    ts, xs, labels, names = [], [], [], []
    for chunk in pd.read_csv(path, chunksize=chunksize, low_memory=False):
        chunk = chunk[chunk["Label"] != "Label"]  # repeated header rows
        t = pd.to_datetime(chunk["Timestamp"], format="%d/%m/%Y %H:%M:%S", errors="coerce")
        chunk, t = chunk[t.notna()], t[t.notna()]
        t = t + pd.to_timedelta(np.where(t.dt.hour < 8, 12, 0), unit="h")
        x, names = _features(chunk)
        ts.append((t.astype("int64") // 10**9).to_numpy(np.float64))
        xs.append(x)
        labels.append(chunk["Label"].astype(str).str.strip().to_numpy())
    ts, x, label = np.concatenate(ts), np.concatenate(xs), np.concatenate(labels)
    # A few rows carry epoch-zero (1970) timestamps; keep only the capture day.
    day = ts // 86400
    days, counts = np.unique(day, return_counts=True)
    keep = day == days[counts.argmax()]
    ts, x, label = ts[keep], x[keep], label[keep]
    benign = np.char.lower(label.astype(str)) == "benign"
    # Flow timestamps have 1 s resolution; keep file order inside each second.
    return EventTable(ts, x, (~benign).astype(np.int8),
                      np.where(benign, BENIGN, label), names)


def load_windows(raw_dir: str | Path, window_s: float = 10.0, max_net: int = 96,
                 cache: str | Path | None = "data/cache", log=print) -> Windows:
    """Window every day separately (days are not contiguous) and concatenate."""
    files = sorted(Path(raw_dir).glob("*TrafficForML_CICFlowMeter.csv"))
    if not files:
        raise FileNotFoundError(f"no CIC-IDS2018 CSVs in {raw_dir}; see docs/DATA.md")
    parts = []
    for f in files:
        cached = Path(cache) / f"cicids2018_{f.stem}_w{int(window_s)}_n{max_net}.npz" if cache else None
        if cached is not None and cached.exists():
            parts.append(Windows.load(cached))
            continue
        table = load_day(f)
        w = build_windows(None, table, window_s, max_net=max_net, require_both=False)
        log(f"{f.name}: {len(table)} flows -> {len(w)} windows, {int(w.label.sum())} attack "
            f"{sorted(set(w.attack[w.label == 1].astype(str)))}")
        if cached is not None:
            cached.parent.mkdir(parents=True, exist_ok=True)
            w.save(cached)
        parts.append(w)
    return concat_windows(parts)
