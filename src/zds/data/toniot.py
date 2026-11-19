"""TON_IoT processed datasets -> paired host + network windows.

Host layer: the Linux process / memory / disk traces (one sampled process every
few seconds, Ubuntu atop output) and, when present, the Windows performance
counters. Each source keeps its own feature block inside one host event vector,
with a one-hot marking which block is filled.

Network layer: the Zeek-derived `Network_dataset_*.csv` flow records.

The files carry no key that links a host record to a flow (host files have no IP
or machine id), so the two layers are paired by time window only.
"""
from __future__ import annotations

import zlib
from pathlib import Path

import numpy as np
import pandas as pd

from .events import BENIGN, EventTable, Windows, build_windows

CMD_BUCKETS = 16
LINUX_SOURCES = {
    "process": ("Linux_process_*.csv", ["TRUN", "TSLPI", "TSLPU", "NICE", "PRI", "RTPR", "CPUNR", "EXC", "CPU"]),
    "memory": ("linux_memory*.csv", ["MINFLT", "MAJFLT", "VSTEXT", "VSIZE", "RSIZE", "VGROW", "RGROW", "MEM"]),
    "disk": ("linux_disk_*.csv", ["RDDSK", "WRDSK", "WCANCL", "DSK"]),
}
NET_NUMERIC = ["duration", "src_bytes", "dst_bytes", "missed_bytes", "src_pkts", "src_ip_bytes", "dst_pkts",
               "dst_ip_bytes", "http_request_body_len", "http_response_body_len", "http_trans_depth"]
NET_CATEGORICAL = {
    "proto": ["tcp", "udp", "icmp"],
    "service": ["-", "dns", "http", "ssl", "ftp", "smb", "dce_rpc", "dhcp"],
    "conn_state": ["SF", "S0", "REJ", "RSTO", "RSTR", "S1", "S2", "S3", "SH", "SHR", "RSTOS0", "RSTRH", "OTH"],
}
NET_FLAGS = ["dns_query", "ssl_version", "http_method", "weird_name"]  # "-" means the field is absent


def _number(col: pd.Series) -> np.ndarray:
    """atop prints sizes like '148.2K' or '1.2M' in some files."""
    if col.dtype != object:
        return col.to_numpy(np.float32, na_value=0.0)
    text = col.astype(str).str.strip()
    mult = text.str[-1].map({"K": 1e3, "M": 1e6, "G": 1e9, "T": 1e12, "%": 1.0}).fillna(1.0)
    digits = text.str.replace(r"[KMGT%]$", "", regex=True)
    return (pd.to_numeric(digits, errors="coerce").fillna(0.0) * mult).to_numpy(np.float32)


def _labels(df: pd.DataFrame):
    label = pd.to_numeric(df["label"], errors="coerce").fillna(0).astype(int).to_numpy()
    kind = df["type"].astype(str).str.strip().str.lower().to_numpy()
    return (label > 0).astype(np.int8), np.where(label > 0, kind, BENIGN)


def _cmd_bucket(cmd: pd.Series) -> np.ndarray:
    codes, names = pd.factorize(cmd.astype(str))
    bucket = np.array([zlib.crc32(n.encode()) % CMD_BUCKETS for n in names], dtype=np.int64)
    return bucket[codes]


def load_linux(raw_dir: str | Path) -> EventTable | None:
    """Merge the Linux traces into one host event stream."""
    folder = Path(raw_dir) / "Processed_datasets" / "Processed_Linux_dataset"
    blocks = list(LINUX_SOURCES)
    width = sum(len(cols) for _, cols in LINUX_SOURCES.values())
    offsets = np.cumsum([0] + [len(LINUX_SOURCES[b][1]) for b in blocks])
    names = [f"{b}:{c}" for b in blocks for c in LINUX_SOURCES[b][1]] + [f"is_{b}" for b in blocks] + \
            [f"cmd_{i}" for i in range(CMD_BUCKETS)]
    ts, xs, labels, kinds = [], [], [], []
    for bi, block in enumerate(blocks):
        pattern, cols = LINUX_SOURCES[block]
        for f in sorted(folder.glob(pattern)):
            df = pd.read_csv(f, low_memory=False)
            t = pd.to_numeric(df["ts"], errors="coerce")
            df, t = df[t.notna()], t[t.notna()]
            x = np.zeros((len(df), width + len(blocks) + CMD_BUCKETS), np.float32)
            for ci, c in enumerate(cols):
                x[:, offsets[bi] + ci] = _number(df[c])
            x[:, width + bi] = 1.0
            x[np.arange(len(df)), width + len(blocks) + _cmd_bucket(df["CMD"])] = 1.0
            label, kind = _labels(df)
            ts.append(t.to_numpy(np.float64)); xs.append(x); labels.append(label); kinds.append(kind)
    if not ts:
        return None
    return EventTable(np.concatenate(ts), np.concatenate(xs), np.concatenate(labels), np.concatenate(kinds), names)


def load_network(raw_dir: str | Path, chunksize: int = 1_000_000) -> EventTable | None:
    folder = Path(raw_dir) / "Processed_datasets" / "Processed_Network_dataset"
    files = sorted(folder.glob("Network_dataset_*.csv"), key=lambda p: int(p.stem.split("_")[-1]))
    names = NET_NUMERIC + ["dst_port_log", "dst_port_wellknown", "src_port_log"] + \
            [f"{c}={v}" for c, vals in NET_CATEGORICAL.items() for v in vals] + [f"has_{c}" for c in NET_FLAGS]
    ts, xs, labels, kinds = [], [], [], []
    for f in files:
        for df in pd.read_csv(f, chunksize=chunksize, low_memory=False):
            t = pd.to_numeric(df["ts"], errors="coerce")
            df, t = df[t.notna()], t[t.notna()]
            cols = [pd.to_numeric(df[c], errors="coerce").fillna(0).to_numpy(np.float32) for c in NET_NUMERIC]
            dport = pd.to_numeric(df["dst_port"], errors="coerce").fillna(0).to_numpy(np.float32)
            sport = pd.to_numeric(df["src_port"], errors="coerce").fillna(0).to_numpy(np.float32)
            cols += [np.log1p(dport), (dport < 1024).astype(np.float32), np.log1p(sport)]
            for c, vals in NET_CATEGORICAL.items():
                text = df[c].astype(str).str.strip()
                cols += [(text == v).to_numpy(np.float32) for v in vals]
            cols += [(df[c].astype(str).str.strip() != "-").to_numpy(np.float32) for c in NET_FLAGS]
            label, kind = _labels(df)
            ts.append(t.to_numpy(np.float64)); xs.append(np.stack(cols, axis=1)); labels.append(label); kinds.append(kind)
    if not ts:
        return None
    return EventTable(np.concatenate(ts), np.concatenate(xs), np.concatenate(labels), np.concatenate(kinds), names)


def load_windows(raw_dir: str | Path = "data/raw/toniot", window_s: float = 60.0, max_host: int = 48,
                 max_net: int = 96, cache: str | Path | None = "data/cache", log=print) -> Windows:
    cached = Path(cache) / f"toniot_w{int(window_s)}_h{max_host}_n{max_net}.npz" if cache else None
    if cached is not None and cached.exists():
        return Windows.load(cached)
    host, net = load_linux(raw_dir), load_network(raw_dir)
    if host is None and net is None:
        raise FileNotFoundError(f"no TON_IoT processed CSVs under {raw_dir}; see docs/DATA.md")
    for name, t in (("host", host), ("network", net)):
        log(f"{name}: " + ("missing" if t is None else f"{len(t)} events, {int(t.label.sum())} attack"))
    w = build_windows(host, net, window_s, max_host, max_net, require_both=host is not None and net is not None)
    log(f"{len(w)} windows, {int(w.label.sum())} attack {sorted(set(w.attack[w.label == 1].astype(str)))}")
    if cached is not None and host is not None and net is not None:  # never cache a partial dataset
        cached.parent.mkdir(parents=True, exist_ok=True)
        w.save(cached)
    return w
