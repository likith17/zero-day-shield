"""Download the CSE-CIC-IDS2018 processed flow CSVs from the official public AWS bucket.

Resumable: re-running continues partial files and skips complete ones.
Usage: python scripts/download_cicids2018.py [out_dir]
"""
import re
import sys
import urllib.parse
import urllib.request
from pathlib import Path

BUCKET = "https://cse-cic-ids2018.s3.ca-central-1.amazonaws.com/"
PREFIX = "Processed Traffic Data for ML Algorithms/"


def listing():
    url = BUCKET + "?list-type=2&prefix=" + urllib.parse.quote(PREFIX)
    xml = urllib.request.urlopen(url, timeout=60).read().decode()
    pairs = re.findall(r"<Key>([^<]*)</Key>.*?<Size>(\d+)</Size>", xml)
    return [(k, int(s)) for k, s in pairs if int(s) > 0]


def fetch(key, size, out):
    dest = out / Path(key).name
    have = dest.stat().st_size if dest.exists() else 0
    if have == size:
        print(f"ok      {dest.name}", flush=True)
        return
    if have > size:
        dest.unlink()
        have = 0
    req = urllib.request.Request(BUCKET + urllib.parse.quote(key), headers={"Range": f"bytes={have}-"})
    with urllib.request.urlopen(req, timeout=120) as r, open(dest, "ab") as f:
        while chunk := r.read(1 << 22):
            f.write(chunk)
    got = dest.stat().st_size
    print(f"{'done' if got == size else 'PARTIAL':7s} {dest.name} {got}/{size}", flush=True)


if __name__ == "__main__":
    out = Path(sys.argv[1] if len(sys.argv) > 1 else "data/raw/cicids2018")
    out.mkdir(parents=True, exist_ok=True)
    for key, size in listing():
        for attempt in range(5):
            try:
                fetch(key, size, out)
                break
            except Exception as e:  # network hiccup: resume on next attempt
                print(f"retry {attempt + 1} {Path(key).name}: {e}", flush=True)
