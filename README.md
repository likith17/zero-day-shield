# zero-day-shield

The primary objective is to determine whether jointly modelling host and network behaviour can identify emerging threats before a complete attack sequence is observed.

ZeroDayShield learns what normal host and network activity look like, and how the two relate, from benign telemetry only. It raises an alert when a time window departs from that learned behaviour. No attack labels are used for training, calibration or the alert threshold. It does not find vulnerabilities; it flags unusual behaviour.

## How it works

1. **Windows.** Host and network events are paired by fixed time window ([events.py](src/zds/data/events.py)). The datasets have no per-event key linking the two layers, so the shared window is the link.
2. **Per-layer temporal encoders** ([zds.py](src/zds/models/zds.py)), trained with masked-event reconstruction and temporal-order prediction.
3. **Cross-view contrastive alignment** between the host and network view of the same window.
4. **Fusion** block trained to rebuild masked events of one layer with the other layer visible.
5. **Score** ([pipeline.py](src/zds/pipeline.py)): reconstruction errors (two-sided: far easier than benign is as abnormal as far harder), order-anomaly logit, host/network mismatch, and nearest-benign-neighbour distance per embedding view plus a window-statistics view. Each is calibrated on benign validation windows; the two largest are averaged. The threshold is the benign validation quantile for a target false-positive rate (default 1%).

## Evaluation protocol

- Benign windows are split by time inside each recording (capture day): 50% train, 10% validation, 40% test. `--cross-day` splits the whole timeline once instead.
- Attack windows never reach the model. They are grouped into episodes; per category half the episodes go to the test set, the other half is available only to supervised baselines.
- **Held-out categories:** the supervised baseline is retrained once per category with that category removed, so it is unseen at test time. For ZeroDayShield every category is unseen by construction.
- **Early detection:** each attack episode is cut to its first 5/10/20/30% and we report whether an alert has fired by then, plus time-to-detection and fraction of the attack observed.
- Baselines ([baselines.py](src/zds/baselines.py)): Isolation Forest, PCA, one-class SVM, autoencoder, network-only variant of the model, Random Forest (closed-set).
- Ablations: no fusion, no contrastive, no temporal information, no order task, no statistics view, host only, network only.

## Run

```bash
pip install -e .[dev]
```

```bash
python -m pytest -q
```

```bash
python -m zds.experiment --dataset synthetic --out runs/synthetic
```

```bash
python -m zds.experiment --dataset cicids2018 --out runs/cicids2018 --seeds 0 1 2
```

Datasets and download steps are in [docs/DATA.md](docs/DATA.md). The synthetic generator exists to test the pipeline; it is not a benchmark.

## Status

| Dataset | Layers | Loader |
|---|---|---|
| Synthetic | host + network | done |
| CSE-CIC-IDS2018 | network | done |
| TON_IoT | host + network | host (Linux) done; network loader written but untested, waiting for `Network_dataset_*.csv` |
| UNSW-NB15 | network | waiting for `UNSW-NB15_1..4.csv` |
