# Datasets

All raw data lives under `data/raw/` (git-ignored). Free for academic research; cite the
dataset papers listed on each project page.

## TON_IoT (principal benchmark, host + network) — manual download

Project page: https://research.unsw.edu.au/projects/toniot-datasets
Folder (Microsoft sign-in required):
https://unsw-my.sharepoint.com/:f:/g/personal/z5025758_ad_unsw_edu_au/EvBTaetotpdGnW7rJQ8fCvYBh8063CNeY9W33MpRsarJaQ?e=yZlnxW

Download the `Processed_datasets` folder (network, Windows, Linux) into `data/raw/toniot/`.
The full processed files are needed, not the `Train_Test_datasets` samples: those are
shuffled subsamples and break the time ordering that windowing depends on.

## UNSW-NB15 (network only) — manual download

Project page: https://research.unsw.edu.au/projects/unsw-nb15-dataset
Folder (Microsoft sign-in required):
https://unsw-my.sharepoint.com/:f:/g/personal/z5025758_ad_unsw_edu_au/EnuQZZn3XuNBjgfcUu4DIVMBLCHyoLHqOswirpOQifr1ag?e=gKWkLS

Download `UNSW-NB15_1.csv` … `UNSW-NB15_4.csv` and `UNSW-NB15_features.csv` into
`data/raw/unsw_nb15/`.

## CSE-CIC-IDS2018 (network only) — scripted

Public AWS bucket `cse-cic-ids2018`, folder "Processed Traffic Data for ML Algorithms"
(10 CSVs, 6.89 GB). Resumable:

```bash
python scripts/download_cicids2018.py data/raw/cicids2018
```
