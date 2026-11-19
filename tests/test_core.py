import numpy as np
import torch

from zds.data.events import BENIGN, EventTable, build_windows
from zds.data.synthetic import make_synthetic
from zds.eval.metrics import detection_metrics, early_detection, find_episodes, first_alert
from zds.models.zds import ModelConfig, ZeroDayShield, swap_segments
from zds.pipeline import RunConfig, run, time_split


def table(ts, label=None, attack=None, f=2):
    n = len(ts)
    label = np.zeros(n) if label is None else np.asarray(label)
    attack = np.where(label == 1, "x", BENIGN) if attack is None else attack
    return EventTable(np.asarray(ts, float), np.arange(n * f, dtype=np.float32).reshape(n, f), label, attack)


def test_windows_pair_by_time_and_label_from_either_layer():
    host = table([0, 30, 70, 130], label=[0, 0, 1, 0])
    net = table([5, 10, 65, 125, 185])
    w = build_windows(host, net, window_s=60, max_host=4, max_net=4)
    # Overlap is [5, 130] -> windows starting at 5, 65, 125 (net event at 185 is outside).
    assert list(w.start) == [5, 65, 125]
    assert list(w.label) == [0, 1, 0] and w.attack[1] == "x"
    assert w.host_mask.sum(1).tolist() == [1, 1, 1] and w.net_mask.sum(1).tolist() == [2, 1, 1]


def test_dense_window_is_thinned_in_time_order():
    net = table(np.linspace(0, 59, 100))
    w = build_windows(None, net, window_s=60, max_net=10, require_both=False)
    assert w.net_mask[0].all() and np.all(np.diff(w.net_dt[0]) > 0)
    assert w.net_dt[0, 0] == 0 and w.net_dt[0, -1] > 55


def test_split_keeps_attacks_out_of_training_and_splits_episodes():
    host, net = make_synthetic(hours=12, n_episodes=8, seed=1)
    w = build_windows(host, net, 60, 12, 48)
    s = time_split(w, 0.5, 0.1, 60, seed=0)
    assert s.train.label.sum() == 0 and s.val.label.sum() == 0
    assert s.train.start.max() < s.val.start.min()
    assert s.test.label.sum() + s.known.label.sum() == w.label.sum()
    assert not set(s.test.start[s.test.label == 1]) & set(s.known.start)


def test_episodes_and_early_detection():
    start = np.arange(20) * 60.0
    label = np.zeros(20, int)
    label[2:12] = 1  # one 10-window episode
    label[16:18] = 1  # second episode after a gap
    attack = np.where(label == 1, "a", BENIGN)
    eps = find_episodes(start, label, attack, 60)
    assert [len(e.idx) for e in eps] == [10, 2]
    score = np.zeros(20)
    score[4] = 5  # third window of the first episode; second episode missed
    res = early_detection(score, start, label, attack, threshold=1, window_s=60)
    o = res["overall"]
    assert o["detected@20%"] == 0.0 and o["detected@30%"] == 0.5 and o["detected@100%"] == 0.5
    assert o["median_ttd_s"] == 180 and abs(o["median_fraction_observed"] - 0.3) < 1e-9
    assert first_alert(np.array([1, 0, 1, 1], bool), persist=2) == 3


def test_metrics_at_threshold():
    m = detection_metrics(np.array([0.9, 0.8, 0.1, 0.7]), np.array([1, 1, 0, 0]), 0.5)
    assert m["recall"] == 1 and m["precision"] == 2 / 3 and m["fpr"] == 0.5


def test_swap_segments_only_moves_valid_events():
    x = torch.arange(12.0).reshape(1, 6, 2).repeat(64, 1, 1)
    mask = torch.zeros(64, 6, dtype=torch.bool)
    mask[:, :4] = True
    torch.manual_seed(0)
    xs, target = swap_segments(x, torch.zeros(64, 6), mask)
    assert 0 < target.sum() < 64
    flipped = xs[target == 1][0]
    assert torch.equal(flipped[:4], x[0, [2, 3, 0, 1]]) and torch.equal(flipped[4:], x[0, 4:])
    assert torch.equal(xs[target == 0][0], x[0])


def test_model_handles_single_layer_and_scores_are_deterministic():
    batch = {"net_x": torch.randn(8, 10, 5), "net_dt": torch.rand(8, 10), "net_mask": torch.ones(8, 10, dtype=torch.bool),
             "net_n": torch.rand(8) * 5}
    batch["net_mask"][0, 3:] = False
    model = ZeroDayShield(ModelConfig(net_dim=5, d_model=32, n_layers=1)).eval()
    loss, parts = model.loss(batch)
    assert torch.isfinite(loss) and "contrast" not in parts
    a, b = model.describe(batch), model.describe(batch)
    assert all(torch.equal(a[k], b[k]) for k in a) and "embed_net" in a


def test_fused_model_detects_cross_layer_mismatch_that_single_layers_miss():
    host, net = make_synthetic(hours=36, n_episodes=24, seed=0)
    w = build_windows(host, net, 60, 12, 48)
    recall = {}
    for name, drop in (("full", ()), ("net", ("host",))):
        res = run(w, RunConfig(epochs=12, model=ModelConfig()), drop=drop, log=lambda *_: None)
        recall[name] = res["by_attack"]["mismatch"]["recall"]
    assert recall["full"] > 0.8 and recall["net"] < 0.3
