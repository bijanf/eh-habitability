"""Offline smoke tests for the illustrative guild-mixture habitability model.

Runs without network and without pytest (``python tests/test_habitability.py``).
These are sanity/plausibility checks on the METHOD using SYNTHETIC data drawn
from published tolerance envelopes -- they are NOT validation against any real
extremophile growth database (none is bundled or fetched).
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from eh_deeptime import habitability as hab  # noqa: E402


def test_synthetic_data_shapes_and_labels():
    rng = np.random.default_rng(0)
    X, y, gid = hab.make_synthetic_data(rng, n=1000)
    assert X.shape == (1000, 3)
    assert y.shape == (1000,) and gid.shape == (1000,)
    assert set(np.unique(y)).issubset({0, 1})
    assert set(np.unique(gid)).issubset(set(range(len(hab.GUILD_NAMES))))
    assert np.all(np.isfinite(X))


def test_p_hab_high_at_box_centre():
    rng = np.random.default_rng(0)
    models = hab.fit_all(rng)
    for g, name in enumerate(hab.GUILD_NAMES):
        c = hab.box_centre(name)[None, :]
        p = float(hab.p_hab(c, models[g])[0])
        assert p > 0.7, f"{name}: p_hab at box centre = {p:.3f} (expected > 0.7)"


def test_p_hab_low_far_outside_all_boxes():
    rng = np.random.default_rng(0)
    models = hab.fit_all(rng)
    # a point well outside every guild box: very hot, very alkaline, very dry
    far = np.array([[150.0, 13.0, 0.30]])
    p = float(hab.p_hab_mixture(far, models)[0])
    assert p < 0.3, f"mixture p_hab far outside all boxes = {p:.3f} (expected < 0.3)"


def test_mixture_dominates_single_guild():
    rng = np.random.default_rng(0)
    models = hab.fit_all(rng)
    rng2 = np.random.default_rng(1)
    X, _, _ = hab.make_synthetic_data(rng2, n=500)
    mix = hab.p_hab_mixture(X, models)
    for m in models:
        single = hab.p_hab(X, m)
        assert np.all(mix >= single - 1e-9), "mixture must be >= each single-guild p"


def test_laplace_ci_ordered_and_bounded():
    rng = np.random.default_rng(0)
    m = hab.fit_one(rng, 0)
    assert "cov" in m and m["cov"].shape[0] == m["cov"].shape[1]
    X = np.array([[80.0, 5.0, 0.95], [150.0, 13.0, 0.30]])  # in-niche, far-out
    mean, lo, hi = hab.p_hab_ci(X, m)
    assert np.all(lo <= mean + 1e-9) and np.all(mean <= hi + 1e-9)   # ordered
    assert np.all((lo >= 0) & (hi <= 1) & (mean >= 0) & (mean <= 1))  # in [0,1]
    assert mean[0] > mean[1]                                          # in-niche higher


def test_cross_validation_metrics_sane():
    rng = np.random.default_rng(0)
    cv = hab.grouped_cross_validate(rng, k=5)
    for key in ("log_loss", "brier", "calibration_slope", "auc", "baseline_brier"):
        assert key in cv, f"missing CV key {key}"
        assert np.isfinite(cv[key]), f"non-finite CV metric {key}"
    assert 0.0 <= cv["auc"] <= 1.0
    assert cv["auc"] > 0.5                              # beats coin-flip ranking
    assert cv["brier"] < cv["baseline_brier"]          # beats the base-rate predictor
    assert 0.5 <= cv["calibration_slope"] <= 1.5       # sane calibration band


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"PASS {name}")
    print("all habitability smoke tests passed")
