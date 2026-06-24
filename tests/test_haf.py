"""Offline smoke tests for the illustrative deep-time HAF (haf.py).

Runs without network and without pytest (``python tests/test_haf.py``).
These are sanity/plausibility checks for an illustration, not validation gates.
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from eh_deeptime import haf  # noqa: E402


def test_default_trajectory_shaped_and_finite():
    t, co2 = haf.default_co2_trajectory(n=120)
    assert t.shape == (120,)
    assert co2.shape == (120,)
    assert np.all(np.isfinite(t))
    assert np.all(np.isfinite(co2))
    assert np.all(np.diff(t) > 0)                      # time increases
    assert np.all(co2 >= haf.CO2_FLOOR_PPM - 1e-9)     # clamped above the floor
    assert np.all(co2 <= haf.CO2_CEIL_PPM + 1e-9)      # clamped below the ceiling


def test_deeptime_haf_finite_and_shaped():
    r = haf.deeptime_haf(seed=0)
    for k in ("t_myr", "co2", "haf", "t_global_C"):
        assert k in r, f"missing key {k}"
        assert r[k].shape == r["t_myr"].shape
        assert np.all(np.isfinite(r[k])), f"non-finite values in {k}"


def test_haf_in_unit_interval():
    r = haf.deeptime_haf(seed=0)
    assert np.all(r["haf"] >= 0.0)
    assert np.all(r["haf"] <= 1.0)


def test_global_temperature_tracks_co2():
    # global-mean temperature should increase monotonically with pCO2: drive the
    # HAF with a monotone CO2 ramp and check the EBM global mean is monotone-ish.
    co2 = np.linspace(250.0, 4000.0, 12)
    r = haf.deeptime_haf(co2_ppm=co2, seed=0)
    tg = r["t_global_C"]
    assert np.all(np.diff(tg) > 0)                     # warmer at higher CO2
    # temperature is ~logarithmic in CO2 (radiative forcing ~ ln CO2), so check a
    # strong positive correlation against ln(CO2) rather than raw CO2.
    assert np.corrcoef(np.log(co2), tg)[0, 1] > 0.95


def test_haf_lower_at_hot_peak_than_temperate_baseline():
    # a hot, high-CO2 peak should be less habitable than a temperate baseline
    temperate = haf.deeptime_haf(co2_ppm=np.array([300.0]), seed=0)
    hot_peak = haf.deeptime_haf(co2_ppm=np.array([5000.0]), seed=0)
    assert hot_peak["t_global_C"][0] > temperate["t_global_C"][0]   # hotter
    assert hot_peak["haf"][0] < temperate["haf"][0]                 # less habitable


def test_haf_minimum_at_hottest_time():
    # on the default synthetic trajectory the HAF minimum should coincide with a
    # hot, high-CO2 time (the least-habitable state), not a cool one.
    r = haf.deeptime_haf(seed=0)
    s = haf.summarise(r)
    assert s["haf_range"] > 0.0                        # HAF actually varies
    i_min = int(np.argmin(r["haf"]))
    # the least-habitable time is warmer than the trajectory's median climate
    assert r["t_global_C"][i_min] > float(np.median(r["t_global_C"]))


def test_reproducible_with_fixed_seed():
    r1 = haf.deeptime_haf(seed=0)
    r2 = haf.deeptime_haf(seed=0)
    assert np.allclose(r1["haf"], r2["haf"])
    assert np.allclose(r1["t_global_C"], r2["t_global_C"])


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"PASS {name}")
    print("all haf smoke tests passed")
