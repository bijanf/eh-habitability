"""Offline smoke tests for the illustrative 1-D energy-balance model (ebm.py).

Runs without network and without pytest (``python tests/test_ebm.py``).
These are sanity/plausibility checks for an illustration, not validation gates.
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from eh_deeptime import ebm  # noqa: E402


def test_solve_ebm_finite_and_shaped():
    r = ebm.solve_ebm(co2_ppm=280.0, n_lat=91)
    for k in ("lat", "T_C", "T_global_C", "ice_latitude_deg"):
        assert k in r, f"missing key {k}"
    assert r["lat"].shape == (91,)
    assert r["T_C"].shape == (91,)
    assert np.all(np.isfinite(r["lat"]))
    assert np.all(np.isfinite(r["T_C"]))
    assert np.isfinite(r["T_global_C"])
    # ice line is either a finite latitude or nan (ice-free); both are valid
    assert np.isfinite(r["ice_latitude_deg"]) or np.isnan(r["ice_latitude_deg"])


def test_equator_warmer_than_poles():
    r = ebm.solve_ebm(co2_ppm=280.0)
    lat, T = r["lat"], r["T_C"]
    eq = T[np.argmin(np.abs(lat))]
    assert eq > T[0]      # warmer than south pole
    assert eq > T[-1]     # warmer than north pole
    assert eq == np.max(T)


def test_higher_co2_warms_global_mean():
    cold = ebm.solve_ebm(co2_ppm=180.0)
    base = ebm.solve_ebm(co2_ppm=280.0)
    warm = ebm.solve_ebm(co2_ppm=560.0)
    assert base["T_global_C"] > cold["T_global_C"]
    assert warm["T_global_C"] > base["T_global_C"]


def test_present_day_plausible_band():
    r = ebm.solve_ebm(co2_ppm=280.0)
    assert 10.0 <= r["T_global_C"] <= 20.0      # plausible present-day global mean


def test_ice_retreats_poleward_as_co2_rises():
    lo = ebm.solve_ebm(co2_ppm=200.0)["ice_latitude_deg"]
    hi = ebm.solve_ebm(co2_ppm=560.0)["ice_latitude_deg"]
    # at low CO2 ice must be present (a finite ice line)
    assert np.isfinite(lo)
    # at high CO2 the ice line has retreated poleward (larger |lat|) or vanished (nan)
    assert np.isnan(hi) or hi >= lo


def test_gradient_right_order():
    s = ebm.summarise(ebm.solve_ebm(co2_ppm=280.0))
    assert 30.0 <= s["gradient_C"] <= 60.0      # equator-to-pole gradient, right order


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"PASS {name}")
    print("all ebm smoke tests passed")
