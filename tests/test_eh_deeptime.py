"""Offline smoke tests for the illustrative deep-time PETM box model.

Runs without network and without pytest (``python tests/test_eh_deeptime.py``).
These are sanity/plausibility checks for an illustration, not validation gates.
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from eh_deeptime import petm  # noqa: E402


def test_run_petm_finite_and_shaped():
    r = petm.run_petm(m_inj=3000.0, t_end=300.0, n_out=321)
    for k in ("kyr", "pco2", "temp", "d13c_surf", "ph", "omega", "dic", "alk"):
        assert k in r, f"missing key {k}"
        assert r[k].shape == r["kyr"].shape
        assert np.all(np.isfinite(r[k])), f"non-finite values in {k}"


def test_control_run_is_steady():
    r = petm.run_petm(m_inj=0.0, t_end=300.0)
    assert np.nanmax(np.abs(r["pco2"] - petm.PCO2_0)) < 1.0      # ppm
    assert np.nanmax(np.abs(r["temp"])) < 1e-3                    # K
    assert np.nanmax(np.abs(r["d13c_surf"] - petm.D13C_0)) < 1e-3


def test_larger_release_warms_more_and_lowers_d13c():
    small = petm.summarise(petm.run_petm(m_inj=2500.0, t_end=400.0))
    big = petm.summarise(petm.run_petm(m_inj=3500.0, t_end=400.0))
    assert big["peak_warming_K"] > small["peak_warming_K"]
    assert big["cie_permil"] < small["cie_permil"]               # more negative


def test_recovery_toward_background():
    r = petm.run_petm(m_inj=3000.0, t_end=400.0)
    peak = np.nanmax(r["pco2"])
    late = r["pco2"][-1]
    assert late < 0.5 * (peak - petm.PCO2_0) + petm.PCO2_0       # well past the peak
    assert late < peak


def test_consensus_plausibility():
    s = petm.summarise(petm.run_petm(m_inj=3000.0, delta_inj=-50.0, ecs=3.0, t_end=400.0))
    assert 3.0 <= s["peak_warming_K"] <= 6.0                     # consensus ~5 +/- 1 K
    assert -4.5 <= s["cie_permil"] <= -2.5                       # consensus CIE
    assert 80.0 <= s["tau_rec_kyr"] <= 250.0                     # consensus ~100-200 kyr


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"PASS {name}")
    print("all eh_deeptime smoke tests passed")
