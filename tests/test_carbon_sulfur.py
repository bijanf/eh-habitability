"""Offline smoke tests for the closed carbon-sulfur-oxygen-alkalinity box model.

Runs without network and without pytest (``python tests/test_carbon_sulfur.py``).
These are sanity/plausibility checks for an illustration, not validation gates.
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from eh_deeptime import carbon_sulfur as cs  # noqa: E402
from eh_deeptime.petm import D13C_0, PCO2_0  # noqa: E402

_KEYS = ("kyr", "pco2", "temp", "ph", "omega", "o2", "d13c",
         "dic", "alk", "corg_cr", "ccarb_cr", "s_pyr", "s_sulf", "so4")


def test_outputs_finite_and_shaped():
    r = cs.run_csys(m_inj=0.0, t_end=300.0, n_out=321)
    for k in _KEYS:
        assert k in r, f"missing key {k}"
        assert r[k].shape == r["kyr"].shape, f"shape mismatch in {k}"
        assert np.all(np.isfinite(r[k])), f"non-finite values in {k}"
    assert "params" in r


def test_control_run_is_steady():
    r = cs.run_csys(m_inj=0.0, t_end=400.0)
    assert cs.steady_drift(r) < 0.01
    assert np.nanmax(np.abs(r["pco2"] - PCO2_0)) < 1.0          # ppm
    assert np.nanmax(np.abs(r["temp"])) < 1e-2                  # K
    assert np.nanmax(np.abs(r["d13c"] - D13C_0)) < 1e-2         # per mil


def test_total_carbon_conservation_control():
    # DIC + Corg_cr + Ccarb_cr changes only by net external input (degassing,
    # weathering, burial all balance in the control) -> ~0 change.
    r = cs.run_csys(m_inj=0.0, t_end=400.0)
    tot = r["dic"] + r["corg_cr"] + r["ccarb_cr"]
    rel = np.nanmax(np.abs(tot - tot[0])) / tot[0]
    assert rel < 1e-11, f"carbon not conserved in control: {rel}"


def test_total_sulfur_conservation_control():
    r = cs.run_csys(m_inj=0.0, t_end=400.0)
    tot = r["s_pyr"] + r["s_sulf"] + r["so4"]
    rel = np.nanmax(np.abs(tot - tot[0])) / tot[0]
    assert rel < 1e-12, f"sulfur not conserved in control: {rel}"


def test_total_sulfur_conservation_under_pulse():
    # The ocean-sulfate reservoir closes the S cycle, so total sulfur is conserved
    # even OFF steady state (under a carbon pulse that accelerates pyrite oxidation).
    r = cs.run_csys(m_inj=4000.0, t_end=400.0)
    tot = r["s_pyr"] + r["s_sulf"] + r["so4"]
    rel = np.nanmax(np.abs(tot - tot[0])) / tot[0]
    assert rel < 1e-12, f"sulfur not conserved under pulse: {rel}"


def test_o2_stays_positive():
    for m in (0.0, 3000.0, 5000.0):
        r = cs.run_csys(m_inj=m, t_end=400.0)
        assert np.all(r["o2"] > 0.0), f"O2 went non-positive for m_inj={m}"


def test_injection_warms_acidifies_and_draws_down_o2():
    ctrl = cs.run_csys(m_inj=0.0, t_end=400.0)
    pert = cs.run_csys(m_inj=3000.0, t_end=400.0)
    sc, sp = cs.summarise(ctrl), cs.summarise(pert)
    assert sp["peak_warming_K"] > sc["peak_warming_K"]          # warmer
    assert sp["cie_permil"] < sc["cie_permil"]                  # more negative CIE
    assert np.nanmin(pert["ph"]) < np.nanmin(ctrl["ph"])        # lower pH
    assert sp["d_o2_PAL"] < -1e-6                               # transient O2 drawdown


def test_larger_release_warms_more_and_lowers_d13c():
    small = cs.summarise(cs.run_csys(m_inj=2500.0, t_end=400.0))
    big = cs.summarise(cs.run_csys(m_inj=3500.0, t_end=400.0))
    assert big["peak_warming_K"] > small["peak_warming_K"]
    assert big["cie_permil"] < small["cie_permil"]


def test_consensus_target_self_consistent():
    # The illustrative flux constants were CHOSEN so a ~3000 GtC pulse lands in the
    # published PETM consensus bands; this checks the model still hits the tuning
    # target it was set to -- a self-consistency check, NOT an independent
    # validation against proxy data.
    s = cs.summarise(cs.run_csys(m_inj=3000.0, delta_inj=-50.0, t_end=400.0))
    assert 3.0 <= s["peak_warming_K"] <= 6.0          # tuning target (consensus ~5 K)
    assert -4.5 <= s["cie_permil"] <= -2.5            # tuning target (consensus CIE)


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"PASS {name}")
    print("all carbon_sulfur smoke tests passed")
