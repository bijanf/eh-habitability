"""Offline smoke tests for the coupled-event deep-time HAF (haf.py).

Runs without network and without pytest (``python tests/test_haf.py``).
These are sanity/plausibility checks for an illustration, not validation gates.

The HAF is driven ENTIRELY by forward-model output: the carbon_sulfur box model's
pCO2(t) and surface-ocean pH(t), the EBM SST response, and a fixed open-marine
water activity. There is no synthetic, analytic or proxy time-series; these tests
also guard that the old fabricated trajectory/proxy code stays removed.
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from eh_deeptime import haf, carbon_sulfur, habitability  # noqa: E402

# small grids keep the offline tests fast while preserving the behaviour
_CSYS_KW = dict(t_dur=5.0, delta_inj=-50.0, t_end=300.0, n_out=121)
_N_LAT = 31


def _models():
    return habitability.fit_all(np.random.default_rng(0))


def test_no_fabricated_trajectory_code_remains():
    # the synthetic CO2 curve and the temperature-driven pH/a_w proxies must be gone
    for gone in ("default_co2_trajectory", "_latitudinal_proxies", "deeptime_haf"):
        assert not hasattr(haf, gone), f"fabricated symbol {gone} still present"


def test_coupled_haf_finite_and_shaped():
    csys = carbon_sulfur.run_csys(m_inj=3000.0, **_CSYS_KW)
    r = haf.coupled_event_haf(csys=csys, n_lat=_N_LAT)
    for k in ("kyr", "co2", "ph_ocean", "temp_anom_K", "t_global_C", "haf"):
        assert k in r, f"missing key {k}"
        assert r[k].shape == r["kyr"].shape
        assert np.all(np.isfinite(r[k])), f"non-finite values in {k}"


def test_haf_in_unit_interval():
    csys = carbon_sulfur.run_csys(m_inj=3000.0, **_CSYS_KW)
    r = haf.coupled_event_haf(csys=csys, n_lat=_N_LAT)
    assert np.all(r["haf"] >= 0.0)
    assert np.all(r["haf"] <= 1.0)


def test_pH_is_the_model_pH_not_fabricated():
    # the pH that drives habitability must be the box model's OWN carbonate-system
    # output, byte-for-byte -- not a temperature-derived proxy.
    csys = carbon_sulfur.run_csys(m_inj=3000.0, **_CSYS_KW)
    r = haf.coupled_event_haf(csys=csys, n_lat=_N_LAT)
    assert np.array_equal(r["ph_ocean"], csys["ph"])
    assert r["a_w"] == haf.A_W_MARINE          # water activity is the stated constant


def test_preonset_haf_is_steady():
    # before the carbon release the system is steady, so HAF must be constant
    csys = carbon_sulfur.run_csys(m_inj=3000.0, **_CSYS_KW)
    r = haf.coupled_event_haf(csys=csys, n_lat=_N_LAT)
    pre = r["kyr"] < 0.0
    assert pre.sum() >= 2
    assert np.ptp(r["haf"][pre]) < 1e-9


def test_global_temperature_tracks_co2():
    # EBM global-mean SST increases monotonically (and ~logarithmically) with pCO2
    models = _models()
    co2 = np.linspace(300.0, 4000.0, 10)
    tg = np.array([haf._haf_at_state(c, 7.8, models, n_lat=_N_LAT)[1] for c in co2])
    assert np.all(np.diff(tg) > 0)                       # warmer at higher CO2
    assert np.corrcoef(np.log(co2), tg)[0, 1] > 0.95


def test_event_lowers_habitability():
    # the observed genuine-model behaviour: a hyperthermal pulse (warming +
    # acidification) depresses HAF below the steady background, with the minimum
    # near the warm peak, then recovers.
    csys = carbon_sulfur.run_csys(m_inj=4000.0, **_CSYS_KW)
    r = haf.coupled_event_haf(csys=csys, n_lat=_N_LAT)
    s = haf.summarise(r)
    assert s["haf_drawdown"] > 0.0                       # event lowers HAF
    i_min = int(np.nanargmin(r["haf"]))
    # the least-habitable time is warmer than the run's median climate
    assert r["temp_anom_K"][i_min] > float(np.median(r["temp_anom_K"]))


def test_coupling_is_genuine_bigger_pulse_bigger_response():
    # a larger carbon release must (through the model) give a lower pH minimum,
    # a larger peak warming AND a larger HAF drawdown -- i.e. the HAF genuinely
    # tracks the box-model perturbation, not a hand-set curve.
    small = haf.coupled_event_haf(
        csys=carbon_sulfur.run_csys(m_inj=1000.0, **_CSYS_KW), n_lat=_N_LAT)
    big = haf.coupled_event_haf(
        csys=carbon_sulfur.run_csys(m_inj=6000.0, **_CSYS_KW), n_lat=_N_LAT)
    ss, sb = haf.summarise(small), haf.summarise(big)
    assert sb["ph_min"] < ss["ph_min"]                  # more acidification
    assert sb["peak_warming_K"] > ss["peak_warming_K"]  # more warming
    assert sb["haf_drawdown"] > ss["haf_drawdown"]      # more habitability loss


def test_reproducible_with_fixed_seed():
    csys = carbon_sulfur.run_csys(m_inj=3000.0, **_CSYS_KW)
    r1 = haf.coupled_event_haf(csys=csys, n_lat=_N_LAT, seed=0)
    r2 = haf.coupled_event_haf(csys=csys, n_lat=_N_LAT, seed=0)
    assert np.allclose(r1["haf"], r2["haf"])
    assert np.allclose(r1["t_global_C"], r2["t_global_C"])


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"PASS {name}")
    print("all haf smoke tests passed")
