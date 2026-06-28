"""Habitable Area Fraction (HAF) through a carbon-release event, fully coupled to
the closed carbon-sulfur-oxygen box model.

This is the deep-time analogue of the shallow-time HAF: a single, area-weighted
scalar in [0, 1] summarising how much of the surface is habitable, tracked as the
climate and ocean-chemistry state evolve. Crucially, EVERY driving time-series is
a forward-model output of the coupled system -- there are NO synthetic, analytic
or proxy time-series anywhere in this module:

    carbon_sulfur.run_csys  -->  pCO2(t), surface-ocean pH(t)   (box-model outputs)
    pCO2(t)  --ebm.solve_ebm-->  zonal sea-surface temperature SST(lat, t)
    (SST(lat), pH(t), a_w)  --habitability.p_hab_mixture-->  habitability p(lat, t)
    HAF(t) = cos(lat)-area-weighted mean of p(lat, t) over latitude.

The carbon-cycle model supplies atmospheric pCO2(t) and the surface-ocean pH(t)
(diagnosed from its carbonate system) directly; the 1-D energy-balance model turns
that pCO2(t) into a zonal SST field; water activity is held at the fixed
open-marine value (the 0-D box model has no hydrological cycle from which to
derive a spatial water-activity field, so it is a STATED CONSTANT, not an invented
series). The across-guild mixture habitability metric is then evaluated per
latitude band and area-weighted into HAF(t).

The carbon release that drives the simulation is an explicitly prescribed,
idealised scenario (e.g. a PETM-scale pulse of a few thousand Gt C); the response
is a forward simulation, labelled as such, NOT a reconstruction of any real event.

This is an ILLUSTRATION / methods demonstration: the box-model and EBM constants
are taken from published parameter envelopes and are NOT calibrated to proxy data.
The coupling, the mass conservation and every plotted series are real model
output; the parameters are not fitted to data and no proxy record is used.

PRODUCTION SWAP: a research-grade deep-time HAF would (a) calibrate the box-model
and climate parameters against compiled proxy records under a Bayesian framework
with the uncertainty propagated, (b) replace the 1-D EBM with a 2-D climate model
carrying real paleogeography, and (c) derive surface water availability and a
spatially resolved ocean pH from a coupled hydrological / multi-box ocean-chemistry
model rather than the single well-mixed surface pH and fixed marine a_w used here.
"""
from __future__ import annotations

import numpy as np

from . import ebm, habitability

# --- fixed environmental constant (a STATED assumption, not a fabricated series) ---
# Open-marine surface seawater has a water activity of ~0.98 essentially
# everywhere; the 0-D carbon-cycle box model has no hydrological cycle to produce
# a spatially or temporally varying a_w, so it is held fixed at this value.
A_W_MARINE = 0.98

# --- defaults for the public entry points ------------------------------------
DEFAULT_N_LAT = 91          # EBM latitude grid (odd -> includes the equator)
DEFAULT_SEED = 0            # fixed seed: the habitability models are fit once here

# default idealised carbon release used when no run is supplied (PETM-scale,
# deliberately at the lower end of published estimates); a prescribed scenario.
DEFAULT_CSYS = dict(m_inj=3000.0, t_dur=5.0, delta_inj=-50.0, t_end=400.0)


def _haf_from_climate(clim, ph, models, a_w):
    """cos(lat)-area-weighted mixture habitability for one climate+chemistry state.

    `clim` is an :func:`eh_deeptime.ebm.solve_ebm` output (zonal SST field); `ph`
    and `a_w` are uniform-in-latitude scalars (the well-mixed surface-ocean pH and
    the fixed marine water activity). No proxy or synthetic field is constructed.
    """
    lat, T_C = clim["lat"], clim["T_C"]
    X = np.column_stack([T_C,
                         np.full_like(T_C, float(ph)),
                         np.full_like(T_C, float(a_w))])   # FEATURES: (T_C, pH, a_w)
    p = habitability.p_hab_mixture(X, models)
    w = np.cos(np.deg2rad(lat))                            # area weight on the sphere
    return float(np.sum(p * w) / np.sum(w))


def _haf_at_state(co2_ppm, ph, models, n_lat=DEFAULT_N_LAT, a_w=A_W_MARINE):
    """HAF and EBM global-mean SST for a single (pCO2, ocean-pH) state.

    Solves the 1-D EBM at ``co2_ppm`` and area-weights the across-guild mixture
    habitability over latitude using that SST field, the uniform ocean ``ph`` and
    the fixed water activity ``a_w``. Every input is a model quantity or a stated
    constant.
    """
    clim = ebm.solve_ebm(co2_ppm=float(co2_ppm), n_lat=n_lat)
    return _haf_from_climate(clim, ph, models, a_w), clim["T_global_C"]


def coupled_event_haf(csys=None, models=None, n_lat=DEFAULT_N_LAT,
                      seed=DEFAULT_SEED, a_w=A_W_MARINE, csys_params=None):
    """HAF(t) through a carbon-release event, driven ENTIRELY by box-model outputs.

    For every time step of a :func:`eh_deeptime.carbon_sulfur.run_csys` simulation,
    the model's atmospheric pCO2(t) forces the 1-D EBM to give the zonal SST field,
    and the model's surface-ocean pH(t) (diagnosed from the carbonate system) sets
    the pH; water activity is held at the fixed open-marine value ``a_w``. The
    across-guild mixture habitability is evaluated per latitude band and
    cos(latitude)-area-weighted into HAF(t).

    NO fabricated or synthetic time-series enters this calculation: pCO2(t) and
    pH(t) are forward outputs of the mass-conserving C-S-O box model responding to
    an explicitly prescribed carbon release, the SST field is the EBM response to
    that pCO2(t), and a_w is a stated constant. The habitability mixture is fit
    once on synthetic draws from the published guild tolerance envelopes (see
    :mod:`eh_deeptime.habitability`) -- a metric calibration, not a time-series.

    This remains an ILLUSTRATION: the box-model and EBM constants come from
    published parameter envelopes and are not calibrated to proxy data, and the
    carbon release is an idealised scenario, not a reconstructed event.

    Parameters
    ----------
    csys : dict | None     a carbon_sulfur.run_csys output; if None one is run with
                           ``csys_params`` (default: a PETM-scale 3000 Gt C pulse).
    models : list | None   pre-fit habitability models; if None they are fit once
                           at ``seed`` on the synthetic guild envelopes.
    n_lat : int            EBM latitude grid points
    seed : int             seed for the one-time habitability-metric fit
    a_w : float            fixed open-marine water activity
    csys_params : dict|None overrides forwarded to carbon_sulfur.run_csys when
                           ``csys`` is None

    Returns
    -------
    dict with 1-D arrays of equal length (all forward-model output):
        kyr          time relative to onset (kyr)
        co2          atmospheric pCO2 (ppm)                 [box-model output]
        ph_ocean     surface-ocean pH                       [box-model output]
        temp_anom_K  box-model surface-warming anomaly (K)  [box-model output]
        t_global_C   EBM global-mean SST (degC)             [EBM output]
        haf          habitable area fraction in [0, 1]
    plus a_w (float, the fixed water activity used).
    """
    from . import carbon_sulfur
    if csys is None:
        kw = dict(DEFAULT_CSYS)
        if csys_params:
            kw.update(csys_params)
        csys = carbon_sulfur.run_csys(**kw)
    if models is None:
        models = habitability.fit_all(np.random.default_rng(seed))

    kyr = np.asarray(csys["kyr"], dtype=float)
    co2 = np.asarray(csys["pco2"], dtype=float)
    ph_ocean = np.asarray(csys["ph"], dtype=float)
    temp_anom = np.asarray(csys["temp"], dtype=float)

    haf = np.empty_like(co2)
    t_global = np.empty_like(co2)
    # EBM(co2) depends only on pCO2, so cache by pCO2 (collapses the long steady
    # pre-onset segment and any repeated values to a single solve each).
    clim_cache = {}
    for i in range(co2.size):
        key = round(float(co2[i]), 4)
        clim = clim_cache.get(key)
        if clim is None:
            clim = ebm.solve_ebm(co2_ppm=float(co2[i]), n_lat=n_lat)
            clim_cache[key] = clim
        haf[i] = _haf_from_climate(clim, ph_ocean[i], models, a_w)
        t_global[i] = clim["T_global_C"]

    return {"kyr": kyr, "co2": co2, "ph_ocean": ph_ocean,
            "temp_anom_K": temp_anom, "t_global_C": t_global,
            "haf": haf, "a_w": float(a_w)}


def summarise(res):
    """Diagnostics for a coupled-event HAF run (illustrative, all model output).

    Returns the steady pre-onset background HAF, the minimum HAF over the event
    and the signed change between them (positive = the event lowers habitability),
    the time and pCO2 at the HAF minimum, the minimum surface-ocean pH, and the
    peak surface warming. Sanity/diagnostic helper, not a validation metric.
    """
    kyr, haf, co2 = res["kyr"], res["haf"], res["co2"]
    pre = kyr < 0.0
    haf_bg = float(np.nanmean(haf[pre])) if np.any(pre) else float(haf[0])
    i_min = int(np.nanargmin(haf))
    i_max = int(np.nanargmax(haf))
    return {
        "haf_background": haf_bg,
        "haf_min": float(haf[i_min]),
        "haf_max": float(haf[i_max]),
        "haf_drawdown": float(haf_bg - haf[i_min]),
        "haf_range": float(haf[i_max] - haf[i_min]),
        "kyr_at_haf_min": float(kyr[i_min]),
        "co2_at_haf_min_ppm": float(co2[i_min]),
        "ph_min": float(np.nanmin(res["ph_ocean"])),
        "peak_warming_K": float(np.nanmax(res["temp_anom_K"])),
    }
