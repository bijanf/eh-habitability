"""An illustrative deep-time Habitable Area Fraction (HAF) trajectory.

This is the deep-time analogue of the shallow-time HAF: a single, area-weighted
scalar in [0, 1] that summarises how much of the planet's surface is habitable,
tracked through geological time as the climate state evolves. Here it is built
from the three illustrative deep-time modules in this package:

    pCO2(t)  --ebm.solve_ebm-->  latitudinal T(lat)
             --illustrative proxies-->  latitudinal pH(lat), a_w(lat)
             --habitability.p_hab_mixture-->  latitudinal habitability p(lat)
    HAF(t) = cos(lat)-area-weighted mean of p(lat) over latitude.

The driving pCO2(t) is a SYNTHETIC, smooth deep-time trajectory -- a few hundred
million years of slowly declining background pCO2 with one or two superimposed
transient hyperthermal-style excursions. It is an ILLUSTRATIVE forcing scenario,
NOT a proxy CO2 reconstruction (no GEOCARB / paleosol / stomatal / boron data are
used or claimed); it merely exercises the climate -> environment -> habitability
chain over a deep-time-like range of CO2.

THE LATITUDINAL pH AND a_w FIELDS ARE ILLUSTRATIVE PROXIES, NOT DATA. The
habitability metric needs three environmental features (temperature, pH, water
activity); the 1-D EBM supplies only temperature. To close the loop for the
illustration we map the EBM temperature field to a surface-water pH and a water
activity (a_w) field through simple, documented, monotone relations:

  * a_w (water activity): highest at temperate latitudes and lowered at both
    thermal extremes -- in the hot tropics by an evaporation/aridity effect, and
    in the cold polar belt by a desiccation/liquid-water-availability effect.
    This is a deliberately crude stand-in for "how much liquid water is available
    to life" as a function of local climate.
  * pH: a weak, near-neutral surface-water pH with a mild warm-equator trend.

These relations carry NO claim of quantitative accuracy; they exist so the
mixture metric has a full (T, pH, a_w) field to act on. With them, the HAF falls
as CO2 (hence temperature) climbs into the hottest excursions, because the hot,
arid low latitudes drop below the water-activity tolerance of most guilds -- the
qualitative behaviour one wants to illustrate, not a reconstructed history.

This is an ILLUSTRATION / methods demonstration of a deep-time habitability
diagnostic, NOT a calibrated or validated reconstruction.

PRODUCTION SWAP: a research-grade deep-time HAF would (a) be driven by a Bayesian
proxy-CO2 reconstruction with its uncertainty propagated, (b) replace the 1-D EBM
with a 2-D climate model carrying real paleogeography, and (c) derive pH and water
availability from a coupled ocean-chemistry / hydrological-cycle model and a
reconstructed land-sea mask, rather than from the illustrative temperature-driven
proxies used here.
"""
from __future__ import annotations

import numpy as np

from . import ebm, habitability

# --- synthetic deep-time CO2 trajectory knobs (ILLUSTRATIVE, not a proxy) -----
# A slowly declining background pCO2 over a few hundred Myr, with two superimposed
# transient excursions (hyperthermal-style spikes). All values illustrative.
T_SPAN_MYR = 300.0          # total length of the synthetic deep-time window (Myr)
CO2_START_PPM = 2200.0      # background pCO2 at the start of the window (ppm)
CO2_END_PPM = 360.0         # background pCO2 at the end of the window (ppm)
EXCURSIONS = (
    # (centre_myr, width_myr, peak_extra_ppm): transient pCO2 spikes added on top
    (95.0, 8.0, 2600.0),    # a large early hyperthermal-style excursion
    (210.0, 5.0, 1200.0),   # a smaller, later excursion
)
CO2_FLOOR_PPM = 150.0       # never let the synthetic trajectory drop below this
CO2_CEIL_PPM = 6000.0       # cap the synthetic trajectory (illustrative)

# --- illustrative latitudinal environmental proxies (functions of local T) ----
# Water-activity (a_w) proxy: temperate optimum, lowered at both thermal extremes.
AW_OPTIMUM = 0.985          # baseline water activity at temperate latitudes
AW_HOT_SLOPE = 0.0040       # a_w drop per degC above AW_HOT_KNEE (aridity tail)
AW_HOT_KNEE_C = 25.0        # degC above which the hot-aridity penalty switches on
AW_COLD_SLOPE = 0.0030      # a_w drop per degC below 0 C (desiccation tail)
AW_MIN = 0.55               # clamp on the proxy water activity
AW_MAX = 0.99               # clamp on the proxy water activity
# Surface-water pH proxy: weak near-neutral field with a mild warm trend.
PH_REF = 7.4                # reference pH at the reference temperature
PH_T_REF_C = 15.0           # reference temperature for the pH trend (degC)
PH_T_SLOPE = 0.006          # pH change per degC away from PH_T_REF_C
PH_MIN = 5.5                # clamp on the proxy pH
PH_MAX = 8.6                # clamp on the proxy pH

# --- defaults for the public entry points ------------------------------------
DEFAULT_N_TIME = 241        # number of points on the deep-time axis
DEFAULT_N_LAT = 91          # EBM latitude grid (odd -> includes the equator)
DEFAULT_SEED = 0            # fixed seed: the habitability models are fit once here


def default_co2_trajectory(n=DEFAULT_N_TIME):
    """Return a SYNTHETIC, smooth deep-time pCO2 trajectory (ILLUSTRATIVE).

    The trajectory is a slowly declining background pCO2 over ``T_SPAN_MYR`` with
    the transient excursions in ``EXCURSIONS`` superimposed as raised-Gaussian
    spikes, clamped to [CO2_FLOOR_PPM, CO2_CEIL_PPM]. This is an illustrative
    forcing scenario, NOT a proxy CO2 reconstruction.

    Parameters
    ----------
    n : int    number of points on the time axis

    Returns
    -------
    t_myr : (n,) float   time in Myr, increasing from 0 to ``T_SPAN_MYR``
    co2_ppm : (n,) float  synthetic atmospheric pCO2 (ppm), same shape as t_myr
    """
    t_myr = np.linspace(0.0, T_SPAN_MYR, n)
    # smooth (cosine-tapered) background decline from start to end
    frac = t_myr / T_SPAN_MYR
    taper = 0.5 * (1.0 - np.cos(np.pi * frac))   # 0 -> 1, smooth at both ends
    co2 = CO2_START_PPM + (CO2_END_PPM - CO2_START_PPM) * taper
    # superimpose the transient excursions (Gaussian bumps)
    for centre, width, peak in EXCURSIONS:
        co2 = co2 + peak * np.exp(-0.5 * ((t_myr - centre) / width) ** 2)
    co2 = np.clip(co2, CO2_FLOOR_PPM, CO2_CEIL_PPM)
    return t_myr, co2


def _latitudinal_proxies(T_C):
    """Illustrative latitudinal (pH, a_w) proxies from the EBM temperature field.

    See the module docstring: a_w is highest at temperate latitudes and lowered at
    both thermal extremes (hot-aridity + cold-desiccation tails); pH is a weak,
    near-neutral field with a mild warm trend. Documented stand-ins, NOT data.

    Parameters
    ----------
    T_C : array   zonal-mean surface temperature (degC)

    Returns
    -------
    pH : array    illustrative surface-water pH, same shape as T_C
    a_w : array   illustrative water activity (-), same shape as T_C
    """
    T_C = np.asarray(T_C, dtype=float)
    hot = np.maximum(T_C - AW_HOT_KNEE_C, 0.0)     # excess warmth -> aridity
    cold = np.maximum(0.0 - T_C, 0.0)              # degrees below freezing
    a_w = AW_OPTIMUM - AW_HOT_SLOPE * hot - AW_COLD_SLOPE * cold
    a_w = np.clip(a_w, AW_MIN, AW_MAX)
    pH = PH_REF + PH_T_SLOPE * (T_C - PH_T_REF_C)
    pH = np.clip(pH, PH_MIN, PH_MAX)
    return pH, a_w


def _haf_at_co2(co2_ppm, models, n_lat=DEFAULT_N_LAT):
    """HAF and global-mean temperature for a single pCO2 value (illustrative).

    Solves the EBM at ``co2_ppm``, builds the latitudinal (T, pH, a_w) field,
    evaluates the across-guild mixture habitability per latitude, and returns the
    cos(lat)-area-weighted mean habitability plus the EBM global-mean temperature.
    """
    clim = ebm.solve_ebm(co2_ppm=float(co2_ppm), n_lat=n_lat)
    lat, T_C = clim["lat"], clim["T_C"]
    pH, a_w = _latitudinal_proxies(T_C)
    X = np.column_stack([T_C, pH, a_w])           # FEATURES order: (T_C, pH, a_w)
    p = habitability.p_hab_mixture(X, models)
    w = np.cos(np.deg2rad(lat))                   # area weight on the sphere
    haf = float(np.sum(p * w) / np.sum(w))
    return haf, clim["T_global_C"]


def deeptime_haf(t_myr=None, co2_ppm=None, n_lat=DEFAULT_N_LAT, seed=DEFAULT_SEED):
    """Compute the illustrative deep-time Habitable Area Fraction trajectory.

    For each time the synthetic pCO2 is mapped to a latitudinal climate via the
    1-D EBM, the latitudinal (T, pH, a_w) environmental field is built (T from the
    EBM; pH and a_w from the illustrative proxies in :func:`_latitudinal_proxies`),
    and HAF is the cos(lat)-area-weighted mean of the across-guild mixture
    habitability over latitude. The habitability models are fit ONCE here, at a
    fixed seed, on SYNTHETIC guild data (see :mod:`eh_deeptime.habitability`).

    Parameters
    ----------
    t_myr : array|None    time axis (Myr); default from :func:`default_co2_trajectory`
    co2_ppm : array|None  pCO2 (ppm) at each time; default from the same trajectory.
                          If ``t_myr`` is given but ``co2_ppm`` is None, the default
                          trajectory is resampled onto the supplied ``t_myr``.
    n_lat : int           EBM latitude grid points
    seed : int            seed for the one-time habitability model fit (fixed)

    Returns
    -------
    dict with keys (all 1-D arrays of equal length):
        t_myr        time axis (Myr)
        co2          synthetic pCO2 (ppm)
        haf          habitable area fraction in [0, 1]
        t_global_C   EBM global-mean surface temperature (degC)
    """
    if t_myr is None and co2_ppm is None:
        t_myr, co2_ppm = default_co2_trajectory()
    elif co2_ppm is None:
        # resample the default synthetic trajectory onto the supplied time axis
        t_myr = np.asarray(t_myr, dtype=float)
        t_def, co2_def = default_co2_trajectory(max(DEFAULT_N_TIME, t_myr.size))
        co2_ppm = np.interp(t_myr, t_def, co2_def)
    elif t_myr is None:
        co2_ppm = np.asarray(co2_ppm, dtype=float)
        t_myr = np.linspace(0.0, T_SPAN_MYR, co2_ppm.size)

    t_myr = np.asarray(t_myr, dtype=float)
    co2_ppm = np.asarray(co2_ppm, dtype=float)
    if t_myr.shape != co2_ppm.shape:
        raise ValueError("t_myr and co2_ppm must have the same shape")

    # fit the habitability mixture ONCE, at a fixed seed (synthetic guild data)
    rng = np.random.default_rng(seed)
    models = habitability.fit_all(rng)

    haf = np.empty_like(co2_ppm, dtype=float)
    t_global = np.empty_like(co2_ppm, dtype=float)
    for i, c in enumerate(co2_ppm):
        haf[i], t_global[i] = _haf_at_co2(c, models, n_lat=n_lat)

    return {"t_myr": t_myr, "co2": co2_ppm, "haf": haf, "t_global_C": t_global}


def summarise(res):
    """Diagnostics for a deep-time HAF run (illustrative).

    Returns the minimum and maximum HAF over the trajectory, the pCO2 and global
    temperature at the HAF minimum (the least-habitable time), and the overall
    HAF range. Sanity/diagnostic helper, not a validation metric.
    """
    haf, co2, tg = res["haf"], res["co2"], res["t_global_C"]
    i_min = int(np.argmin(haf))
    i_max = int(np.argmax(haf))
    return {
        "haf_min": float(haf[i_min]),
        "haf_max": float(haf[i_max]),
        "haf_range": float(haf[i_max] - haf[i_min]),
        "co2_at_haf_min_ppm": float(co2[i_min]),
        "t_global_at_haf_min_C": float(tg[i_min]),
    }
