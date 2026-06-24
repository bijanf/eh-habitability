"""Variance-based sensitivity and Jensen-bias aggregation for the deep-time model.

This module provides two illustrative analyses that sit on top of the core
:mod:`eh_deeptime` modules (carbon_sulfur, ebm, habitability). Neither produces a
calibrated result; both are methods demonstrations.

1. SALTELLI/SOBOL VARIANCE-BASED SENSITIVITY (:func:`sobol_indices`).
   First-order (S1) and total-effect (ST) Sobol indices estimated with the
   standard radial A/B/AB sampling design (Saltelli 2010, Comput. Phys. Commun.).
   S1 uses the Sobol/Saltelli (2010) estimator and ST uses the Jansen (1999)
   estimator -- both are the choices recommended by Saltelli et al. (2010) for
   robustness at small sample size. The base sample is a scrambled Sobol
   low-discrepancy sequence (scipy.stats.qmc.Sobol): the A and B matrices are the
   two halves of one 2d-dimensional Sobol set, the standard SALib construction.
   This converges far faster than a pseudo-random sample, so the ST >= S1
   inequality holds at modest n_base instead of being swamped by Monte-Carlo noise.
   The default model is the peak PETM warming of :mod:`eh_deeptime.carbon_sulfur`
   for a fixed ~3000 GtC pulse, as a function of a handful of its DEFAULT_PARAMS.

2. JENSEN-BIAS AGGREGATION (:func:`jensen_bias`, Task 4.5).
   Because the habitability metric is a nonlinear (convex/concave) function of the
   environment, the spatial mean of habitability over a heterogeneous field is NOT
   equal to habitability evaluated at the mean environment -- the classic
   Jensen-inequality "aggregation bias". For a sequence of CO2 levels we build an
   illustrative latitudinal environment (temperature from the 1-D EBM; ILLUSTRATIVE
   latitudinal pH and water-activity proxies, documented below), then compare the
   area-weighted (cos-lat) mean of the mixture habitability to the habitability of
   the area-mean environment. The gap delta_J is the aggregation bias.

This is an ILLUSTRATION / methods demonstration, NOT a calibrated or validated
analysis. The Sobol inputs vary illustrative model parameters over illustrative
ranges; the latitudinal pH / a_w fields are illustrative proxies, not data. No
real measurements are used or claimed.

PRODUCTION SWAP: a research-grade study would (a) run at much larger n_base with
bootstrap confidence intervals on the indices (e.g. via SALib), (b) drive the
Sobol analysis with the full closed
biogeochemical model under its calibrated posterior parameter envelope rather than
hand-set +-ranges, and (c) replace the illustrative latitudinal pH / a_w proxies
with reconstructed or modelled environmental fields.
"""
from __future__ import annotations

import numpy as np
from scipy.stats import qmc

from . import carbon_sulfur, ebm, habitability

# --- Sobol design ------------------------------------------------------------
# Fractional half-widths of the uniform ranges placed AROUND carbon_sulfur
# DEFAULT_PARAMS for the default Sobol problem. Illustrative envelopes only.
DEFAULT_SOBOL_NAMES = ("ecs", "f_degas", "f_runoff", "n_silw", "Ea")
DEFAULT_SOBOL_HALFWIDTH = {
    "ecs": 0.30,        # +-30% around ECS (K per doubling)
    "f_degas": 0.40,    # +-40% around the degassing multiplier
    "f_runoff": 0.30,   # +-30% around the runoff multiplier
    "n_silw": 0.25,     # +-25% around the silicate-weathering exponent
    "Ea": 0.20,         # +-20% around the activation energy
}

# Fixed PETM-scale pulse used by the default Sobol model output.
SOBOL_PULSE_GTC = 3000.0    # Gt C, illustrative consensus-scale release
SOBOL_PULSE_DUR = 5.0       # kyr, release duration

# --- Jensen-bias latitudinal proxies -----------------------------------------
# ILLUSTRATIVE latitudinal environmental proxies (NOT data). Surface ocean pH and
# water activity vary weakly with latitude in the real world; here we impose
# simple, smooth, plausible latitudinal shapes purely so that the environment is
# heterogeneous and the aggregation bias is non-trivial. They are documented as
# illustrative and must not be read as a reconstruction.
PH_EQUATOR = 7.9        # illustrative surface-ocean pH near the equator
PH_POLE = 8.2           # illustrative surface-ocean pH near the poles
AW_EQUATOR = 0.985      # illustrative water-activity proxy near the equator
AW_POLE = 0.965         # illustrative water-activity proxy near the poles

DEFAULT_CO2_GRID = np.array(
    [280.0, 400.0, 560.0, 700.0, 840.0, 1120.0, 1400.0, 2000.0]
)

# Fixed seed for the one-off habitability fit used by jensen_bias, so the
# aggregation-bias curve is reproducible.
HAB_FIT_SEED = 12345


def _default_bounds(names=DEFAULT_SOBOL_NAMES):
    """(names, bounds) for the default Sobol problem around carbon_sulfur defaults."""
    base = carbon_sulfur.DEFAULT_PARAMS
    bounds = []
    for nm in names:
        c = float(base[nm])
        hw = DEFAULT_SOBOL_HALFWIDTH[nm]
        bounds.append((c * (1.0 - hw), c * (1.0 + hw)))
    return list(names), bounds


def default_model_fn(param_array, names=DEFAULT_SOBOL_NAMES):
    """Default scalar model for the Sobol analysis: PETM peak warming (K).

    Maps a parameter vector (in the order of ``names``) onto the
    :func:`carbon_sulfur.DEFAULT_PARAMS`, runs a fixed ~3000 GtC pulse through
    :func:`carbon_sulfur.run_csys`, and returns the peak surface-temperature
    anomaly. Illustrative output, not a calibrated prediction.
    """
    overrides = {nm: float(v) for nm, v in zip(names, param_array)}
    res = carbon_sulfur.run_csys(params=overrides, m_inj=SOBOL_PULSE_GTC,
                                 t_dur=SOBOL_PULSE_DUR, t_end=400.0, n_out=261)
    return float(carbon_sulfur.summarise(res)["peak_warming_K"])


def _scale_sample(unit, bounds):
    """Map a (.,d) unit-cube sample to the hyper-rectangle given by bounds."""
    lo = np.array([b[0] for b in bounds])
    hi = np.array([b[1] for b in bounds])
    return lo + unit * (hi - lo)


def sobol_indices(model_fn=None, bounds=None, names=None,
                  n_base=256, seed=0):
    """First-order (S1) and total-effect (ST) Sobol indices (Saltelli 2010 design).

    Uses the standard radial A/B/AB estimator: two independent base samples A and
    B (each n_base x d), and d "AB^(i)" matrices where column i of A is replaced by
    column i of B. The model is evaluated on A, B and every AB^(i), giving
    n_base*(d+2) runs. S1 is the Sobol/Saltelli (2010) estimator and ST is the
    Jansen (1999) estimator, both recommended by Saltelli et al. (2010).

    Parameters
    ----------
    model_fn : callable | None  maps a length-d parameter vector to a scalar; if
                                None, uses :func:`default_model_fn` (PETM peak
                                warming) with the default carbon_sulfur problem.
    bounds : list[(lo,hi)]|None per-input uniform ranges; default around
                                carbon_sulfur.DEFAULT_PARAMS.
    names : list[str]|None       input names (length d); default DEFAULT_SOBOL_NAMES.
    n_base : int                 base-sample size (total runs = n_base*(d+2)).
    seed : int                   RNG seed for the base samples.

    Returns
    -------
    dict with keys 'names' (list), 'S1' (ndarray, d), 'ST' (ndarray, d).
    Indices are sanity/illustrative; not a calibrated attribution.
    """
    if names is None and bounds is None:
        names, bounds = _default_bounds()
    elif names is None:
        names = [f"x{i}" for i in range(len(bounds))]
    elif bounds is None:
        _, bounds = _default_bounds(names)
    if model_fn is None:
        model_fn = lambda v: default_model_fn(v, names=names)  # noqa: E731

    d = len(names)
    # A and B are the two halves of one scrambled 2d-dimensional Sobol set
    # (the standard SALib construction); n_base is rounded up to a power of two,
    # the size on which Sobol sequences are balanced.
    m = int(np.ceil(np.log2(max(n_base, 2))))
    pts = qmc.Sobol(d=2 * d, scramble=True, seed=seed).random_base2(m=m)
    n_base = pts.shape[0]
    A = _scale_sample(pts[:, :d], bounds)
    B = _scale_sample(pts[:, d:], bounds)

    def _eval(M):
        return np.array([model_fn(row) for row in M])

    yA = _eval(A)
    yB = _eval(B)

    # AB^(i): A with column i taken from B.
    yAB = np.empty((n_base, d))
    for i in range(d):
        AB = A.copy()
        AB[:, i] = B[:, i]
        yAB[:, i] = _eval(AB)

    # total output variance (use the combined A,B sample for a stable estimate)
    var = float(np.var(np.concatenate([yA, yB]), ddof=1))
    S1 = np.zeros(d)
    ST = np.zeros(d)
    if var <= 0.0:
        return {"names": list(names), "S1": S1, "ST": ST}

    for i in range(d):
        # Sobol/Saltelli (2010) first-order estimator
        S1[i] = float(np.mean(yB * (yAB[:, i] - yA))) / var
        # Jansen (1999) total-effect estimator
        ST[i] = float(0.5 * np.mean((yA - yAB[:, i]) ** 2)) / var

    return {"names": list(names), "S1": S1, "ST": ST}


# --- Jensen-bias aggregation -------------------------------------------------
def _lat_environment(co2_ppm, n_lat, models=None):
    """Build the illustrative latitudinal environment field at a given CO2.

    Returns (lat, X, weights) where X is (n_lat, 3) in habitability.FEATURES order
    (T_C, pH, a_w) and weights are cos(lat) area weights. Temperature comes from
    the 1-D EBM; pH and a_w are ILLUSTRATIVE smooth latitudinal proxies.
    """
    sol = ebm.solve_ebm(co2_ppm=co2_ppm, n_lat=n_lat)
    lat = sol["lat"]
    T_C = sol["T_C"]
    # |sin(lat)| goes 0 at equator -> 1 at poles; blend the equator/pole proxies.
    s = np.abs(np.sin(np.deg2rad(lat)))
    pH = PH_EQUATOR + (PH_POLE - PH_EQUATOR) * s
    a_w = AW_EQUATOR + (AW_POLE - AW_EQUATOR) * s
    X = np.column_stack([T_C, pH, a_w])
    w = np.cos(np.deg2rad(lat))
    return lat, X, w


def jensen_bias(co2_array=None, n_lat=91, models=None, rng=None):
    """Aggregation (Jensen) bias of the habitability metric over latitude.

    For each CO2 level: get the latitudinal temperature field from the 1-D EBM,
    attach illustrative latitudinal pH and water-activity proxies, then compare

      f_hab_integrated  = area-weighted (cos lat) mean over latitude of the
                          across-guild mixture habitability p_hab_mixture(X);
      p_hab_globalmean  = p_hab_mixture evaluated at the area-MEAN environment.

    Their difference delta_J = f_hab_integrated - p_hab_globalmean is the
    Jensen-inequality aggregation bias (non-zero because habitability is a
    nonlinear function of the environment); sigma_agg = rms(delta_J) over the CO2
    grid summarises its magnitude. Illustrative, not a calibrated result.

    Parameters
    ----------
    co2_array : array|None   CO2 levels (ppm); default DEFAULT_CO2_GRID.
    n_lat : int              latitude grid size passed to the EBM.
    models : list|None       fitted habitability models; if None, fit once with a
                             fixed seed (HAB_FIT_SEED).
    rng : seed|Generator|None  only used to fit models if models is None and a
                             non-default seed is wanted.

    Returns
    -------
    dict with keys 'co2', 'f_hab_integrated', 'p_hab_globalmean', 'delta_J'
    (each an ndarray over the CO2 grid) and 'sigma_agg' (float).
    """
    co2 = DEFAULT_CO2_GRID if co2_array is None else np.asarray(co2_array, dtype=float)
    if models is None:
        seed = HAB_FIT_SEED if rng is None else rng
        models = habitability.fit_all(seed)

    f_int = np.empty(len(co2))
    p_mean = np.empty(len(co2))
    for k, c in enumerate(co2):
        _lat, X, w = _lat_environment(float(c), n_lat, models)
        p_lat = habitability.p_hab_mixture(X, models)        # (n_lat,)
        f_int[k] = float(np.sum(p_lat * w) / np.sum(w))      # mean of f(env)
        x_mean = np.sum(X * w[:, None], axis=0) / np.sum(w)  # area-mean env
        p_mean[k] = float(habitability.p_hab_mixture(x_mean[None, :], models)[0])

    delta = f_int - p_mean
    sigma = float(np.sqrt(np.mean(delta ** 2)))
    return {
        "co2": co2,
        "f_hab_integrated": f_int,
        "p_hab_globalmean": p_mean,
        "delta_J": delta,
        "sigma_agg": sigma,
    }


def summarise(sobol=None, jb=None):
    """One-line diagnostic bundle for the two illustrative analyses."""
    if sobol is None:
        sobol = sobol_indices(n_base=128)
    if jb is None:
        jb = jensen_bias()
    order = np.argsort(sobol["ST"])[::-1]
    return {
        "sobol_names": [sobol["names"][i] for i in order],
        "sobol_S1": sobol["S1"][order],
        "sobol_ST": sobol["ST"][order],
        "jensen_sigma_agg": jb["sigma_agg"],
        "jensen_delta_J_range": (float(np.min(jb["delta_J"])),
                                 float(np.max(jb["delta_J"]))),
    }
