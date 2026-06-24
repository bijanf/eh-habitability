"""A 1-D latitudinal energy-balance model (EBM) for an illustrative climate field.

North (1981)-class diffusive energy-balance model on x = sin(latitude). The annual-mean,
zonal-mean surface temperature T(x) satisfies, at equilibrium,

    0 = S0/4 * s(x) * (1 - alpha(x,T)) - (A_eff + B*T) + d/dx[ D*(1-x^2) dT/dx ],

where
    s(x)   = 1 - 0.482*P2(x),  P2 = (3x^2 - 1)/2          (mean annual insolation shape),
    A_eff  = A - 5.35*ln(CO2/CO2_0)                        (CO2 radiative forcing folded in),
    alpha  = a_ice where T < T_ICE (ice-covered) else a_land (ice-albedo feedback),
    (A + B*T) is the linearised outgoing longwave radiation,
    D*(1-x^2) dT/dx is meridional heat diffusion (the (1-x^2) is the metric on the sphere).

The steady state is solved by finite differences: the diffusion operator + the -B*T sink
form a tridiagonal linear system that is solved for T at fixed albedo, then the ice-albedo
feedback (alpha from the new T) is re-applied and the linear solve repeated until the
ice mask stops changing (Picard/fixed-point iteration on the albedo).

Constants (A, B, D, the albedos) are chosen so that at CO2 = 280 ppm the global-mean
temperature is ~14 C with an equator-to-pole gradient of the right order (~40-50 C) and a
mid-to-high-latitude ice line that retreats poleward (and ultimately vanishes) as CO2 rises
-- the qualitative behaviour of the North (1981) model. They are hand-set to give a
present-day-like global mean (~14 C) and equator-pole gradient; they are NOT formally
calibrated or fitted to an observational climatology.

This is an ILLUSTRATION of the climate module honestly reduced to 1-D, not a calibrated or
validated climate model. The 2-D high-resolution version in the Perspective would require a
proper spectral/finite-volume solver (e.g. JAX-accelerated) and reconstructed land-sea,
orography and insolation boundary conditions that are not available here.

PRODUCTION SWAP: a research-grade implementation would (i) move to 2-D with realistic
geography and seasonally resolved insolation, (ii) replace the linear OLR (A + B*T) with a
radiative-transfer or band model, (iii) use a state-dependent diffusivity D(x), and (iv) be
calibrated against an observational climatology rather than hand-set to plausible values.
"""
from __future__ import annotations

import numpy as np

# --- fixed forcing / radiation constants -------------------------------------
S0 = 1361.0          # W m^-2, total solar irradiance (solar "constant")
CO2_0 = 280.0        # ppm, pre-industrial reference CO2 (sets the forcing zero)
CO2_FORCING = 5.35   # W m^-2, radiative-forcing coefficient for ln(CO2/CO2_0)
S2 = 0.482           # 2nd-Legendre coefficient of the annual-mean insolation shape s(x)

# --- linearised outgoing-longwave-radiation (OLR = A + B*T) ------------------
A_OLR = 205.0        # W m^-2, OLR intercept (illustrative; sets the mean climate)
B_OLR = 1.90         # W m^-2 K^-1, OLR slope (climate-feedback / restoring strength)

# --- meridional heat diffusion ----------------------------------------------
D_DIFF = 0.38        # W m^-2 K^-1, diffusion coefficient (sets the equator-pole gradient)

# --- ice-albedo feedback -----------------------------------------------------
ALBEDO_ICE = 0.62    # ice / snow-covered surface albedo
ALBEDO_LAND = 0.30   # ice-free (land+ocean) surface albedo
T_ICE_C = -10.0      # degC, threshold below which the surface is treated as ice-covered

DEFAULT_PARAMS = {
    "S0": S0, "CO2_0": CO2_0, "CO2_forcing": CO2_FORCING, "S2": S2,
    "A": A_OLR, "B": B_OLR, "D": D_DIFF,
    "albedo_ice": ALBEDO_ICE, "albedo_land": ALBEDO_LAND, "T_ice_C": T_ICE_C,
}

_MAX_ITER = 200      # Picard iterations on the ice-albedo feedback


def _insolation_shape(x, s2):
    """Annual-mean, zonal-mean insolation distribution s(x), x = sin(latitude)."""
    p2 = (3.0 * x * x - 1.0) / 2.0
    return 1.0 - s2 * p2


def _build_diffusion_operator(x, d):
    """Tridiagonal finite-difference matrix L for d/dx[ D*(1-x^2) dT/dx ] on grid x.

    Second-order centred differences on the (uniform-in-latitude, non-uniform-in-x) grid,
    with zero-flux (Neumann) boundary conditions at the poles (no heat through |lat|=90).
    """
    n = x.size
    L = np.zeros((n, n))
    for i in range(1, n - 1):
        xm = 0.5 * (x[i - 1] + x[i])          # diffusivity sampled at cell faces
        xp = 0.5 * (x[i] + x[i + 1])
        gm = d * (1.0 - xm * xm)
        gp = d * (1.0 - xp * xp)
        dxm = x[i] - x[i - 1]
        dxp = x[i + 1] - x[i]
        dxc = 0.5 * (x[i + 1] - x[i - 1])
        a_left = gm / (dxm * dxc)
        a_right = gp / (dxp * dxc)
        L[i, i - 1] += a_left
        L[i, i + 1] += a_right
        L[i, i] += -(a_left + a_right)
    # poles: zero-flux one-sided closures (the (1-x^2) factor already -> 0 at |x|=1)
    xp = 0.5 * (x[0] + x[1]); gp = d * (1.0 - xp * xp); dxp = x[1] - x[0]
    L[0, 0] += -gp / (dxp * dxp); L[0, 1] += gp / (dxp * dxp)
    xm = 0.5 * (x[n - 2] + x[n - 1]); gm = d * (1.0 - xm * xm); dxm = x[n - 1] - x[n - 2]
    L[n - 1, n - 1] += -gm / (dxm * dxm); L[n - 1, n - 2] += gm / (dxm * dxm)
    return L


def solve_ebm(co2_ppm=280.0, params=None, n_lat=91):
    """Solve the steady 1-D diffusive energy-balance model at a given CO2.

    Parameters
    ----------
    co2_ppm : float    atmospheric CO2 (ppm); forcing is 5.35*ln(co2_ppm/CO2_0)
    params : dict|None  overrides for DEFAULT_PARAMS (A, B, D, albedos, ...)
    n_lat : int        number of latitude grid points from -90 to +90 (odd -> includes 0)

    Returns
    -------
    dict with keys:
        lat              latitude grid (deg), length n_lat
        T_C              zonal-mean surface temperature (degC), array length n_lat
        T_global_C       area-weighted (cos lat) global-mean temperature (degC), float
        ice_latitude_deg highest |lat| at which T > T_ice (the ice line), or nan if ice-free
    """
    p = dict(DEFAULT_PARAMS)
    if params:
        p.update(params)

    lat = np.linspace(-90.0, 90.0, n_lat)
    x = np.sin(np.deg2rad(lat))
    s = _insolation_shape(x, p["S2"])
    a_eff = p["A"] - p["CO2_forcing"] * np.log(co2_ppm / p["CO2_0"])
    diag_B = p["B"]
    L = _build_diffusion_operator(x, p["D"])

    # fixed-point iteration on the ice-albedo feedback
    albedo = np.full(n_lat, p["albedo_land"])
    T = np.full(n_lat, 14.0)
    for it in range(_MAX_ITER):
        absorbed = p["S0"] / 4.0 * s * (1.0 - albedo)
        # (L - B I) T = -(absorbed - A_eff)  ->  steady-state balance
        M = L - diag_B * np.eye(n_lat)
        rhs = -(absorbed - a_eff)
        T_new = np.linalg.solve(M, rhs)
        albedo_new = np.where(T_new < p["T_ice_C"], p["albedo_ice"], p["albedo_land"])
        if it > 0 and np.array_equal(albedo_new, albedo):
            T = T_new
            break
        albedo = albedo_new
        T = T_new

    w = np.cos(np.deg2rad(lat))
    t_global = float(np.sum(T * w) / np.sum(w))

    ice_free = T > p["T_ice_C"]
    if np.all(ice_free):
        ice_latitude = float("nan")            # no ice anywhere
    else:
        ice_latitude = float(np.max(np.abs(lat[ice_free])))

    return {
        "lat": lat,
        "T_C": T,
        "T_global_C": t_global,
        "ice_latitude_deg": ice_latitude,
    }


def summarise(res):
    """Diagnostics for an EBM solution: global mean, equator-pole gradient, ice line."""
    lat, T = res["lat"], res["T_C"]
    eq = float(T[np.argmin(np.abs(lat))])      # temperature at the equator
    pole = float(np.min(T))                    # coldest (polar) temperature
    return {
        "T_global_C": res["T_global_C"],
        "equator_C": eq,
        "pole_C": pole,
        "gradient_C": eq - pole,
        "ice_latitude_deg": res["ice_latitude_deg"],
    }
