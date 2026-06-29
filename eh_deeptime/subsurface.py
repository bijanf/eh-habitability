"""An illustrative deep subsurface-biosphere carbon box (Hypothesis H3).

The deep continental subsurface hosts a large microbial biomass. Published global
estimates:
  - Magnabosco et al. 2018, Nat. Geosci. 11, 707 (doi 10.1038/s41561-018-0221-6):
    continental subsurface prokaryotic biomass ~23-31 Pg C.
  - Bar-On, Phillips & Milo 2018, PNAS 115, 6506 (doi 10.1073/pnas.1711842115):
    total subsurface bacteria + archaea ~60-70 Pg C (deep marine + terrestrial).
  - The proposal's Hypothesis H3 target: present-day habitable subsurface carbon
    ~20 +- 10 Pg C.

WHAT THIS IS. A minimal, transparent box for the *continental* habitable
subsurface. The microbial cell-abundance-with-depth profile is an exponential
decline (the shape seen in global compilations, e.g. McMahon & Parnell 2014); the
deep habitability floor is the ~122 C upper temperature limit for cultured life
(Takai et al. 2008, doi 10.1073/pnas.0712334105) reached at a depth set by the
geothermal gradient. The box integrates habitable cell abundance over depth.

HONEST SCOPE. The ABSOLUTE carbon scale is ANCHORED to the published Magnabosco
(2018) total -- it is NOT predicted from first principles. So H3 is stated as a
*consistency / parameter* statement ("the model carries a subsurface C stock
consistent with the published 23-31 Pg C"), never as an independent confirmation.
The genuine, non-circular content is the MECHANISTIC RESPONSE: as the surface warms
(or the geothermal gradient steepens) the 122 C isotherm shoals, shrinking the
habitable depth window and the carbon it holds. No fabricated data: the anchor and
the profile shape are cited published values; the response is forward model output.
"""
from __future__ import annotations

import numpy as np

# --- published estimates (Pg C) ---------------------------------------------
MAGNABOSCO2018_PGC = (23.0, 31.0)     # continental subsurface prokaryotic biomass
BARON2018_TOTAL_PGC = (60.0, 70.0)    # total subsurface (marine + terrestrial)
H3_TARGET_PGC = (10.0, 30.0)          # proposal H3: 20 +- 10 Pg C

# --- box parameters (illustrative; profile shape from published compilations) ---
T_MAX_LIFE_C = 122.0       # upper temperature limit for cultured life (Takai 2008)
GEOTHERM_C_PER_KM = 25.0   # continental geothermal gradient (deg C / km), typical
Z_SCALE_KM = 2.0           # e-folding depth of microbial cell abundance (km)
T_SURFACE_REF_C = 15.0     # present-day mean surface temperature (anchor reference)

# absolute scale anchored so the present-day continental stock = midpoint of the
# Magnabosco (2018) range (this is an ANCHOR to the literature, not a prediction).
ANCHOR_PGC = float(np.mean(MAGNABOSCO2018_PGC))


def habitable_depth_km(t_surface_C=T_SURFACE_REF_C, geotherm=GEOTHERM_C_PER_KM,
                       t_max=T_MAX_LIFE_C):
    """Depth (km) at which the geotherm reaches the upper temperature limit for life.

    z_hab = (T_max - T_surface) / geotherm. Below this the rock is too hot to be
    habitable; a warmer surface or steeper geotherm shoals it.
    """
    return max(0.0, (t_max - float(t_surface_C)) / float(geotherm))


def _relative_habitable_carbon(t_surface_C, geotherm, z_scale=Z_SCALE_KM,
                               t_max=T_MAX_LIFE_C):
    """Depth-integral of an exponential cell-abundance profile over [0, z_hab].

    integral_0^z_hab exp(-z/z_scale) dz = z_scale * (1 - exp(-z_hab/z_scale)).
    Returns a RELATIVE measure (km-equivalent); the absolute Pg C scale is applied
    separately by anchoring to the published present-day total.
    """
    z_hab = habitable_depth_km(t_surface_C, geotherm, t_max)
    return z_scale * (1.0 - np.exp(-z_hab / z_scale))


# normalisation constant so that present-day conditions give ANCHOR_PGC
_REF_REL = _relative_habitable_carbon(T_SURFACE_REF_C, GEOTHERM_C_PER_KM)
_PGC_PER_REL = ANCHOR_PGC / _REF_REL


def subsurface_carbon(t_surface_C=T_SURFACE_REF_C, geotherm=GEOTHERM_C_PER_KM,
                      z_scale=Z_SCALE_KM, t_max=T_MAX_LIFE_C):
    """Present-day-anchored continental habitable subsurface carbon stock (Pg C).

    Scaled so that at reference conditions (T_surface=15 C, geotherm=25 C/km) the
    stock equals the Magnabosco (2018) midpoint (~27 Pg C) BY CONSTRUCTION. The
    value-add is the response to changed conditions, not the absolute number.
    """
    rel = _relative_habitable_carbon(t_surface_C, geotherm, z_scale, t_max)
    return float(_PGC_PER_REL * rel)


def h3_consistency(stock_pgc=None):
    """State Hypothesis H3 as a consistency check against the published envelopes.

    Returns the present-day stock and whether it lies within the Magnabosco (2018)
    range and the proposal's H3 target. By construction the reference stock matches
    Magnabosco; this is a consistency/parameter statement, not an independent test.
    """
    if stock_pgc is None:
        stock_pgc = subsurface_carbon()
    lo_m, hi_m = MAGNABOSCO2018_PGC
    lo_h, hi_h = H3_TARGET_PGC
    return {
        "present_day_PgC": float(stock_pgc),
        "habitable_depth_km": habitable_depth_km(),
        "magnabosco2018_PgC": MAGNABOSCO2018_PGC,
        "h3_target_PgC": H3_TARGET_PGC,
        "within_magnabosco": bool(lo_m <= stock_pgc <= hi_m),
        "within_h3_target": bool(lo_h <= stock_pgc <= hi_h),
        "note": "absolute scale ANCHORED to Magnabosco (2018); consistency/parameter "
                "statement, not an independent prediction",
    }


def warming_response(delta_T_C=None, geotherm=GEOTHERM_C_PER_KM):
    """Mechanistic response: habitable subsurface carbon vs surface warming.

    A warmer surface raises the whole geotherm, shoaling the 122 C isotherm and
    shrinking the habitable depth window. Returns the stock (Pg C) and habitable
    depth (km) across a surface-warming sweep -- forward model output, not anchored
    away. (The reference, delta_T=0, equals the anchored present-day stock.)
    """
    if delta_T_C is None:
        delta_T_C = np.linspace(0.0, 20.0, 21)
    delta_T_C = np.asarray(delta_T_C, dtype=float)
    stock = np.array([subsurface_carbon(T_SURFACE_REF_C + dT, geotherm)
                      for dT in delta_T_C])
    depth = np.array([habitable_depth_km(T_SURFACE_REF_C + dT, geotherm)
                      for dT in delta_T_C])
    return {"delta_T_C": delta_T_C, "stock_PgC": stock, "habitable_depth_km": depth}
