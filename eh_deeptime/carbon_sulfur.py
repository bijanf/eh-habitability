"""A closed carbon-sulfur-oxygen-alkalinity box model with carbon isotopes.

This module generalises the 3-state PETM box model in :mod:`eh_deeptime.petm`
(DIC, ALK, d13C) to a CLOSED 9-state biogeochemical system (eight reservoirs
plus the d13C isotope tracer) in the spirit of
COPSE (Lenton et al. 2018) and GEOCARBSULF (Berner 2006), reduced to the minimum
needed for an illustration. It directly answers the referee point that the
carbon core was "a single ODE for 8 declared state variables, with no closed
C-S-O system and no d13C / O2 / sulfur subsystem".

State vector (9 components: 8 surface/crustal reservoirs + the isotope tracer):
    DIC      - dissolved inorganic carbon (mol)
    ALK      - alkalinity (mol equivalents)
    O2       - atmosphere+ocean dioxygen (mol)
    Corg_cr  - crustal/sedimentary organic carbon (mol)
    Ccarb_cr - crustal/sedimentary carbonate carbon (mol)
    S_pyr    - crustal pyrite sulfur (mol)
    S_sulf   - crustal gypsum/evaporite sulfur (mol)
    SO4      - ocean sulfate sulfur (mol); the reservoir that closes the S cycle so
               weathered sulfur has a destination and total S is conserved exactly
    d13C_DIC - delta-13C of the DIC pool (per mil); carried through the conserved
               tracer C13 = d13C_DIC * DIC

Atmospheric pCO2, pH and the calcite saturation state Omega are diagnosed from
(DIC, ALK) by the same explicit carbonate-equilibrium solve used in petm.py
(imported from .petm). Surface-temperature anomaly is diagnosed from pCO2 via a
climate-sensitivity (ECS) link.

Budgets (mol kyr^-1), COPSE/GEOCARBSULF-style:
    dDIC/dt      = F_degas + F_meta + W_carb + W_oxid - B_carb - B_org   (+ injection)
    dALK/dt      = 2*W_sil + 2*W_carb - 2*B_carb + Alk_S
    dO2/dt       = B_org + (15/8)*B_pyr - W_oxid - (15/8)*W_pyr - F_red
    dCorg_cr/dt  = B_org - W_oxid
    dCcarb_cr/dt = B_carb - W_carb - F_degas   (degassing recycles buried carbonate)
    dS_pyr/dt    = B_pyr - W_pyr
    dS_sulf/dt   = B_gyp - W_gyp
    dSO4/dt      = W_pyr + W_gyp - B_pyr - B_gyp   (closes the S cycle; total S conserved)
    d(d13C_DIC*DIC)/dt = sum_i F_i*delta_i - B_org*(d13C_DIC - Delta_B) - B_carb*d13C_DIC

The sulfur alkalinity term Alk_S follows the COPSE/GEOCARBSULF sign convention:
oxidative pyrite weathering produces sulfuric acid and REMOVES carbonate alkalinity,
while pyrite burial (microbial sulfate reduction) ADDS it, so Alk_S = 2*B_pyr - 2*W_pyr
(gypsum dissolution/precipitation is carbonate-alkalinity-neutral and omitted). Pyrite
and gypsum burial scale with the ocean sulfate inventory (SO4/SO4_0), so the loop closes
dynamically and total sulfur S_pyr + S_sulf + SO4 is conserved exactly (like carbon).

The CO2-consuming silicate weathering carries the COPSE feedback,
    W_sil = W0 * f_runoff * (pCO2/pCO2_0)^n_silw * exp[(Ea/R)(1/T0 - 1/T)],
oxidative weathering of crustal organic C and pyrite scale with O2 as
(O2/O2_0)^0.5, and carbonate compensation (Omega-dependent burial + seafloor
dissolution) is taken straight from petm.py. Reference fluxes are fixed so the
no-injection control is in EXACT steady state by construction (see
:func:`_calibrate_background`), exactly as in petm._calibrate_background.

This is an ILLUSTRATION / methods demonstration of closed multi-sphere
biogeochemical coupling, NOT a calibrated or validated reconstruction. All
constants are illustrative, chosen from published Phanerozoic parameter
envelopes; no real proxy data are used or claimed.

PRODUCTION SWAP: a research-grade version would (a) replace the single-box ocean
with a multi-box LOSCAR-class geometry, (b) calibrate the dozens of rate/feedback
constants against compiled proxy records under a Bayesian framework (PyMC/Stan or
a JAX-based ODE-adjoint), and (c) carry sulfur and oxygen isotopes (d34S, d18O)
and a phosphorus/nutrient subsystem. None of that is done here.
"""
from __future__ import annotations

import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import brentq

from .petm import (
    GTC_TO_MOL,
    M_OCEAN,
    PCO2_0,
    DIC0_CONC,
    D13C_0,
    _carbonate_solve,
)

# --- gas constant -------------------------------------------------------------
R_GAS = 8.314462618              # J mol^-1 K^-1
T0_K = 288.0                     # K, reference surface temperature

# --- background carbon reservoirs (mol) --------------------------------------
#   crustal C reservoirs are huge relative to the ocean; values are order-of-
#   magnitude Phanerozoic estimates (e.g. ~1.2e21 mol carbonate, ~1.25e21 mol
#   organic C) used only to set finite, plausible turnover times.
CORG_CR_0 = 1.25e21              # mol, crustal organic carbon (illustrative)
CCARB_CR_0 = 5.0e21             # mol, crustal carbonate carbon (illustrative)
S_PYR_0 = 2.0e20                # mol, crustal pyrite sulfur (illustrative)
S_SULF_0 = 2.0e20              # mol, crustal gypsum/evaporite sulfur (illustrative)
SO4_OCN_0 = 4.0e19            # mol, ocean sulfate inventory (illustrative, ~present)

# --- background O2 ------------------------------------------------------------
O2_0 = 3.7e19                    # mol, present atmospheric O2 inventory (PAL)

# --- background carbon fluxes (mol kyr^-1) -----------------------------------
#   tuned to the same scale as petm.py so the carbonate sub-system behaves the
#   same; the silicate/organic/pyrite branches are added on top.
F_DEGAS = 8.0e15                 # volcanic + metamorphic CO2 degassing
F_META = 0.0                     # extra metamorphic decarbonation (folded into F_DEGAS)
F_CARBW0 = 10.0e15              # carbonate weathering at background pCO2
F_BORG0 = 4.0e15                # organic-carbon burial at background
F_WOXID0 = 4.0e15              # oxidative weathering of crustal organic C at background

# --- background sulfur fluxes (mol kyr^-1) -----------------------------------
F_BPYR0 = 1.0e15               # pyrite burial at background
F_WPYR0 = 1.0e15              # oxidative pyrite weathering at background
F_BGYP0 = 0.5e15             # gypsum burial at background
F_WGYP0 = 0.5e15            # gypsum/sulfate weathering at background

# --- weathering-feedback exponents -------------------------------------------
N_SILW = 0.80                    # silicate-weathering pCO2 sensitivity (COPSE)
N_CARBW = 0.30                   # carbonate-weathering pCO2 sensitivity
N_OXIDW = 0.50                   # O2 sensitivity of oxidative weathering
EA_SILW = 42000.0               # J mol^-1, silicate-weathering activation energy
EA_OXIDW = 50000.0             # J mol^-1, kinetic T-sensitivity of oxidative
#   weathering: warming accelerates oxidation of crustal organic C and pyrite,
#   a recognised O2 sink and the dominant illustrative driver of the transient
#   O2 drawdown during a warming pulse. Equals 1.0 at the background T0, so the
#   control steady state is unaffected by construction.
F_RUNOFF = 1.0                  # runoff/hydrology scaling factor (illustrative)

# --- carbonate-compensation constants (as in petm) ---------------------------
P_BURIAL = 2.0                  # CaCO3-burial saturation exponent
K_DISS = 6.0e16               # seafloor CaCO3 dissolution scale (mol kyr^-1)

# --- isotopic signatures (per mil) -------------------------------------------
DELTA_DEGAS = -5.0              # degassed CO2
DELTA_B = 28.0                  # organic-burial fractionation (Corg ~ DIC - DELTA_B)
ECS_DEFAULT = 3.0               # K per CO2 doubling

# --- oxygen stoichiometry ----------------------------------------------------
#   pyrite burial 2 Fe2O3 + 8 SO4^2- ... releases 15/8 O2 per mol S buried.
PYR_O2 = 15.0 / 8.0

# Public default-parameter dictionary (referenced across modules).
DEFAULT_PARAMS = {
    "ecs": ECS_DEFAULT,         # K per CO2 doubling
    "f_degas": 1.0,             # degassing multiplier (1.0 = background)
    "f_runoff": F_RUNOFF,       # runoff/hydrology multiplier
    "n_silw": N_SILW,           # silicate-weathering pCO2 exponent
    "Ea": EA_SILW,              # silicate-weathering activation energy (J/mol)
    "delta_b": DELTA_B,         # organic-burial isotopic fractionation (per mil)
    "delta_degas": DELTA_DEGAS,  # degassed-CO2 delta-13C (per mil)
    "n_oxidw": N_OXIDW,         # O2 exponent of oxidative weathering
    "ea_oxidw": EA_OXIDW,       # T-sensitivity of oxidative weathering (J/mol)
}


def _temp_from_pco2(pco2, ecs):
    """Surface temperature (K) diagnosed from pCO2 via climate sensitivity."""
    return T0_K + ecs * np.log2(max(pco2, 1e-6) / PCO2_0)


def _calibrate_background(params):
    """Fix reference fluxes so the no-injection control is in EXACT steady state.

    Mirrors petm._calibrate_background: choose ALK0 to hit PCO2_0 at DIC0, then
    solve the carbon, alkalinity, oxygen, sulfur and isotope balances for the
    derived burial / weathering constants. Returns a dict of derived constants.
    """
    dic0 = DIC0_CONC
    # ALK0 (mol kg^-1) that yields PCO2_0 at DIC0
    alk0 = brentq(lambda a: _carbonate_solve(dic0, a)[0] - PCO2_0,
                  1.5e-3, 3.5e-3, maxiter=200, xtol=1e-12)
    _, _, omega0 = _carbonate_solve(dic0, alk0)

    f_degas = F_DEGAS * params["f_degas"]

    # --- carbon balance: dDIC/dt = 0
    #   f_degas + F_carbw0 + F_woxid0 = B_carb0 + B_org0
    f_bcarb0 = f_degas + F_CARBW0 + F_WOXID0 - F_BORG0
    # --- alkalinity balance: dALK/dt = 0
    #   2 f_silw0 + 2 F_carbw0 - 2 B_carb0 + Alk_S0 = 0
    #   COPSE sign: pyrite burial adds alkalinity, pyrite weathering removes it;
    #   gypsum is carbonate-alkalinity-neutral. At background W_pyr0 = B_pyr0 so
    #   Alk_S0 = 0 and the control alkalinity balance is unchanged.
    alk_s0 = 2.0 * F_BPYR0 - 2.0 * F_WPYR0
    f_silw0 = f_bcarb0 - F_CARBW0 - 0.5 * alk_s0
    # CaCO3-burial saturation constant matched to f_bcarb0 at omega0
    k_burial = f_bcarb0 / (omega0 - 1.0) ** P_BURIAL

    # --- delta-13C balance: at steady state d(d13C*DIC)/dt = 0 with d13C = D13C_0
    #   inputs: degassing, carbonate weathering, oxidative weathering of org C
    #   outputs (as fractionation): organic burial removes (D13C_0 - DELTA_B);
    #   carbonate burial removes at the mean.
    #   sum F_i (delta_i - D13C_0) - B_org*(-DELTA_B) = 0  -> solve delta_carbw
    #   Treat oxidative-weathering CO2 as carrying the crustal-organic signature
    #   delta_org = D13C_0 - DELTA_B.
    delta_degas = params["delta_degas"]
    delta_b = params["delta_b"]
    delta_org = D13C_0 - delta_b
    #   f_degas*(d_degas - d0) + F_carbw0*(d_carbw - d0)
    #     + F_woxid0*(d_org - d0) + B_org0*delta_b = 0
    delta_carbw = D13C_0 - (
        f_degas * (delta_degas - D13C_0)
        + F_WOXID0 * (delta_org - D13C_0)
        + F_BORG0 * delta_b
    ) / F_CARBW0

    return {
        "dic0_mol": dic0 * M_OCEAN,
        "alk0_mol": alk0 * M_OCEAN,
        "omega0": omega0,
        "f_degas": f_degas,
        "f_bcarb0": f_bcarb0,
        "f_silw0": f_silw0,
        "k_burial": k_burial,
        "alk_s0": alk_s0,
        "delta_carbw": delta_carbw,
        "delta_org": delta_org,
    }


def _injection_rate(t, m_inj_mol, t_dur):
    """Carbon-release rate (mol kyr^-1) at time t (kyr): a smooth pulse on [0, t_dur]."""
    if t < 0.0 or t > t_dur:
        return 0.0
    # raised-cosine pulse integrating to m_inj_mol over [0, t_dur]
    return m_inj_mol / t_dur * (1.0 - np.cos(2.0 * np.pi * t / t_dur))


def _rhs(t, y, bg, params, m_inj_mol, t_dur, delta_inj):
    """Right-hand side of the 8-state closed C-S-O-ALK system.

    State y = [DIC, ALK, O2, Corg_cr, Ccarb_cr, S_pyr, S_sulf, SO4, C13]
    where C13 = d13C_DIC * DIC is the conserved isotope tracer.
    """
    dic_mol, alk_mol, o2_mol, corg, ccarb, s_pyr, s_sulf, so4, c13 = y
    dic_mol = max(dic_mol, 1e6)
    o2_mol = max(o2_mol, 1e-30)
    d13c = c13 / dic_mol

    dic_conc = dic_mol / M_OCEAN
    alk_conc = alk_mol / M_OCEAN
    pco2, _ph, omega = _carbonate_solve(dic_conc, alk_conc)

    temp = _temp_from_pco2(pco2, params["ecs"])
    r_co2 = max(pco2 / PCO2_0, 1e-6)
    o2_ratio = max(o2_mol / O2_0, 1e-9)
    so4_ratio = max(so4 / SO4_OCN_0, 1e-9)

    # --- carbon weathering / burial ------------------------------------------
    arr = np.exp((params["Ea"] / R_GAS) * (1.0 / T0_K - 1.0 / temp))
    arr_ox = np.exp((params["ea_oxidw"] / R_GAS) * (1.0 / T0_K - 1.0 / temp))
    f_silw = bg["f_silw0"] * params["f_runoff"] * r_co2 ** params["n_silw"] * arr
    f_carbw = F_CARBW0 * params["f_runoff"] * r_co2 ** N_CARBW
    f_woxid = F_WOXID0 * o2_ratio ** params["n_oxidw"] * arr_ox
    f_bcarb = bg["k_burial"] * max(omega - 1.0, 0.0) ** P_BURIAL
    f_diss = K_DISS * max(1.0 - omega, 0.0)   # carbonate compensation
    f_borg = F_BORG0                          # constant organic-C burial
    f_inj = _injection_rate(t, m_inj_mol, t_dur)

    # --- sulfur weathering / burial ------------------------------------------
    f_wpyr = F_WPYR0 * o2_ratio ** params["n_oxidw"] * arr_ox   # pyrite oxidation
    f_wgyp = F_WGYP0                                     # gypsum dissolution
    # burial scales with the ocean sulfate inventory so the S cycle closes
    # dynamically: weathered S accumulates in SO4 and is removed by burial.
    f_bpyr = F_BPYR0 * so4_ratio                         # microbial sulfate reduction
    f_bgyp = F_BGYP0 * so4_ratio                         # evaporite (gypsum) burial

    # --- budgets -------------------------------------------------------------
    ddic = bg["f_degas"] + F_META + f_carbw + f_woxid + f_inj + f_diss - f_bcarb - f_borg
    # COPSE sulfur alkalinity: pyrite burial (sulfate reduction) ADDS alkalinity,
    # oxidative pyrite weathering REMOVES it; gypsum is carbonate-alk-neutral.
    alk_s = 2.0 * f_bpyr - 2.0 * f_wpyr
    dalk = 2.0 * f_silw + 2.0 * f_carbw + 2.0 * f_diss - 2.0 * f_bcarb + alk_s
    do2 = f_borg + PYR_O2 * f_bpyr - f_woxid - PYR_O2 * f_wpyr
    dcorg = f_borg - f_woxid
    # degassing recycles buried crustal carbonate (volcanic/metamorphic
    # decarbonation), so it is a sink on Ccarb_cr -> total carbon is conserved.
    dccarb = f_bcarb - f_carbw - f_diss - bg["f_degas"]
    ds_pyr = f_bpyr - f_wpyr
    ds_sulf = f_bgyp - f_wgyp
    # ocean sulfate closes the S cycle: weathering sources it, burial removes it,
    # so total sulfur S_pyr + S_sulf + SO4 is conserved exactly.
    dso4 = f_wpyr + f_wgyp - f_bpyr - f_bgyp

    # --- carbon isotope tracer C13 = d13C * DIC ------------------------------
    #   inputs carry their source delta; organic burial removes (d13C - DELTA_B);
    #   carbonate burial removes at the DIC mean (d13C). Injection carries delta_inj.
    dc13 = (
        bg["f_degas"] * params["delta_degas"]
        + f_carbw * bg["delta_carbw"]
        + f_woxid * bg["delta_org"]
        + f_diss * D13C_0
        + f_inj * delta_inj
        - f_borg * (d13c - params["delta_b"])
        - f_bcarb * d13c
    )

    return [ddic, dalk, do2, dcorg, dccarb, ds_pyr, ds_sulf, dso4, dc13]


def run_csys(params=None, m_inj=0.0, t_dur=5.0, delta_inj=-50.0,
             t_end=500.0, n_out=521):
    """Run the illustrative closed carbon-sulfur-oxygen-alkalinity box model.

    Parameters
    ----------
    params : dict | None   overrides for DEFAULT_PARAMS
    m_inj : float          total carbon release (Gt C); 0.0 = steady control
    t_dur : float          release duration (kyr)
    delta_inj : float      injected-carbon delta-13C (per mil)
    t_end : float          end time (kyr)
    n_out : int            number of output points (grid linspace(-20, t_end))

    Returns
    -------
    dict with keys kyr, pco2 (ppm), temp (K anomaly), ph, omega, o2 (fraction of
    present atmospheric level), d13c (per mil), dic (mol), alk (mol),
    corg_cr (mol), ccarb_cr (mol), s_pyr (mol), s_sulf (mol), so4 (mol), params.
    All arrays share the shape of 'kyr'. The control (m_inj=0) is steady by
    construction.
    """
    p = dict(DEFAULT_PARAMS)
    if params:
        p.update(params)
    bg = _calibrate_background(p)

    kyr = np.linspace(-20.0, t_end, n_out)
    m_inj_mol = m_inj * GTC_TO_MOL

    y0 = [
        bg["dic0_mol"], bg["alk0_mol"], O2_0,
        CORG_CR_0, CCARB_CR_0, S_PYR_0, S_SULF_0, SO4_OCN_0,
        D13C_0 * bg["dic0_mol"],
    ]
    teval = kyr[kyr >= 0.0]
    atol = [1e10, 1e10, 1e12, 1e14, 1e14, 1e12, 1e12, 1e12, 1e10]
    sol = solve_ivp(_rhs, (0.0, kyr.max()), y0, t_eval=teval,
                    args=(bg, p, m_inj_mol, t_dur, delta_inj),
                    method="LSODA", rtol=1e-8, atol=atol, max_step=2.0)
    if not sol.success:
        raise RuntimeError(f"C-S-O integration failed: {sol.message}")

    npts = len(kyr)
    out = {k: np.full(npts, np.nan) for k in
           ("pco2", "temp", "ph", "omega", "o2", "d13c",
            "dic", "alk", "corg_cr", "ccarb_cr", "s_pyr", "s_sulf", "so4")}

    # pre-onset = steady background
    pre = kyr < 0.0
    p0, ph0, om0 = _carbonate_solve(bg["dic0_mol"] / M_OCEAN, bg["alk0_mol"] / M_OCEAN)
    out["pco2"][pre] = PCO2_0
    out["temp"][pre] = 0.0
    out["ph"][pre] = ph0
    out["omega"][pre] = om0
    out["o2"][pre] = 1.0
    out["d13c"][pre] = D13C_0
    out["dic"][pre] = bg["dic0_mol"]
    out["alk"][pre] = bg["alk0_mol"]
    out["corg_cr"][pre] = CORG_CR_0
    out["ccarb_cr"][pre] = CCARB_CR_0
    out["s_pyr"][pre] = S_PYR_0
    out["s_sulf"][pre] = S_SULF_0
    out["so4"][pre] = SO4_OCN_0

    post = np.where(kyr >= 0.0)[0]
    for k, idx in enumerate(post):
        dic_mol, alk_mol, o2_mol = sol.y[0, k], sol.y[1, k], sol.y[2, k]
        corg, ccarb, s_pyr, s_sulf = sol.y[3, k], sol.y[4, k], sol.y[5, k], sol.y[6, k]
        so4, c13 = sol.y[7, k], sol.y[8, k]
        pc, phk, omk = _carbonate_solve(dic_mol / M_OCEAN, alk_mol / M_OCEAN)
        out["pco2"][idx] = pc
        out["temp"][idx] = p["ecs"] * np.log2(pc / PCO2_0)
        out["ph"][idx] = phk
        out["omega"][idx] = omk
        out["o2"][idx] = o2_mol / O2_0
        out["d13c"][idx] = c13 / dic_mol
        out["dic"][idx] = dic_mol
        out["alk"][idx] = alk_mol
        out["corg_cr"][idx] = corg
        out["ccarb_cr"][idx] = ccarb
        out["s_pyr"][idx] = s_pyr
        out["s_sulf"][idx] = s_sulf
        out["so4"][idx] = so4

    out["kyr"] = kyr
    out["params"] = {"m_inj": m_inj, "t_dur": t_dur, "delta_inj": delta_inj, **p}
    return out


def summarise(res):
    """Diagnostics for an illustrative run.

    Returns peak_warming_K, cie_permil (most-negative d13C excursion), tau_rec_kyr
    (recovery to within 10% of background pCO2 after the peak), and d_o2_PAL (the
    maximum O2 drawdown, in present-atmospheric-level fraction; negative = loss).
    """
    kyr, pco2, temp, d13c, o2 = (res["kyr"], res["pco2"], res["temp"],
                                 res["d13c"], res["o2"])
    peak_warming = float(np.nanmax(temp))
    cie = float(np.nanmin(d13c) - D13C_0)

    post = kyr >= 0.0
    exc = pco2[post] - PCO2_0
    kk = kyr[post]
    kpeak = kk[int(np.nanargmax(exc))]
    thresh = 0.10 * PCO2_0
    rec_after = kk[(kk > kpeak) & (exc <= thresh)]
    tau_rec = float(rec_after[0]) if len(rec_after) else float("nan")

    d_o2 = float(np.nanmin(o2) - 1.0)   # fractional O2 change vs PAL (<=0)
    return {"peak_warming_K": peak_warming, "cie_permil": cie,
            "tau_rec_kyr": tau_rec, "d_o2_PAL": d_o2}


def steady_drift(res):
    """Max fractional drift of the key reservoirs over a run (~0 for a control).

    For the no-injection control the system is steady by construction, so every
    reservoir should hold its background value; this returns the largest relative
    excursion seen across DIC, ALK, O2 and the four crustal reservoirs.
    """
    keys_ref = {
        "dic": res["dic"][0],
        "alk": res["alk"][0],
        "corg_cr": res["corg_cr"][0],
        "ccarb_cr": res["ccarb_cr"][0],
        "s_pyr": res["s_pyr"][0],
        "s_sulf": res["s_sulf"][0],
        "so4": res["so4"][0],
    }
    drift = 0.0
    for k, ref in keys_ref.items():
        arr = res[k]
        if ref == 0.0:
            continue
        drift = max(drift, float(np.nanmax(np.abs(arr - ref)) / abs(ref)))
    # O2 is reported as PAL fraction (background = 1.0)
    drift = max(drift, float(np.nanmax(np.abs(res["o2"] - 1.0))))
    return drift
