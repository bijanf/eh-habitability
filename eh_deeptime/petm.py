"""A compact 0-D carbon-cycle box model for an illustrative PETM hindcast.

LOSCAR/COPSE-class reduced model (see package docstring). State vector (well-mixed ocean):
    DIC  - dissolved inorganic carbon (mol)
    ALK  - alkalinity (mol equivalents)
    d13C - delta-13C of the DIC pool (per mil)

Atmospheric pCO2, pH and the calcite saturation state (Omega) are diagnosed from
(DIC, ALK) by an explicit carbonate-equilibrium solve; surface-temperature anomaly is
diagnosed from pCO2 via a climate-sensitivity (ECS) link. Atmospheric carbon mass
(~few % of the ocean) is neglected in the budget, standard for an illustrative box model.

Fluxes (mol kyr^-1): volcanic/metamorphic degassing, CO2-dependent silicate and carbonate
weathering (the COPSE feedback), Omega-dependent CaCO3 burial (carbonate compensation),
a constant organic-carbon burial (sets the d13C baseline), and the PETM injection.
Flux constants are fixed so the no-injection control is in steady state by construction.

This is an ILLUSTRATION of deep-time multi-sphere coupling, not a calibrated reconstruction.
"""
from __future__ import annotations

import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import brentq

# --- fixed physical constants -------------------------------------------------
GTC_TO_MOL = 1e15 / 12.011        # 1 Gt C -> mol C
M_OCEAN = 1.32e21                  # kg seawater
CA = 1.028e-2                      # mol kg^-1, ~constant ocean [Ca2+]
PPM_PER_ATM = 1.0e6               # pCO2 in ppm = atm * 1e6

# carbonate-system equilibrium constants (mol kg^-1; illustrative, seawater ~T0)
K0 = 2.84e-2                       # CO2 solubility (mol kg^-1 atm^-1), Henry
K1 = 1.39e-6                       # first dissociation  (pK1 ~ 5.86)
K2 = 1.20e-9                       # second dissociation (pK2 ~ 8.92)
KSP = 4.27e-7                      # calcite solubility product (mol kg^-1)^2

# --- background (pre-PETM) targets -------------------------------------------
PCO2_0 = 900.0                    # ppm, PETM-background atmospheric CO2
DIC0_CONC = 2.30e-3               # mol kg^-1, background ocean DIC
D13C_0 = 1.0                      # per mil, background ocean DIC delta-13C

# --- flux scales (mol kyr^-1) and isotopic signatures ------------------------
F_VOLC = 8.0e15                   # volcanic + metamorphic degassing
F_CARBW0 = 10.0e15               # carbonate weathering at background pCO2
F_BORG0 = 1.0e15                  # organic-carbon burial (constant)
N_SILW = 0.80                     # effective silicate-weathering pCO2 sensitivity
#   (bundles the direct CO2, temperature and runoff dependences of the COPSE
#    weathering feedback into one illustrative exponent; sets the recovery timescale)
N_CARBW = 0.30                    # carbonate-weathering CO2 exponent
P_BURIAL = 2.0                    # CaCO3-burial saturation exponent
K_DISS = 6.0e16                   # seafloor CaCO3 dissolution scale (carbonate compensation)
DELTA_VOLC = -5.0                 # per mil, degassed CO2
DELTA_ORG_FRAC = 25.0             # per mil, organic-burial fractionation


def _carbonate_solve(dic_conc, alk_conc):
    """Given DIC, ALK (mol kg^-1), return (pco2_ppm, pH, omega) via a carbonate solve.

    Carbonate alkalinity ALK ~= [HCO3-] + 2[CO3 2-]; borate/water terms folded into the
    illustrative constants. Solves for [H+] by bracketing on pH in [5.5, 9.5].
    """
    def alk_residual(h):
        denom = h * h + K1 * h + K1 * K2
        hco3 = dic_conc * K1 * h / denom
        co3 = dic_conc * K1 * K2 / denom
        return hco3 + 2.0 * co3 - alk_conc

    # bracket [H+] over pH 4-11 -- wide enough that even strongly perturbed
    # (DIC, ALK) states keep the root inside it; guard a degenerate same-sign bracket
    # so a clear error is raised instead of brentq's opaque ValueError.
    lo, hi = 10.0 ** -11, 10.0 ** -4
    if alk_residual(lo) * alk_residual(hi) > 0.0:
        raise ValueError(
            f"carbonate solve: no pH root in [4, 11] for DIC={dic_conc:.3e}, "
            f"ALK={alk_conc:.3e} mol/kg")
    h = brentq(alk_residual, lo, hi, maxiter=200, xtol=1e-18)
    denom = h * h + K1 * h + K1 * K2
    co2aq = dic_conc * h * h / denom
    co3 = dic_conc * K1 * K2 / denom
    pco2 = co2aq / K0 * PPM_PER_ATM
    omega = CA * co3 / KSP
    return pco2, -np.log10(h), omega


def _calibrate_background():
    """Pick ALK0 so the background (DIC0, ALK0) gives PCO2_0, then set burial/weathering
    reference fluxes and delta_carbw so the no-injection control is in exact steady state.

    Returns a dict of derived constants used by the RHS.
    """
    dic0 = DIC0_CONC
    # find ALK0 (mol kg^-1) that yields PCO2_0 at DIC0
    def pco2_err(alk):
        return _carbonate_solve(dic0, alk)[0] - PCO2_0
    alk0 = brentq(pco2_err, 1.5e-3, 3.5e-3, maxiter=200, xtol=1e-12)
    _, _, omega0 = _carbonate_solve(dic0, alk0)

    # carbon balance: F_volc + F_carbw0 = F_bcarb0 + F_borg0
    f_bcarb0 = F_VOLC + F_CARBW0 - F_BORG0
    # alkalinity balance: 2 F_silw0 + 2 F_carbw0 = 2 F_bcarb0
    f_silw0 = f_bcarb0 - F_CARBW0
    k_burial = f_bcarb0 / (omega0 - 1.0) ** P_BURIAL
    # delta steady state: F_volc(dv-d0)+F_carbw0(dc-d0)+F_borg0*frac = 0  -> solve dc
    delta_carbw = D13C_0 - (F_VOLC * (DELTA_VOLC - D13C_0) + F_BORG0 * DELTA_ORG_FRAC) / F_CARBW0

    return {
        "dic0_mol": dic0 * M_OCEAN, "alk0_mol": alk0 * M_OCEAN,
        "omega0": omega0, "f_bcarb0": f_bcarb0, "f_silw0": f_silw0,
        "k_burial": k_burial, "delta_carbw": delta_carbw,
    }


_BG = _calibrate_background()


def _injection_rate(t, m_inj_mol, t_dur):
    """Carbon-release rate (mol kyr^-1) at time t (kyr): a smooth pulse over [0, t_dur]."""
    if t < 0.0 or t > t_dur:
        return 0.0
    # raised-cosine pulse integrating to m_inj_mol over [0, t_dur]
    return m_inj_mol / t_dur * (1.0 - np.cos(2.0 * np.pi * t / t_dur))


def _rhs(t, y, m_inj_mol, t_dur, delta_inj):
    dic_mol, alk_mol, d13c = y
    dic_conc = dic_mol / M_OCEAN
    alk_conc = alk_mol / M_OCEAN
    pco2, _ph, omega = _carbonate_solve(dic_conc, alk_conc)

    r = max(pco2 / PCO2_0, 1e-6)
    f_silw = _BG["f_silw0"] * r ** N_SILW
    f_carbw = F_CARBW0 * r ** N_CARBW
    f_bcarb = _BG["k_burial"] * max(omega - 1.0, 0.0) ** P_BURIAL
    # seafloor carbonate dissolution when undersaturated (carbonate compensation): the
    # acidity-neutralising feedback that damps the pCO2 spike and drives fast recovery
    f_diss = K_DISS * max(1.0 - omega, 0.0)
    f_inj = _injection_rate(t, m_inj_mol, t_dur)

    ddic = F_VOLC + f_carbw + f_inj + f_diss - f_bcarb - F_BORG0
    dalk = 2.0 * f_silw + 2.0 * f_carbw + 2.0 * f_diss - 2.0 * f_bcarb
    # d13C of DIC (carbonate burial removes at the mean -> no effect; organic burial is light)
    dd13c = (F_VOLC * (DELTA_VOLC - d13c)
             + f_carbw * (_BG["delta_carbw"] - d13c)
             + f_diss * (D13C_0 - d13c)
             + f_inj * (delta_inj - d13c)
             + F_BORG0 * DELTA_ORG_FRAC) / dic_mol
    return [ddic, dalk, dd13c]


def run_petm(m_inj=3000.0, t_dur=5.0, delta_inj=-50.0, ecs=3.0,
             kyr=None, t_end=300.0, n_out=641):
    """Run the illustrative PETM box model.

    Parameters
    ----------
    m_inj : float      total carbon release (Gt C); consensus ~3000-7000
    t_dur : float      release duration (kyr)
    delta_inj : float  injected-carbon delta-13C (per mil); methane ~-60, organic/volcanic higher
    ecs : float        equilibrium climate sensitivity (K per CO2 doubling)
    kyr : array|None   output time grid (kyr, relative to onset); default linspace(-20, t_end)
    t_end : float      end time (kyr)
    n_out : int        number of output points if kyr is None

    Returns
    -------
    dict with keys: kyr, pco2 (ppm), temp (K anomaly), d13c_surf (per mil),
                    ph, omega, dic (mol), alk (mol)
    """
    if kyr is None:
        kyr = np.linspace(-20.0, t_end, n_out)
    kyr = np.asarray(kyr, dtype=float)
    m_inj_mol = m_inj * GTC_TO_MOL

    y0 = [_BG["dic0_mol"], _BG["alk0_mol"], D13C_0]
    teval = kyr[kyr >= 0.0]
    sol = solve_ivp(_rhs, (0.0, kyr.max()), y0, t_eval=teval,
                    args=(m_inj_mol, t_dur, delta_inj), method="LSODA",
                    rtol=1e-7, atol=[1e10, 1e10, 1e-6], max_step=2.0)
    if not sol.success:
        raise RuntimeError(f"PETM integration failed: {sol.message}")

    # assemble outputs on the full grid (pre-onset = steady background)
    npts = len(kyr)
    pco2 = np.full(npts, np.nan); temp = np.full(npts, np.nan)
    d13c = np.full(npts, np.nan); ph = np.full(npts, np.nan); omega = np.full(npts, np.nan)
    dic = np.full(npts, np.nan); alk = np.full(npts, np.nan)
    pre = kyr < 0.0
    pco2[pre] = PCO2_0; temp[pre] = 0.0; d13c[pre] = D13C_0
    p0, ph0, om0 = _carbonate_solve(_BG["dic0_mol"] / M_OCEAN, _BG["alk0_mol"] / M_OCEAN)
    ph[pre] = ph0; omega[pre] = om0
    dic[pre] = _BG["dic0_mol"]; alk[pre] = _BG["alk0_mol"]

    post = np.where(kyr >= 0.0)[0]
    for k, idx in enumerate(post):
        dic_mol, alk_mol, dd = sol.y[0, k], sol.y[1, k], sol.y[2, k]
        pc, phk, omk = _carbonate_solve(dic_mol / M_OCEAN, alk_mol / M_OCEAN)
        pco2[idx] = pc; ph[idx] = phk; omega[idx] = omk
        temp[idx] = ecs * np.log2(pc / PCO2_0)
        d13c[idx] = dd; dic[idx] = dic_mol; alk[idx] = alk_mol

    return {"kyr": kyr, "pco2": pco2, "temp": temp, "d13c_surf": d13c,
            "ph": ph, "omega": omega, "dic": dic, "alk": alk,
            "params": {"m_inj": m_inj, "t_dur": t_dur, "delta_inj": delta_inj, "ecs": ecs}}


def summarise(res):
    """Diagnostics for an illustrative run: peak warming, CIE magnitude, recovery time."""
    kyr, pco2, temp, d13c = res["kyr"], res["pco2"], res["temp"], res["d13c_surf"]
    peak_warming = float(np.nanmax(temp))
    cie = float(np.nanmin(d13c) - D13C_0)
    # recovery time: first kyr after the peak at which pCO2 returns to within 10% of
    # background (the standard "recovery to within 10% of background" definition)
    post = kyr >= 0.0
    exc = pco2[post] - PCO2_0
    kk = kyr[post]
    kpeak = kk[int(np.nanargmax(exc))]
    thresh = 0.10 * PCO2_0
    rec_after = kk[(kk > kpeak) & (exc <= thresh)]
    tau_rec = float(rec_after[0]) if len(rec_after) else float("nan")
    return {"peak_warming_K": peak_warming, "cie_permil": cie, "tau_rec_kyr": tau_rec}
