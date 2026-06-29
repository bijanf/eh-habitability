"""Nature-family vector figure for the illustrative deep-time PETM run.

Mirrors the rcParams / size conventions of eh_shallow/plots.py (Springer/Nature figure
guide): vector PDF backend, sans-serif <=7 pt, RGB, 1-col = 88 mm / 2-col = 180 mm,
pdf.fonttype 42 (editable text).
"""
from __future__ import annotations

import matplotlib
matplotlib.use("pdf")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
    "font.size": 6, "axes.labelsize": 7, "axes.titlesize": 7,
    "xtick.labelsize": 6, "ytick.labelsize": 6, "legend.fontsize": 6,
    "savefig.dpi": 300, "pdf.fonttype": 42, "ps.fonttype": 42,
})
MM = 1 / 25.4
ONE_COL, TWO_COL = 88 * MM, 180 * MM

_CARBON = "#b35806"     # carbon (orange-brown)
_CLIM = "#b2182b"       # climate (red)
_ISO = "#2166ac"        # carbon isotopes (blue)
_OCEAN = "#1b7837"      # ocean chemistry (green)
_O2 = "#762a83"         # oxygen (purple)
_SULF = "#8c510a"       # sulfur (brown)
_HAB = "#35978f"        # habitability / HAF (teal)


def _panel(ax, kyr, central, lo, hi, color, ylabel, title, target=None):
    m = kyr <= 300.0
    ax.fill_between(kyr[m], lo[m], hi[m], color=color, alpha=0.20, lw=0)
    ax.plot(kyr[m], central[m], color=color, lw=1.0)
    if target is not None:
        ax.axhspan(target[0], target[1], color="0.6", alpha=0.18, lw=0, zorder=0)
    ax.axvline(0.0, color="0.5", lw=0.5, ls=":")
    ax.set_xlim(-20, 300)
    ax.set_xlabel("Time relative to onset (kyr)")
    ax.set_ylabel(ylabel)
    ax.set_title(title, loc="left", fontweight="bold")


def plot_petm_illustration(ens, path):
    """4-panel illustration of deep-time multi-sphere coupling for a PETM-scale release.

    `ens` is the dict returned by eh_deeptime.run.build_ensemble: central trajectory plus
    a low/high envelope from a consensus carbon-release band.
    """
    kyr = ens["kyr"]
    fig, axs = plt.subplots(2, 2, figsize=(TWO_COL, 95 * MM))
    _panel(axs[0, 0], kyr, ens["pco2"][1], ens["pco2"][0], ens["pco2"][2], _CARBON,
           "Atmospheric CO$_2$ (ppm)", "a  Carbon: atmospheric CO$_2$")
    _panel(axs[0, 1], kyr, ens["temp"][1], ens["temp"][0], ens["temp"][2], _CLIM,
           "Surface warming (K)", "b  Climate: surface temperature", target=(4.0, 6.0))
    _panel(axs[1, 0], kyr, ens["d13c"][1], ens["d13c"][0], ens["d13c"][2], _ISO,
           r"Surface $\delta^{13}$C (‰)", r"c  Biosphere: carbon-isotope excursion",
           target=(ens["d13c0"] - 4.5, ens["d13c0"] - 2.5))
    _panel(axs[1, 1], kyr, ens["ph"][1], ens["ph"][0], ens["ph"][2], _OCEAN,
           "Surface ocean pH", "d  Ocean: acidification")
    fig.tight_layout(pad=0.6)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


# --- framework figures (illustrative deep-time EH scaffold) -------------------
def _line(ax, x, y, color, ylabel, title, x0line=True,
          xlabel="Time relative to onset (kyr)"):
    ax.plot(x, y, color=color, lw=1.0)
    if x0line:
        ax.axvline(0.0, color="0.5", lw=0.5, ls=":")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title, loc="left", fontweight="bold")


def plot_carbon_sulfur(res, path, tmax=400.0):
    """6-panel response of the CLOSED C-S-O-alkalinity box model to a carbon pulse.

    `res` is an eh_deeptime.carbon_sulfur.run_csys output. Shows the coupled
    carbon -> climate -> isotope -> ocean-chemistry -> oxygen -> sulfur response;
    total carbon and total sulfur are conserved to machine precision (illustration).
    """
    kyr = res["kyr"]
    m = kyr <= tmax
    fig, axs = plt.subplots(2, 3, figsize=(TWO_COL, 95 * MM))
    _line(axs[0, 0], kyr[m], res["pco2"][m], _CARBON,
          "Atmospheric CO$_2$ (ppm)", "a  Carbon")
    _line(axs[0, 1], kyr[m], res["temp"][m], _CLIM,
          "Surface warming (K)", "b  Climate")
    _line(axs[0, 2], kyr[m], res["d13c"][m], _ISO,
          r"DIC $\delta^{13}$C (‰)", "c  Carbon isotopes")
    _line(axs[1, 0], kyr[m], res["ph"][m], _OCEAN,
          "Surface ocean pH", "d  Ocean acidification")
    _line(axs[1, 1], kyr[m], res["o2"][m] * 100.0, _O2,
          "O$_2$ (% of present)", "e  Oxygen")
    _line(axs[1, 2], kyr[m], (res["so4"][m] - res["so4"][0]) / 1e15, _SULF,
          r"$\Delta$ ocean SO$_4$ (10$^{15}$ mol)", "f  Sulfur")
    fig.tight_layout(pad=0.6)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_ebm(sols, sweep, path):
    """1-D EBM climate field: T(latitude) at several CO2 + global mean vs CO2.

    `sols` = list of dicts {co2, lat, T_C}; `sweep` = dict {co2: array, t_global: array}.
    """
    fig, axs = plt.subplots(1, 2, figsize=(TWO_COL, 70 * MM))
    n = max(len(sols) - 1, 1)
    for i, s in enumerate(sols):
        axs[0].plot(s["lat"], s["T_C"], lw=1.0, color=plt.cm.viridis(i / n),
                    label=f"{int(s['co2'])} ppm")
    axs[0].axhline(0.0, color="0.6", lw=0.5, ls=":")
    axs[0].set_xlabel("Latitude (°)")
    axs[0].set_ylabel("Temperature (°C)")
    axs[0].set_title("a  Zonal-mean climate", loc="left", fontweight="bold")
    axs[0].legend(fontsize=5, frameon=False)
    axs[1].semilogx(sweep["co2"], sweep["t_global"], "o-", color=_CLIM, lw=1.0, ms=3)
    axs[1].set_xlabel("Atmospheric CO$_2$ (ppm)")
    axs[1].set_ylabel("Global-mean T (°C)")
    axs[1].set_title("b  Climate response", loc="left", fontweight="bold")
    fig.tight_layout(pad=0.6)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_habitability(grid, slices, path):
    """Guild-mixture habitability: P_hab over (T, pH) + 1-D guild-niche slices.

    `grid` = {T, pH, P (2-D), a_w}; `slices` = {T, per_guild{name:p}, mixture, pH, a_w}.
    """
    fig, axs = plt.subplots(1, 2, figsize=(TWO_COL, 70 * MM))
    pcm = axs[0].pcolormesh(grid["T"], grid["pH"], grid["P"], cmap="YlGnBu",
                            vmin=0.0, vmax=1.0, shading="auto")
    cb = fig.colorbar(pcm, ax=axs[0])
    cb.set_label("$P_{hab}$")
    axs[0].set_xlabel("Temperature (°C)")
    axs[0].set_ylabel("pH")
    axs[0].set_title(f"a  Mixture habitability ($a_w$={grid['a_w']:.2f})",
                     loc="left", fontweight="bold")
    for name, p in slices["per_guild"].items():
        axs[1].plot(slices["T"], p, lw=0.8, label=name)
    axs[1].plot(slices["T"], slices["mixture"], color="k", lw=1.4, label="mixture")
    axs[1].set_xlabel("Temperature (°C)")
    axs[1].set_ylabel("$P_{hab}$")
    axs[1].set_title(f"b  Guild niches (pH={slices['pH']:.1f}, "
                     f"$a_w$={slices['a_w']:.2f})", loc="left", fontweight="bold")
    axs[1].legend(fontsize=4.5, frameon=False)
    fig.tight_layout(pad=0.6)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_smc(post, path):
    """Identical-twin SMC: weighted posterior marginals with the known truth marked."""
    theta, w, names, truth = (post["theta"], post["weights"],
                              post["names"], post["truth"])
    d = len(names)
    fig, axs = plt.subplots(1, d, figsize=(TWO_COL, 55 * MM))
    axs = np.atleast_1d(axs)
    for j, ax in enumerate(axs):
        x = theta[:, j]
        ax.hist(x, bins=25, weights=w, color="#4393c3", alpha=0.85, density=True)
        ax.axvline(truth[j], color=_CLIM, lw=1.3, label="truth")
        ax.axvline(float(np.sum(w * x)), color="k", lw=0.8, ls="--",
                   label="post. mean")
        ax.set_xlabel(names[j])
        ax.set_title("abcdef"[j], loc="left", fontweight="bold")
    axs[0].set_ylabel("posterior density")
    axs[0].legend(fontsize=5, frameon=False)
    fig.tight_layout(pad=0.6)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


def _ci_err(point, ci):
    """Convert a (d,2) [p05,p95] CI to matplotlib yerr (2,d) [lower, upper]."""
    point = np.asarray(point)
    ci = np.asarray(ci)
    lo = np.clip(point - ci[:, 0], 0, None)
    hi = np.clip(ci[:, 1] - point, 0, None)
    return np.vstack([lo, hi])


def plot_sensitivity(sobol, jb, path, shap=None):
    """Sobol S1/ST (with bootstrap CIs if present) + optional Shapley effects + Jensen.

    `sobol` may carry 'S1_ci'/'ST_ci' (drawn as error bars). If `shap` is given
    (a sensitivity.shapley_effects output) a Shapley-effects panel is added.
    """
    ncol = 3 if shap is not None else 2
    fig, axs = plt.subplots(1, ncol, figsize=(TWO_COL, 65 * MM))
    names = sobol["names"]
    xpos = np.arange(len(names))
    width = 0.38
    e1 = _ci_err(sobol["S1"], sobol["S1_ci"]) if "S1_ci" in sobol else None
    eT = _ci_err(sobol["ST"], sobol["ST_ci"]) if "ST_ci" in sobol else None
    axs[0].bar(xpos - width / 2, sobol["S1"], width, yerr=e1, capsize=2,
               error_kw={"lw": 0.6}, label="$S_1$ (first-order)", color="#4393c3")
    axs[0].bar(xpos + width / 2, sobol["ST"], width, yerr=eT, capsize=2,
               error_kw={"lw": 0.6}, label="$S_T$ (total)", color="#b2182b")
    axs[0].set_xticks(xpos)
    axs[0].set_xticklabels(names, rotation=30, ha="right")
    axs[0].set_ylabel("Sobol index")
    axs[0].set_title("a  Sobol sensitivity (peak warming)", loc="left", fontweight="bold")
    axs[0].legend(fontsize=5, frameon=False)

    if shap is not None:
        sh = np.asarray(shap["shapley"])
        axs[1].bar(np.arange(len(shap["names"])), sh, color="#5aae61")
        axs[1].set_xticks(np.arange(len(shap["names"])))
        axs[1].set_xticklabels(shap["names"], rotation=30, ha="right")
        axs[1].set_ylabel("Shapley effect")
        axs[1].set_title(f"b  Shapley effects ($\\Sigma$={np.nansum(sh):.2f})",
                         loc="left", fontweight="bold")

    axj = axs[ncol - 1]
    axj.axhline(0.0, color="0.6", lw=0.5)
    axj.semilogx(jb["co2"], jb["delta_J"], "o-", color=_OCEAN, lw=1.0, ms=3)
    axj.set_xlabel("Atmospheric CO$_2$ (ppm)")
    axj.set_ylabel(r"$\delta_J = f_{hab} - P_{hab}(\bar{x})$")
    axj.set_title(f"{'c' if shap is not None else 'b'}  Aggregation (Jensen) bias",
                  loc="left", fontweight="bold")
    fig.tight_layout(pad=0.6)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_coupled_haf(res, path, tmax=300.0):
    """HAF through a carbon-release event, driven by the closed C-S-O box model.

    Every panel is a forward-model time-series on a common time axis (NO synthetic
    or proxy series): the box model's atmospheric CO2(t), surface warming(t) and
    surface-ocean pH(t), and the habitable area fraction HAF(t) they imply via the
    1-D EBM + guild-mixture metric at fixed open-marine water activity.

    `res` is an eh_deeptime.haf.coupled_event_haf output.
    """
    kyr = res["kyr"]
    m = kyr <= tmax
    fig, axs = plt.subplots(2, 2, figsize=(TWO_COL, 95 * MM))
    _line(axs[0, 0], kyr[m], res["co2"][m], _CARBON,
          "Atmospheric CO$_2$ (ppm)", "a  Carbon (model)")
    _line(axs[0, 1], kyr[m], res["temp_anom_K"][m], _CLIM,
          "Surface warming (K)", "b  Climate (model)")
    _line(axs[1, 0], kyr[m], res["ph_ocean"][m], _OCEAN,
          "Surface ocean pH", "c  Ocean acidification (model)")
    _line(axs[1, 1], kyr[m], res["haf"][m], _HAB,
          "Habitable area fraction", "d  Habitability (HAF)")
    fig.tight_layout(pad=0.6)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_subsurface(resp, h3, path):
    """Subsurface-biosphere carbon (H3): present-day anchor + warming response.

    `resp` = subsurface.warming_response output; `h3` = subsurface.h3_consistency.
    Left: stock vs surface warming with the Magnabosco (2018) 23-31 Pg C band
    shaded (the anchored present-day value). Right: habitable depth vs warming.
    The absolute scale is anchored to the literature; the response is model output.
    """
    fig, axs = plt.subplots(1, 2, figsize=(TWO_COL, 62 * MM))
    lo, hi = h3["magnabosco2018_PgC"]
    axs[0].axhspan(lo, hi, color="#5aae61", alpha=0.18, lw=0,
                   label="Magnabosco 2018 (23–31 Pg C)")
    axs[0].plot(resp["delta_T_C"], resp["stock_PgC"], color=_HAB, lw=1.2)
    axs[0].set_xlabel("Surface warming (K)")
    axs[0].set_ylabel("Habitable subsurface C (Pg C)")
    axs[0].set_title("a  Subsurface carbon (anchored)", loc="left", fontweight="bold")
    axs[0].legend(fontsize=5, frameon=False, loc="upper right")
    axs[1].plot(resp["delta_T_C"], resp["habitable_depth_km"], color=_CLIM, lw=1.2)
    axs[1].set_xlabel("Surface warming (K)")
    axs[1].set_ylabel("Habitable depth to 122 °C (km)")
    axs[1].set_title("b  Habitable depth window", loc="left", fontweight="bold")
    fig.tight_layout(pad=0.6)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_structural_benchmark(comp, path):
    """Forest plot: our box model vs PUBLISHED community-model + proxy diagnostics.

    `comp` is an eh_deeptime.benchmark.structural_comparison output. Three panels
    (peak warming, CIE, recovery); each published study is a point or range
    (blue = model, green = proxy/observational), our box model is a red line.
    This is an inter-model structural-spread INDICATOR + proxy plausibility check,
    NOT validation: LOSCAR/cGENIE/iLOSCAR are models, not data.
    """
    diags = [("peak_warming_K", "Peak warming (K)", "a  Climate"),
             ("cie_permil", r"CIE ($\delta^{13}$C, ‰)", "b  Carbon isotopes"),
             ("recovery_kyr", "Recovery time (kyr)", "c  Recovery")]
    fig, axs = plt.subplots(1, 3, figsize=(TWO_COL, 70 * MM))
    for ax, (field, xlabel, title) in zip(axs, diags):
        rows = [r for r in comp["published"] if r.get(field) is not None]
        y = np.arange(len(rows))
        for i, r in enumerate(rows):
            lo, hi = r[field]
            col = _O2 if r["kind"] == "model" else _HAB
            if hi > lo:
                ax.plot([lo, hi], [i, i], color=col, lw=2.0, alpha=0.8,
                        solid_capstyle="butt")
            ax.plot([(lo + hi) / 2], [i], "o", color=col, ms=4)
        ax.set_yticks(y)
        ax.set_yticklabels([r["key"] for r in rows], fontsize=4.5)
        v = comp["by_diag"][field]["ours"]
        if v is not None:
            ax.axvline(v, color=_CLIM, lw=1.2, ls="--",
                       label=f"this box model ({v:.1f})")
            ax.legend(fontsize=4.5, frameon=False, loc="lower right")
        ax.set_xlabel(xlabel)
        ax.set_title(title, loc="left", fontweight="bold")
        ax.margins(y=0.15)
    fig.tight_layout(pad=0.6)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_deeptime_combined(csys, hafres, path, tmax=300.0):
    """Combined deep-time illustration for the Perspective.

    Panels a-e: the CLOSED C-S-O-alkalinity-isotope box model's response to a
    PETM-scale carbon pulse, showing one perturbation cascading carbon ->
    climate -> isotopes -> ocean chemistry -> oxygen + sulfur in a system that
    conserves total C and total S to machine precision. Panel f: the Habitable
    Area Fraction HAF(t) implied by the coupled chain, on the SAME time axis --
    the box model's pCO2(t) and ocean pH(t) drive the EBM + guild-mixture metric.

    `csys` = carbon_sulfur.run_csys output; `hafres` = haf.coupled_event_haf
    output computed from the SAME csys. Every panel is forward-model output; no
    synthetic, analytic or proxy time-series appears anywhere in this figure.
    This is an ILLUSTRATION of operationalisability, not a reconstruction.
    """
    kyr = csys["kyr"]
    m = kyr <= tmax
    fig, axs = plt.subplots(2, 3, figsize=(TWO_COL, 95 * MM))
    _line(axs[0, 0], kyr[m], csys["pco2"][m], _CARBON,
          "Atmospheric CO$_2$ (ppm)", "a  Carbon")
    _line(axs[0, 1], kyr[m], csys["temp"][m], _CLIM,
          "Surface warming (K)", "b  Climate")
    _line(axs[0, 2], kyr[m], csys["d13c"][m], _ISO,
          r"DIC $\delta^{13}$C (‰)", "c  Carbon isotopes")
    _line(axs[1, 0], kyr[m], csys["ph"][m], _OCEAN,
          "Surface ocean pH", "d  Ocean acidification")
    # panel e: the two closed-system reservoirs that make this a C-S-O cycle
    axe = axs[1, 1]
    axe.plot(kyr[m], csys["o2"][m] * 100.0, color=_O2, lw=1.0)
    axe.axvline(0.0, color="0.5", lw=0.5, ls=":")
    axe.set_xlabel("Time relative to onset (kyr)")
    axe.set_ylabel("O$_2$ (% of present)", color=_O2)
    axe.tick_params(axis="y", labelcolor=_O2)
    axe.set_title("e  Closed O$_2$ + S reservoirs", loc="left", fontweight="bold")
    axe2 = axe.twinx()
    axe2.plot(kyr[m], (csys["so4"][m] - csys["so4"][0]) / 1e15, color=_SULF, lw=1.0)
    axe2.set_ylabel(r"$\Delta$ ocean SO$_4$ (10$^{15}$ mol)", color=_SULF)
    axe2.tick_params(axis="y", labelcolor=_SULF)
    # panel f: HAF(t) on the same time axis as a-e (the coupled chain's output)
    mh = hafres["kyr"] <= tmax
    axf = axs[1, 2]
    axf.plot(hafres["kyr"][mh], hafres["haf"][mh], color=_HAB, lw=1.2)
    axf.axvline(0.0, color="0.5", lw=0.5, ls=":")
    axf.set_xlim(-20, tmax)
    axf.set_xlabel("Time relative to onset (kyr)")
    axf.set_ylabel("Habitable area fraction")
    axf.set_title("f  Habitability (HAF)", loc="left", fontweight="bold")
    fig.tight_layout(pad=0.6)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path
