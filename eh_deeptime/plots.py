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


def plot_sensitivity(sobol, jb, path):
    """Sobol first/total indices (peak warming) + Jensen aggregation bias vs CO2."""
    fig, axs = plt.subplots(1, 2, figsize=(TWO_COL, 65 * MM))
    names = sobol["names"]
    xpos = np.arange(len(names))
    width = 0.38
    axs[0].bar(xpos - width / 2, sobol["S1"], width, label="$S_1$ (first-order)",
               color="#4393c3")
    axs[0].bar(xpos + width / 2, sobol["ST"], width, label="$S_T$ (total)",
               color="#b2182b")
    axs[0].set_xticks(xpos)
    axs[0].set_xticklabels(names, rotation=30, ha="right")
    axs[0].set_ylabel("Sobol index")
    axs[0].set_title("a  Sensitivity of peak warming", loc="left", fontweight="bold")
    axs[0].legend(fontsize=5, frameon=False)
    axs[1].axhline(0.0, color="0.6", lw=0.5)
    axs[1].semilogx(jb["co2"], jb["delta_J"], "o-", color=_OCEAN, lw=1.0, ms=3)
    axs[1].set_xlabel("Atmospheric CO$_2$ (ppm)")
    axs[1].set_ylabel(r"$\delta_J = f_{hab} - P_{hab}(\bar{x})$")
    axs[1].set_title("b  Aggregation (Jensen) bias", loc="left", fontweight="bold")
    fig.tight_layout(pad=0.6)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_deeptime_haf(res, path):
    """Illustrative deep-time HAF: synthetic CO2 forcing -> climate -> habitability."""
    t = res["t_myr"]
    fig, axs = plt.subplots(3, 1, figsize=(ONE_COL * 1.7, 110 * MM), sharex=True)
    axs[0].semilogy(t, res["co2"], color=_CARBON, lw=1.0)
    axs[0].set_ylabel("CO$_2$ (ppm)")
    axs[0].set_title("a  Forcing (synthetic, illustrative)", loc="left",
                     fontweight="bold")
    axs[1].plot(t, res["t_global_C"], color=_CLIM, lw=1.0)
    axs[1].set_ylabel("Global T (°C)")
    axs[1].set_title("b  Climate", loc="left", fontweight="bold")
    axs[2].plot(t, res["haf"], color=_OCEAN, lw=1.0)
    axs[2].set_ylabel("HAF")
    axs[2].set_xlabel("Model time (Myr, illustrative)")
    axs[2].set_title("c  Habitable area fraction", loc="left", fontweight="bold")
    fig.tight_layout(pad=0.6)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path
