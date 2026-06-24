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
