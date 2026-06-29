"""Structural-uncertainty benchmark: our box model vs PUBLISHED community models.

This compares the illustrative closed C-S-O box model
(:func:`eh_deeptime.carbon_sulfur.run_csys`) against PETM diagnostics REPORTED IN
THE LITERATURE by structurally different community carbon-cycle / Earth-system
models (LOSCAR, cGENIE, iLOSCAR) and against the proxy/observational consensus.

WHAT THIS IS / IS NOT.
  * It is an honest *model-vs-model structural-spread INDICATOR*: how far apart do
    independent published models land on the same PETM diagnostics, and does our
    reduced box model fall inside that envelope? That directly addresses the
    "structural uncertainty / no external benchmark" gap.
  * It is NOT validation. LOSCAR/cGENIE/iLOSCAR are MODELS, not data, so agreement
    with them is inter-comparison, never validation. The one genuinely
    observational row (McInerney & Wing 2011) is the proxy CONSENSUS band; our box
    model is illustrative and tuned, so landing in that band is a plausibility /
    consistency check, not a fit.
  * Every number below is transcribed from the cited paper with its DOI and only
    where the study explicitly reports it; gaps are ``None`` ("not reported"), never
    guessed.

Sources (DOIs Crossref-verified):
  - Zeebe, Zachos & Dickens 2009, Nat. Geosci. 2, 576 (LOSCAR)           10.1038/ngeo578
  - Gutjahr et al. 2017, Nature 548, 573 (cGENIE + boron inversion)      10.1038/nature23646
  - Penman et al. 2014, Paleoceanography 29, 357 (delta-11B PROXY)       10.1002/2014PA002621
  - Ridgwell & Schmidt 2010, Nat. Geosci. 3, 196 (cGENIE)               10.1038/ngeo755
  - Li, Zeebe & Zhang 2024, Glob. Planet. Change 236, 104413 (iLOSCAR)  10.1016/j.gloplacha.2024.104413
  - McInerney & Wing 2011, Annu. Rev. Earth Planet. Sci. 39, 489
    (OBSERVATIONAL / PROXY CONSENSUS, not a model)                       10.1146/annurev-earth-040610-133431
"""
from __future__ import annotations

import numpy as np

from . import carbon_sulfur

# Each record: explicitly-reported PETM diagnostics. None = not reported by that
# study (never inferred). kind: 'model' (a carbon-cycle/ESM run) or 'proxy'
# (observational consensus). peak_warming_K, cie_permil are the comparable scalars;
# ranges are stored as (lo, hi); point values as (v, v).
PUBLISHED = [
    {"key": "Zeebe2009", "model": "LOSCAR", "kind": "model",
     "peak_warming_K": (1.0, 3.5),       # CO2-alone forcing; "insufficient" vs proxy
     "cie_permil": None,                  # tuned to match, not a reported diagnostic
     "carbon_GtC": (3000.0, 4500.0),      # ~3000 pulse + ~1480 bleed
     "recovery_kyr": None,
     "doi": "10.1038/ngeo578"},
    {"key": "Gutjahr2017", "model": "cGENIE", "kind": "model",
     "peak_warming_K": (3.6, 3.6),        # modelled annual-mean SST rise
     "cie_permil": (-3.4, -3.4),          # benthic, Site 401
     "carbon_GtC": (5700.0, 20000.0),     # best ~10200
     "recovery_kyr": None,
     "doi": "10.1038/nature23646"},
    {"key": "Ridgwell2010", "model": "cGENIE", "kind": "model",
     "peak_warming_K": (5.0, 6.0),        # surface ocean, study context
     "cie_permil": None,
     "carbon_GtC": None,                  # not extractable / not verified -> None
     "recovery_kyr": None,
     "doi": "10.1038/ngeo755"},
    {"key": "Li2024", "model": "iLOSCAR", "kind": "model",
     "peak_warming_K": None,              # LOSCAR has no full climate module
     "cie_permil": None,
     "carbon_GtC": (3000.0, 3000.0),      # reproduces Zeebe 2009 forward example
     "recovery_kyr": None,
     "doi": "10.1016/j.gloplacha.2024.104413"},
    {"key": "Penman2014", "model": "delta-11B proxy (+LOSCAR)", "kind": "proxy",
     "peak_warming_K": (5.0, 5.0),        # Mg/Ca, Site 1209
     "cie_permil": None,
     "carbon_GtC": (3000.0, 9000.0),      # quoted range, not an independent fit
     "recovery_kyr": None,
     "doi": "10.1002/2014PA002621"},
    {"key": "McInerneyWing2011", "model": "proxy consensus", "kind": "proxy",
     "peak_warming_K": (5.0, 8.0),        # global
     "cie_permil": (-4.7, -2.8),          # terrestrial .. marine compilation means
     "carbon_GtC": (1200.0, 10000.0),     # source-dependent mass-balance
     "recovery_kyr": (83.0, 113.0),       # CIE body ~113 kyr; recovery ~83 kyr
     "doi": "10.1146/annurev-earth-040610-133431"},
]

_DIAGS = ("peak_warming_K", "cie_permil", "recovery_kyr")


def _spread(records, field):
    """(lo, hi) envelope of a field across records that report it, else None."""
    vals = []
    for r in records:
        v = r.get(field)
        if v is not None:
            vals.extend([v[0], v[1]])
    if not vals:
        return None
    return (float(min(vals)), float(max(vals)))


def our_model(m_inj=3000.0, t_dur=5.0, delta_inj=-50.0, t_end=400.0):
    """Run our box model and return its comparable PETM diagnostics."""
    res = carbon_sulfur.run_csys(m_inj=m_inj, t_dur=t_dur, delta_inj=delta_inj,
                                 t_end=t_end)
    s = carbon_sulfur.summarise(res)
    return {"peak_warming_K": float(s["peak_warming_K"]),
            "cie_permil": float(s["cie_permil"]),
            "recovery_kyr": float(s["tau_rec_kyr"]),
            "carbon_GtC": float(m_inj)}


def structural_comparison(ours=None, m_inj=3000.0):
    """Compare our box model to the published model envelope + proxy consensus.

    Returns a dict with our diagnostics, the published per-study values, the
    MODEL-only spread (structural-uncertainty indicator), the PROXY-consensus band,
    and, per diagnostic, whether our value lands inside each. This is an inter-model
    consistency indicator, NOT a validation.
    """
    if ours is None:
        ours = our_model(m_inj=m_inj)
    models = [r for r in PUBLISHED if r["kind"] == "model"]
    proxies = [r for r in PUBLISHED if r["kind"] == "proxy"]

    out = {"ours": ours, "published": PUBLISHED, "by_diag": {}}
    for f in _DIAGS:
        model_env = _spread(models, f)
        proxy_env = _spread(proxies, f)
        v = ours.get(f)
        within_model = (model_env is not None and v is not None
                        and model_env[0] <= v <= model_env[1])
        within_proxy = (proxy_env is not None and v is not None
                        and proxy_env[0] <= v <= proxy_env[1])
        out["by_diag"][f] = {
            "ours": v,
            "model_spread": model_env,
            "proxy_consensus": proxy_env,
            "model_spread_width": (None if model_env is None
                                   else float(model_env[1] - model_env[0])),
            "within_model_spread": bool(within_model),
            "within_proxy_consensus": bool(within_proxy),
        }
    return out


def summarise(comp=None):
    """Compact, JSON-friendly summary of the structural comparison."""
    if comp is None:
        comp = structural_comparison()
    s = {"ours": comp["ours"], "n_published_models":
         sum(1 for r in PUBLISHED if r["kind"] == "model"),
         "diagnostics": {}}
    for f, d in comp["by_diag"].items():
        s["diagnostics"][f] = {
            "ours": d["ours"],
            "published_model_spread": d["model_spread"],
            "proxy_consensus": d["proxy_consensus"],
            "within_model_spread": d["within_model_spread"],
            "within_proxy_consensus": d["within_proxy_consensus"],
        }
    return s
