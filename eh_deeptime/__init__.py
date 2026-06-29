"""eh_deeptime: a compact, illustrative deep-time Earth-habitability framework.

This package is a light, transparent, fully offline ILLUSTRATION accompanying the
Nature Reviews Earth & Environment Perspective: it shows that the multi-sphere
approach to Earth habitability is operationalisable in deep time. It is deliberately
NOT a research-grade model -- there is no calibration against real proxy data, no
out-of-sample validation, and no fabricated data anywhere. Every input is either a
published parameter envelope or an explicitly-labelled synthetic/illustrative field.

Modules
-------
petm          0-D carbon-cycle box model (DIC, ALK, d13C) for a PETM hindcast figure.
carbon_sulfur closed 9-state carbon-sulfur-oxygen-alkalinity-isotope box model
              (generalises petm; total C and total S conserved to machine precision).
ebm           1-D North-class diffusive energy-balance (climate) model.
habitability  guild-mixture Bayesian-logistic habitability metric, fit to SYNTHETIC
              draws from published tolerance envelopes (NOT a real growth database).
smc           tempered-SMC identical-twin parameter recovery (a sampler
              self-consistency demonstration on synthetic pseudo-data -- NOT
              calibration to proxies, NOT validation).
sensitivity   Saltelli/Sobol (with bootstrap CIs) + Shapley effects + Jensen-bias.
benchmark     structural-uncertainty benchmark: our box model vs PUBLISHED LOSCAR/
              cGENIE/iLOSCAR PETM diagnostics + the proxy consensus (cited; an
              inter-model indicator, NOT validation).
subsurface    deep subsurface-biosphere carbon box (H3) anchored to Magnabosco
              (2018, 23-31 Pg C); mechanistic shrinkage under surface warming.
deeptime_data REAL public deep-time data ingestion -- LIVE: Foster 2017 CO2
              compilation (~1200 proxies, 0-423 Ma), LR04 benthic d18O, Macrostrat
              lithology, PINT(QPI) geomagnetic dipole moment (~640 dets 0-3458 Ma),
              and PBDB GPlates palaeocoordinates; + no-fabrication guard (refuses,
              never substitutes).
extremophiles BacDive (DSMZ) per-strain cardinal-limit ingestion (credential-gated)
              + guild cardinal-range aggregator -- a real-data niche backbone.
haf           deep-time Habitable Area Fraction HAF(t) through a carbon-release
              event, driven ENTIRELY by the carbon_sulfur box model's own pCO2(t)
              and ocean pH(t) (carbon -> climate -> ocean chemistry ->
              habitability); no synthetic, analytic or proxy time-series.
plots         Nature-style vector figures for all of the above.
framework     one-call driver: `python -m eh_deeptime.framework` runs every module
              and writes the figures + a metrics.json.

The carbon modules follow the structure of LOSCAR-class long-term carbon-cycle box
models (Zeebe 2012, Geosci. Model Dev.; the iLOSCAR Python re-implementation, Liu
et al. 2024) and the COPSE/GEOCARBSULF silicate-weathering feedback (Berner 2006;
Lenton et al. 2018), reduced to the minimum needed for an illustration. Constants are
illustrative: the control runs are steady by construction, and a ~3000 Gt C release
gives a consensus-SCALE PETM peak warming (~4-5 K), carbon-isotope excursion (-2.5 to
-4.5 per mil) and ~100-200 kyr recovery. Those bands are the tuning targets the
constants were set to hit; they are NOT an independent validation against proxy data.
"""
from __future__ import annotations

__all__ = [
    "petm", "carbon_sulfur", "ebm", "habitability",
    "smc", "sensitivity", "benchmark", "subsurface", "haf",
    "deeptime_data", "extremophiles", "plots", "framework",
]
__version__ = "0.2.0"
