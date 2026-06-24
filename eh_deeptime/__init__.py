"""eh_deeptime: a compact, illustrative deep-time carbon-cycle box model.

This package provides ONE thing for the Nature Reviews Earth & Environment
Perspective: a light, transparent illustration that the multi-sphere approach is
operationalisable in deep time, via a 0-D (well-mixed) carbon-cycle response to a
PETM-scale carbon release.

It is deliberately NOT a research-grade model: there is no Bayesian calibration, no
out-of-sample validation, and no parameter inference. It runs a forward simulation at
consensus PETM parameters and produces a single illustrative figure showing the coupled
carbon -> climate -> ocean-chemistry response and its recovery.

The model follows the structure of LOSCAR-class long-term carbon-cycle box models
(Zeebe 2012, Geosci. Model Dev.; the iLOSCAR Python re-implementation, Liu et al. 2024)
and the COPSE/GEOCARBSULF silicate-weathering feedback (Berner 2006; Lenton et al. 2018),
reduced to the minimum needed for an illustration. Constants are illustrative, chosen so
the control run is steady and a ~5000 Gt C release reproduces the consensus PETM peak
warming (~5 C), carbon-isotope excursion (-2.5 to -4.5 per mil) and ~100-200 kyr recovery.
"""
from __future__ import annotations

__all__ = ["petm", "plots"]
__version__ = "0.1.0"
