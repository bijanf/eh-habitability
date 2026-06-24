# eh_deeptime — illustrative deep-time carbon-cycle box model

A **light, transparent illustration** for the Nature Reviews Earth & Environment Perspective
*"Deep time, shallow time, and the multi-sphere model of Earth habitability."* It shows that the
multi-sphere approach is operationalisable in deep time, via a 0-D (well-mixed) carbon-cycle response
to a PETM-scale carbon release.

**What it is / is not.** This is an *illustration*, not a research-grade model: there is **no**
Bayesian calibration, **no** out-of-sample validation, and **no** parameter inference. It runs a
forward simulation at consensus PETM parameters (with a carbon-release band) and produces one figure.

It follows the structure of LOSCAR-class long-term carbon-cycle box models (Zeebe 2012; the iLOSCAR
Python re-implementation, Liu et al. 2024) and the COPSE/GEOCARBSULF silicate-weathering feedback
(Berner 2006; Lenton et al. 2018), reduced to the minimum needed for an illustration. Flux constants
are illustrative: chosen so the no-injection control is in exact steady state, and a ~3000 Gt C release
reproduces the consensus PETM peak warming (~4–5 K), carbon-isotope excursion (−2.5…−4.5 ‰) and
~100–200 kyr recovery, with attendant ocean acidification.

## Run

```bash
python -m eh_deeptime.run            # writes eh_deeptime/out/petm_illustration.pdf (+ .json)
python tests/test_eh_deeptime.py     # offline smoke tests (also: pytest tests/test_eh_deeptime.py)
```

Runs fully offline (numpy + scipy + matplotlib only). The figure is the deep-time panel of the
Perspective (Fig. 3): coupled carbon → climate → biosphere (δ¹³C) → ocean-chemistry response.

## Files
- `petm.py` — the box model (`run_petm`, `summarise`) and carbonate-chemistry solver.
- `plots.py` — the Nature-style 4-panel figure.
- `run.py` — builds the consensus ensemble and writes the figure + metadata.
