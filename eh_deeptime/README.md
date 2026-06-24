# eh_deeptime — illustrative deep-time Earth-habitability framework

A **light, transparent, fully offline** illustration accompanying the Nature Reviews Earth &
Environment Perspective. It demonstrates that the multi-sphere approach to Earth habitability is
*operationalisable in deep time* — coupling a closed biogeochemical cycle, a climate model, and a
probabilistic habitability metric into a deep-time Habitable Area Fraction.

> **What it is / is not.** This is an **illustration / methods demonstration**, *not* a research-grade
> model. There is **no calibration to real proxy data, no out-of-sample validation, and no fabricated
> datasets**. Every input is either a published parameter envelope or an explicitly-labelled
> synthetic / illustrative field. Each module carries an honest scope statement and a `PRODUCTION SWAP`
> note describing what a real implementation would add (Bayesian calibration against compiled proxies,
> 2-D geography, PyMC/Stan or JAX, sulfur/oxygen isotopes, etc.).

## Modules

| Module | What it does |
|---|---|
| `petm.py` | 0-D carbon-cycle box model (DIC, ALK, δ¹³C) → the PETM hindcast figure (Fig. of the Perspective). |
| `carbon_sulfur.py` | **Closed 9-state** carbon–sulfur–oxygen–alkalinity–isotope box model (generalises `petm`). Total C **and** total S conserved to machine precision; control steady by construction. Directly answers the referee point that the carbon core was "a single ODE … no closed C–S–O system". |
| `ebm.py` | 1-D North-class diffusive energy-balance climate model: `T(latitude)`, global mean, ice line. |
| `habitability.py` | Guild-mixture Bayesian-logistic habitability metric, fit by penalised IRLS to **synthetic draws from published tolerance envelopes** (four guilds). `P_hab(x) = max_g P_hab^(g)(x)`. |
| `smc.py` | Tempered Sequential Monte Carlo **identical-twin parameter recovery** — recovers a known truth from synthetic pseudo-data (sampler self-consistency check, *not* calibration). |
| `sensitivity.py` | Saltelli/Sobol variance-based sensitivity (low-discrepancy Sobol sequence) + Jensen-bias spatial-aggregation analysis. |
| `haf.py` | Illustrative **deep-time Habitable Area Fraction**: synthetic CO₂ forcing → climate → habitability over latitude. |
| `plots.py` | Nature-style vector figures (88/180 mm, ≤7 pt sans-serif, `pdf.fonttype 42`). |
| `framework.py` | One-call driver that runs every module and writes the figures + `framework_metrics.json`. |

## Run

```bash
python -m eh_deeptime.run                  # the PETM illustration figure (Fig. 3)
python -m eh_deeptime.framework            # full framework: 6 figures + metrics.json -> out/
python -m eh_deeptime.framework --quick    # faster smoke run (smaller SMC/Sobol samples)
python -m pytest -q tests/                 # offline test suite (49 sanity/plausibility checks)
```

Everything runs offline with numpy + scipy + matplotlib only. `framework.py` writes, into `out/`:
`csys_response.pdf` (closed C–S–O response), `ebm_climate.pdf`, `habitability.pdf`, `smc_recovery.pdf`,
`sensitivity.pdf`, `deeptime_haf.pdf`, plus `framework_metrics.json`.

## Honesty checks baked into the tests

The tests are **sanity / plausibility checks, not validation gates**: mass conservation (C and S to
~1e-13), control steadiness, monotone responses, SMC bracketing of the planted truth, Sobol `Sₜ ≥ S₁`,
unit-interval probabilities, and self-consistency with the tuning targets the constants were set to hit
(explicitly *not* an independent comparison with proxy data).
