# Pre-registration & freeze — Earth-habitability framework (illustrative)

**Status: ILLUSTRATIVE METHODS PRE-REGISTRATION.** This document freezes the
*honest current state* of the `eh-habitability` code and pre-registers the
falsifiable hypotheses and rejection criteria for a *future* research-grade run.
It is **not** a claim that calibrated, validated deep-time results exist now — the
deep-time code uses synthetic / published-envelope inputs only, with no calibration
to real proxy data and no out-of-sample validation. Freeze it on Zenodo/OSF with
that label.

## 1. What is frozen
- **Code:** a tagged release of `eh_shallow/` + `eh_deeptime/` + `tests/`
  (archive the GitHub release to a versioned Zenodo DOI).
- **Config:** module default parameters (`carbon_sulfur.DEFAULT_PARAMS`,
  `ebm.DEFAULT_PARAMS`, guild boxes in `habitability.GUILDS`, sensitivity ranges),
  fixed RNG seeds, and the environment lockfile (`requirements-lock.txt`).
- **"Posterior":** the only ensemble that exists is the tempered-SMC **identical-
  twin recovery on SYNTHETIC pseudo-data** (`eh_deeptime/smc.py`). It is frozen
  **explicitly labelled as a sampler self-consistency demonstration, not a
  calibration to observations.** No real-proxy posterior exists to freeze yet.

## 2. Pre-registered hypotheses & rejection criteria (for the future real run)
Registered *before* any calibration to withheld proxy data. Rejection criteria are
fixed here so they cannot be tuned post hoc.

- **H1 — Carbon-cycle two-stage recovery.** A PETM-scale carbon release produces a
  *fast* (carbonate-compensation) and a *slow* (silicate-weathering) recovery.
  Pre-registered intervals: CIE recovery **30–120 kyr**; atmospheric-CO2 recovery
  **100–250 kyr**. *Reject* if the frozen model's prior-predictive 90% interval
  does not bracket the proxy-derived estimates.
  Caveat: in the present illustrative code the recovery timescale is set by the
  tuned silicate-weathering exponent, so H1 is a *consistency* check until the
  weathering law is calibrated to data.

- **H3 — Deep subsurface biosphere.** Present-day continental habitable subsurface
  carbon **20 ± 10 Pg C** (proposal target), consistent with Magnabosco et al.
  (2018) 23–31 Pg C. The box (`eh_deeptime/subsurface.py`) carries a stock
  *anchored* to that literature value; the registered, non-circular prediction is
  the **sign and approximate magnitude of its response to surface warming**
  (shoaling 122 °C isotherm → shrinking habitable depth window).

- **H2 — Geomagnetic → UV-B — NOT pre-registered / WITHDRAWN.** Prior expert review
  found the proposed UV-B magnitude (~28%) is ~10× over the photochemical
  ozone-column ceiling, magnetic fields do not shield UV-B photons, the coupling is
  absent from the model equations, and the OSL "paleo-UV-B" proxy does not measure
  UV-B. **No UV-B magnitude is registered or claimed.** Any future work must first
  implement a photochemically-bounded mechanism and find an instrument-grade proxy.

## 3. Pre-registered out-of-sample validation plan (future)
- Pleistocene hindcast vs LR04 + ice-core CO2 (Spearman > 0.8) — report the
  achieved value verbatim, pass or fail.
- PETM warming/CIE vs the proxy consensus (warming 5 ± 1 °C; CIE −2.5 to −4.5 ‰) —
  as a likelihood-based test once a real training set exists; until then this is a
  labelled plausibility comparison, **not** an "out-of-sample / K>3" claim.
- Structural-uncertainty benchmark vs published LOSCAR/cGENIE/iLOSCAR diagnostics
  (`eh_deeptime/benchmark.py`) — an inter-model indicator, never validation.

## 4. Integrity commitments
- No fabricated time-series or data, ever. Synthetic inputs are labelled synthetic;
  real public data carry source DOIs + provenance; model outputs are labelled model
  outputs; literature anchors are cited.
- The Zenodo/OSF freeze records the commit hash and timestamp; results reported
  later must reference this frozen artifact.
