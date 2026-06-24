# eh-habitability

Illustrative code accompanying the Perspective **"A multi-sphere perspective on Earth
habitability across deep time and the Anthropocene"** (M. Rostami, B. Fallah, L.-Y. Fu).

The repository contains the two compact, transparent prototypes referenced in the paper.
They are **illustrative proofs-of-concept** that demonstrate the multi-sphere approach can be
operationalised — they are **not** calibrated reconstructions or operational forecasts.

## Modules

- **`eh_shallow/`** — shallow-time (1750–2300) prototype: a reduced-complexity Earth-system
  emulator and the Composite Hazard Score (CHS) → Habitable Area Fraction (HAF) metric, with a
  Random-Forest weighting trained on a Water Hazard Index target, Bayesian SMC calibration, and
  consistency checks (human-climate-niche, crop-yield, pattern stationarity, structural error).
- **`eh_deeptime/`** — deep-time prototype: a 0-D (well-mixed) carbon-cycle box model of the
  LOSCAR/COPSE class, used for the PETM carbon → climate → ocean-chemistry illustration.

## Install & run

```bash
pip install -r eh_shallow/requirements.txt        # numpy, scipy, matplotlib, scikit-learn, ...
python -m eh_deeptime.run                          # writes the deep-time PETM figure
python -m eh_shallow.niche                          # HAF vs human climate niche
python -m eh_shallow.structural                     # leave-one-reconstruction-out structural error
pytest tests/                                        # offline smoke tests
```

Python ≥ 3.10. The public input datasets (HadCRUT5, IPCC AR6 ERF, NASA GISTEMP, Berkeley Earth,
CRU, NOAA/NCEI ocean heat content, GDHY crop yields) are downloaded and cached automatically on
first run; afterwards the code runs fully offline.

## Data availability

All input datasets used by the illustrative figures are openly available and fetched
automatically (see above). The **raw, high-resolution Water Hazard Index (WHI) rasters and their
predictor stack** used by `eh_shallow/whi.py` are from a separate, as-yet **unpublished** aquifer
study and are **not** distributed here; they are available from the corresponding author on
reasonable request. Point `eh_shallow/whi.py` at local copies via the `EH_WHI_PATH` and
`EH_WHI_PRED_DIR` environment variables. Every other figure is reproducible from the public data.

## License

Code is released under the [MIT License](LICENSE). An archived release will be deposited to
Zenodo with a citable DOI on publication.
