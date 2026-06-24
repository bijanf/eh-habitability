"""Produce the illustrative deep-time PETM figure for the NREE Perspective.

    python -m eh_deeptime.run [--out DIR]

Runs the 0-D carbon-cycle box model at consensus-central PETM parameters plus a
carbon-release band (2500-3500 Gt C), and writes a single vector PDF showing the coupled
carbon -> climate -> biosphere(d13C) -> ocean-chemistry response. No calibration, no
inference: this is an illustration that the multi-sphere approach is operationalisable in
deep time, not a validated reconstruction.
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np

from . import petm, plots

# consensus-central illustration + carbon-release band (Gt C)
M_CENTRAL = 3000.0
M_BAND = (2500.0, 2750.0, 3000.0, 3250.0, 3500.0)
ECS = 3.0
DELTA_INJ = -50.0
T_DUR = 5.0
T_END = 400.0


def build_ensemble():
    """Run the carbon-release band; return central trajectory + (lo, central, hi) envelopes."""
    kyr = np.linspace(-20.0, T_END, 841)
    runs = [petm.run_petm(m_inj=m, t_dur=T_DUR, delta_inj=DELTA_INJ, ecs=ECS, kyr=kyr)
            for m in M_BAND]
    icen = M_BAND.index(M_CENTRAL)

    def env(key):
        stack = np.vstack([r[key] for r in runs])
        return (np.nanmin(stack, axis=0), runs[icen][key], np.nanmax(stack, axis=0))

    return {
        "kyr": kyr,
        "pco2": env("pco2"), "temp": env("temp"),
        "d13c": env("d13c_surf"), "ph": env("ph"),
        "d13c0": petm.D13C_0,
        "_central": runs[icen], "_band": M_BAND,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(os.path.dirname(__file__), "out"))
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    ens = build_ensemble()
    pdf = os.path.join(args.out, "petm_illustration.pdf")
    plots.plot_petm_illustration(ens, pdf)

    s = petm.summarise(ens["_central"])
    meta = {
        "figure": pdf,
        "illustration": "0-D LOSCAR/COPSE-class carbon-cycle box model; not a calibration",
        "central_release_GtC": M_CENTRAL, "release_band_GtC": [M_BAND[0], M_BAND[-1]],
        "ecs": ECS, "delta_inj_permil": DELTA_INJ,
        "central_peak_warming_K": round(s["peak_warming_K"], 2),
        "central_CIE_permil": round(s["cie_permil"], 2),
        "central_recovery_kyr": round(s["tau_rec_kyr"], 0),
    }
    with open(os.path.join(args.out, "petm_illustration.json"), "w") as fh:
        json.dump(meta, fh, indent=2)
    print(f"wrote {pdf}")
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
