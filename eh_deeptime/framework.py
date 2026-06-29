"""One-call driver for the illustrative deep-time Earth-habitability framework.

    python -m eh_deeptime.framework [--out DIR] [--quick]

Runs every module in :mod:`eh_deeptime` and writes a set of Nature-style vector
figures plus a ``framework_metrics.json`` into the output directory:

    csys_response.pdf   closed C-S-O-alkalinity box-model response to a carbon pulse
    ebm_climate.pdf     1-D energy-balance climate field T(lat) + global mean vs CO2
    habitability.pdf    guild-mixture habitability surface + guild niches
    smc_recovery.pdf    identical-twin SMC parameter recovery (synthetic pseudo-data)
    sensitivity.pdf     Sobol sensitivity (peak warming) + Jensen aggregation bias
    coupled_haf.pdf     HAF(t) through a carbon pulse, driven by the C-S-O model

EVERYTHING here is an ILLUSTRATION: synthetic / published-envelope inputs only, no
calibration to real proxy data, no out-of-sample validation, and no fabricated
datasets. See the package docstring and each module's docstring for the scope.
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np

from . import (carbon_sulfur, ebm, habitability, smc, sensitivity, haf, plots,
               benchmark, subsurface, deeptime_data)


def _carbon_sulfur_fig(out):
    res = carbon_sulfur.run_csys(m_inj=3000.0, t_dur=5.0, delta_inj=-50.0, t_end=400.0)
    plots.plot_carbon_sulfur(res, os.path.join(out, "csys_response.pdf"))
    s = carbon_sulfur.summarise(res)
    tot_s = res["s_pyr"] + res["s_sulf"] + res["so4"]
    tot_c = res["dic"] + res["corg_cr"] + res["ccarb_cr"]
    s["S_conservation_rel"] = float(np.nanmax(np.abs(tot_s - tot_s[0])) / tot_s[0])
    s["C_conservation_rel_minus_input"] = float(
        np.nanmax(np.abs(tot_c - tot_c[0]))
        - 3000.0 * carbon_sulfur.GTC_TO_MOL) / tot_c[0]
    s["control_steady_drift"] = float(carbon_sulfur.steady_drift(
        carbon_sulfur.run_csys(m_inj=0.0, t_end=400.0)))
    return s


def _ebm_fig(out):
    co2_list = [200.0, 280.0, 400.0, 560.0, 840.0, 1200.0]
    sols = [{"co2": c, **{k: ebm.solve_ebm(co2_ppm=c)[k] for k in ("lat", "T_C")}}
            for c in co2_list]
    sweep_co2 = np.array([180, 240, 280, 350, 420, 560, 700, 840, 1120, 1400, 2000.0])
    t_global = np.array([ebm.solve_ebm(co2_ppm=float(c))["T_global_C"]
                         for c in sweep_co2])
    plots.plot_ebm(sols, {"co2": sweep_co2, "t_global": t_global},
                   os.path.join(out, "ebm_climate.pdf"))
    pd = ebm.summarise(ebm.solve_ebm(co2_ppm=280.0))
    return {"present_day_global_C": pd["T_global_C"],
            "present_day_gradient_C": pd["gradient_C"],
            "ice_latitude_deg_280ppm": pd["ice_latitude_deg"]}


def _habitability_fig(out, seed=0):
    models = habitability.fit_all(np.random.default_rng(seed))
    aw = 0.95
    T = np.linspace(-20.0, 130.0, 90)
    pH = np.linspace(0.0, 12.0, 70)
    TT, PP = np.meshgrid(T, pH)
    X = np.column_stack([TT.ravel(), PP.ravel(), np.full(TT.size, aw)])
    P = habitability.p_hab_mixture(X, models).reshape(PP.shape)
    grid = {"T": T, "pH": pH, "P": P, "a_w": aw}
    Ts = np.linspace(-20.0, 130.0, 160)
    Xs = np.column_stack([Ts, np.full(Ts.size, 7.0), np.full(Ts.size, aw)])
    per = {m["name"]: habitability.p_hab(Xs, m) for m in models}
    mix = habitability.p_hab_mixture(Xs, models)
    slices = {"T": Ts, "per_guild": per, "mixture": mix, "pH": 7.0, "a_w": aw}
    plots.plot_habitability(grid, slices, os.path.join(out, "habitability.pdf"))
    return habitability.grouped_cross_validate(np.random.default_rng(seed))


def _smc_fig(out, quick=False):
    cfg = dict(n_particles=150, n_temps=6, n_rejuv=2) if quick \
        else dict(n_particles=300, n_temps=8, n_rejuv=3)
    post = smc.run_smc(seed=0, **cfg)
    plots.plot_smc(post, os.path.join(out, "smc_recovery.pdf"))
    return {"truth": {n: float(t) for n, t in zip(post["names"], post["truth"])},
            "posterior": smc.posterior_summary(post), "config": cfg}


def _sensitivity_fig(out, quick=False):
    # fast peak-warming model_fn (short integration), shared by Sobol AND Shapley so
    # the two are mutually consistent; peak warming is reached well within 60 kyr.
    names, bounds = sensitivity._default_bounds()

    def fast_pw(v):
        ov = {n: float(x) for n, x in zip(names, v)}
        r = carbon_sulfur.run_csys(params=ov, m_inj=sensitivity.SOBOL_PULSE_GTC,
                                   t_dur=sensitivity.SOBOL_PULSE_DUR,
                                   t_end=60.0, n_out=61)
        return float(carbon_sulfur.summarise(r)["peak_warming_K"])

    sob = sensitivity.sobol_indices(model_fn=fast_pw, bounds=bounds, names=names,
                                    n_base=(32 if quick else 128),
                                    n_boot=(0 if quick else 400))
    jb = sensitivity.jensen_bias()
    shap = None
    if not quick:   # Shapley is the heavier (double-loop) estimator
        shap = sensitivity.shapley_effects(model_fn=fast_pw, bounds=bounds,
                                           names=names, n_outer=32, n_inner=4, n_var=256)
    plots.plot_sensitivity(sob, jb, os.path.join(out, "sensitivity.pdf"), shap=shap)
    order = np.argsort(sob["ST"])[::-1]
    m = {"sobol_names": [sob["names"][i] for i in order],
         "sobol_ST": [float(sob["ST"][i]) for i in order],
         "sobol_S1": [float(sob["S1"][i]) for i in order],
         "jensen_sigma_agg": float(jb["sigma_agg"])}
    if shap is not None:
        os_ = np.argsort(shap["shapley"])[::-1]
        m["shapley_names"] = [shap["names"][i] for i in os_]
        m["shapley"] = [float(shap["shapley"][i]) for i in os_]
        m["shapley_sum"] = float(np.nansum(shap["shapley"]))
    return m


def _haf_fig(out):
    res = haf.coupled_event_haf()
    plots.plot_coupled_haf(res, os.path.join(out, "coupled_haf.pdf"))
    return haf.summarise(res)


def _proxy_co2_fig(out):
    """REAL Phanerozoic CO2 proxy compilation (Foster 2017), fetched live. Network-
    graceful: if the download fails (offline) the figure is skipped, never faked."""
    deeptime_data.reset_provenance()
    co2 = deeptime_data.load_foster2017_co2()
    if not co2["rows"]:
        print("    [skip] Foster 2017 CO2 unavailable (offline); no figure (not faked)")
        return {"status": "unavailable_offline", "n": 0}
    plots.plot_proxy_co2(co2, os.path.join(out, "proxy_co2_foster2017.pdf"))
    import collections
    fam = collections.Counter(r["proxy_family"] for r in co2["rows"])
    return {"n": co2["n"], "coverage_Ma": co2["coverage_Ma"], "doi": co2["doi"],
            "by_family": dict(fam), "source": co2["source"]}


def _benchmark_fig(out):
    """Structural-uncertainty benchmark: our box model vs published community
    models + the proxy consensus (model-vs-model indicator, not validation)."""
    comp = benchmark.structural_comparison()
    plots.plot_structural_benchmark(comp, os.path.join(out, "structural_benchmark.pdf"))
    return benchmark.summarise(comp)


def _subsurface_fig(out):
    """Subsurface-biosphere carbon box (H3): anchored present-day stock + the
    mechanistic shrinkage of the habitable depth window under surface warming."""
    resp = subsurface.warming_response()
    h3 = subsurface.h3_consistency()
    plots.plot_subsurface(resp, h3, os.path.join(out, "subsurface_h3.pdf"))
    return h3


def _deeptime_combined_fig(out):
    """Combined deep-time figure for the Perspective: closed C-S-O response to a
    PETM-scale pulse (a-e) + the HAF(t) it implies (f), on a common time axis.

    Panel (f) is computed from the SAME carbon-sulfur run as panels (a-e), so the
    whole figure is one self-consistent forward simulation with no synthetic,
    analytic or proxy time-series anywhere."""
    csys = carbon_sulfur.run_csys(m_inj=3000.0, t_dur=5.0, delta_inj=-50.0,
                                  t_end=400.0)
    hafres = haf.coupled_event_haf(csys=csys)
    plots.plot_deeptime_combined(csys, hafres,
                                 os.path.join(out, "deeptime_framework.pdf"))
    cs = carbon_sulfur.summarise(csys)
    hs = haf.summarise(hafres)
    return {"peak_warming_K": cs["peak_warming_K"], "cie_permil": cs["cie_permil"],
            "tau_rec_kyr": cs["tau_rec_kyr"],
            "haf_background": hs["haf_background"], "haf_min": hs["haf_min"],
            "haf_drawdown": hs["haf_drawdown"],
            "co2_at_haf_min_ppm": hs["co2_at_haf_min_ppm"]}


def main():
    ap = argparse.ArgumentParser(
        description="Illustrative deep-time EH framework driver.")
    ap.add_argument("--out", default=os.path.join(os.path.dirname(__file__), "out"))
    ap.add_argument("--quick", action="store_true",
                    help="smaller SMC/Sobol samples for a faster smoke run")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    metrics = {"illustration": "deep-time EH framework scaffold; synthetic/illustrative "
                               "inputs only, not calibrated or validated against proxy data"}
    print("[framework] carbon-sulfur-oxygen-alkalinity response ...")
    metrics["carbon_sulfur"] = _carbon_sulfur_fig(args.out)
    print("[framework] 1-D energy-balance climate ...")
    metrics["ebm"] = _ebm_fig(args.out)
    print("[framework] guild-mixture habitability ...")
    metrics["habitability_cv"] = _habitability_fig(args.out)
    print("[framework] tempered-SMC identical-twin recovery ...")
    metrics["smc"] = _smc_fig(args.out, quick=args.quick)
    print("[framework] Sobol sensitivity + Jensen-bias aggregation ...")
    metrics["sensitivity"] = _sensitivity_fig(args.out, quick=args.quick)
    print("[framework] coupled HAF(t) through a carbon pulse ...")
    metrics["coupled_haf"] = _haf_fig(args.out)
    print("[framework] structural benchmark vs published models ...")
    metrics["structural_benchmark"] = _benchmark_fig(args.out)
    print("[framework] subsurface-biosphere carbon box (H3) ...")
    metrics["subsurface_h3"] = _subsurface_fig(args.out)
    print("[framework] REAL Phanerozoic CO2 proxies (Foster 2017, live) ...")
    metrics["proxy_co2_foster2017"] = _proxy_co2_fig(args.out)
    print("[framework] combined deep-time figure (closed C-S-O + HAF) ...")
    metrics["deeptime_combined"] = _deeptime_combined_fig(args.out)

    with open(os.path.join(args.out, "framework_metrics.json"), "w") as fh:
        json.dump(metrics, fh, indent=2, default=float)
    print(f"\nwrote 7 figures + framework_metrics.json to {args.out}")
    print(json.dumps(metrics, indent=2, default=float))


if __name__ == "__main__":
    main()
