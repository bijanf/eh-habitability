"""Offline smoke tests for the illustrative sensitivity / Jensen-bias module.

Runs without network and without pytest (``python tests/test_sensitivity.py``).
These are sanity/plausibility checks for an illustration, not validation gates.
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from eh_deeptime import sensitivity  # noqa: E402


def _toy_problem():
    """A cheap analytic Sobol problem (Ishigami-like) for fast, robust checks."""
    names = ["x0", "x1", "x2"]
    bounds = [(-np.pi, np.pi)] * 3

    def model(v):
        a, b = 7.0, 0.1
        return (np.sin(v[0]) + a * np.sin(v[1]) ** 2
                + b * v[2] ** 4 * np.sin(v[0]))

    return model, bounds, names


def test_sobol_indices_in_range():
    model, bounds, names = _toy_problem()
    res = sensitivity.sobol_indices(model_fn=model, bounds=bounds, names=names,
                                    n_base=512, seed=1)
    S1, ST = res["S1"], res["ST"]
    assert res["names"] == names
    assert np.all(np.isfinite(S1)) and np.all(np.isfinite(ST))
    assert np.all(S1 >= -0.05) and np.all(S1 <= 1.05)
    assert np.all(ST >= -0.05) and np.all(ST <= 1.05)


def test_sobol_total_geq_first():
    model, bounds, names = _toy_problem()
    res = sensitivity.sobol_indices(model_fn=model, bounds=bounds, names=names,
                                    n_base=512, seed=2)
    # total effect >= first order for every input (within Monte-Carlo tolerance)
    assert np.all(res["ST"] >= res["S1"] - 0.10)


def test_sobol_not_all_zero():
    model, bounds, names = _toy_problem()
    res = sensitivity.sobol_indices(model_fn=model, bounds=bounds, names=names,
                                    n_base=512, seed=3)
    assert np.max(np.abs(res["ST"])) > 0.05    # something is influential


def test_sobol_default_model_runs():
    # default carbon_sulfur PETM-peak-warming problem, tiny n_base for speed
    res = sensitivity.sobol_indices(n_base=8, seed=0)
    assert len(res["names"]) == len(res["S1"]) == len(res["ST"])
    assert np.all(np.isfinite(res["S1"])) and np.all(np.isfinite(res["ST"]))


def test_sobol_bootstrap_ci():
    model, bounds, names = _toy_problem()
    res = sensitivity.sobol_indices(model_fn=model, bounds=bounds, names=names,
                                    n_base=256, seed=4, n_boot=200)
    for k in ("S1_ci", "ST_ci"):
        assert k in res and res[k].shape == (len(names), 2)
        assert np.all(res[k][:, 1] >= res[k][:, 0])          # p95 >= p05
    # the point estimate should sit inside (or essentially at) its CI
    assert np.all(res["S1"] >= res["S1_ci"][:, 0] - 0.05)
    assert np.all(res["S1"] <= res["S1_ci"][:, 1] + 0.05)


def test_shapley_sums_to_one_and_ranks():
    # additive model y = 3*x0 + 1*x1 + 0*x2 on the unit cube: no interactions, so
    # Shapley effects ~ variance shares (~0.9, 0.1, 0.0) and MUST sum to 1.
    names = ["x0", "x1", "x2"]
    bounds = [(0.0, 1.0)] * 3
    res = sensitivity.shapley_effects(model_fn=lambda v: 3.0 * v[0] + v[1],
                                      bounds=bounds, names=names,
                                      n_outer=120, n_inner=8, n_var=1024, seed=1)
    sh = res["shapley"]
    assert abs(float(np.nansum(sh)) - 1.0) < 1e-6      # exact partition of variance
    assert sh[0] > sh[1] > sh[2]                        # correct ranking
    assert sh[0] > 0.6 and sh[2] < 0.2                  # x0 dominant, x2 ~ negligible


def test_jensen_bias_finite_and_shaped():
    jb = sensitivity.jensen_bias(n_lat=61)
    for k in ("co2", "f_hab_integrated", "p_hab_globalmean", "delta_J"):
        assert jb[k].shape == jb["co2"].shape
        assert np.all(np.isfinite(jb[k]))
    assert np.isfinite(jb["sigma_agg"])


def test_jensen_bias_probabilities_in_unit_interval():
    jb = sensitivity.jensen_bias(n_lat=61)
    for k in ("f_hab_integrated", "p_hab_globalmean"):
        assert np.all(jb[k] >= -1e-9) and np.all(jb[k] <= 1.0 + 1e-9)


def test_jensen_bias_nonzero_aggregation_bias():
    # convex/concave habitability => spatial mean != value at mean environment
    jb = sensitivity.jensen_bias(n_lat=61)
    assert jb["sigma_agg"] > 0.0
    assert np.max(np.abs(jb["delta_J"])) > 1e-4


def test_jensen_bias_custom_co2_grid():
    jb = sensitivity.jensen_bias(co2_array=[300.0, 800.0, 1500.0], n_lat=41)
    assert jb["co2"].shape == (3,)
    assert np.all(np.isfinite(jb["delta_J"]))


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"PASS {name}")
    print("all sensitivity smoke tests passed")
