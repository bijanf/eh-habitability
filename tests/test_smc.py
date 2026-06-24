"""Offline smoke tests for the tempered-SMC identical-twin parameter recovery.

Runs without network and without pytest (``python tests/test_smc.py``). These are
sanity/plausibility checks for an ILLUSTRATION (identical-twin recovery of a known
synthetic truth), NOT a calibration to real data and NOT a validation gate. They
use a fixed seed and a small particle count so the whole suite runs in well under
a minute offline.
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from eh_deeptime import smc  # noqa: E402

# Small, fixed configuration shared across tests so the suite stays fast.
_CFG = dict(n_particles=200, n_temps=8, n_rejuv=3, seed=0)

# Cache one SMC run across all tests (the sampler is the only slow part; running
# it once keeps the whole offline suite well under a minute).
_POST = None


def _run():
    global _POST
    if _POST is None:
        _POST = smc.run_smc(**_CFG)
    return _POST


def test_posterior_runs_and_is_well_formed():
    post = _run()
    for k in ("theta", "weights", "names", "truth", "loglik", "log"):
        assert k in post, f"missing key {k}"
    n, d = post["theta"].shape
    assert d == len(post["names"]) == len(post["truth"])
    assert post["weights"].shape == (n,)
    assert post["loglik"].shape == (n,)


def test_weights_finite_and_normalised():
    post = _run()
    w = post["weights"]
    assert np.all(np.isfinite(w)), "non-finite weights"
    assert np.all(w >= 0.0), "negative weights"
    assert abs(float(w.sum()) - 1.0) < 1e-8, "weights not normalised"
    assert np.all(np.isfinite(post["theta"])), "non-finite particles"


def test_best_identified_param_recovered():
    # 'ecs' is the best-identified parameter (the temperature channel responds to
    # it almost one-for-one); its truth must sit inside the 5-95% interval and the
    # weighted mean must be close to truth. Lenient tolerance, fixed seed.
    post = _run()
    summ = smc.posterior_summary(post)
    truth = dict(zip(post["names"], post["truth"]))
    s = summ["ecs"]
    t = truth["ecs"]
    assert s["p05"] <= t <= s["p95"], f"truth ecs={t} outside [{s['p05']},{s['p95']}]"
    assert abs(s["mean"] - t) / t < 0.25, f"ecs mean {s['mean']} far from truth {t}"


def test_all_truths_bracketed():
    # Every inferred parameter's truth should fall within its 5-95% posterior
    # interval -- a basic credibility-interval coverage check for the twin.
    post = _run()
    summ = smc.posterior_summary(post)
    truth = dict(zip(post["names"], post["truth"]))
    for name in post["names"]:
        s = summ[name]
        t = truth[name]
        assert s["p05"] <= t <= s["p95"], (
            f"truth {name}={t} outside [{s['p05']},{s['p95']}]")


def test_posterior_sharper_than_prior_for_ecs():
    # The data should tighten 'ecs' relative to its prior sd (a sign the
    # likelihood is actually informing the cloud, not just echoing the prior).
    post = _run()
    j = post["names"].index("ecs")
    x, w = post["theta"][:, j], post["weights"]
    mean = np.sum(w * x)
    post_sd = np.sqrt(np.sum(w * (x - mean) ** 2))
    prior_sd = smc.PRIOR["ecs"][1]
    assert post_sd < prior_sd, f"posterior sd {post_sd} not below prior sd {prior_sd}"


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"PASS {name}")
    print("all smc smoke tests passed")
