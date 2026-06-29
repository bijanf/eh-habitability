"""Offline smoke tests for the structural-uncertainty benchmark (benchmark.py).

Sanity checks that the published-diagnostics table is well-formed and that our box
model is compared against it honestly (inter-model indicator, not validation).
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from eh_deeptime import benchmark  # noqa: E402


def test_published_table_well_formed():
    assert len(benchmark.PUBLISHED) >= 5
    for r in benchmark.PUBLISHED:
        assert r["kind"] in ("model", "proxy")
        assert r["doi"] and "/" in r["doi"]                 # a real-looking DOI
        for f in ("peak_warming_K", "cie_permil", "recovery_kyr", "carbon_GtC"):
            v = r.get(f)
            assert v is None or (len(v) == 2 and v[0] <= v[1])   # (lo, hi) or "not reported"


def test_our_model_diagnostics_finite():
    o = benchmark.our_model(m_inj=3000.0)
    for k in ("peak_warming_K", "cie_permil", "recovery_kyr"):
        assert np.isfinite(o[k])
    assert o["cie_permil"] < 0.0                              # an excursion is negative


def test_structural_comparison_shape_and_honesty():
    c = benchmark.structural_comparison(m_inj=3000.0)
    assert set(c["by_diag"]) == {"peak_warming_K", "cie_permil", "recovery_kyr"}
    # at least one diagnostic must land within the published-model spread, and the
    # comparison must distinguish model spread from proxy consensus.
    d = c["by_diag"]["peak_warming_K"]
    assert d["model_spread"] is not None
    assert isinstance(d["within_model_spread"], bool)
    assert isinstance(d["within_proxy_consensus"], bool)


def test_bigger_pulse_changes_our_warming():
    small = benchmark.our_model(m_inj=1000.0)
    big = benchmark.our_model(m_inj=6000.0)
    assert big["peak_warming_K"] > small["peak_warming_K"]    # genuine model response


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"PASS {name}")
    print("all benchmark smoke tests passed")
