"""Offline-safe tests for real deep-time data ingestion (deeptime_data.py).

The provenance guard is tested offline; the live Macrostrat fetch is network-graceful.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from eh_deeptime import deeptime_data as dd  # noqa: E402


def test_no_fabrication_guard():
    dd.reset_provenance()
    assert dd._record("Macrostrat ... downloaded") and not dd.fallbacks_used()
    dd.assert_real_data("t")                               # real -> no raise
    dd._record("Foster CO2 FAILED")
    assert dd.fallbacks_used()
    try:
        dd.assert_real_data("t")
        raised = False
    except RuntimeError:
        raised = True
    assert raised, "guard must refuse a failed source, not fabricate"
    dd.reset_provenance()


def test_proxy_source_catalogue_has_dois():
    srcs = dd.proxy_sources()
    assert len(srcs) >= 4
    for s in srcs:
        assert "/" in s["doi"]                             # a real-looking DOI
        assert s["kind"] in ("measurement", "model_synthesis", "reconstruction")


def test_macrostrat_live_or_graceful():
    dd.reset_provenance()
    m = dd.load_macrostrat_lithologies()
    if m["lithologies"]:                                   # network available
        assert m["n"] > 50
        assert "CC-BY" in m["license"].upper().replace(" ", "") or "CC-BY" in m["license"]
        assert not dd.fallbacks_used()
    else:                                                  # offline
        assert dd.fallbacks_used()                         # recorded a failure, no stand-in
    dd.reset_provenance()


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"PASS {name}")
    print("all deeptime_data tests passed")
