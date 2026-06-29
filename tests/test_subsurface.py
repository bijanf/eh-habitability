"""Offline smoke tests for the subsurface-biosphere carbon box (H3, subsurface.py)."""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from eh_deeptime import subsurface as ss  # noqa: E402


def test_present_day_stock_anchored_to_literature():
    stock = ss.subsurface_carbon()
    lo, hi = ss.MAGNABOSCO2018_PGC
    assert lo <= stock <= hi                       # anchored within 23-31 Pg C
    h3 = ss.h3_consistency()
    assert h3["within_magnabosco"] and h3["within_h3_target"]


def test_habitable_depth_from_geotherm():
    # (122 - 15) / 25 = 4.28 km; warmer surface -> shallower habitable floor
    assert abs(ss.habitable_depth_km() - (122.0 - 15.0) / 25.0) < 1e-6
    assert ss.habitable_depth_km(t_surface_C=40.0) < ss.habitable_depth_km(t_surface_C=15.0)


def test_warming_shrinks_subsurface_carbon():
    r = ss.warming_response(np.linspace(0, 20, 11))
    assert np.all(np.diff(r["stock_PgC"]) <= 1e-9)        # monotone non-increasing
    assert np.all(np.diff(r["habitable_depth_km"]) <= 1e-9)
    assert r["stock_PgC"][0] > r["stock_PgC"][-1]          # genuinely responds


def test_steeper_geotherm_shrinks_habitable_window():
    assert ss.subsurface_carbon(geotherm=40.0) < ss.subsurface_carbon(geotherm=25.0)


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"PASS {name}")
    print("all subsurface smoke tests passed")
