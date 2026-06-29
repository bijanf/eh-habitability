"""Offline tests for the BacDive niche backbone (extremophiles.py).

The live BacDive pull needs DSMZ credentials; here we test only the no-fabrication
credential guard, the guild-assignment logic, the salt->a_w derivation, and the
cardinal-range aggregator on a small synthetic TEST FIXTURE (a unit-test input,
not presented as real data).
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from eh_deeptime import extremophiles as E  # noqa: E402


def test_refuses_without_credentials():
    # ensure env creds are absent for the test
    old = (os.environ.pop("EH_BACDIVE_EMAIL", None),
           os.environ.pop("EH_BACDIVE_PASSWORD", None))
    try:
        raised = False
        try:
            E.load_bacdive()
        except RuntimeError:
            raised = True
        assert raised, "must refuse (not fabricate) without BacDive credentials"
    finally:
        if old[0] is not None:
            os.environ["EH_BACDIVE_EMAIL"] = old[0]
        if old[1] is not None:
            os.environ["EH_BACDIVE_PASSWORD"] = old[1]


def test_guild_assignment_thresholds():
    assert E.assign_guild(t_opt=90.0) == "thermophile"
    assert E.assign_guild(t_opt=5.0) == "psychrophile"
    assert E.assign_guild(ph_opt=2.0) == "acidophile"
    assert E.assign_guild(nacl_g_per_l=200.0) == "halophile/xerophile"
    assert E.assign_guild(t_opt=30.0, ph_opt=7.0) == "mesophile"


def test_nacl_to_aw_anchors():
    assert abs(E._nacl_to_aw(0.0) - 1.0) < 1e-9         # fresh water
    assert abs(E._nacl_to_aw(360.0) - 0.75) < 1e-6      # ~saturated NaCl
    assert E._nacl_to_aw(1e6) >= 0.60                    # clamped floor


def test_cardinal_range_aggregator():
    fixture = [
        {"guild": "thermophile", "T_opt": 80.0, "pH_opt": 6.5, "a_w": 0.98},
        {"guild": "thermophile", "T_opt": 95.0, "pH_opt": 7.0, "a_w": 0.99},
        {"guild": "acidophile", "T_opt": 40.0, "pH_opt": 2.0, "a_w": 0.97},
    ]
    r = E.guild_cardinal_ranges(fixture)
    assert r["thermophile"]["n_strains"] == 2
    lo, hi = r["thermophile"]["T_C"]
    assert 80.0 <= lo <= hi <= 95.0
    assert r["psychrophile"]["n_strains"] == 0            # none in fixture
    assert r["psychrophile"]["T_C"] is None               # no fabrication for empty


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"PASS {name}")
    print("all extremophiles tests passed")
