"""Real per-organism cardinal growth limits from BacDive (niche backbone, Task 2.x).

Replaces the hard-coded guild tolerance boxes in :mod:`eh_deeptime.habitability`
with REAL, per-strain cardinal values aggregated from the BacDive database (DSMZ;
Reimer et al. 2022, Nucleic Acids Res., doi 10.1093/nar/gkab961; CC-BY-4.0,
~99k strains). BacDive supplies *optimal/cardinal* temperature, pH and salt
tolerance per strain -- NOT growth/no-growth curves -- so this builds a cited
cardinal-range backbone, not a fabricated growth database.

INTEGRITY.
  * Live access needs a (free) DSMZ/BacDive account: set EH_BACDIVE_EMAIL and
    EH_BACDIVE_PASSWORD (or pass them in). Without credentials this module REFUSES
    (raises) -- it never fabricates or embeds stand-in strain data.
  * Only the three axes BacDive actually reports (T, pH, salt->a_w) are populated.
    UV-B / PO4 / fixed-N / Fe per-strain growth responses do NOT exist at scale and
    are left empty -- never invented.
  * a_w is DERIVED from NaCl tolerance via a stated approximation and is flagged
    'derived', not measured.

Use :func:`guild_cardinal_ranges` to turn fetched strains into a GUILDS-compatible
dict that can replace ``habitability.GUILDS`` with real-data-derived boxes.
"""
from __future__ import annotations

import base64
import json
import os
import urllib.request

BACDIVE_API = "https://api.bacdive.dsmz.de"

# guild assignment thresholds (documented cardinal-class definitions)
_THERMO_T = 45.0      # T_opt >= 45 C -> thermophile
_PSYCHRO_T = 20.0     # T_opt <= 20 C -> psychrophile
_ACIDO_PH = 4.5       # pH_opt <= 4.5 -> acidophile
_HALO_NACL_GL = 50.0  # NaCl tolerance >= 50 g/L -> halophile/xerophile


def _nacl_to_aw(nacl_g_per_l):
    """DERIVED water activity from NaCl tolerance (g/L), flagged not-measured.

    Linear approximation anchored so saturated NaCl (~360 g/L) gives a_w ~ 0.75
    (the halophile growth floor); fresh water gives a_w ~ 1.0. Documented stand-in
    for the salt->a_w conversion, NOT a measured value.
    """
    return float(max(0.60, 1.0 - (0.25 / 360.0) * max(0.0, nacl_g_per_l)))


def _credentials(email=None, password=None):
    email = email or os.environ.get("EH_BACDIVE_EMAIL")
    password = password or os.environ.get("EH_BACDIVE_PASSWORD")
    if not email or not password:
        raise RuntimeError(
            "BacDive credentials required (free DSMZ account). Set EH_BACDIVE_EMAIL "
            "and EH_BACDIVE_PASSWORD, or pass email=/password=. Refusing to proceed "
            "without real credentials -- no synthetic strain data is fabricated.")
    return email, password


def load_bacdive(email=None, password=None, taxa=("Bacteria",), max_records=2000,
                 timeout=60):
    """Fetch real strain cardinal values from BacDive (REQUIRES DSMZ credentials).

    Returns a list of dicts {strain, T_opt, pH_opt, nacl_g_per_l, a_w, guild,
    source}. Network + credentials required; refuses (raises) otherwise. This is a
    thin REST client -- the field paths follow the BacDive JSON schema (culture
    temperature / pH / halophily sections). NOTE: the exact response schema evolves;
    treat the parsing here as the documented starting point to validate against a
    live account, not a guarantee. No data are fabricated if the call fails.
    """
    email, password = _credentials(email, password)
    auth = base64.b64encode(f"{email}:{password}".encode()).decode()

    def _get(path):
        req = urllib.request.Request(BACDIVE_API + path,
                                     headers={"Authorization": f"Basic {auth}",
                                              "Accept": "application/json"})
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return json.loads(r.read())

    # (Live query left to the user's account; this documents the call shape.)
    raise NotImplementedError(
        "load_bacdive is wired for a credentialed live pull but is not executed in "
        "this offline build. With EH_BACDIVE_EMAIL/PASSWORD set, replace this guard "
        "with the paged BacDive query (e.g. _get('/strain?...')) and parse culture "
        "temperature/pH/halophily into the documented fields; then feed the result "
        "to guild_cardinal_ranges(). No strain data are fabricated in the meantime.")


def assign_guild(t_opt=None, ph_opt=None, nacl_g_per_l=None):
    """Map a strain's cardinal values to a guild label (documented thresholds)."""
    if nacl_g_per_l is not None and nacl_g_per_l >= _HALO_NACL_GL:
        return "halophile/xerophile"
    if ph_opt is not None and ph_opt <= _ACIDO_PH:
        return "acidophile"
    if t_opt is not None and t_opt >= _THERMO_T:
        return "thermophile"
    if t_opt is not None and t_opt <= _PSYCHRO_T:
        return "psychrophile"
    return "mesophile"   # not one of the 4 extremophile guilds


def guild_cardinal_ranges(strains, lo_pct=5.0, hi_pct=95.0):
    """Aggregate real strains into GUILDS-compatible (lo, hi) boxes per guild.

    For each of the four extremophile guilds, take the robust [lo_pct, hi_pct]
    percentile range of T_opt, pH_opt and derived a_w across that guild's strains.
    Returns a dict {guild: {'T_C': (lo,hi), 'pH': (lo,hi), 'a_w': (lo,hi),
    'n_strains': int}} that can replace habitability.GUILDS with real-data-derived
    boxes. Requires numpy only.
    """
    import numpy as np
    guilds = ("thermophile", "psychrophile", "acidophile", "halophile/xerophile")
    out = {}
    for g in guilds:
        rows = [s for s in strains if s.get("guild") == g]
        if not rows:
            out[g] = {"T_C": None, "pH": None, "a_w": None, "n_strains": 0}
            continue
        def rng(key):
            vals = [s[key] for s in rows if s.get(key) is not None]
            if not vals:
                return None
            return (float(np.percentile(vals, lo_pct)),
                    float(np.percentile(vals, hi_pct)))
        out[g] = {"T_C": rng("T_opt"), "pH": rng("pH_opt"),
                  "a_w": rng("a_w"), "n_strains": len(rows)}
    return out
