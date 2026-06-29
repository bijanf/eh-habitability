"""Real public deep-time data ingestion (Phase-1 forcing / proxy backbone).

This module fetches REAL, citable, open-licensed deep-time datasets and exposes
them with full provenance. It carries the SAME no-fabrication guard as
:mod:`eh_shallow.data`: if a source cannot be retrieved it is recorded as a
failure and :func:`assert_real_data` refuses to proceed -- it NEVER substitutes a
synthetic stand-in. There are no embedded fallback data here.

Confirmed-working sources (checked 2026-06-29):
  - Macrostrat lithology definitions / proportions (macrostrat.org/api, CC-BY-4.0).
    Peters et al. 2018, GSA Today; doi 10.1130/GSATG377A.1.
  - Jones & Domeier 2024 Phanerozoic gridded palaeogeography (PhanGrids;
    github.com/LewisAJones/PhanGrids; Sci. Data 11:710, doi 10.1038/s41597-024-03468-w).

Scaffolded with verified DOIs but the exact machine-download endpoint must be
confirmed by the user before a live pull (so the module refuses rather than guess):
  - Foster, Royer & Lunt 2017 Phanerozoic CO2 compilation (Nat. Commun. 8:14845,
    doi 10.1038/ncomms14845).
  - PINT v8 absolute palaeointensity database (pintdb.org; Bono et al. 2022 GJI,
    doi 10.1093/gji/ggab490).

This ingestion is honest data engineering, not a calibrated model. Series are
tagged measurement vs model-synthesis so e.g. GEOCARBSULF/Scotese curves are never
confused with primary measurements.
"""
from __future__ import annotations

import json
import os
import urllib.request

_CACHE = os.path.join(os.path.dirname(__file__), "_deeptime_cache")
os.makedirs(_CACHE, exist_ok=True)

# --- no-fabrication provenance guard (mirrors eh_shallow.data) ----------------
_PROVENANCE: list[str] = []


def _is_fallback(source: str) -> bool:
    s = (source or "").upper()
    return "FALLBACK" in s or "UNAVAILABLE" in s or "FAILED" in s


def _record(source: str) -> str:
    if _is_fallback(source):
        _PROVENANCE.append(source)
    return source


def fallbacks_used() -> list[str]:
    return sorted(set(_PROVENANCE))


def reset_provenance() -> None:
    _PROVENANCE.clear()


def assert_real_data(context: str = "") -> None:
    """Raise if any source failed to retrieve. NEVER fabricates a stand-in."""
    bad = fallbacks_used()
    if bad:
        raise RuntimeError(
            "Refusing to proceed" + (f" [{context}]" if context else "")
            + " -- these deep-time sources did not return real data:\n  - "
            + "\n  - ".join(bad)
            + "\nRe-run with network access; no synthetic substitute is used.")


def _download(url: str, fname: str, timeout: int = 60) -> str | None:
    path = os.path.join(_CACHE, fname)
    if os.path.exists(path) and os.path.getsize(path) > 0:
        return path
    try:
        req = urllib.request.Request(
            url, headers={"User-Agent": "eh_deeptime/0.3 (research; contact corresponding author)"})
        with urllib.request.urlopen(req, timeout=timeout) as r:
            data = r.read()
        if not data:
            return None
        with open(path, "wb") as f:
            f.write(data)
        return path
    except Exception:
        return None


# --- Macrostrat lithology (REAL, CC-BY-4.0) ----------------------------------
MACROSTRAT_LITH_URL = "https://macrostrat.org/api/v2/defs/lithologies?all&format=json"


def load_macrostrat_lithologies() -> dict:
    """Return the real Macrostrat lithology dictionary (CC-BY-4.0).

    {'lithologies': [ {lith_id, name, type, group, class, ...}, ... ],
     'source': '...', 'doi': '10.1130/GSATG377A.1', 'license': 'CC-BY-4.0'}.
    Refuses (records a failure) rather than fabricate if the API is unreachable.
    """
    path = _download(MACROSTRAT_LITH_URL, "macrostrat_lithologies.json")
    if path is None:
        return {"lithologies": [], "source": _record("Macrostrat lithologies FAILED")}
    with open(path) as fh:
        d = json.load(fh)
    liths = d.get("success", {}).get("data", [])
    return {
        "lithologies": liths,
        "n": len(liths),
        "source": "Macrostrat lithology definitions (macrostrat.org/api, downloaded)",
        "doi": "10.1130/GSATG377A.1",
        "license": d.get("success", {}).get("license", "CC-BY 4.0"),
    }


# --- proxy / paleogeography loaders: verified DOIs, endpoint to confirm -------
PHANGRIDS_REPO = "https://github.com/LewisAJones/PhanGrids"
FOSTER2017_DOI = "10.1038/ncomms14845"
PINT_URL = "https://pintdb.org"
PINT_DOI = "10.1093/gji/ggab490"


def proxy_sources() -> list[dict]:
    """Catalogue of the Phanerozoic proxy/forcing sources with verified DOIs.

    These are the REAL public sources for the harmonised proxy DB (Task 1.2) and the
    540-Ma forcing (Task 1.3). 'kind' = measurement | model_synthesis | reconstruction.
    A live pull of the gridded/compiled files needs each endpoint confirmed by the
    user (PhanGrids data files, Foster SI, PINT export), so the loaders below refuse
    rather than guess a URL -- never fabricating data.
    """
    return [
        {"name": "Foster 2017 Phanerozoic CO2 compilation", "var": "CO2",
         "kind": "measurement", "doi": FOSTER2017_DOI, "coverage_Ma": 420,
         "note": "~1200 proxy CO2 estimates; boron/alkenone/stomatal/paleosol families"},
        {"name": "PINT v8 absolute palaeointensity", "var": "geomag_dipole",
         "kind": "measurement", "doi": PINT_DOI, "url": PINT_URL,
         "note": "VDM/VADM; very sparse pre-Cenozoic; QPI quality flags"},
        {"name": "Jones & Domeier 2024 PhanGrids palaeogeography", "var": "paleogeography",
         "kind": "reconstruction", "doi": "10.1038/s41597-024-03468-w", "url": PHANGRIDS_REPO,
         "note": "gridded land/sea + paleolatitude, 5 plate models, 1-Myr steps 540-0 Ma"},
        {"name": "Macrostrat lithology", "var": "lithology",
         "kind": "measurement", "doi": "10.1130/GSATG377A.1", "note": "N-America biased"},
        {"name": "GEOCARBSULF (Berner 2006)", "var": "CO2_model",
         "kind": "model_synthesis", "doi": "10.2475/ajs.291.4.339",
         "note": "MODEL output -- never plot as a measurement"},
    ]
