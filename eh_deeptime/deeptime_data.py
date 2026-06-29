"""Real public deep-time data ingestion (Phase-1 forcing / proxy backbone).

This module fetches REAL, citable, open-licensed deep-time datasets and exposes
them with full provenance. It carries the SAME no-fabrication guard as
:mod:`eh_shallow.data`: if a source cannot be retrieved it is recorded as a
failure and :func:`assert_real_data` refuses to proceed -- it NEVER substitutes a
synthetic stand-in. There are no embedded fallback data here.

LIVE sources (real downloads confirmed working 2026-06-29):
  - Foster, Royer & Lunt 2017 Phanerozoic CO2 compilation (~520 proxy estimates,
    0-423 Ma) from the paper's Springer Supplementary workbook. Nat. Commun. 8:14845,
    doi 10.1038/ncomms14845. -> load_foster2017_co2()
  - LR04 global benthic d18O stack (Lisiecki & Raymo 2005), via the NOAA NCEI mirror
    (PANGAEA blocks programmatic download). doi 10.1029/2004PA001071. -> load_lr04()
  - Macrostrat lithology definitions (macrostrat.org/api, CC-BY-4.0). Peters et al.
    2018, GSA Today; doi 10.1130/GSATG377A.1. -> load_macrostrat_lithologies()

Catalogue-only (verified DOIs; endpoint still to confirm, so NOT auto-pulled):
  - PINT v8 absolute palaeointensity (pintdb.org reset under load; try MagIC).
    Bono et al. 2022 GJI, doi 10.1093/gji/ggab490.
  - Jones & Domeier 2024 PhanGrids palaeogeography (github.com/LewisAJones/PhanGrids;
    Sci. Data 11:710, doi 10.1038/s41597-024-03468-w) -- confirm the data-file path.
  NB: PANGAEA's ?format=textfile endpoint returns HTTP 400/406 to scripts, so proxy
  series there are sourced from NOAA mirrors instead where available.

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


# --- LIVE proxy loaders (real downloads confirmed reachable 2026-06-29) -------
FOSTER2017_DOI = "10.1038/ncomms14845"
FOSTER2017_XLSX = (
    "https://static-content.springer.com/esm/art%3A10.1038%2Fncomms14845/"
    "MediaObjects/41467_2017_BFncomms14845_MOESM2874_ESM.xlsx")
LR04_URL = ("https://www.ncei.noaa.gov/pub/data/paleo/contributions_by_author/"
            "lisiecki2005/lisiecki2005.txt")
LR04_DOI = "10.1029/2004PA001071"


def load_foster2017_co2() -> dict:
    """Foster, Royer & Lunt (2017) Phanerozoic CO2 proxy compilation (REAL).

    Downloads the paper's Supplementary Data workbook (Springer) and parses the
    'proxies' sheet into rows {proxy_family, age_Ma, co2_ppm, co2_lo, co2_hi}.
    ~520 proxy CO2 estimates spanning ~0-420 Ma (paleosols, stomata, boron,
    alkenones, ...). Real measurements; refuses (records a failure) if unreachable.
    """
    import pandas as pd
    path = _download(FOSTER2017_XLSX, "foster2017_co2.xlsx", timeout=120)
    if path is None:
        return {"rows": [], "n": 0, "source": _record("Foster 2017 CO2 FAILED")}
    df = pd.read_excel(path, sheet_name="proxies", header=None)

    def _num(x):
        try:
            return float(x)
        except (TypeError, ValueError):
            return None

    # The sheet lays the proxy families out in HORIZONTAL column blocks; each block
    # begins at a "Reference" column (row index 2) with, relative to it:
    # Age(Ma)=+2, CO2(ppm)=+5, CO2 low=+7, CO2 high=+8; the family name is in row 1.
    hdr = df.iloc[2].astype(str).str.strip()
    starts = [j for j in range(df.shape[1]) if hdr.iloc[j] == "Reference"]
    rows = []
    for b in starts:
        fam = str(df.iloc[1, b]).strip()
        fam = fam if fam and fam != "nan" else None
        for i in range(4, df.shape[0]):
            age, co2 = _num(df.iloc[i, b + 2]), _num(df.iloc[i, b + 5])
            if age is not None and co2 is not None and 0 < age < 600 and 0 < co2 < 1e5:
                rows.append({"proxy_family": fam, "age_Ma": age, "co2_ppm": co2,
                             "co2_lo": _num(df.iloc[i, b + 7]),
                             "co2_hi": _num(df.iloc[i, b + 8])})
    return {
        "rows": rows, "n": len(rows), "var": "CO2", "kind": "measurement",
        "source": "Foster, Royer & Lunt 2017 CO2 compilation (Springer SI, downloaded)",
        "doi": FOSTER2017_DOI, "coverage_Ma": (min(r["age_Ma"] for r in rows),
                                               max(r["age_Ma"] for r in rows)) if rows else None,
    }


def load_lr04() -> dict:
    """LR04 global benthic d18O stack, Lisiecki & Raymo 2005 (REAL, NOAA mirror).

    {age_kyr, d18O, error} over 0-5320 ka -- the standard Plio-Pleistocene benthic
    d18O (temperature + ice-volume) proxy used for the 5.2a hindcast. Refuses
    rather than fabricate if the NOAA file is unreachable.
    """
    import re
    path = _download(LR04_URL, "lr04_lisiecki2005.txt")
    if path is None:
        return {"rows": [], "n": 0, "source": _record("LR04 FAILED")}
    with open(path, encoding="utf-8", errors="replace") as fh:
        lines = fh.read().splitlines()
    data = []
    for ln in lines:
        if re.match(r"^\s*\d+(\.\d+)?\s+[\-\d.]+\s+[\-\d.]+\s*$", ln):
            a, d, e = ln.split()
            data.append({"age_kyr": float(a), "d18O": float(d), "error": float(e)})
    return {
        "rows": data, "n": len(data), "var": "d18O", "kind": "measurement",
        "source": "LR04 benthic d18O stack (Lisiecki & Raymo 2005, NOAA, downloaded)",
        "doi": LR04_DOI,
    }


# --- still catalogue-only (verified DOIs; endpoint to confirm) ----------------
PHANGRIDS_REPO = "https://github.com/LewisAJones/PhanGrids"
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
         "kind": "measurement", "doi": FOSTER2017_DOI, "live": True,
         "loader": "load_foster2017_co2",
         "note": "~520 proxy CO2 estimates 0-423 Ma; boron/alkenone/stomatal/paleosol"},
        {"name": "LR04 benthic d18O stack (Lisiecki & Raymo 2005)", "var": "d18O",
         "kind": "measurement", "doi": LR04_DOI, "live": True, "loader": "load_lr04",
         "note": "0-5320 ka Plio-Pleistocene benthic d18O (NOAA mirror)"},
        {"name": "Macrostrat lithology", "var": "lithology", "kind": "measurement",
         "doi": "10.1130/GSATG377A.1", "live": True,
         "loader": "load_macrostrat_lithologies", "note": "CC-BY-4.0; N-America biased"},
        {"name": "PINT v8 absolute palaeointensity", "var": "geomag_dipole",
         "kind": "measurement", "doi": PINT_DOI, "url": PINT_URL, "live": False,
         "note": "endpoint to confirm (pintdb.org reset; try MagIC). Sparse pre-Cenozoic"},
        {"name": "Jones & Domeier 2024 PhanGrids palaeogeography", "var": "paleogeography",
         "kind": "reconstruction", "doi": "10.1038/s41597-024-03468-w", "url": PHANGRIDS_REPO,
         "live": False, "note": "gridded land/sea + paleolat, 5 plate models; confirm data file"},
        {"name": "GEOCARBSULF (Berner 2006)", "var": "CO2_model",
         "kind": "model_synthesis", "doi": "10.2475/ajs.291.4.339", "live": False,
         "note": "MODEL output -- never plot as a measurement"},
    ]


def proxy_db():
    """Assemble the available REAL proxy series into one provenanced dict.

    Pulls every live loader (Foster CO2, LR04 d18O, Macrostrat lithology) and
    returns {series: {...}, provenance: [...], any_failed: bool}. Each series keeps
    its source + DOI + measurement/model tag. If a live source fails to download it
    is recorded (no synthetic substitute); call :func:`assert_real_data` to refuse
    on any failure. This is honest aggregation of real public data, not a model.
    """
    series = {
        "co2_foster2017": load_foster2017_co2(),
        "d18O_lr04": load_lr04(),
        "lithology_macrostrat": load_macrostrat_lithologies(),
    }
    return {
        "series": series,
        "provenance": [{"key": k, "source": v.get("source"), "doi": v.get("doi"),
                        "n": v.get("n")} for k, v in series.items()],
        "any_failed": bool(fallbacks_used()),
    }
