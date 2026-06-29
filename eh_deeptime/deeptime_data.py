"""Real public deep-time data ingestion (Phase-1 forcing / proxy backbone).

This module fetches REAL, citable, open-licensed deep-time datasets and exposes
them with full provenance. It carries the SAME no-fabrication guard as
:mod:`eh_shallow.data`: if a source cannot be retrieved it is recorded as a
failure and :func:`assert_real_data` refuses to proceed -- it NEVER substitutes a
synthetic stand-in. There are no embedded fallback data here.

LIVE sources (real downloads confirmed working 2026-06-29):
  - Foster, Royer & Lunt 2017 Phanerozoic CO2 compilation (~1200 proxy estimates,
    0-423 Ma) from the paper's Springer Supplementary workbook. Nat. Commun. 8:14845,
    doi 10.1038/ncomms14845. -> load_foster2017_co2()
  - LR04 global benthic d18O stack (Lisiecki & Raymo 2005), via the NOAA NCEI mirror
    (PANGAEA blocks programmatic download). doi 10.1029/2004PA001071. -> load_lr04()
  - Macrostrat lithology definitions (macrostrat.org/api, CC-BY-4.0). Peters et al.
    2018, GSA Today; doi 10.1130/GSATG377A.1. -> load_macrostrat_lithologies()
  - PINT(QPI) absolute palaeointensity database (Veikkolainen et al. 2017, Sci. Data;
    PINT v8 Bono et al. 2022 GJI). ~640 dipole-moment determinations 0-3458 Ma from a
    CC0 Zenodo/Dryad deposit (doi 10.5061/dryad.63g17), read with a tiny built-in .ods
    parser. -> load_pint()
  - PBDB GPlates-reconstructed palaeocoordinates (Paleobiology Database data service,
    Peters & McClennen 2016, doi 10.1017/pab.2015.39): the canonical primary
    palaeogeography source. -> load_pbdb_paleocoords()

Catalogue-only (verified DOI; the derived product is impractical to fetch in-sandbox):
  - Jones & Domeier 2024 PhanGrids palaeogeography GRIDS (Sci. Data 11:710,
    doi 10.1038/s41597-024-03468-w; data doi 10.5281/zenodo.10607398) -- the gridded
    deposit is 3.4 GB and R-serialized, so we ingest the underlying PBDB+GPlates
    palaeocoordinates live via load_pbdb_paleocoords() instead.
  NB: PANGAEA's ?format=textfile endpoint returns HTTP 400/406 to scripts, and
  pintdb.org is unreachable, so those series are sourced from NOAA / Zenodo mirrors.

This ingestion is honest data engineering, not a calibrated model. Series are
tagged measurement vs model-synthesis so e.g. GEOCARBSULF/Scotese curves are never
confused with primary measurements.
"""
from __future__ import annotations

import csv
import io
import json
import os
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
import zipfile

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


def _download_zip_member(url: str, zip_fname: str, member: str,
                         timeout: int = 120) -> bytes | None:
    """Download a zip (cached) and return the raw bytes of one member, or None."""
    path = _download(url, zip_fname, timeout=timeout)
    if path is None:
        return None
    try:
        with zipfile.ZipFile(path) as z:
            return z.read(member)
    except Exception:
        return None


# --- minimal pure-Python OpenDocument-spreadsheet reader (no odfpy needed) -----
_ODS_T = "urn:oasis:names:tc:opendocument:xmlns:table:1.0"
_ODS_O = "urn:oasis:names:tc:opendocument:xmlns:office:1.0"


def _read_ods(ods_bytes: bytes, sheet_name: str) -> list[list]:
    """Return one sheet of an .ods workbook as a list of row-lists.

    A cell's ``office:value`` is used when present (numbers), else its concatenated
    text. ``number-columns-repeated`` / ``number-rows-repeated`` are expanded
    (capped, since they are mostly trailing blanks). Empty rows are dropped. This is
    a tiny dependency-free reader -- the sandbox has no odfpy -- sufficient for the
    flat PINT(QPI) tables; it is not a general ODS implementation.
    """
    try:
        root = ET.fromstring(zipfile.ZipFile(io.BytesIO(ods_bytes)).read("content.xml"))
    except Exception:
        return []
    for tbl in root.iter(f"{{{_ODS_T}}}table"):
        if tbl.get(f"{{{_ODS_T}}}name") != sheet_name:
            continue
        rows: list[list] = []
        for r in tbl.iter(f"{{{_ODS_T}}}table-row"):
            rrep = min(int(r.get(f"{{{_ODS_T}}}number-rows-repeated", "1")), 5)
            cells: list = []
            for c in list(r):
                if c.tag.split("}")[-1] not in ("table-cell", "covered-table-cell"):
                    continue
                crep = min(int(c.get(f"{{{_ODS_T}}}number-columns-repeated", "1")), 200)
                val = c.get(f"{{{_ODS_O}}}value")
                if val is None:
                    val = "".join(t for t in c.itertext())
                cells.extend([val] * crep)
            if any(str(x).strip() for x in cells):
                rows.extend([cells] * rrep)
        return rows
    return []


def _num(x):
    try:
        return float(str(x).strip())
    except (TypeError, ValueError):
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


# --- PINT(QPI) absolute palaeointensity (REAL, CC0) --------------------------
PINT_ZENODO = ("https://zenodo.org/api/records/4983322/files/"
               "paleomag_databases.zip/content")
PINT_DATA_DOI = "10.5061/dryad.63g17"        # the deposited PINT(QPI)+PALEOMAGIA tables
PINT_PAPER_DOI = "10.1038/sdata.2017.68"     # Veikkolainen et al. 2017, Sci. Data (QPI db)
PINT_V8_DOI = "10.1093/gji/ggab490"          # Bono et al. 2022 GJI (PINT v8 compilation)


def load_pint() -> dict:
    """PINT(QPI) absolute palaeointensity database (REAL, CC0).

    Downloads the deposited PINT(QPI) workbook (Veikkolainen, Biggin, Pesonen,
    Evans & Jarboe; the quality-graded PINT compilation) from Zenodo and parses its
    ``PINTData`` sheet into per-determination rows giving the virtual (axial) dipole
    moment ``VDM/VADM`` (in 1e22 A m^2), the site age (Ma) and the QPI reliability
    score (0-7). ~640 determinations spanning 0-3458 Ma (Archean -> present): the
    real geomagnetic dipole-moment backbone for the dynamo strand. Refuses (records
    a failure) rather than fabricate if the deposit is unreachable.
    """
    raw = _download_zip_member(PINT_ZENODO, "pint_qpi_paleomag.zip", "PINT_Qpi_web.ods")
    if raw is None:
        return {"rows": [], "n": 0, "source": _record("PINT(QPI) FAILED")}
    grid = _read_ods(raw, "PINTData")
    if not grid:
        return {"rows": [], "n": 0, "source": _record("PINT(QPI) parse FAILED")}
    hdr = [str(x).strip() for x in grid[0]]

    def col(name):
        return hdr.index(name) if name in hdr else None

    iAGE, iDAGE, iVDM, iQPI = col("AGE"), col("DAGE"), col("VDM/VADM"), col("QPI")
    iID, iLAT, iLON = col("IDENT"), col("SLAT"), col("SLONG")
    iCONT, iCTRY, iGRP, iTYPE = col("Continent"), col("Country"), col("GROUP"), col("TYPE")

    def get(r, i):
        return r[i] if (i is not None and i < len(r)) else None

    def txt(r, i):
        v = get(r, i)
        v = str(v).strip() if v is not None else ""
        return v or None

    rows = []
    for r in grid[1:]:
        age, vdm = _num(get(r, iAGE)), _num(get(r, iVDM))
        if age is None or vdm is None or age <= 0 or vdm <= 0:
            continue
        rows.append({
            "ident": txt(r, iID), "age_Ma": age, "dage_Ma": _num(get(r, iDAGE)),
            "vdm_e22_Am2": vdm, "qpi": _num(get(r, iQPI)),
            "site_lat": _num(get(r, iLAT)), "site_lon": _num(get(r, iLON)),
            "continent": txt(r, iCONT), "country": txt(r, iCTRY),
            "group": txt(r, iGRP), "rock_type": txt(r, iTYPE),
        })
    return {
        "rows": rows, "n": len(rows), "var": "geomag_dipole_moment",
        "kind": "measurement", "units": "VDM/VADM in 1e22 A m^2",
        "source": "PINT(QPI) absolute palaeointensity database "
                  "(Veikkolainen et al.; Zenodo/Dryad, downloaded)",
        "doi": PINT_DATA_DOI, "paper_doi": PINT_PAPER_DOI, "compilation_doi": PINT_V8_DOI,
        "license": "CC0",
        "coverage_Ma": (min(x["age_Ma"] for x in rows),
                        max(x["age_Ma"] for x in rows)) if rows else None,
    }


# --- PBDB reconstructed palaeocoordinates (REAL) -----------------------------
# PhanGrids' own derived grid product is a 3.4 GB / R-serialized Zenodo deposit
# (10.5281/zenodo.10607398) -- impractical to fetch + unparseable in pure Python in
# this sandbox -- so the framework ingests the canonical PRIMARY palaeogeography
# source PhanGrids itself is built from: PBDB fossil collections with GPlates-
# reconstructed palaeocoordinates, live via the official data service.
PBDB_COLLS = "https://paleobiodb.org/data1.2/colls/list.csv"
PBDB_DOI = "10.1017/pab.2015.39"             # Peters & McClennen 2016, PBDB data API
PHANGRIDS_REPO = "https://github.com/LewisAJones/PhanGrids"
PHANGRIDS_DOI = "10.1038/s41597-024-03468-w"
PHANGRIDS_DATA_DOI = "10.5281/zenodo.10607398"


def load_pbdb_paleocoords(base_name: str = "Trilobita", limit: int = 5000,
                          timeout: int = 90) -> dict:
    """Real GPlates-reconstructed palaeocoordinates from the Paleobiology Database.

    Queries the official PBDB data service for fossil *collections* of ``base_name``
    and parses each collection's reconstructed palaeolatitude/-longitude and
    palaeoage into rows {collection_no, age_Ma, paleolat, paleolng, modern_lat,
    modern_lng, plate}. This is the canonical primary palaeogeography source (the
    same PBDB-occurrence + GPlates-plate-model basis PhanGrids is built on); ``limit``
    bounds the fully reproducible query. Refuses (records a failure) rather than
    fabricate if the service is unreachable.
    """
    url = (f"{PBDB_COLLS}?base_name={urllib.parse.quote(base_name)}"
           f"&show=paleoloc&limit={int(limit)}")
    fname = f"pbdb_{base_name.lower().replace(' ', '_')}_{int(limit)}.csv"
    path = _download(url, fname, timeout=timeout)
    if path is None:
        return {"rows": [], "n": 0,
                "source": _record(f"PBDB paleocoords ({base_name}) FAILED")}
    rows = []
    with open(path, encoding="utf-8", errors="replace") as fh:
        for d in csv.DictReader(fh):
            plat, plng = _num(d.get("paleolat")), _num(d.get("paleolng"))
            if plat is None or plng is None:
                continue
            mx, mn = _num(d.get("max_ma")), _num(d.get("min_ma"))
            age = (mx + mn) / 2 if (mx is not None and mn is not None) else None
            rows.append({"collection_no": d.get("collection_no"), "age_Ma": age,
                         "paleolat": plat, "paleolng": plng,
                         "modern_lat": _num(d.get("lat")), "modern_lng": _num(d.get("lng")),
                         "plate": d.get("geoplate")})
    ages = [x["age_Ma"] for x in rows if x["age_Ma"] is not None]
    return {
        "rows": rows, "n": len(rows), "var": "paleogeography",
        "kind": "reconstruction", "base_name": base_name,
        "paleomodel": "GPlates (PBDB default plate model)",
        "source": f"Paleobiology Database collection palaeocoordinates "
                  f"({base_name}, n<={limit}), paleobiodb.org/data1.2 (downloaded)",
        "doi": PBDB_DOI, "url": "https://paleobiodb.org",
        "coverage_Ma": (min(ages), max(ages)) if ages else None,
    }


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
        {"name": "PINT(QPI) absolute palaeointensity", "var": "geomag_dipole_moment",
         "kind": "measurement", "doi": PINT_DATA_DOI, "live": True, "loader": "load_pint",
         "note": "~640 determinations 0-3458 Ma; VDM/VADM dipole moment + QPI grade; "
                 "Zenodo/Dryad CC0 (Veikkolainen 2017; PINT v8 Bono 2022)"},
        {"name": "PBDB GPlates palaeocoordinates", "var": "paleogeography",
         "kind": "reconstruction", "doi": PBDB_DOI, "url": "https://paleobiodb.org",
         "live": True, "loader": "load_pbdb_paleocoords",
         "note": "real reconstructed palaeolat/lon of fossil collections; the primary "
                 "source PhanGrids is built from (PBDB + GPlates plate model)"},
        {"name": "Jones & Domeier 2024 PhanGrids palaeogeography grids", "var": "paleogeography",
         "kind": "reconstruction", "doi": PHANGRIDS_DOI, "url": PHANGRIDS_REPO,
         "live": False, "data_doi": PHANGRIDS_DATA_DOI,
         "note": "derived H3 land/sea + paleolat grids, 5 plate models; the deposit is "
                 "3.4 GB / R-serialized -- impractical in-sandbox, so use load_pbdb_paleocoords"},
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
        "dipole_pint": load_pint(),
        "paleocoords_pbdb": load_pbdb_paleocoords(),
    }
    return {
        "series": series,
        "provenance": [{"key": k, "source": v.get("source"), "doi": v.get("doi"),
                        "n": v.get("n")} for k, v in series.items()],
        "any_failed": bool(fallbacks_used()),
    }
