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


def test_foster2017_co2_live_or_graceful():
    dd.reset_provenance()
    co2 = dd.load_foster2017_co2()
    if co2["rows"]:                                        # network available
        assert co2["n"] > 800                             # ~1200 across all families
        ages = [r["age_Ma"] for r in co2["rows"]]
        assert max(ages) > 300                            # reaches the deep Phanerozoic
        fams = {r["proxy_family"] for r in co2["rows"]}
        assert len(fams) >= 3                              # multiple proxy families parsed
        assert all(r["co2_ppm"] > 0 for r in co2["rows"])
    else:
        assert dd.fallbacks_used()
    dd.reset_provenance()


def test_lr04_live_or_graceful():
    dd.reset_provenance()
    lr = dd.load_lr04()
    if lr["rows"]:
        assert lr["n"] > 1000
        assert max(r["age_kyr"] for r in lr["rows"]) > 1000   # spans the Pleistocene+
        assert all(2.0 < r["d18O"] < 6.0 for r in lr["rows"][:50])  # benthic d18O range
    else:
        assert dd.fallbacks_used()
    dd.reset_provenance()


def test_pint_live_or_graceful():
    dd.reset_provenance()
    p = dd.load_pint()
    if p["rows"]:                                         # network available
        assert p["n"] > 400                               # ~640 graded determinations
        ages = [r["age_Ma"] for r in p["rows"]]
        assert max(ages) > 2500                           # reaches the Archean
        assert all(r["vdm_e22_Am2"] > 0 for r in p["rows"])
        assert all(0 < r["vdm_e22_Am2"] < 100 for r in p["rows"])  # 1e22 A m^2 scale
        assert p["doi"] == "10.5061/dryad.63g17"
        assert not dd.fallbacks_used()
    else:
        assert dd.fallbacks_used()                        # recorded a failure, no stand-in
    dd.reset_provenance()


def test_pbdb_paleocoords_live_or_graceful():
    dd.reset_provenance()
    pg = dd.load_pbdb_paleocoords(base_name="Trilobita", limit=500)
    if pg["rows"]:                                        # network available
        assert pg["n"] > 50
        assert all(-90.0 <= r["paleolat"] <= 90.0 for r in pg["rows"])
        assert all(-180.0 <= r["paleolng"] <= 180.0 for r in pg["rows"])
        assert "/" in pg["doi"]
        assert not dd.fallbacks_used()
    else:
        assert dd.fallbacks_used()
    dd.reset_provenance()


def test_proxy_db_assembles_real_series():
    dd.reset_provenance()
    db = dd.proxy_db()
    assert set(db["series"]) == {"co2_foster2017", "d18O_lr04", "lithology_macrostrat",
                                 "dipole_pint", "paleocoords_pbdb"}
    for p in db["provenance"]:
        assert "/" in (p["doi"] or "")                    # each carries a real DOI
    # if everything fetched, the guard passes; otherwise it must have recorded failures
    if not db["any_failed"]:
        dd.assert_real_data("proxy_db")
    dd.reset_provenance()


def test_valid_content_rejects_error_pages(tmpdir=None):
    import tempfile
    d = tempfile.mkdtemp()
    def w(name, b):
        p = os.path.join(d, name)
        with open(p, "wb") as fh:
            fh.write(b)
        return p
    html = w("e.html", b"<!DOCTYPE html><html><head><title>503</title></head></html>")
    assert not dd._valid_content(html, "csv")          # error page is NOT valid csv/text
    assert not dd._valid_content(html, "json")
    assert not dd._valid_content(html, "zip")
    assert dd._valid_content(w("ok.json", b'{"success": {}}'), "json")
    assert dd._valid_content(w("ok.csv", b'"a","b"\n1,2\n'), "csv")
    assert dd._valid_content(w("ok.zip", b"PK\x03\x04rest"), "zip")
    assert not dd._valid_content(w("notzip.bin", b"not a zip"), "zip")


def test_guard_refuses_successful_but_wrong_schema():
    # A download that succeeds (HTTP 200) but returns the WRONG schema must be
    # recorded as a FAILURE, never returned as empty-but-"downloaded" real data.
    import tempfile
    d = tempfile.mkdtemp()
    bad = os.path.join(d, "bad.csv")
    with open(bad, "w") as fh:
        fh.write("collection_no,lng,lat\n1,2,3\n")     # no paleolat/paleolng columns
    orig = dd._download
    try:
        dd._download = lambda *a, **k: bad
        dd.reset_provenance()
        pg = dd.load_pbdb_paleocoords(base_name="Trilobita", limit=10)
        assert pg["rows"] == [] and pg["n"] == 0
        assert dd.fallbacks_used(), "wrong-schema 200 must be recorded as a failure"
        raised = False
        try:
            dd.assert_real_data("t")
        except RuntimeError:
            raised = True
        assert raised, "guard must refuse, not pass a mislabeled empty success"
    finally:
        dd._download = orig
        dd.reset_provenance()


def test_read_ods_does_not_duplicate_value_rows():
    # number-rows-repeated on a value-bearing row must NOT inflate the data.
    row = ('<table:table-row table:number-rows-repeated="4">'
           '<table:table-cell office:value-type="float" office:value="100">'
           '<text:p>100</text:p></table:table-cell></table:table-row>')
    content = ('<?xml version="1.0"?>'
               '<office:document-content '
               'xmlns:office="urn:oasis:names:tc:opendocument:xmlns:office:1.0" '
               'xmlns:table="urn:oasis:names:tc:opendocument:xmlns:table:1.0" '
               'xmlns:text="urn:oasis:names:tc:opendocument:xmlns:text:1.0">'
               '<office:body><office:spreadsheet>'
               '<table:table table:name="S">' + row + '</table:table>'
               '</office:spreadsheet></office:body></office:document-content>')
    import io as _io
    import zipfile as _zip
    buf = _io.BytesIO()
    with _zip.ZipFile(buf, "w") as z:
        z.writestr("content.xml", content)
    grid = dd._read_ods(buf.getvalue(), "S")
    assert len(grid) == 1, "a single value-bearing row must appear once, not 4x"
    assert grid[0][0] == "100"


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"PASS {name}")
    print("all deeptime_data tests passed")
