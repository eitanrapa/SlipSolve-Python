"""Download every input of the Türkiye 2023 slip model into this folder, and check it against SHA256SUMS.

    python get_data.py            # download what is missing (about 3.9 GB), then verify everything
    python get_data.py --check    # verify only

Standard library only. Files already present with the right checksum are skipped, so it can be re-run after an
interruption. Each file lands where the scripts look for it:

    ceda_ou2024/inputs/...   Sentinel-1 range/azimuth offsets, their E/N/U coefficients and empirical sigmas (CEDA)
    licsar/<frame>/...       LiCSAR interferograms (unwrapped phase, coherence) and look vectors, six frames (CEDA)
    gnss/                    UNR per-event GNSS offsets, one file per event
    traces/                  USGS simplified rupture traces (ScienceBase) and the GEM active-faults GeoJSON
    crust1.0/                CRUST1.0's four model files
"""
import hashlib
import io
import sys
import tarfile
import urllib.request
import zipfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
SUMS = HERE / "SHA256SUMS"

CEDA_OFFSETS = "https://dap.ceda.ac.uk/neodc/comet/publications_data/Turkey_earthquake/v1.0/inputs"
CEDA_LICSAR = "https://dap.ceda.ac.uk/neodc/comet/data/licsar_products"
UNR = "https://geodesy.unr.edu/news_items/20230213"
USGS_ZIP = ("https://www.sciencebase.gov/catalog/file/get/644ad9afd34e45f6ddccf736"
            "?name=simple_fault_v1.1_202402_allformats.zip")
GEM = ("https://raw.githubusercontent.com/GEMScienceTools/gem-global-active-faults/"
       "850fd05b48841eb806d61a37043b5567f5bb99dd/geojson/gem_active_faults_harmonized.geojson")
CRUST1 = "https://igppweb.ucsd.edu/~gabi/crust1/crust1.0.tar.gz"

# Sentinel-1 offset pairs (Ou et al., CEDA): track -> (LiCSAR frame, pair)
OFFSETS = {"014A": ("014A_05232_242525", "20230128_20230209"), "021D": ("021D_05266_252525", "20230129_20230210"),
           "116A": ("116A_05207_252525", "20230204_20230228"), "123D": ("123D_05292_242525", "20230205_20230217")}
# LiCSAR interferograms: (track number as CEDA spells it, frame, pair)
LICSAR = [("14", "014A_05232_242525", "20230128_20230209"), ("21", "021D_05266_252525", "20230129_20230210"),
          ("116", "116A_05207_252525", "20230204_20230216"), ("123", "123D_05292_242525", "20230205_20230217"),
          ("94", "094D_05288_130913", "20230203_20230215"), ("94", "094D_05100_131313", "20230203_20230215")]


def plan():
    """{relative path: url} for every file downloaded directly."""
    out = {}
    for trk, (frame, pair) in OFFSETS.items():
        for c in ("rng", "azi"):
            out[f"ceda_ou2024/inputs/sentinel-1/{frame}/{pair}.geo.{c}.tif"] = \
                f"{CEDA_OFFSETS}/sentinel-1/{frame}/{pair}.geo.{c}.tif?download=1"
            for k in "ENU":
                out[f"ceda_ou2024/inputs/component_vector_coefficients/{trk}_{c}_{k}.tif"] = \
                    f"{CEDA_OFFSETS}/component_vector_coefficients/{trk}_{c}_{k}.tif?download=1"
            out[f"ceda_ou2024/inputs/empirical_offset_uncertainties/{trk}_{c}_sig.tif"] = \
                f"{CEDA_OFFSETS}/empirical_offset_uncertainties/{trk}_{c}_sig.tif?download=1"
    for trk, frame, pair in LICSAR:
        for kind in ("unw", "cc"):
            out[f"licsar/{frame}/{pair}.geo.{kind}.tif"] = \
                f"{CEDA_LICSAR}/{trk}/{frame}/{pair}/{pair}.geo.{kind}.tif?download=1"
        for k in "ENU":
            out[f"licsar/{frame}/{frame}.geo.{k}.tif"] = f"{CEDA_LICSAR}/{trk}/{frame}/metadata/{frame}.geo.{k}.tif?download=1"
    for ev in ("us6000jllz", "us6000jlqa"):                     # Mw 7.8, Mw 7.7
        out[f"gnss/{ev}_final5min.txt"] = f"{UNR}/{ev}_final5min.txt"
    out["traces/gem/gem_active_faults_harmonized.geojson"] = GEM
    return out


def expected():
    rows = {}
    for line in SUMS.read_text().splitlines():
        if line.strip() and not line.startswith("#"):
            digest, size, rel = line.split(None, 2)
            rows[rel] = (digest, int(size))
    return rows


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        while block := fh.read(1 << 22):
            h.update(block)
    return h.hexdigest()


def good(rel, want):
    p = HERE / rel
    return p.exists() and p.stat().st_size == want[rel][1] and sha256(p) == want[rel][0]


def fetch(url):
    req = urllib.request.Request(url, headers={"User-Agent": "slipsolve-example"})
    for attempt in range(3):
        try:
            with urllib.request.urlopen(req, timeout=600) as r:
                return r.read()
        except Exception as e:                                  # transient failures: retry twice
            print(f"  retry {attempt + 1}: {e}", flush=True)
    sys.exit(f"failed to download {url}")


def save(rel, data):
    p = HERE / rel
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_name(p.name + ".part")
    tmp.write_bytes(data)
    tmp.rename(p)


def download(want):
    for rel, url in plan().items():
        if good(rel, want):
            continue
        print(f"get {rel}", flush=True)
        save(rel, fetch(url))
    rel = "traces/simple_fault/simple_fault_v1.1_202402.geojson"
    if not good(rel, want):
        print(f"get {rel} (from the USGS zip)", flush=True)
        with zipfile.ZipFile(io.BytesIO(fetch(USGS_ZIP))) as z:
            save(rel, z.read("simple_fault_v1.1_202402.geojson"))
    names = [f"crust1.0/crust1.{k}" for k in ("bnds", "vp", "vs", "rho")]
    if not all(good(n, want) for n in names):
        print("get crust1.0/ (from crust1.0.tar.gz)", flush=True)
        with tarfile.open(fileobj=io.BytesIO(fetch(CRUST1)), mode="r:gz") as t:
            members = {Path(m.name).name: m for m in t.getmembers() if m.isfile()}
            for n in names:
                save(n, t.extractfile(members[Path(n).name]).read())


def verify(want):
    bad = [rel for rel in want if not good(rel, want)]
    for rel in bad:
        print(f"MISSING OR DIFFERENT: {rel}")
    print(f"{len(want) - len(bad)} of {len(want)} files match SHA256SUMS")
    return not bad


if __name__ == "__main__":
    want = expected()
    if "--check" not in sys.argv[1:]:
        download(want)
    sys.exit(0 if verify(want) else 1)
