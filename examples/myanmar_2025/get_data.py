"""Download every input of the Myanmar 2025 slip model into this folder, and check it against SHA256SUMS.

    python get_data.py            # download what is missing (a 740 MB archive; 0.8 GB is kept), then verify
    python get_data.py --check    # verify only

Standard library only. Files already present with the right checksum are skipped, so it can be re-run after an
interruption. Each file lands where the scripts look for it:

    xu2025/<product>/   Xu et al.'s ALOS-2 interferograms and Sentinel-1 pixel offsets: the files the run reads, out
                        of their LOS_DEFO.zip, which is deleted afterwards (its LuTan-1 offsets are not used)
    trace/              He & Yun's rupture trace
    crust1.0/           CRUST1.0's four model files
"""
import hashlib
import io
import shutil
import sys
import tarfile
import urllib.request
import zipfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
SUMS = HERE / "SHA256SUMS"

LOS_DEFO = "https://zenodo.org/records/17113746/files/LOS_DEFO.zip?download=1"
LOS_DEFO_MD5 = "fc1f50bb52a87256d9de6b84540f0e45"     # as Zenodo lists it
TRACE = "https://zenodo.org/records/17595200/files/fault_trace_from_NS.txt?download=1"
CRUST1 = "https://igppweb.ucsd.edu/~gabi/crust1/crust1.0.tar.gz"

#: product -> the value file; with it the run reads enu.geo and, for phase, geo_filt.cor, for offsets the value's Std
PRODUCTS = {
    "ALOS2_A152_20250211_20250408_IFG": "geo_filt_msk_ERA5_SET.unw",
    "ALOS2_D041_20250216_20250330_IFG": "geo_filt_msk_ION_ERA5_SET.unw",
    **{f"{p}_OFF_{c}": f"off{c.title()}.geo"
       for p in ("S1_A070_20250322_20250403", "S1_A143_20250327_20250408", "S1_D033_20250319_20250331",
                 "S1_D106_20250324_20250405") for c in ("RG", "AZ")},
}


def files(product):
    stem = PRODUCTS[product]
    second = "geo_filt.cor" if product.endswith("_IFG") else stem.replace(".geo", "Std.geo")
    return [f"xu2025/{product}/{n}{ext}" for n in (stem, second, "enu.geo") for ext in ("", ".rsc")]


def expected():
    rows = {}
    for line in SUMS.read_text().splitlines():
        if line.strip() and not line.startswith("#"):
            digest, size, rel = line.split(None, 2)
            rows[rel] = (digest, int(size))
    return rows


def digest(path, kind="sha256"):
    h = hashlib.new(kind)
    with open(path, "rb") as fh:
        while block := fh.read(1 << 22):
            h.update(block)
    return h.hexdigest()


def good(rel, want):
    p = HERE / rel
    return rel in want and p.exists() and p.stat().st_size == want[rel][1] and digest(p) == want[rel][0]


def fetch(url, to=None):
    """The bytes at `url`, or with `to` streamed into that file."""
    req = urllib.request.Request(url, headers={"User-Agent": "slipsolve-example"})
    for attempt in range(3):
        try:
            with urllib.request.urlopen(req, timeout=600) as r:
                if to is None:
                    return r.read()
                to.parent.mkdir(parents=True, exist_ok=True)
                part = to.with_name(to.name + ".part")
                with open(part, "wb") as out:
                    shutil.copyfileobj(r, out, 1 << 22)
                part.rename(to)
                return to
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
    missing = [p for p in PRODUCTS if not all(good(rel, want) for rel in files(p))]
    if missing:
        archive = HERE / "xu2025" / "LOS_DEFO.zip"
        if not archive.exists() or digest(archive, "md5") != LOS_DEFO_MD5:
            print("get xu2025/LOS_DEFO.zip (740 MB)", flush=True)
            fetch(LOS_DEFO, to=archive)
            if digest(archive, "md5") != LOS_DEFO_MD5:
                sys.exit("LOS_DEFO.zip does not match Zenodo's checksum: delete it and run again")
        with zipfile.ZipFile(archive) as outer:
            for product in missing:
                print(f"unpack {product}", flush=True)
                with zipfile.ZipFile(io.BytesIO(outer.read(f"LOS_DEFO/{product}.zip"))) as inner:
                    for rel in files(product):
                        save(rel, inner.read(rel.split("/", 1)[1]))
        archive.unlink()
    if not good("trace/fault_trace_from_NS.txt", want):
        print("get trace/fault_trace_from_NS.txt", flush=True)
        save("trace/fault_trace_from_NS.txt", fetch(TRACE))
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
