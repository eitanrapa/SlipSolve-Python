"""Download every input of the Ridgecrest 2019 slip model into this folder, and check it against SHA256SUMS.

    python get_data.py            # download what is missing (about 170 MB), then verify everything
    python get_data.py --check    # verify only

Python's standard library and numpy, and one program to open the Sentinel-2 offsets' RAR archive: `bsdtar` (macOS's
`tar`), `unar`, `7z` or `unrar`. Files already present with the right checksum are skipped, so it can be re-run after an
interruption. Each file lands where the scripts look for it:

    ucsd/        UCSD's line-of-sight sets: ALOS-2 tracks 65 and 66, Sentinel-1 tracks 64 and 71
    s2/          the Sentinel-2 east and north offsets (COSI-Corr, detrended)
    gnss/        the Nevada Geodetic Laboratory's daily series of every station within 120 km of the epicentres, and
                 the offsets computed from them (ngl_offsets.csv)
    crust1.0/    CRUST1.0's four model files

The GNSS series are live files that grow every day, so only the offsets computed from them are checked. If NGL
reprocesses its solutions they can differ from ours by a little; the run then uses the new values.
"""
import datetime as dt
import hashlib
import io
import math
import shutil
import subprocess
import sys
import tarfile
import tempfile
import urllib.request
import zipfile
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
SUMS = HERE / "SHA256SUMS"

UCSD = "https://topex.ucsd.edu/SV_7.1/web/highres/SV_los_original.zip"
UCSD_SETS = ("ALOS2_los_T065", "ALOS2_los_T066", "S1_los_T064", "S1_los_T071")
S2 = "https://zenodo.org/records/3275073/files/COSI_20190628_20190708.rar?download=1"
S2_FILES = ("COSI_20190628_20190708_EW_DETR.tif", "COSI_20190628_20190708_NS_DETR.tif")
NGL_HOLDINGS = "https://geodesy.unr.edu/NGLStationPages/DataHoldings.txt"
NGL_SERIES = "https://geodesy.unr.edu/gps_timeseries/IGS20/tenv3/IGS20/{}.tenv3"
CRUST1 = "https://igppweb.ucsd.edu/~gabi/crust1/crust1.0.tar.gz"

#: USGS ci38443183 (Mw 6.4) and ci38457511 (Mw 7.1): longitude, latitude, origin time (Unix seconds)
ORIGINS = ((-117.5038333, 35.7053333, 1562261629.0), (-117.5993333, 35.7695, 1562383193.04))
MAX_KM = 120.0                             # stations within this of either epicentre
PRE_DAYS, POST_DAYS, MIN_DAYS = 10, 5, 4   # the windows: see `offsets`
GNSS_TABLE = "gnss/ngl_offsets.csv"


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        while block := fh.read(1 << 22):
            h.update(block)
    return h.hexdigest()


def expected():
    rows = {}
    for line in SUMS.read_text().splitlines():
        if line.strip() and not line.startswith("#"):
            digest, size, rel = line.split(None, 2)
            rows[rel] = (digest, int(size))
    return rows


def good(rel, want):
    p = HERE / rel
    return rel in want and p.exists() and p.stat().st_size == want[rel][1] and sha256(p) == want[rel][0]


def fetch(url, required=True):
    req = urllib.request.Request(url, headers={"User-Agent": "slipsolve-example"})
    for attempt in range(3):
        try:
            with urllib.request.urlopen(req, timeout=600) as r:
                return r.read()
        except Exception as e:                                  # transient failures: retry twice
            print(f"  retry {attempt + 1}: {e}", flush=True)
    if required:
        sys.exit(f"failed to download {url}")
    print(f"  skipped: {url}", flush=True)
    return None


def save(rel, data):
    p = HERE / rel
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_name(p.name + ".part")
    tmp.write_bytes(data)
    tmp.rename(p)


def unrar(data, names, dest):
    """Extract `names` from a RAR archive held in `data` into `dest`, with the first program that can."""
    commands = (["bsdtar", "-xf", "{a}", "-C", "{d}"], ["tar", "-xf", "{a}", "-C", "{d}"],
                ["unar", "-q", "-f", "-o", "{d}", "{a}"], ["7z", "x", "-y", "-o{d}", "{a}"], ["unrar", "x", "-o+", "{a}", "{d}/"])
    with tempfile.TemporaryDirectory() as tmp:
        archive = Path(tmp) / "archive.rar"
        archive.write_bytes(data)
        out = Path(tmp) / "out"
        for cmd in commands:
            if shutil.which(cmd[0]) is None:
                continue
            out.mkdir(exist_ok=True)
            run = [c.replace("{a}", str(archive)).replace("{d}", str(out)) for c in cmd]
            if subprocess.run(run, capture_output=True).returncode == 0:
                found = {p.name: p for p in out.rglob("*") if p.is_file()}
                if all(n in found for n in names):
                    dest.mkdir(parents=True, exist_ok=True)
                    for n in names:
                        shutil.move(str(found[n]), dest / n)
                    return
            shutil.rmtree(out)
    sys.exit("could not open the Sentinel-2 RAR archive: install bsdtar (libarchive), unar, 7z or unrar")


def distance_km(lat1, lon1, lat2, lon2):
    p1, p2, dl = math.radians(lat1), math.radians(lat2), math.radians(lon2 - lon1)
    c = math.sin(p1) * math.sin(p2) + math.cos(p1) * math.cos(p2) * math.cos(dl)
    return 6371.0 * math.acos(max(-1.0, min(1.0, c)))


def mjd(unix_seconds):
    return unix_seconds / 86400.0 + 40587.0


def offsets(text, first_origin, last_origin):
    """One station's coseismic offset from its daily series (NGL tenv3, metres).

    The offset is the median of days +1..+5 after the last origin's day minus the median of the 10 days before the
    first origin's day; the origin days themselves are left out, since a daily solution there mixes both states. Each
    sigma is the windows' robust scatter (1.4826 MAD over sqrt(n)) in quadrature. A station with fewer than 4 days
    in either window is left out (None).
    """
    rows = []
    for line in text.splitlines()[1:]:
        p = line.split()
        if len(p) >= 16:
            rows.append((float(p[3]), float(p[7]) + float(p[8]), float(p[9]) + float(p[10]), float(p[11]) + float(p[12])))
    if not rows:
        return None
    a = np.array(rows)
    day0, day1 = math.floor(mjd(first_origin)), math.floor(mjd(last_origin))
    pre = a[(a[:, 0] >= day0 - PRE_DAYS) & (a[:, 0] < day0)][:, 1:]
    post = a[(a[:, 0] > day1) & (a[:, 0] <= day1 + POST_DAYS)][:, 1:]
    if len(pre) < MIN_DAYS or len(post) < MIN_DAYS:
        return None

    def scatter(x):
        return 1.4826 * np.median(np.abs(x - np.median(x, axis=0)), axis=0) / np.sqrt(len(x))

    return np.median(post, axis=0) - np.median(pre, axis=0), np.hypot(scatter(pre), scatter(post)), len(pre), len(post)


def gnss():
    """Every NGL station within MAX_KM of an epicentre with data across both windows -> its series and the offsets."""
    t0, t1 = min(o[2] for o in ORIGINS), max(o[2] for o in ORIGINS)
    utc = dt.timezone.utc
    need_from = (dt.datetime.fromtimestamp(t0, utc) - dt.timedelta(days=PRE_DAYS)).strftime("%Y-%m-%d")
    need_to = (dt.datetime.fromtimestamp(t1, utc) + dt.timedelta(days=POST_DAYS)).strftime("%Y-%m-%d")
    stations = []
    for row in fetch(NGL_HOLDINGS).decode().splitlines()[1:]:
        r = row.split()
        try:
            lat, lon = float(r[1]), (float(r[2]) + 180.0) % 360.0 - 180.0
        except (ValueError, IndexError):
            continue
        km = min(distance_km(lat, lon, o[1], o[0]) for o in ORIGINS)
        if float(f"{km:.1f}") <= MAX_KM and r[7] <= need_from and r[8] >= need_to:
            stations.append((r[0], lon, lat, km))
    print(f"{len(stations)} NGL stations within {MAX_KM:.0f} km with data {need_from} .. {need_to}", flush=True)
    lines = ["sta,lon,lat,dist_km,de_m,dn_m,du_m,sde_m,sdn_m,sdu_m,n_pre,n_post"]
    for name, lon, lat, km in stations:
        rel = f"gnss/series/{name}.tenv3"
        if not (HERE / rel).exists():
            data = fetch(NGL_SERIES.format(name), required=False)
            if data is None:
                continue
            save(rel, data)
        res = offsets((HERE / rel).read_text(), t0, t1)
        if res is not None:
            d, s, n_pre, n_post = res
            lines.append(f"{name},{lon:.5f},{lat:.5f},{km:.1f}," + ",".join(f"{v:.5f}" for v in (*d, *s))
                         + f",{n_pre},{n_post}")
    save(GNSS_TABLE, ("\n".join(lines) + "\n").encode())
    print(f"offsets at {len(lines) - 1} stations -> {GNSS_TABLE}", flush=True)


def download(want):
    if not all(good(f"ucsd/{n}.lltnde", want) for n in UCSD_SETS):
        print("get ucsd/ (from SV_los_original.zip)", flush=True)
        with zipfile.ZipFile(io.BytesIO(fetch(UCSD))) as z:
            for n in UCSD_SETS:
                save(f"ucsd/{n}.lltnde", z.read(f"{n}.lltnde"))
    if not all(good(f"s2/{n}", want) for n in S2_FILES):
        print("get s2/ (from COSI_20190628_20190708.rar, 47 MB)", flush=True)
        unrar(fetch(S2), S2_FILES, HERE / "s2")
    if not (HERE / GNSS_TABLE).exists():                         # delete it to compute it again
        gnss()
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
        if rel == GNSS_TABLE and (HERE / rel).exists():
            print(f"DIFFERENT: {rel} (NGL may have reprocessed its series; the run uses these values)")
        else:
            print(f"MISSING OR DIFFERENT: {rel}")
    print(f"{len(want) - len(bad)} of {len(want)} files match SHA256SUMS")
    return all(rel == GNSS_TABLE and (HERE / rel).exists() for rel in bad)


if __name__ == "__main__":
    want = expected()
    if "--check" not in sys.argv[1:]:
        download(want)
    sys.exit(0 if verify(want) else 1)
