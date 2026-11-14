"""Fault traces for the 2023 Kahramanmaraş doublet, as lon/lat polylines.

Sources (all public, in ``traces/``):
- USGS simplified rupture traces (Reitman et al. 2023, doi:10.5066/P985I7U2), mapped from
  Sentinel-1 offsets at 1:750,000. The ruptured parts of every branch.
- GEM global active faults (EMME model), for the EAF east of the rupture's end, the Pütürge
  stretch where Pereiaslov et al. (2026) put the seismic gap (their A-A' runs ~365 km).
- The Lomax (2023) relocated aftershocks (``catalogs/``), for the northern branch east of
  its mapped surface rupture, into the Nurhak fault complexity (NFC): the paper extends that
  branch "according to the east-northeast-trending plane that enters the NFC".

The branches, as in Pereiaslov et al. 2026 (their Fig. 2):
- ``south``: Amanos - Pazarcık - Erkenek (Mw 7.8), plus the GEM Pütürge extension. Vertical.
- ``north``: Çardak and its continuation across the NFC (Mw 7.7). 70 deg N in the paper.
- ``narli``: the Narlı splay, where the Mw 7.8 nucleated. A secondary of the south branch.
- ``ns``: the 21 km N-S strand at 36.73 E mapped for the Mw 7.7. A secondary of the north branch.

USGS feature indices below are positions in ``simple_fault_v1.1_202402.geojson``.
"""

import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
USGS = HERE / "traces" / "simple_fault" / "simple_fault_v1.1_202402.geojson"
GEM = HERE / "traces" / "gem" / "gem_active_faults_harmonized.geojson"

# USGS pieces in along-strike order, south-west to north-east. Feature 7 (a 7.5 km strand
# beside the Türkoğlu bend) and the sub-5 km pieces 10, 17-21 are left out: one sheet per
# branch, as in the paper.
SOUTH_PIECES = [5, 3, 4, 2, 1, 0, 22, 6]
NORTH_PIECES = [12, 11, 15, 16]
NARLI_PIECE = 9
NS_PIECE = 8

# Northern branch east of the mapped rupture (USGS ends at 38.017 E, 38.186 N): medians of
# the Lomax aftershocks after the Mw 7.7 in 0.1 deg longitude bins (errH < 5 km).
NFC_EXTENSION = [(38.10, 38.205), (38.20, 38.230), (38.30, 38.255)]

# The GEM EMME fault that continues the EAF east of the Erkenek end (38.537 E, 38.067 N):
# start (38.52, 38.08), 63 km, sinistral, dip (85, 80, 90).
GEM_PUTURGE_START = (38.52, 38.08)
GEM_PUTURGE_END = (39.15, 38.36)
PUTURGE_FROM_LON = 38.56          # keep GEM vertices east of the USGS end


def _pieces(path=USGS):
    g = json.loads(Path(path).read_text())
    out = []
    for f in g["features"]:
        geom = f["geometry"]
        parts = geom["coordinates"] if geom["type"] == "MultiLineString" else [geom["coordinates"]]
        out.append([np.asarray([[p[0], p[1]] for p in part], dtype=float) for part in parts])
    return out


def _km(a, b):
    lat = np.radians(0.5 * (a[1] + b[1]))
    return float(np.hypot((b[0] - a[0]) * 111.32 * np.cos(lat), (b[1] - a[1]) * 110.57))


def _chain(arrays, max_gap_km=3.0):
    """Join polylines end to end, flipping each to start nearest the previous end."""
    line = arrays[0]
    # Orient the first piece toward the second.
    nxt = arrays[1]
    ends = [min(_km(line[i], nxt[0]), _km(line[i], nxt[-1])) for i in (0, -1)]
    if ends[0] < ends[1]:
        line = line[::-1]
    for piece in arrays[1:]:
        if _km(line[-1], piece[-1]) < _km(line[-1], piece[0]):
            piece = piece[::-1]
        gap = _km(line[-1], piece[0])
        if gap > max_gap_km:
            raise ValueError(f"gap of {gap:.1f} km between chained pieces")
        if gap < 1e-6:
            piece = piece[1:]
        line = np.vstack([line, piece])
    return line


def _gem_feature(start, end, path=GEM, tol_deg=0.02):
    g = json.loads(Path(path).read_text())
    for f in g["features"]:
        geom = f["geometry"]
        if geom is None:
            continue
        parts = geom["coordinates"] if geom["type"] == "MultiLineString" else [geom["coordinates"]]
        for part in parts:
            a = np.asarray([[p[0], p[1]] for p in part], dtype=float)
            for line in (a, a[::-1]):
                if (np.abs(line[0] - start).max() < tol_deg
                        and np.abs(line[-1] - end).max() < tol_deg):
                    return line
    raise LookupError(f"no GEM feature from {start} to {end}")


def traces():
    """``{name: (n, 2) lon/lat array}`` for the four sheets, each ordered SW/W to NE/E."""
    pieces = _pieces()
    one = lambda i: pieces[i][0]

    south = _chain([one(i) for i in SOUTH_PIECES])
    puturge = _gem_feature(GEM_PUTURGE_START, GEM_PUTURGE_END)
    south = np.vstack([south, puturge[puturge[:, 0] > PUTURGE_FROM_LON]])

    north = _chain([one(i) for i in NORTH_PIECES])
    north = np.vstack([north, np.asarray(NFC_EXTENSION)])

    narli = one(NARLI_PIECE)
    if narli[0, 1] > narli[-1, 1]:          # south to north, ending at the EAF
        narli = narli[::-1]
    ns = one(NS_PIECE)
    if ns[0, 1] > ns[-1, 1]:                # south to north
        ns = ns[::-1]
    return {"south": south, "north": north, "narli": narli, "ns": ns}


def length_km(line):
    return float(sum(_km(a, b) for a, b in zip(line[:-1], line[1:])))


if __name__ == "__main__":
    for name, line in traces().items():
        steps = [_km(a, b) for a, b in zip(line[:-1], line[1:])]
        print(f"{name:6s} {len(line):4d} vertices, {length_km(line):6.1f} km, "
              f"({line[0, 0]:.3f}, {line[0, 1]:.3f}) -> ({line[-1, 0]:.3f}, {line[-1, 1]:.3f}), "
              f"max step {max(steps):.1f} km")
