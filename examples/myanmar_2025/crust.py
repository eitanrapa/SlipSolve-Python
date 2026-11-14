"""The crust: CRUST1.0 averaged over the 1-degree cells the fault traces cross.

CRUST1.0 (Laske et al. 2013; `get_data.py` puts its files in `crust1.0/`) is read at every cell a trace passes
through, and the cells are averaged layer by layer. The rock surface is put at depth 0, the sediments are folded into
the upper crust, and the profile keeps the upper, middle and lower crust and the mantle.
"""
import json

import numpy as np

from slipsolve import RemainderTables, VelocityModel

MANTLE_BOTTOM = 80e3                       # m; the profile's last depth


def cells(lines):
    """The (lat, lon) centres of the 1-degree cells the traces' vertices fall in, each line a sequence of (lon, lat)."""
    return sorted({(float(np.floor(lat) + 0.5), float(np.floor(lon) + 0.5)) for line in lines for lon, lat in line})


def profile(folder, cell_list):
    """Per cell: thicknesses (km) of the upper (with the sediments), middle and lower crust, and vp, vs, rho of those
    three and the mantle; then their means."""
    b, vp, vs, rho = (np.loadtxt(folder / f"crust1.{k}") for k in ("bnds", "vp", "vs", "rho"))
    lats, lons = 89.5 - np.arange(180), -179.5 + np.arange(360)
    rows = []
    for la, lo in cell_list:
        k = int(np.argmin(np.abs(lats - la))) * 360 + int(np.argmin(np.abs(lons - lo)))
        thick = [b[k][5] - b[k][6], b[k][6] - b[k][7], b[k][7] - b[k][8]]
        thick[0] += b[k][2] - b[k][5]                      # the sediments into the upper crust
        rows.append(dict(cell=[la, lo], thick_km=[float(t) for t in thick], vp=[float(v) for v in vp[k][5:9]],
                         vs=[float(v) for v in vs[k][5:9]], rho=[float(v) for v in rho[k][5:9]]))
    mean = {k: np.mean([r[k] for r in rows], axis=0) for k in ("thick_km", "vp", "vs", "rho")}
    return rows, mean


def velocity_model(lines, folder, cache, name):
    """The averaged profile as a `VelocityModel`, kept in the JSON file `cache` after the first call."""
    if not cache.exists():
        cell_list = cells(lines)
        rows, mean = profile(folder, cell_list)
        bases = np.cumsum(mean["thick_km"]) * 1e3
        depth, vp, vs, rho = [0.0], [], [], []
        for i, base in enumerate(list(bases) + [MANTLE_BOTTOM]):
            depth += [float(round(base))] * (2 if i < 3 else 1)
            vp += [1e3 * mean["vp"][i]] * 2
            vs += [1e3 * mean["vs"][i]] * 2
            rho += [1e3 * mean["rho"][i]] * 2
        mu = [rho[2 * i] * vs[2 * i] ** 2 / 1e9 for i in range(4)]
        cache.write_text(json.dumps(dict(source=f"CRUST1.0 (Laske et al. 2013), {len(cell_list)} cells", cells=rows,
                                         depth_m=depth, vp=vp, vs=vs, rho=rho, mu_gpa=mu), indent=1))
    d = json.loads(cache.read_text())
    return VelocityModel(depth=d["depth_m"], vp=d["vp"], vs=d["vs"], rho=d["rho"], name=name)


def tables(model, path, workers=3):
    """The layered Green's-function tables for `model`, computed and saved to `path` the first time (a few minutes)."""
    if not path.exists():
        print("computing the layered tables (a few minutes)", flush=True)
        RemainderTables.compute(model, workers=workers).save(path)
    return RemainderTables.load(path)
