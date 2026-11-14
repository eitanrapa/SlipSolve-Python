"""Slip on every sheet with the named segments marked, their magnitudes, and a map of where each one is.

    python plot_slip_map.py runs/prod [zmax=26] [mesh=1] [out=slip_map.png] [title=...]

zmax= hides the elements at and below that depth (a model cut there) and ends each panel at it. mesh=1
draws the element edges on the slip panels, and on the map each sheet's bottom edge at zmax (where a dipping sheet sits
at depth).

Segment boundaries are the junctions of the USGS rupture pieces (geometry.py): the Türkoğlu bend (36.656 E,
37.113 N), the Gölbaşı stepover (37.487 E, 37.685 N), the east end of the 2023 rupture (38.537 E, 38.067 N), past which
the GEM Pütürge trace continues; on the north branch, the end of the mapped Çardak rupture (37.812 E, 38.071 N), past
which the branch follows the aftershocks into the Nurhak fault complexity (NFC). Each is placed at the arc length of
the nearest surface node of its sheet, so it lines up with the slip panel.

Mw: the events and branches are the integrated moment on the layered rigidity (as summary.json); a segment's is its
share of the node-lumped moment, which partitions the total exactly (the lumped rule reads ~0.005 high).
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import matplotlib.tri as mtri  # noqa: E402
import numpy as np  # noqa: E402

from slipsolve import SlipModel  # noqa: E402

import geometry  # noqa: E402

FAULTS = ["south", "narli", "north", "ns"]
JUNCTIONS = {"south": [("Türkoğlu bend", (36.656, 37.113)), ("Gölbaşı stepover", (37.487, 37.685)),
                       ("2023 rupture end", (38.537, 38.067))],
             "north": [("end of mapped rupture", (37.812, 38.071))]}
SEGMENTS = {"south": ["Amanos", "Pazarcık", "Erkenek", "Pütürge (gap)"], "north": ["Çardak", "NFC"]}
COLORS = {"Amanos": "#1f77b4", "Pazarcık": "#ff7f0e", "Erkenek": "#2ca02c", "Pütürge (gap)": "#8c564b",
          "Çardak": "#d62728", "NFC": "#9467bd", "Narlı": "#e377c2", "N-S strand": "#7f7f7f"}
# Lomax (2023) relocations of the two mainshocks
EPICENTRES = [("Mw 7.8, 01:17 UTC", 37.040, 37.238), ("Mw 7.7, 10:24 UTC", 37.230, 38.061)]


def mw(m0):
    return (2.0 / 3.0) * (np.log10(m0) - 9.1)


def main():
    run = Path(sys.argv[1])
    opts = dict(a.split("=", 1) for a in sys.argv[2:])
    zmax = float(opts["zmax"]) if "zmax" in opts else None
    show_mesh = opts.get("mesh") == "1"
    model = SlipModel.load(run / "model.slip.zip")
    summary = json.loads((run / "summary.json").read_text())
    # the older drivers wrote {"moments": {k: {"Mw": ...}}}; slipsolve.production writes {"Mw": {k: ...}}
    moments = summary.get("moments") or {k: {"Mw": v} for k, v in summary["Mw"].items()}
    mesh = model.mesh
    counts = np.asarray(mesh.attrs["joined_nodes"], dtype=int)
    starts = np.r_[0, np.cumsum(counts)]
    mag = model.slip_magnitude
    params = mesh.params
    frame = mesh.frame
    tri_all = np.asarray(mesh.triangles)
    lumped = None
    try:
        areas = model.basis.lumped_areas()
        mu = np.asarray(model._default_shear_modulus(), dtype=float)
        lumped = np.broadcast_to(mu, areas.shape) * areas * mag
    except Exception:  # noqa: BLE001
        pass

    def sheet(k):
        a, b = starts[k], starts[k + 1]
        t = tri_all[np.all((tri_all >= a) & (tri_all < b), axis=1)] - a
        return a, b, t

    def boundaries(name):
        """Arc length (km) of each junction: the nearest surface node of that sheet."""
        a, b, _ = sheet(FAULTS.index(name))
        nodes, s = mesh.nodes[a:b], params[a:b, 0]
        top = np.abs(nodes[:, 2]) < 1.0
        out = []
        for label, (lon, lat) in JUNCTIONS.get(name, []):
            x, y = frame.to_local(lon, lat)
            d = np.hypot(nodes[top, 0] - x, nodes[top, 1] - y)
            out.append((label, float(s[top][np.argmin(d)] / 1e3)))
        return out

    def segment_mw(name):
        a, b, _ = sheet(FAULTS.index(name))
        s = params[a:b, 0] / 1e3
        edges = [0.0] + [v for _, v in boundaries(name)] + [np.inf]
        out = []
        for seg, lo, hi in zip(SEGMENTS[name], edges[:-1], edges[1:]):
            sel = (s >= lo) & (s < hi)
            m0 = float(lumped[a:b][sel].sum()) if lumped is not None else np.nan
            out.append((seg, lo, min(hi, s.max()), mw(m0) if m0 > 0 else None))
        return out

    vmax = float(np.nanpercentile(mag, 99.8))
    cut = zmax is not None
    # a cut model's panels are shallower: shorter rows, or the equal-aspect panels float in empty space
    fig = plt.figure(figsize=(17, 8.6 if cut else 11))
    gs = fig.add_gridspec(3, 3, height_ratios=[0.55, 0.8, 1.0] if cut else [1.0, 1.0, 0.95],
                          width_ratios=[1, 1, 1.05], hspace=0.62 if cut else 0.55, wspace=0.25,
                          top=0.92 if cut else 0.88)
    ax_s = fig.add_subplot(gs[0, :])
    ax_n = fig.add_subplot(gs[1, :2])
    ax_na = fig.add_subplot(gs[2, 0])
    ax_ns = fig.add_subplot(gs[2, 1])
    ax_map = fig.add_subplot(gs[1:, 2])

    titles = {
        "south": f"South branch (event 1): Mw {moments['south']['Mw']:.2f}",
        "north": f"North branch (event 2): Mw {moments['north']['Mw']:.2f}",
        "narli": f"Narlı fault (event 1): Mw {moments['narli']['Mw']:.2f}",
        "ns": f"N-S strand (event 2): Mw {moments['ns']['Mw']:.2f}",
    }
    for name, ax in (("south", ax_s), ("north", ax_n), ("narli", ax_na), ("ns", ax_ns)):
        a, b, t = sheet(FAULTS.index(name))
        s, z = params[a:b, 0] / 1e3, -params[a:b, 1] / 1e3
        tr = mtri.Triangulation(s, z, t)
        if zmax is not None:
            tr.set_mask(z[t].min(axis=1) >= zmax - 0.5)          # elements wholly at or below the cut carry no slip
        pc = ax.tripcolor(tr, mag[a:b], shading="gouraud", cmap="magma_r", vmin=0, vmax=vmax)
        ax.tricontour(tr, mag[a:b], levels=np.arange(1, vmax, 1.0), colors="k", linewidths=0.4)
        if show_mesh:
            ax.triplot(tr, color="0.45", lw=0.3, alpha=0.8)
        zbot = zmax if zmax is not None else z.max()
        ax.set_ylim(zbot, 0)
        ax.set_xlim(s.min(), s.max())
        ax.set_aspect("equal")
        ax.set_xlabel("along strike (km)")
        ax.set_ylabel("depth (km)")
        peak = float(mag[a:b].max())
        sep = "\n" if cut and name in ("narli", "ns") else ", "      # a cut model's small panels: titles wrap
        ax.set_title(f"{titles[name]}{sep}peak {peak:.1f} m", fontsize=11, loc="left")
        if name in SEGMENTS:
            for label, v in boundaries(name):
                ax.axvline(v, color="w", lw=1.6, ls="--")
                ax.axvline(v, color="k", lw=0.8, ls="--")
            for seg, lo, hi, m in segment_mw(name):
                txt = f"{seg}" + (f"\nMw {m:.2f}" if m else "")
                ax.text(0.5 * (lo + hi), zbot * 0.93, txt, ha="center", va="bottom", fontsize=9.5,
                        color="k", bbox=dict(boxstyle="round,pad=0.25", fc="w", ec=COLORS[seg], lw=1.8))
        else:
            ax.text(0.03, 0.06, "Narlı" if name == "narli" else "N-S strand", transform=ax.transAxes, fontsize=9.5,
                    bbox=dict(boxstyle="round,pad=0.25", fc="w",
                              ec=COLORS["Narlı" if name == "narli" else "N-S strand"], lw=1.8))
    # The colourbar sits in the south panel's empty top-right corner (the Pütürge stretch carries no slip); a cut model's
    # panel is too shallow for it beside the Pütürge label, so it goes to the map's empty north-west corner instead.
    from mpl_toolkits.axes_grid1.inset_locator import inset_axes
    if cut:
        cax = fig.add_axes([0.80, 0.958, 0.13, 0.013])         # the figure's empty top-right corner
    else:
        cax = inset_axes(ax_s, width="16%", height="7%", loc="upper right", borderpad=1.2)
    cb = fig.colorbar(pc, cax=cax, orientation="horizontal")
    cb.set_label("slip (m)", fontsize=9)
    cb.ax.tick_params(labelsize=8)

    # map: each sheet's trace coloured by segment, the same names, the epicentres
    lines = geometry.traces()
    for name in FAULTS:
        lon, lat = lines[name][:, 0], lines[name][:, 1]
        x, y = frame.to_local(lon, lat)
        sa = np.r_[0, np.cumsum(np.hypot(np.diff(x), np.diff(y)))] / 1e3
        if name in SEGMENTS:
            edges = [0.0] + [v for _, v in boundaries(name)] + [np.inf]
            for seg, lo, hi in zip(SEGMENTS[name], edges[:-1], edges[1:]):
                sel = (sa >= lo - 1.5) & (sa <= hi + 1.5)
                ax_map.plot(lon[sel], lat[sel], "-", color=COLORS[seg], lw=3.2, solid_capstyle="round")
                mid = np.flatnonzero(sel)[len(np.flatnonzero(sel)) // 2]
                off = {"Amanos": (-0.45, 0.0), "Pazarcık": (-0.32, 0.12), "Erkenek": (0.15, -0.22),
                       "Pütürge (gap)": (0.2, -0.32), "Çardak": (-0.1, 0.18), "NFC": (0.05, 0.16)}[seg]
                ax_map.text(lon[mid] + off[0], lat[mid] + off[1], seg, fontsize=9.5, color=COLORS[seg],
                            fontweight="bold", ha="center")
        else:
            seg = "Narlı" if name == "narli" else "N-S strand"
            ax_map.plot(lon, lat, "-", color=COLORS[seg], lw=3.2)
            ax_map.text(lon.mean() + (0.3 if name == "narli" else -0.42), lat.mean() - (0.05 if name == "narli" else 0.12),
                        seg, fontsize=9.5,
                        color=COLORS[seg], fontweight="bold", ha="center")
    if show_mesh and zmax is not None:
        # each sheet's bottom edge at zmax, mapped: where a dipping sheet sits at depth
        for k, name in enumerate(FAULTS):
            a, b, _ = sheet(k)
            row = np.flatnonzero(np.abs(np.abs(mesh.nodes[a:b, 2]) / 1e3 - zmax) < 0.5)
            if not row.size:
                continue
            row = row[np.argsort(params[a:b, 0][row])]
            top_ = np.flatnonzero(np.abs(mesh.nodes[a:b, 2]) < 1.0)
            top_ = top_[np.argsort(params[a:b, 0][top_])]
            s_row, s_top = params[a:b, 0][row], params[a:b, 0][top_]
            dx = mesh.nodes[a:b][row, 0] - np.interp(s_row, s_top, mesh.nodes[a:b][top_, 0])
            dy = mesh.nodes[a:b][row, 1] - np.interp(s_row, s_top, mesh.nodes[a:b][top_, 1])
            if np.hypot(dx, dy).max() < 500.0:          # a vertical sheet: its bottom edge is under its trace
                continue
            blon, blat = frame.to_lonlat(mesh.nodes[a:b][row, 0], mesh.nodes[a:b][row, 1])
            ax_map.plot(blon, blat, "--", color="0.35", lw=1.0, zorder=4, label=f"{name} sheet at {zmax:g} km depth")
        ax_map.legend(loc="lower right", fontsize=8.5, framealpha=0.9)
    for label, lon, lat in EPICENTRES:
        ax_map.plot(lon, lat, "*", ms=18, mfc="gold", mec="k", mew=1.0, zorder=5)
        ax_map.text(lon + 0.08, lat + 0.07, label, fontsize=8.5, zorder=6)
    ax_map.set_xlim(35.9, 39.4)
    ax_map.set_ylim(36.0, 38.6)
    ax_map.set_aspect(1 / np.cos(np.radians(37.3)))
    ax_map.grid(alpha=0.3)
    ax_map.set_xlabel("longitude (°E)")
    ax_map.set_ylabel("latitude (°N)")
    ax_map.set_title(f"Event 1 Mw {moments['e1']['Mw']:.2f}   Event 2 Mw {moments['e2']['Mw']:.2f}",
                     fontsize=11, loc="left")

    if "title" in opts:
        fig.suptitle(opts["title"], fontsize=13, x=0.42, y=0.985)
    out = run / opts.get("out", "slip_map.png")
    fig.savefig(out, dpi=120, bbox_inches="tight")
    print(out)


if __name__ == "__main__":
    main()
