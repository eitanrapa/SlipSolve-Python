"""The slip on each sheet of a run's model, along strike against depth, with each sheet's Mw.

    python plot_slip.py runs/prod [out=slip.png]
"""
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from slipsolve import SlipModel  # noqa: E402


def main():
    run = Path(sys.argv[1])
    opts = dict(a.split("=", 1) for a in sys.argv[2:] if "=" in a)
    model = SlipModel.load(run / "model.slip.zip")
    summary = json.loads((run / "summary.json").read_text())
    names = [s["name"] for s in summary["sheets"]]
    slip = np.hypot(model.strike_slip, model.dip_slip)            # per node
    top = float(np.max(slip))

    sheets = model.mesh.sheets()
    span = max(np.ptp(m.params[:, 0]) for m in sheets) / 1e3        # km, the longest sheet
    depths = [-m.params[:, 1].min() / 1e3 for m in sheets]
    stretch = max(1, round(span / (5 * max(depths))))                 # vertical exaggeration for a long fault
    width = 9.0                                                       # inches for the longest sheet
    heights = [width * stretch * d / span for d in depths]
    fig, axes = plt.subplots(len(sheets), 1, figsize=(width + 0.8, sum(heights) + 0.5 * len(sheets) + 1.3),
                             squeeze=False, layout="constrained", gridspec_kw=dict(height_ratios=heights))
    start = 0
    for ax, mesh, name in zip(axes[:, 0], sheets, names):
        nodes = slice(start, start + len(mesh.nodes))
        start += len(mesh.nodes)
        s, z = mesh.params[:, 0] / 1e3, mesh.params[:, 1] / 1e3
        coll = ax.tripcolor(s, z, mesh.triangles, slip[nodes], shading="gouraud", cmap="magma_r", vmin=0, vmax=top)
        ax.set_xlim(s.min(), s.min() + span)
        ax.set_ylim(z.min(), 0.0)
        ax.set_aspect(stretch)
        ax.set_ylabel("depth (km)" if stretch == 1 else f"depth (km, ×{stretch})")
        ax.set_title(f"{name}: Mw {summary['Mw'][name]:.2f}, peak {slip[nodes].max():.1f} m", loc="left", fontsize=10)
    axes[-1, 0].set_xlabel("distance along strike (km)")
    fig.colorbar(coll, ax=axes[:, 0].tolist(), label="slip (m)", orientation="horizontal", shrink=0.4, aspect=30)
    fig.suptitle(f"Mw {summary['Mw']['total']:.2f}, rank {summary['rank']:.0f}", x=0.02, ha="left")
    out = run / opts.get("out", "slip.png")
    fig.savefig(out, dpi=120, bbox_inches="tight")
    print(out)


if __name__ == "__main__":
    main()
