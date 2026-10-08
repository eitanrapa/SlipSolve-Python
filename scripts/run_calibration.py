"""Stage 0: what wavelength can this data resolve, and at what rank and budget?

    cd ~/Documents/slipsolve-python
    KMP_DUPLICATE_LIB_OK=TRUE \
    nohup ~/miniforge3/envs/remote_sensing/bin/python -u scripts/run_calibration.py \
          > workdir/calibration.log 2>&1 &
    tail -f workdir/calibration.log

Drives :mod:`slipsolve.calibrate` on the configured scenes. Plants a checkerboard
of known slip, forward-models it through the *real* look geometry and coverage,
adds noise drawn from each scene's own fitted covariance, and inverts. What comes
back is the rung that minimises error against a truth that is known, which is a
different question from the L-curve's -- and measured, a different answer:
on the six-scene geometry the corner sat at rank 21 while error against truth was
still falling at 79, a 35% error reduction the corner leaves on the table.

    stage 0  discretisation ceiling        pure geometry, seconds
    stage 1  wavelength x rank ladder      one budget, one layout, one seed
    stage 2  seeds at the best wavelength  is that optimum a realisation?
    stage 3  budget x layout at that cell  are they second-order, as assumed?

Run this BEFORE ``run_lcurve.py`` if the question is "how many parameters should
I be solving for"; run the L-curve if the question is "what weight on this mesh".
They are not substitutes.

Writes, as each loop finishes, so a killed run keeps what it did:

    model_sampling/calibration.csv        one row per rank rung
    model_sampling/calibration_plan.json  the search, and what stage 0 dropped
    model_sampling/covariance.json        the fitted per-scene noise

Resumable: rows already in the CSV are skipped by case name. Pass a stage number
to run only that stage (``run_calibration.py 3``).

⚠️ **Cost.** One loop is one ``iterate_sampling``; the rank rungs after it
re-solve on a factorisation that already exists and are nearly free. Measured on
seven real scenes at budget 660, a loop was 1500-1700 s of which all six rungs
were ~60 s. The staged plan is 14 loops where a full factorial would be 90, so
budget most of a day and expect stage 1 to answer in about an hour.

⚠️ **Every number here is an upper bound if the variograms are still rising at
maximum lag**, which on real scenes they usually are. The fitted sill is then a
lower bound on the noise, the synthetic worlds are quieter than the real one, and
the best rank is optimistic.
"""

import json
import sys
import time

import pandas as pd

from slipsolve import calibrate
from slip_config import (
    BOUNDS, CALIBRATION_BUDGETS, CALIBRATION_CELLS, CALIBRATION_LAYOUTS,
    CALIBRATION_MAX_ITER, CALIBRATION_MAX_ROUNDS, CALIBRATION_RANKS,
    CALIBRATION_SEEDS, CALIBRATION_SIZE_MAX, OUT_DIR, banner,
    calibration_covariances, geometry, geometry_kwargs, gridded_scenes,
    inversion_kwargs, mesh_summary, sampling_parameters, workspace,
)

TABLE = OUT_DIR / "calibration.csv"


def _append_csv(frame, path):
    """Append rows to ``path``, reconciling the header if the schema has moved.

    ⚠️ ``to_csv(path, mode="a", header=not path.exists())`` writes a header only
    for a new file, so a run whose columns have CHANGED appends wide rows under a
    narrow header and the file stops parsing --- ``Expected 21 fields, saw 27``.
    That happened on 2026-09-08 when ``run_case`` gained six ``*_tent`` scoring
    columns and this driver appended to a table written in September. The rows
    were fine; only the header was wrong, and the failure surfaced an hour later
    in ``report()`` rather than at the append.

    Reconciling costs one read of a file that is a few tens of kB.
    """
    import pandas as pd

    if path.exists():
        old = pd.read_csv(path)
        if list(old.columns) != list(frame.columns):
            pd.concat([old, frame], ignore_index=True).to_csv(path, index=False)
            return
    frame.to_csv(path, mode="a", header=not path.exists(), index=False)


def main():
    only = int(sys.argv[1]) if len(sys.argv) > 1 else None
    started = time.time()

    banner("stage 0: calibration")
    ws = workspace()
    trace, frame, mesh = geometry()
    print(f"    {trace!r}\n    {mesh!r}\n    {mesh_summary(mesh)}\n", flush=True)

    gridded = gridded_scenes(ws, frame)
    sampling = sampling_parameters(gridded, trace, frame, mesh)
    print("\nfitting per-scene noise:", flush=True)
    covariances = calibration_covariances(gridded, trace, frame)

    plan = calibrate.calibration_plan(
        mesh, cells=CALIBRATION_CELLS, budgets=CALIBRATION_BUDGETS,
        layouts=CALIBRATION_LAYOUTS, ranks=CALIBRATION_RANKS,
        seeds=CALIBRATION_SEEDS)
    _report_ceiling(plan)
    (OUT_DIR / "calibration_plan.json").write_text(json.dumps(
        {k: v for k, v in plan.items() if k != "cases"}, indent=2))

    # ⚠️ `size_max` is not optional. Left to its default the size field caps
    # growth at `current.max() * max_ratio` -- measured, 21.6 km on a 40 km fault
    # -- and the loop spends its budget along strike, collapsing the depth levels
    # under the very checkerboard it is scoring. `run_case` rechecks the ceiling
    # on the adapted mesh, so a collapse is recorded either way, but the cap is
    # what prevents it.
    common = dict(
        scenes=gridded, mesh0=mesh, trace=trace, frame=frame,
        sample_kwargs=sampling, inversion_kwargs=inversion_kwargs(),
        solve_kwargs={**BOUNDS, "max_iter": CALIBRATION_MAX_ITER},
        geometry=geometry_kwargs(trace, frame), covariances=covariances,
        max_rounds=CALIBRATION_MAX_ROUNDS,
        mesh_adapt={"size_max": CALIBRATION_SIZE_MAX},
    )

    finished = _done()
    cell = None
    for stage in (1, 2, 3):
        if only is not None and stage != only:
            continue
        cases = [c for c in plan["cases"] if c.stage == stage]
        if stage > 1:
            cell = best_cell() if cell is None else cell
            if cell is None:
                print(f"\nstage {stage}: nothing from stage 1 to pick a wavelength "
                      "from -- run stage 1 first", flush=True)
                continue
            cases = [type(c)(**{**c.__dict__, "cell": cell}) for c in cases]

        print(f"\n=== stage {stage}: {len(cases)} loop(s) ===", flush=True)
        for case in cases:
            if case.name in finished:
                print(f"  {case.name}: done already, skipping", flush=True)
                continue
            rows = calibrate.run_case(case, **common)
            for row in rows:
                row["case"] = case.name
            _append(rows)

    print(f"\ntotal {time.time() - started:.0f}s", flush=True)
    report()


def _report_ceiling(plan):
    """Stage 0, which runs inside :func:`calibration_plan` rather than as a loop."""
    print("\n=== stage 0: discretisation ceiling (on the STARTING mesh) ===",
          flush=True)
    for row in plan["ceiling"]:
        print("  %5.0f km  along %4.2f  down %4.2f  amplitude %.3f  corr %.3f  %s"
              % (row["cell"] / 1e3, row["samples_per_cell"],
                 row["samples_down_dip"], row["amplitude_retained"],
                 row["pattern_correlation"],
                 "USABLE" if row["usable"] else "DROPPED"), flush=True)
    if plan["dropped_cells"]:
        print(f"  dropped: {[c / 1e3 for c in plan['dropped_cells']]} km -- the "
              "mesh cannot carry them, so a failure there would say nothing about "
              "the data", flush=True)
    print(f"\n{plan['n_loops']} loops / {plan['n_solves']} solves "
          f"(a full factorial would be {plan['factorial_loops']} loops)", flush=True)
    print("  ⚠️ this ceiling is the starting mesh's. Each case adapts away from it "
          "and rechecks; read `ceiling_usable` per row.", flush=True)


def _done():
    """Case names already in the table, so a killed run resumes."""
    if not TABLE.exists():
        return set()
    return set(pd.read_csv(TABLE)["case"].unique())


def _append(rows):
    frame = pd.DataFrame(rows)
    _append_csv(frame, TABLE)


def best_cell():
    """Stage 1's best-*resolved* wavelength: the smallest cell that cleared the bar.

    ⚠️ **Snap UP, never to the nearest rung.**
    :func:`~slipsolve.calibrate.resolving_wavelength` interpolates in log cell and
    returns the crossing, which generally falls *between* two planted rungs --
    measured, 25.7 km between an unresolved 20 km (peak correlation 0.354) and a
    resolved 40 km (0.759). Snapping to the nearest picks 20 km and spends stages
    2 and 3 planting a checkerboard the data cannot recover, which is the one
    thing a resolution ladder must not do. It cost 4.7 h of loops before it was
    noticed, and nothing in the output says it happened.
    """
    if not TABLE.exists():
        return None
    stage1 = pd.read_csv(TABLE).query("stage == 1")
    if stage1.empty:
        return None

    peak = stage1.groupby("cell")["correlation"].max()
    cleared = [float(c) for c in sorted(peak.index)
               if peak[c] >= calibrate.RESOLVED_CORRELATION]
    crossing = calibrate.resolving_wavelength(stage1.to_dict("records"))
    if cleared:
        print(f"  resolving wavelength "
              + (f"{crossing / 1e3:.1f} km" if crossing else "not bracketed")
              + f"; smallest planted cell clearing "
              f"{calibrate.RESOLVED_CORRELATION} is {cleared[0] / 1e3:.0f} km",
              flush=True)
        return cleared[0]

    cell = float(peak.idxmax())
    print(f"  ⚠️  nothing cleared {calibrate.RESOLVED_CORRELATION}; falling back to "
          f"the best-correlated cell, {cell / 1e3:.0f} km (peak {peak.max():.3f}). "
          "The ladder does not bracket this data -- plant longer wavelengths.",
          flush=True)
    return cell


def _rung(table):
    """Label each row with the rank rung it was *aiming* at.

    ⚠️ Rows carry the rank they *achieved*, and that is not repeatable across
    cases: the weight is interpolated onto the requested rung, so a different
    mesh or a different noise seed lands at 9.70 where another landed at 9.82.
    Grouping or pivoting on the raw value therefore splits one rung into one row
    per case and silently undoes the averaging it was meant to do -- five seeds
    become five groups of one.
    """
    import numpy as np

    rungs = np.asarray(CALIBRATION_RANKS, dtype=float)
    achieved = table["rank"].to_numpy(dtype=float)
    return rungs[np.abs(np.log(achieved[:, None]) - np.log(rungs)).argmin(axis=1)]


def report():
    if not TABLE.exists():
        return
    table = pd.read_csv(TABLE)
    table["rung"] = _rung(table)

    stage1 = table.query("stage == 1")
    if not stage1.empty:
        print("\n=== stage 1: correlation on the oscillating part ===")
        print(stage1.pivot_table(index="rung", columns="cell",
                                 values="correlation").to_string())
        print("\n    relative error against truth:")
        print(stage1.pivot_table(index="rung", columns="cell",
                                 values="rel_error").to_string())
        crossing = calibrate.resolving_wavelength(stage1.to_dict("records"))
        print("\n    resolving wavelength: "
              + (f"{crossing / 1e3:.1f} km" if crossing
                 else "NONE cleared the threshold"))

    stage2 = table.query("stage == 2")
    if not stage2.empty:
        # Seed 1 lives in stage 1, so the realisations are pooled across the two
        # -- but only at the wavelength stage 2 actually planted, since stage 1
        # planted several and they are not comparable to each other.
        cell = stage2["cell"].iloc[0]
        pooled = table.query("stage <= 2 and cell == @cell")
        summary = pooled.groupby("rung")["rel_error"].agg(["mean", "std", "count"])
        print(f"\n=== stage 2: relative error at {cell / 1e3:.0f} km over "
              f"{int(summary['count'].max())} noise realisation(s) ===")
        print(summary.to_string())
        best = summary["mean"].idxmin()
        floor = summary["mean"].min() + summary["std"].fillna(0.0).max()
        plateau = summary.index[summary["mean"] <= floor].tolist()
        print(f"\n    best rank {best:.0f}; within one seed sd of it: "
              f"{[f'{r:.0f}' for r in plateau]}")
        print("    prefer the LOW end of that plateau -- the same error for fewer "
              "effective parameters, and an understated noise level biases the "
              "apparent optimum upward.")

    stage3 = table.query("stage == 3")
    if not stage3.empty:
        # `calibration_plan` skips the reference budget-and-layout in stage 3
        # because stage 1 already ran it -- so it has to be folded back in here,
        # or the grid has a hole in exactly the cell every other one is being
        # compared against.
        cell = stage3["cell"].iloc[0]
        grid = pd.concat([stage3,
                          table.query("stage == 1 and cell == @cell and seed == 1")])
        print("\n=== stage 3: budget x layout ===")
        print(grid.pivot_table(index="rung", columns=["budget", "layout"],
                               values="rel_error").to_string())
        print("\n    ⚠️ margins here are often 0.5-2 sd of the seed scatter stage 2 "
              "measured. Read a consistent SIGN across budgets, not one cell.")

    _warn_about(table, "converged", False, ["case", "rank"],
                "hit the iteration cap and score their own non-convergence. "
                "Raise NISAR_CAL_MAX_ITER and re-run those cases")
    _warn_about(table, "ceiling_usable", False, ["case", "rank", "ceiling_amplitude"],
                "were solved on a mesh that cannot carry their wavelength, so "
                "their failure is discretisation and not resolution. Lower "
                "NISAR_CAL_SIZE_MAX or raise the budget")


def _warn_about(table, column, bad, show, why):
    """Call out rows that scored something other than the data, if any did.

    Guarded on the column existing: ``ceiling_usable`` only arrived once the
    ceiling started being rechecked on the adapted mesh, and a table written
    before that is still worth reporting on -- minus this check, which it has no
    answer for.
    """
    if column not in table:
        print(f"\n    (no {column!r} column -- this table predates that check)")
        return
    rows = table[table[column] == bad]
    if len(rows):
        print(f"\n⚠️  {len(rows)} row(s) {why}:")
        print(rows[show].to_string(index=False))


if __name__ == "__main__":
    main()
