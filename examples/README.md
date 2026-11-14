# Examples

Three earthquakes solved end to end from public data with slipsolve's production recipe (`slipsolve.production`):
the slip models shown at <https://eitanrapa.github.io/research/slip-models/>. Each folder downloads its own data,
checks it against the checksums of the files we used, runs the inversion and draws the result. Its README describes
the model, the data and what the run should give.

| folder | earthquake | what it shows | data | run |
|---|---|---|---|---|
| [`ridgecrest_2019`](ridgecrest_2019/) | 2019 Ridgecrest, California: Mw 6.4 and 7.1 | two crossing faults; InSAR, optical offsets and GNSS together | 170 MB | 11 min |
| [`myanmar_2025`](myanmar_2025/) | 2025 Mandalay, Myanmar: Mw 7.7 | a 500 km fault with a dip per segment (`CurvedSheet`); SAR pixel offsets; a rigidity contrast across the fault, added to the Green's functions in a last stage (`correction=`) | 740 MB, 0.8 GB kept | 2¾ h |
| [`turkiye_2023`](turkiye_2023/) | 2023 Kahramanmaraş, Türkiye: Mw 7.8 and 7.7 | two earthquakes in one inversion: four faults, one dipping, each earthquake's GNSS seeing only its own faults (`sees=`, `events=`) | 3.9 GB | 2¼ h |

The run times are from a 6-core Intel Mac mini, Myanmar's from a 10-core Intel iMac Pro. Ridgecrest is the quickest
place to start.

Each one runs the same way, in its own folder:

    python get_data.py              # download the data and check it
    python run.py                   # the inversion, into runs/prod/ (run.log there follows it)
    python plot_slip.py runs/prod   # the slip (Türkiye: plot_slip_map.py)

They need slipsolve from this repository: `pip install -e .` in a clone, or
`pip install "slipsolve-python @ git+https://github.com/eitanrapa/SlipSolve-Python.git"`. Its dependencies cover the
scripts; Ridgecrest's Sentinel-2 offsets also need a program that opens RAR archives (`bsdtar`, `unar`, `7z` or
`unrar`). Each run is checkpointed stage by stage, so an interrupted run resumes where it stopped.
