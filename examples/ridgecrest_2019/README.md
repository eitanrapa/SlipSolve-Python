# Slip model of the 2019 Ridgecrest (California) earthquakes

This folder solves, from public data, the slip model of the 4 July 2019 Mw 6.4 and 6 July Mw 7.1 Ridgecrest
earthquakes shown at <https://eitanrapa.github.io/research/slip-models/#ridgecrest-2019>. Every data set spans both
shocks, so they are solved in one inversion.

1. `python get_data.py` downloads the data (170 MB, about a minute) and checks it.
2. `python run.py` runs the inversion: about 11 minutes on 5 cores.
3. `python plot_slip.py runs/prod` draws the slip.

## 1. Software

- **slipsolve** from this repository: `pip install -e .` in a clone, or
  `pip install "slipsolve-python @ git+https://github.com/eitanrapa/SlipSolve-Python.git"`. Its dependencies cover
  everything the scripts use. The [Production guide](https://eitanrapa.github.io/SlipSolve-Python/guide/production/)
  describes the runner's stages and settings.
- **A program that opens RAR archives**, for the Sentinel-2 offsets: `bsdtar` (macOS's `tar` is one; on Linux the
  `libarchive-tools` package, or `conda install -c conda-forge libarchive`), `unar`, `7z` or `unrar`.
- **Hardware.** We ran on a 6-core Intel Mac mini with 64 GB. The Green's functions run in `workers=` processes
  (default 5). Disk: 170 MB of data and about 120 MB of caches, mostly `runs/prod/g_cache/`, which can be deleted
  afterwards.

## 2. Data

`python get_data.py` downloads everything below into this folder. It then checks every file against `SHA256SUMS`, the
checksums of the files we used; `python get_data.py --check` only verifies. All sources are public.

| data | source | files |
|---|---|---|
| InSAR line of sight: ALOS-2 ascending tracks 65 and 66, Sentinel-1 ascending track 64 and descending track 71, with per-point look vectors | Xu, Sandwell & Smith-Konter (2020), from UCSD: [topex.ucsd.edu/SV_7.1](https://topex.ucsd.edu/SV_7.1/) (`SV_los_original.zip`) | `ucsd/`: 4 files |
| Sentinel-2 east and north offsets, 2019-06-28 → 07-08, COSI-Corr at 20 m, detrended by the author | Valkaniotis (2019), Zenodo, [doi:10.5281/zenodo.3275073](https://doi.org/10.5281/zenodo.3275073), CC BY 4.0 | `s2/`: 2 files |
| GNSS: the daily series of every station within 120 km of the epicentres, and the offsets computed from them | Nevada Geodetic Laboratory (Blewitt et al. 2018), IGS20 `tenv3` series | `gnss/` |
| crust | CRUST1.0 (Laske et al. 2013), [igppweb.ucsd.edu/~gabi/crust1.html](https://igppweb.ucsd.edu/~gabi/crust1.html) | `crust1.0/` |

- **GNSS offsets.** For each station, the median of days +1 to +5 after the Mw 7.1's day minus the median of the 10
  days before the Mw 6.4's day; the origin days are left out. σ is the two windows' robust scatter (1.4826 MAD over
  √n) in quadrature. Stations with fewer than 4 days in either window are left out: 64 remain. The offsets include
  about 5 days of postseismic motion.
  - The series are live files that grow every day, so `SHA256SUMS` checks only the offsets computed from them
    (`gnss/ngl_offsets.csv`). If NGL reprocesses its solutions, the check reports a difference and the run uses the
    new values.
- **Fault traces** are in `traces.json`, as we drew them:
  - the Mw 7.1's fault: the NW-striking lines of Valkaniotis's Sentinel-2 trace (Zenodo 3275073,
    `Fault_Trace_S2.kmz`), joined into one line;
  - the Mw 6.4's fault: the NE-striking ruptures of the Mw 6.4 longer than 200 m in Ponti et al.'s map (2020,
    [doi:10.5066/P9BZ5IJ9](https://doi.org/10.5066/P9BZ5IJ9)), joined into one line.

  To join the pieces, their vertices are projected on the pieces' principal axis; each 1 km bin along it takes the
  median offset across it, a 3-bin running mean smooths the line, and the mapped ends are kept.

## 3. Running

    python run.py [stages...] [name=prod] [workers=5]

Run it in this folder, after `get_data.py`. It:
1. reads the data and resamples it onto one 500 m lattice, each cell the median of its pixels (`work/`, kept for
   later runs);
2. builds the crust profile and the layered Green's-function tables on its first run (`crust1_ridgecrest.json`,
   `remainder_ridgecrest_crust1.npz`; a few minutes);
3. hands the faults, data and recipe to `slipsolve.production.Production`, which writes to `runs/<name>/` and logs to
   `runs/<name>/run.log`.

The stages, each checkpointed (an interrupted run resumes, and `python run.py C D E report` re-runs from C). The times
are from our test run.

| stage | what | time |
|---|---|---|
| A | 3 km start meshes; the first samples drawn from the data alone; rank 40 | 1 min (and 1.5 min before it, the first time: the layered tables) |
| B | meshes adapted to the resolution until the element count is stable | 4 min |
| C0 | samples redrawn from the model until stable (4 rounds at most) | 2.5 min |
| C | the rank, by 12-slab blocked cross-validation | 0.5 min |
| D | meshes re-adapted at that rank: **the final model** | 1.5 min |
| E | cross-validation on the final mesh: is the rank inside its 1% band? | 0.5 min |
| report | `model.slip.zip`, `summary.json` | seconds |

In our test run (11 min in all) the meshes settled in B (6 maps, 943 elements) and D (2 maps), and the samples in
C0's third round.

Then `python plot_slip.py runs/prod` writes `runs/prod/slip.png`.

**Outputs** in `runs/prod/`:
- `model.slip.zip` opens with `slipsolve.SlipModel.load`. It carries the mesh, the slip and the samples.
- `summary.json` holds Mw per sheet, the rank and λ, the fit and row count of each data series, and the recipe.
- `converge_B.csv` and `converge_D.csv` follow the mesh adaptation, `sampling_C0.csv` the resampling, and `cv_C.csv`
  and `cv_E.csv` are the cross-validation curves.

**What to expect.** Our test run gave Mw 7.137 in all, 7.097 on the Mw 7.1's sheet and 6.540 on the Mw
6.4's, at CV rank 45 (1% band 42.6–47.7), with 946 elements, 1101 samples and a peak slip of 6.2 m. The fits (rms):
ALOS-2 14 and 18 mm, Sentinel-1 17 and 27 mm, GNSS 4.5 mm, Sentinel-2 131 mm east and 189 mm north. The model on the
website is Mw 7.135, at rank 44.6. It was made by our earlier scripts, which put the data on the lattice with GDAL's
median: that takes the lower of the two middle pixels in a cell with an even count, and the symmetric median here moved
Mw by 0.0015.

## 4. What the model is

**Faults.** Two vertical sheets, 20 km deep, with zero slip on their bottoms and ends.
- The Mw 7.1's fault, 50.7 km long, drawn NW to SE.
- The Mw 6.4's NE-striking fault, 23.3 km long, drawn SW to NE. It crosses the Mw 7.1's fault near 117.52° W,
  35.68° N.
- The Mw 6.4 also broke NW-striking faults, in short scattered pieces. They are not a sheet: their slip lies on the
  Mw 7.1's fault, and data spanning both shocks cannot tell the two apart. The Mw 7.1 sheet's moment includes it.

**Data and noise.**
- *InSAR.* Masked within 2 km of the traces. Each ALOS-2 track has a cubic ramp and each Sentinel-1 track a plane;
  each connected region of at least 100 km² has its own offset. The quadtree cells split while the model varies by more
  than one σ.
- *Sentinel-2 offsets.* 500 m block medians, used from 2 to 30 km of the traces, a plane each (the author removed a
  trend). Cells split while the model varies by more than half a σ.
- *σ* per series is the rms of its whole residual, re-measured from the model each sampling round.
- *GNSS.* East, north and up, σ × 3 and at least 2 cm, used as they are.

**Forward model.** A layered elastic half-space: CRUST1.0 at the one 1° cell the traces cross (35.5° N, 117.5° W).

| layer top | Vs | μ |
|---|---|---|
| 0 km | 3.55 km/s | 34.5 GPa |
| 10.7 km | 3.65 km/s | 37.0 GPa |
| 21.4 km | 3.60 km/s | 37.1 GPa |
| 32.3 km (mantle) | 4.32 km/s | 59.7 GPa |

- Slip is piecewise linear over triangles (nodal), computed by slipsolve's `LayeredTDE`.
- The moment is integrated over this rigidity.

**Inversion.**
- Laplacian smoothing, chosen by blocked cross-validation over 12 along-strike slabs of equal width across all the
  data, held-out residuals weighted by σ. Stages A and B work at rank 40; stage C chooses the rank.
- Bounds: strike-slip and dip-slip ±20 m and ±15 m, either sign.
- Meshes are unstructured, from 3 km start meshes. Element sizes are 0.35 of the local resolving width, within
  1.5–10 km.

**Not modelled.** No near-trace (shallow slip deficit) test; no correlated noise (the weights are diagonal); no
postseismic correction (the SAR pairs and the GNSS offsets include some days after the Mw 7.1).

## 5. Files

| file | |
|---|---|
| `get_data.py`, `SHA256SUMS` | the download and its checksums |
| `run.py` | the model: faults, data, settings; runs slipsolve's production recipe |
| `products.py` | reads the UCSD sets and the Sentinel-2 offsets |
| `crust.py` | the CRUST1.0 profile and the layered tables |
| `traces.json` | the two fault traces |
| `plot_slip.py` | the slip on both sheets |

## 6. References

- Blewitt, G., Hammond, W. C., & Kreemer, C. (2018). Harnessing the GPS data explosion for interdisciplinary science.
  *Eos*, 99.
- Laske, G., Masters, G., Ma, Z., & Pasyanos, M. (2013). Update on CRUST1.0, a 1-degree global model of Earth's crust.
  *Geophys. Res. Abstracts*, 15, EGU2013-2658.
- Ponti, D. J., Pickering, A. J., Blair, J. L., Rosa, C. M., Thomas, K., & Dawson, T. (2020). Digital datasets
  documenting surface fault rupture and ground deformation features produced by the Ridgecrest M6.4 and M7.1
  earthquake sequence of July 4 and 5, 2019. U.S. Geological Survey data release, doi:10.5066/P9BZ5IJ9.
- Valkaniotis, S. (2019). Subpixel optical correlation co-seismic offsets for the Mw 6.4 and Mw 7.1 Ridgecrest,
  California earthquakes, from Copernicus Sentinel 2 data. Zenodo, doi:10.5281/zenodo.3275073.
- Xu, X., Sandwell, D. T., & Smith-Konter, B. (2020). Coseismic displacements and surface fractures from Sentinel-1
  InSAR: 2019 Ridgecrest earthquakes. *Seismological Research Letters*, 91(4), 1979–1985, doi:10.1785/0220190275.
