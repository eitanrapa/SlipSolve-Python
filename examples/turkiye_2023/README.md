# Slip model of the 2023 Kahramanmaraş (Türkiye) earthquakes

This folder solves, from public data, the slip model of the 6 February 2023 Mw 7.8 and Mw 7.7 earthquakes shown at
<https://eitanrapa.github.io/research/slip-models/#turkiye-2023>. Both events are solved in one inversion.

1. `python get_data.py` downloads the data (3.9 GB) and checks it.
2. `python run.py` runs the inversion: about 2¼ h on 6 cores.
3. `python plot_slip_map.py runs/prod` draws the slip.

## 1. Software

- **slipsolve** from this repository: `pip install -e .` in a clone, or
  `pip install "slipsolve-python @ git+https://github.com/eitanrapa/SlipSolve-Python.git"`. Its dependencies cover
  everything the scripts use. The [Production guide](https://eitanrapa.github.io/SlipSolve-Python/guide/production/)
  describes the runner's stages and settings.
- **Hardware.** We ran on a 6-core Intel Mac mini with 64 GB.
  - The Green's functions and surface fields run in `workers=` processes (default 6).
  - Disk: 3.9 GB of data, plus about 4 GB of caches. Most of that is `runs/prod/g_cache/`, which can be deleted
    afterwards.

## 2. Data

`python get_data.py` downloads everything below into this folder. It then checks every file against `SHA256SUMS`, the
checksums of the files we used; `python get_data.py --check` only verifies. All sources are public. In our test the
whole download took about 10 minutes.

| data | source | files |
|---|---|---|
| Sentinel-1 range and azimuth pixel offsets, tracks 014A, 021D, 116A, 123D, with per-pixel E/N/U coefficients and empirical σ | Ou et al. (2023), *3D Displacements and Strain from the 2023 February Turkey Earthquakes*, version 1, CEDA, [doi:10.5285/df93e92a3adc46b9a5c4bd3a547cd242](https://doi.org/10.5285/df93e92a3adc46b9a5c4bd3a547cd242) ([files](https://data.ceda.ac.uk/neodc/comet/publications_data/Turkey_earthquake/v1.0)), Open Government Licence v3 | `ceda_ou2024/inputs/`: 40 files, 2.9 GB |
| Sentinel-1 interferograms (unwrapped phase, coherence) and look vectors, 6 frames | COMET LiCSAR (Lazecký et al. 2020), from CEDA's mirror ([data.ceda.ac.uk/neodc/comet/data/licsar_products](https://data.ceda.ac.uk/neodc/comet/data/licsar_products)) | `licsar/<frame>/`: 30 files, 1.0 GB |
| GNSS coseismic offsets, one file per event | Nevada Geodetic Laboratory: [us6000jllz_final5min.txt](https://geodesy.unr.edu/news_items/20230213/us6000jllz_final5min.txt) (Mw 7.8) and [us6000jlqa_final5min.txt](https://geodesy.unr.edu/news_items/20230213/us6000jlqa_final5min.txt) (Mw 7.7) | `gnss/` |
| fault traces | USGS simplified rupture traces v1.1 (Reitman et al. 2023, [doi:10.5066/P985I7U2](https://doi.org/10.5066/P985I7U2)) | `traces/simple_fault/` |
| fault east of the rupture (Pütürge) | GEM Global Active Faults (Styron & Pagani 2020), [GitHub](https://github.com/GEMScienceTools/gem-global-active-faults), commit `850fd05` | `traces/gem/` |
| crust | CRUST1.0 (Laske et al. 2013), [igppweb.ucsd.edu/~gabi/crust1.html](https://igppweb.ucsd.edu/~gabi/crust1.html) | `crust1.0/` |

The pairs:

| track | offsets (CEDA) | interferogram (LiCSAR frame) |
|---|---|---|
| 014A | 2023-01-28 → 02-09 | 014A_05232_242525, 01-28 → 02-09 |
| 021D | 01-29 → 02-10 | 021D_05266_252525, 01-29 → 02-10 |
| 116A | 02-04 → **02-28** | 116A_05207_252525, 02-04 → 02-16 |
| 123D | 02-05 → 02-17 | 123D_05292_242525, 02-05 → 02-17 |
| 094D | — | 094D_05288_130913 and 094D_05100_131313, 02-03 → 02-15 |

- The 116A offsets span the 20 February Mw 6.4 near Antakya. Their pixels within 35 km of its epicentre are left
  out.
- The CEDA GeoTIFFs carry no CRS. Their transforms are plain longitude/latitude at 0.001°, and
  `slipsolve.readers` takes a raster without a CRS as longitude/latitude.
- Use `https://` for the GNSS files. The `http://` address currently redirects to a broken host name.

## 3. Running

    python run.py [stages...] [name=prod] [workers=6]

Run it in this folder, after `get_data.py`. It:
1. reads the rasters and resamples them onto one 250 m lattice, each cell the median of its pixels (`work/`, kept
   for later runs; a few minutes);
2. builds the crust profile and the layered Green's-function tables on its first run (`crust1_turkiye.json`,
   `remainder_turkiye_crust1.npz`; a few minutes);
3. hands the faults, data and recipe to `slipsolve.production.Production`, which writes to `runs/<name>/` and logs to
   `runs/<name>/run.log`.

The stages, each checkpointed (an interrupted run resumes, and `python run.py C D E report` re-runs from C). The times
are from our test run.

| stage | what | time |
|---|---|---|
| A | 4 km start meshes; the first samples drawn from the data alone; rank 78 | 3 min (and 5 min before it, the first time: every lattice pixel's distance to the traces) |
| B | meshes adapted to the resolution until the element count is stable | 7 min |
| C0 | samples redrawn from the model until stable (4 rounds at most) | 76 min |
| C | the rank, by 24-slab blocked cross-validation | 6 min |
| D | meshes re-adapted at that rank: **the final model** | 16 min |
| E | cross-validation on the final mesh: is the rank inside its 1% band? | 5 min |
| report | `model.slip.zip`, `summary.json` | seconds |

In our test run (2 h in all) the samples were still changing after C0's fourth round (7% of the rows), and the
meshes settled in B and D. Most of the run shared the machine with another job, so the times are upper bounds.

Then `python plot_slip_map.py runs/prod` writes `runs/prod/slip_map.png` (`mesh=1` adds the element edges).

**Outputs** in `runs/prod/`:
- `model.slip.zip` opens with `slipsolve.SlipModel.load`. It carries the mesh, the slip and the samples.
- `summary.json` holds Mw per event and per sheet, the rank and λ, the fit and row count of each data series, and the
  recipe.
- `converge_B.csv` and `converge_D.csv` follow the mesh adaptation, `sampling_C0.csv` the resampling, and `cv_C.csv`
  and `cv_E.csv` are the cross-validation curves.

**What to expect.** Our test run gave event 1 Mw 7.838 and event 2 Mw 7.688, at CV rank 77 (1% band 59–108), with
2356 elements, 13,413 samples and a peak slip of 8.4 m. The model on the website is Mw 7.838 and 7.689, at rank 97.5.
It was made by our earlier scripts, which started from an intermediate run's samples rather than from the data, and
which put the data on the lattice with GDAL's median: that takes the lower of the two middle pixels in a cell with an
even count, and the symmetric median here moved either event's Mw by 0.001. The cross-validation curve is flat near its
minimum: across the four runs we made of this recipe the rank went from 77 to 98, and either event's Mw moved by 0.003
at most.

## 4. What the model is

**Faults.** Four sheets, all 26 km deep with zero slip on their bottoms and ends.
- The south sheet (Amanos–Pazarcık–Erkenek) is extended east along the GEM fault through Pütürge. It is vertical.
- The north sheet (Çardak) is extended east into the Nurhak complexity by three points, the medians of the Lomax (2023)
  aftershocks. It dips 65° N: each node at depth d is moved d·cot 65° along the trace's normal, smoothed over 10 km.
- The Narlı splay and the N–S strand at 36.73° E are vertical.

**Events.** Every SAR pair spans both earthquakes, but the GNSS resolves them. Event 1's stations see only the south
and Narlı sheets, and event 2's only the north sheet and the N–S strand. This is exact if no sheet slipped in both.

**Data and noise.**
- *First samples.* Drawn from the data alone. Offsets: cells of at least 6 km, split while a cell's scatter exceeds
  0.6 m (range) or 2.4 m (azimuth), weighted by σ 0.15 m and 0.6 m. Phase: split above 2 cm, σ 2 cm.
- *Offsets.* Quadtree cells take the median, with a 2 km floor, and split while the model varies by more than half a
  σ. Pixels within 2 km of a trace are left out.
  - σ per series is the rms of its whole residual, re-estimated each sampling round.
  - Each series has one constant as its nuisance term.
- *Phase.*
  - Masked within 15 km of every trace and below coherence 0.25.
  - Cells split while the model varies by more than one σ.
  - Each connected region of at least 100 km² has its own offset, and each frame a linear ramp.
  - LOS = −range change (positive toward the satellite), measured against the model.
- *GNSS.* Horizontals only, σ × 3, from stations 2–300 km from a trace. CYL2 is dropped from both events, and EKZ1
  (on the Çardak trace) from event 2.

**Forward model.** A layered elastic half-space: CRUST1.0 averaged over the eight 1° cells the traces cross.

| layer top | Vs | μ |
|---|---|---|
| 0 km | 3.52 km/s | 33.7 GPa |
| 19.0 km | 3.68 km/s | 37.8 GPa |
| 27.9 km | 3.82 km/s | 41.6 GPa |
| 37.6 km (mantle) | 4.47 km/s | 66.2 GPa |

- Slip is piecewise linear over triangles (nodal), computed by slipsolve's `LayeredTDE`.
- The moment is integrated over this rigidity.

**Inversion.**
- Laplacian smoothing, chosen by blocked cross-validation over 24 along-strike slabs of equal width across all the
  data (about 36 km each), held-out residuals weighted by σ.
- Bounds: strike-slip left-lateral, 0–20 m; dip-slip ±5 m.
- Meshes are unstructured. Element sizes are 0.35 of the local resolving width, within 1.5–10 km.

**Fixed choices and where they came from.** The geometry, the depth and the starting rank are fixed in `run.py`.
They come from cross-validation tests on an earlier, fixed-mesh layered model:
- *North dip.* CV over 50–90° N has its minimum at ~65° N (70° N is 1.0% worse), between the Çardak aftershock clusters
  (~61°) and the Nurhak fit (77° N).
- *South sheet.* Vertical on all three segments, within 0.2% of the best arm. Amanos at 70–80° W ties.
- *Depth.* CV by fault bottom is 1.982 / 1.836 / **1.812** / 1.860 at 15 / 20 / 26 / 35 km.
  - In planted tests (the real residuals as noise) the ladder follows slip at depth.
  - The real data's penalty for going to 35 km (+2.6%) matches a planted model with no slip at or below 20 km, not
    the planted deep cases (+0.8–1.4%).
- *Starting rank.* 78 is the CV rank of a fixed-mesh model with these choices.

Those tests need further scripts and intermediate runs, and are not part of this example.

**Not modelled.**
- No near-trace (shallow slip deficit) test.
- No correlated noise: the weights are diagonal.
- No postseismic correction. The SAR pairs end 3–22 days after the events, and the GNSS offsets are 5-minute
  solutions.

## 5. Files

| file | |
|---|---|
| `get_data.py`, `SHA256SUMS` | the download and its checksums |
| `run.py` | the model: faults, data, settings; runs slipsolve's production recipe |
| `crust.py` | the CRUST1.0 profile and the layered tables |
| `geometry.py` | the four traces |
| `plot_slip_map.py` | the slip maps |

`run.py` uses only slipsolve's public functions.

## 6. References

- Lazecký, M., et al. (2020). LiCSAR: an automatic InSAR tool for measuring and monitoring tectonic and volcanic
  activity. *Remote Sensing*, 12, 2430.
- Laske, G., Masters, G., Ma, Z., & Pasyanos, M. (2013). Update on CRUST1.0, a 1-degree global model of Earth's crust.
  *Geophys. Res. Abstracts*, 15, EGU2013-2658.
- Lomax, A. (2023). Precise, NLL-SSST-coherence hypocenter catalog for the 2023 Mw 7.8 and Mw 7.6 SE Turkey earthquake
  sequence. Zenodo, doi:10.5281/zenodo.7699882.
- Ou, Q., Lazecký, M., Watson, C. S., Maghsoudi, Y., & Wright, T. (2023). 3D displacements and strain from the 2023
  February Turkey earthquakes, version 1. NERC EDS Centre for Environmental Data Analysis,
  doi:10.5285/df93e92a3adc46b9a5c4bd3a547cd242.
- Reitman, N. G., et al. (2023). Fault rupture mapping of the 6 February 2023 Kahramanmaraş, Türkiye, earthquake
  sequence from satellite data (v1.1, February 2024). U.S. Geological Survey data release, doi:10.5066/P985I7U2.
- Styron, R., & Pagani, M. (2020). The GEM Global Active Faults Database. *Earthquake Spectra*, 36(1_suppl), 160–180.
- Nevada Geodetic Laboratory, University of Nevada, Reno: coseismic offsets of events us6000jllz and us6000jlqa
  (geodesy.unr.edu).
