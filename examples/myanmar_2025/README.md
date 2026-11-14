# Slip model of the 2025 Mandalay (Myanmar) earthquake

This folder solves, from public data, the slip model of the 28 March 2025 Mw 7.7 earthquake on the Sagaing fault shown
at <https://eitanrapa.github.io/research/slip-models/#myanmar-2025>. The rupture is about 500 km long. The model has
one fault surface that changes dip along strike, and a rigidity contrast across the fault, solved exactly in a last
stage.

1. `python get_data.py` downloads the data (a 740 MB archive, about 3 minutes) and checks it.
2. `python run.py` runs the inversion: about 2¾ hours on 6 cores.
3. `python plot_slip.py runs/prod` draws the slip.

## 1. Software

- **slipsolve** from this repository: `pip install -e .` in a clone, or
  `pip install "slipsolve-python @ git+https://github.com/eitanrapa/SlipSolve-Python.git"`. Its dependencies cover
  everything the scripts use. The [Production guide](https://eitanrapa.github.io/SlipSolve-Python/guide/production/)
  describes the runner's stages and settings, including the curved sheet and the correction stage used here.
- **Hardware.** We ran on a 10-core Intel iMac Pro with 128 GB, with `workers=6`. The Green's functions run in
  `workers=` processes (default 5). Disk: 0.8 GB of data, plus about 7 GB of caches, mostly `runs/prod/g_cache/`,
  which can be deleted afterwards.

## 2. Data

`python get_data.py` downloads everything below into this folder. It then checks every file against `SHA256SUMS`, the
checksums of the files we used; `python get_data.py --check` only verifies. All sources are public.

| data | source | files |
|---|---|---|
| ALOS-2 interferograms, ascending track 152 (2025-02-11 → 04-08) and descending track 41 (02-16 → 03-30): unwrapped phase, coherence and look vectors | Xu et al. (2025), Zenodo, [doi:10.5281/zenodo.17113746](https://doi.org/10.5281/zenodo.17113746), CC BY 4.0 (`LOS_DEFO.zip`) | `xu2025/`: 10 products, 60 files |
| Sentinel-1 range and azimuth pixel offsets, tracks A070, A143, D033, D106, with their σ | the same | |
| the rupture trace, mapped from Sentinel-2 north-south offsets | He & Yun (2025), Zenodo, [doi:10.5281/zenodo.17595200](https://doi.org/10.5281/zenodo.17595200), CC BY 4.0 | `trace/` |
| crust | CRUST1.0 (Laske et al. 2013), [igppweb.ucsd.edu/~gabi/crust1.html](https://igppweb.ucsd.edu/~gabi/crust1.html) | `crust1.0/` |

- The products are ISCE-2 geocoded binaries, read through their `.rsc` headers (`products.py`). Their `.vrt` files name
  files that are not in the archive.
- The ascending interferogram (A152) is corrected for the troposphere and tides but not the ionosphere; the
  descending one is.
- Signs: each product's `enu.geo` points along a positive value. For phase and range offsets it runs from the
  satellite to the ground (the value is a range increase), so both are negated to give a line of sight positive toward
  the satellite. We checked every product's sign against Xu et al.'s own 3-D displacement field.
- The products are on one 10″ grid, 0.8 GB unpacked.
- Xu et al.'s archive also holds LuTan-1 range and azimuth offsets (track A052), which the model leaves out. Each fit
  worse than any Sentinel-1 series of its kind (rms 104 mm in range, 293 mm in azimuth), and a run with them gives the
  same slip (Mw 0.0001 higher).

## 3. Running

    python run.py [stages...] [name=prod] [workers=5]

Run it in this folder, after `get_data.py`. It:
1. reads the products and resamples them onto one 500 m lattice, each cell the median of its pixels (`work/`, kept for
   later runs);
2. builds the crust profile and the layered Green's-function tables on its first run (`crust1_myanmar.json`,
   `remainder_myanmar_crust1.npz`; a few minutes);
3. hands the fault, data and recipe to `slipsolve.production.Production`, which writes to `runs/<name>/` and logs to
   `runs/<name>/run.log`.

The stages, each checkpointed (an interrupted run resumes, and `python run.py C D E X report` re-runs from C). The
times are from our run.

| stage | what | time |
|---|---|---|
| A | a 5 km start mesh; the first samples drawn from the data alone; rank 80 | 4.5 min (and 2 min before it, the first time: the lattice and the layered tables) |
| B | the mesh adapted to the resolution until the element count is stable | 27 min |
| C0 | samples redrawn from the model until stable (4 rounds at most) | 22 min |
| C | the rank, by 24-slab blocked cross-validation | 5 min |
| D | the mesh re-adapted at that rank: the final mesh, and the model without the contrast | 18 min |
| E | cross-validation on the final mesh: is the rank inside its 1% band? | 6.5 min |
| X | the final mesh and samples solved again with the contrast (`contrast.py`) at the final rank, and cross-validated: **the final model** | 73 min, 66 of them the contrast's correction to the Green's functions |
| report | `model.slip.zip`, `model_uncorrected.slip.zip`, `summary.json` | seconds |

In our run (2 h 38 min in all) the mesh settled in B (6 maps, 2164 elements) and D (3 maps, 2445), and the samples in
C0's second round.

Then `python plot_slip.py runs/prod` writes `runs/prod/slip.png`.

**Outputs** in `runs/prod/`:
- `model.slip.zip` is the model with the contrast; `model_uncorrected.slip.zip` is the same mesh and samples without
  it. Both open with `slipsolve.SlipModel.load`, and carry the mesh, the slip and the samples.
- `summary.json` holds Mw, the rank and λ, the fit and row count of each data series, and the recipe.
- `converge_B.csv` and `converge_D.csv` follow the mesh adaptation, `sampling_C0.csv` the resampling, and `cv_C.csv`,
  `cv_E.csv` and `cv_X.csv` are the cross-validation curves.

**What to expect.** Our run gave Mw 7.904 (7.895 without the contrast, in `model_uncorrected.slip.zip`), at CV rank
90 (1% band 62.9–113.5; stage X's corrected system alone would choose 89.4, inside it), with 2445 elements, 22,821
samples and a peak slip of 5.7 m. The fits (rms): ALOS-2 21 and 25 mm; Sentinel-1 range offsets 55–57 mm and azimuth
offsets 140–198 mm. This is the model on the website, which quotes Mw 7.88: the moment with the two blocks'
harmonic-mean rigidity above 10 km. To get it, in this folder:

```python
from slipsolve import SlipModel
import contrast, run                     # this folder's modules

model = SlipModel.load("runs/prod/model.slip.zip")
model.options["bimaterial"] = {"contrast": contrast.softer_side(run.fault_trace()).to_dict()}
print(model.moment_magnitude)            # 7.879
```

## 4. What the model is

**Fault.** One sheet along the rupture trace, 30 km deep, with zero slip on its bottom and ends.
- The trace is He & Yun's (506 km), with its vertices thinned to about 1 km apart.
- The surface is `FaultMesh.curved`'s, fitted on the trace resampled at 4 km, with one dip for each eighth of the trace.
  South to north: 90 90 95 100 110 120 120 120 degrees, where a dip above 90 leans east. The south is vertical and the
  north half dips 60–70° east.
- Right-lateral: strike-slip of one sign, up to 20 m; dip-slip within ±5 m.

**Data and noise.**
- *Phase.* Masked within 5 km of the trace and below coherence 0.3. Each interferogram has a cubic ramp, and each
  connected region of at least 100 km² its own offset. Cells split while the model varies by more than one σ.
- *Offsets.* Used from 2 km of the trace, each series with a plane. Cells split while the model varies by more than
  half a σ.
  - The first samples are drawn from the data alone: cells of at least 6 km, split while their scatter exceeds 0.6 m
    (range) or 2.4 m (azimuth), weighted by the median of each product's own σ layer.
- *σ* per series is the rms of its whole residual, re-measured from the model each sampling round.
- No GNSS.

**Rigidity contrast (stage X).** Xu et al. (2025) measured a softer west side (the Central Myanmar Basin): Vs in the
top 10 km about 2.9 km/s west of the fault and 3.3 km/s east of it, a rigidity ratio of about 1.3. Stage X solves the
final mesh and samples again, at the final rank, with the west block 1.3 times softer than the 1-D crust above 10 km,
solved exactly by `slipsolve.bimaterial` on the layered crust (`contrast.py`). The report quotes that model.

**Forward model.** A layered elastic half-space: CRUST1.0 averaged over the six 1° cells the trace crosses.

| layer top | Vs | μ |
|---|---|---|
| 0 km | 3.49 km/s | 33.0 GPa |
| 15.0 km | 3.63 km/s | 36.5 GPa |
| 26.9 km | 3.93 km/s | 45.3 GPa |
| 35.0 km (mantle) | 4.45 km/s | 65.5 GPa |

- Slip is piecewise linear over triangles (nodal), computed by slipsolve's `LayeredTDE`.
- The moment is integrated over this rigidity: the east block's. The website quotes it with the two blocks'
  harmonic-mean rigidity above 10 km, 0.025 lower in Mw (see *What to expect*).

**Inversion.**
- Laplacian smoothing, chosen by blocked cross-validation over 24 along-strike slabs of equal width across all the
  data (35 km each), held-out residuals weighted by σ. Stages A and B work at rank 80; stage C chooses the rank.
- Meshes are unstructured, from 5 km start meshes. Element sizes are 0.35 of the local resolving width, within
  1.5–10 km.

**Fixed choices and where they came from.** The geometry, the depth, the ramps and the contrast are fixed in `run.py`.
They come from cross-validation tests on earlier, fixed-mesh layered models:
- *Dips.* A sweep of one dip per eighth of the trace, scored by 24-slab CV with each arm at its own rank. The north
  half wants about 64° E (held-out misfit 7.2% below a vertical fault), the south a vertical fault. The east dip comes
  from the azimuth offsets: 2–8 km from the trace between 20.8 and 22.6° N, the east side moved 1.7–2.5 times as far
  along the fault as the west side, in three Sentinel-1 tracks.
- *Depth.* The CV curve by fault bottom is lowest at 20–25 km and 0.6–0.7% higher at 30 km. A model planted with a
  30 km bottom gives the same curve: cross-validation reads a deep bottom about one step shallow here. The model uses
  30 km.
- *Ramps.* Cubic ramps on the interferograms improve the held-out misfit by 1.4% over planes; planes on the offsets
  by 2.6% over constants.
- *Contrast.* The measured contrast improves the held-out misfit by 0.5%: consistent with the data, a prior rather than
  a requirement.

Those tests need further scripts and intermediate runs, and are not part of this example.

**Not modelled.** No near-trace (shallow slip deficit) test; no correlated noise (the weights are diagonal); no
postseismic correction (the pairs end 2–11 days after the earthquake).

## 5. Files

| file | |
|---|---|
| `get_data.py`, `SHA256SUMS` | the download and its checksums |
| `run.py` | the model: fault, data, settings; runs slipsolve's production recipe |
| `products.py` | reads Xu et al.'s products |
| `contrast.py` | the rigidity contrast's correction to the Green's functions (stage X) |
| `crust.py` | the CRUST1.0 profile and the layered tables |
| `plot_slip.py` | the slip on the fault |

## 6. References

- He, L., & Yun, S.-H. (2025). Surface rupture trace of the 2025 Myanmar earthquake from Sentinel-2 derived
  north-south displacement map. Zenodo, doi:10.5281/zenodo.17595200.
- Laske, G., Masters, G., Ma, Z., & Pasyanos, M. (2013). Update on CRUST1.0, a 1-degree global model of Earth's crust.
  *Geophys. Res. Abstracts*, 15, EGU2013-2658.
- Xu, L., et al. (2025). Bimaterial effect and favorable energy ratio enabled supershear rupture in the 2025 Mandalay
  earthquake. *Science*, doi:10.1126/science.ady6100.
- Xu, L., Zhang, Y., Yang, Y., Wang, Y., & Hu, C. (2025). Geodetic deformation, DPM products, and source imaging
  results in "Bimaterial effect and favorable energy ratio enabled supershear rupture in the 2025 Mandalay
  earthquake". Zenodo, doi:10.5281/zenodo.17113746.
