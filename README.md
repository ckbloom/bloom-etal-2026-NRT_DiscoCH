# DiscoCH 2026 Operational

Near-real-time mapping of early leaf discoloration in Swiss forests (mainly European beech) from Sentinel-2.
Each run turns FORCE vegetation-index time series into a probability of discoloration for every 10 m pixel,
on a 5-day grid of dates through the season. It then masks the results to forest, validates them against
mapped polygons, and summarizes them nationally, per district, and on hexagon maps. Every output can be
compared with the 2018 drought year.

## Workflow

```
FORCE level-3 TSS (raw Sentinel-2 VIs, one band per acquisition)
   │  transfer.py            copy the year's tiles from the network share
   ▼
   │  interpolate.py         ensemble RBF filter -> regular 5-day series (*_TSS_TSI.tif)
   ▼                         near-real-time: each date uses data up to date + lag_days
   │  apply_model.py         running seasonal min/max normalization of 5 VIs
   ▼                         -> pickled sklearn model -> Disco_Proba_<date>.tif per tile + mosaic
   │  mask.py                reproject to EPSG:2056, blank outside the forest mask, int16 scaled
   ▼
masked Disco_Proba_<date>.tif   (stored 0–10000, band scale 0.0001 → probability 0–1)
   │
   ├─ validate.py            pixel-level metrics against 1/0 ground-truth polygons, threshold sweep
   ├─ QAQC.py                stratified random sampling + design-based accuracy (Olofsson 2014, Stehman 2014)
   ├─ disco_statistics.py    discolored share and hectares per date, national and per zone
   ├─ compare.py             seasonal curves of several years on one month/day axis
   ├─ hexmap.py              month-end hexagon maps, years × months
   └─ plot_date_grid.py      every date of a season on the swisstopo grey map
```

### Model inputs

The model uses five VIs: **NDVI, EVI, NDMI, CIRE (CIre), CCI**. For each pixel and date, each VI is scaled to
`(value − min) / (max − min)`, where min and max are taken over the season from `start_date` up to and including
that date. Nothing after the date is used. The model runs only in `model_months` (default June–September).
The feature names follow the pre-trained pipeline (`MODEL_FEATURES` in `apply_model.py`).

File names abbreviate VIs differently from year to year (e.g. 2026 `CIre`, 2018 `CRE`/`NDV`/`NDM`).
Pass a `vi_keys` dict that maps each canonical name to the token used in the file names.

### Thresholds and scaling

Probability rasters store values from 0 to 10000. Thresholds in `validate.py`, `disco_statistics.py`,
`compare.py` and `hexmap.py` use that stored scale, so `4000` means 40 %. `QAQC.py` works in probabilities
from 0 to 1 (`scale=10000`).

## Repository layout

```
data/
  CH_FORCE_Grids.csv                                  FORCE tiles covering Switzerland (Tile_ID, Tile_X, Tile_Y)
  DiscoCH_2026_Mask.tif                               forest mask, EPSG:2056 (1 = keep)
  empirical_discoloration_model_pipeline_2025_6_2.pkl pre-trained sklearn pipeline (not tracked in git)
  example_extents/                                    example area shapefile (Herbetswil)
notebooks/
  DiscoCH.ipynb                    operational run: transfer → interpolate → model → mask → validate → stats → maps
  DiscoCH_NRT_vs_Full_Data.ipynb   rerun a finished season with all data and compare with the NRT series
src/disco_ch_operational/          current pipeline modules (see below)
src/dep/                           superseded scripts kept for reference (rbf.py, apply_discoch.py, ...)
output/                            run outputs (not tracked)
scratch/                           ad-hoc scripts
```

### Modules (`src/disco_ch_operational/`)

| Module | Main entry points | Purpose |
|---|---|---|
| `transfer.py` | `transfer()` | Resumable copy of files whose names match a string, keeping the folder structure |
| `interpolate.py` | `interpolate_tiles()`, `run_full_data_season()` | Ensemble RBF interpolation to a 5-day series. NRT mode with `as_of` / `lag_days`, appends only new dates |
| `apply_model.py` | `apply_discoch_tiles()`, `plot_probability()` | Stepwise normalization, model prediction and mosaicking. Resumable through a min/max cache |
| `mask.py` | `mask_and_scale_dir()` | Reproject to the mask grid (EPSG:2056), mask, write scaled int16 |
| `mask_builder.py` | `vi_date_tiles()`, `snapshot_tiles()`, `mosaic_and_mask()`, `difference()` | Single-date VI / min-max layers as masked Swiss mosaics, and year-to-year differences |
| `train_model.py` | `train_discoch_svc()`, `plot_cv()` | Train SVC or logistic regression on labelled polygons, with nested spatial block CV |
| `validate.py` | `validate_threshold()`, `validate_thresholds()` | Confusion matrix and metrics (accuracy, F1, kappa, IoU, ...) against polygons |
| `QAQC.py` | `qaqc_sample()`, `qaqc_assess()`, `qaqc_report()`, `polygon_recall()` | Stratified sampling of probability rasters and design-based accuracy and area estimates |
| `disco_statistics.py` | `summarize_discoloration()`, `discoloration_stats()` | Per-date discolored share and hectares, total and per zone (chunked, multithreaded) |
| `compare.py` | `plot_year_comparison()` | Multi-year seasonal curves, overlaid or one panel per region |
| `hexmap.py` | `plot_hex_maps()` | Hexagon maps over a relief basemap, with GeoPackage export |
| `plot_date_grid.py` | `plot_date_grid()` | Every date of a season as a map grid on the swisstopo WMS |
| `visualize_interpolation.py` | `plot_pixel()`, `effective_bandwidth_stats()` | Single-pixel diagnostics of the RBF interpolation |
| `config.py` | `load_config()` | Reads the machine-specific paths from `config.yaml` |
| `utils.py` | — | Shared helpers: tile discovery, band dates, scaled-int I/O, mosaicking, run logs |

Each module has a docstring with a full usage example at the top.

## Setup

Create the conda environment from `environment.yml` (conda-forge, Python ≥ 3.11):

```bash
conda env create -f environment.yml
conda activate discoch
```

The pre-trained model is a pickled scikit-learn pipeline. If loading it gives version warnings or errors,
pin `scikit-learn` in `environment.yml` to the version the model was saved with.

### Configure paths for your machine

Paths that depend on the machine (FORCE data, validation polygons, boundaries, the 2018 mosaics) are kept in
`config.yaml` at the project root. Git ignores this file. Copy the template and fill in your own paths:

```bash
cp config.example.yaml config.yaml
```

Relative paths are resolved from the project root, so the defaults for `data/` and `output/` work as they are.
Leave an entry empty if you don't have that dataset. The step that needs it will stop with a message naming
the missing key. To use a different config file, for example on an operational server, set the environment
variable `DISCOCH_CONFIG` to its path.

The notebooks and scripts read the config like this:

```python
from src.disco_ch_operational.config import load_config

cfg = load_config()
force_directory = cfg["force_dir"]
districts = cfg.require("districts")   # stops with an error if it isn't set
```

### Imports

The modules are imported as `src.disco_ch_operational.<module>`, so the project root must be on `sys.path`.
The notebooks use paths relative to `notebooks/`, so start Jupyter in `notebooks/` and either mark the project
root as a source root in your IDE (PyCharm does this) or add it yourself:

```python
import sys, os; sys.path.insert(0, os.path.abspath(".."))
```

## Running the operational pipeline

Open `notebooks/DiscoCH.ipynb`, set the run parameters in the first cell (year, the date to run to,
the threshold), and run the cells in order. Paths come from `config.yaml`. The core calls are:

```python
from datetime import date
from src.disco_ch_operational.config import load_config
from src.disco_ch_operational.interpolate import interpolate_tiles
from src.disco_ch_operational.apply_model import apply_discoch_tiles
from src.disco_ch_operational.mask import mask_and_scale_dir

cfg = load_config()
vi_keys = {"NDVI": "NDVI", "EVI": "EVI", "NDMI": "NDMI", "CIRE": "CIre", "CCI": "CCI"}

interpolate_tiles(tss_dir=cfg["force_dir"], out_dir=rbf_output, vi_keys=vi_keys,
                  date_range=(date(2026, 1, 1), date(2026, 12, 31)),
                  as_of=date.today(), lag_days=5, rbf_sigma=(10, 20, 30),
                  tiles_csv=cfg["tiles_csv"], max_workers=10)

apply_discoch_tiles(rbf_dir=rbf_output, out_dir=discoCH_output, year=2026,
                    model_path=cfg["model_path"], vi_keys=vi_keys, tiles_csv=cfg["tiles_csv"],
                    model_months=(6, 7, 8, 9), export_minmax=True, max_workers=10)

mask_and_scale_dir(input_dir=f"{discoCH_output}/mosaic", mask_path=cfg["mask_path"],
                   output_dir=f"{discoCH_output}/masked", pattern="Disco_Proba_*.tif")
```

All three steps can be run again. Interpolation only appends dates that have become eligible, the model
picks up after the last processed date, and masking redoes an output only when its input is newer. In
operation, run them daily with `as_of=date.today()`.

### Outputs

Under `output/DiscoCH_rbf_10_20_30/`, named after the RBF sigmas:

```
level4_rbf/<year>/<tile>/*_<VI>_TSS_TSI.tif   interpolated 5-day VIs (int16, nodata −9999)
<year>/<tile>/Disco_Proba_<date>.tif          per-tile probability
<year>/mosaic/Disco_Proba_<date>.tif          national mosaic
<year>/masked/Disco_Proba_<date>.tif          forest-masked, EPSG:2056, the product to use
<year>/minmax/, minmax_by_date/, vi_norm/     normalization cache and optional per-date snapshots
<year>/validation/, statistics/               metrics, CSV tables, figures
<year>_full_data/                             retrospective all-data rerun (DiscoCH_NRT_vs_Full_Data.ipynb)
```

## Data not in the repository

- FORCE level-3 TSS time series (network share, copied locally with `transfer()`)
- Validation polygons (e.g. `Mapping_2026.gpkg`, with `Discolored` = 1/0 and one layer per survey)
- District boundaries (`fgeolib_geolib_g1b20_LV95.shp`, field `bzname`) and biogeographic regions
- 2018 discoloration mosaics (Bloom et al.), used for the year comparisons

Set their locations in `config.yaml` (see [Configure paths](#configure-paths-for-your-machine)).

## Contact

Colin Bloom, WSL, colin.bloom@wsl.ch
