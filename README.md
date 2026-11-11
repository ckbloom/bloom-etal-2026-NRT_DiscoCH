# Near-real-time monitoring of early leaf discoloration signals unprecedented drought stress in Switzerland’s European beech forests

Colin K. Bloom, Dominique Weber, Marius Rütschi, Tiziana L. Koch, Frank Krumm, Katrin Meusburger, Andri Baltensweiler

This repository contains the analysis code for the manuscript above. The pre-trained early leaf discoloration
model of Bloom et al. (2026) was applied to a near-real-time interpolated Sentinel-2 time series 
to identify early leaf discoloration in Swiss European beech forests every five-days throughout the 2026 growing season.

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
masked DiscoCH_<RUN>_<date>.tif   (e.g. DiscoCH_NRT_2026-08-14.tif; stored 0–10000, band scale 0.0001 → probability 0–1)
   │
   ├─ QAQC.py                point validation: stratified random sample + design-based accuracy (Olofsson 2014,
   │                         Stehman 2014); recall against opportunistically mapped polygons
   ├─ validate.py            pixel-level metrics against 1/0 ground-truth polygons, threshold sweep
   ├─ disco_statistics.py    discolored share and hectares per date, national and per zone
   ├─ clear_sky.py           cumulative clear-sky EVI observations per date (raw TSS, up to date + lag), uint8 maps, mean ± SD plot
   ├─ compare.py             seasonal curves of several years on one month/day axis
   ├─ hexmap.py              month-end hexagon maps, months × years
   └─ plot_date_grid.py      every date of a season on the swisstopo grey map
```

Two runs of a season go through these steps:

- **Near real time (NRT):** each date *d* is interpolated from the observations up to *d* + `lag_days` (5) and
  produced once *d* + `lag_days` ≤ the date the pipeline is run to. Each date's min/max is the running min/max of
  the NRT series from 1 January up to *d*.
- **Retrospective (Retro):** after the season, each date is interpolated from all observations within the RBF
  kernel, before and after it, and normalized with the running min/max of the retrospective series from 1 January
  up to *d*. A third variant (**Retro_NRTminmax**) uses the all-data interpolation with the NRT run's per-date
  min/max, so only the interpolation differs from NRT.

### Model inputs

The model uses five VIs: **NDVI, EVI, NDMI, CIRE (CIre), GRVI (CCI)**. For each pixel and date, each VI is scaled to
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
  DiscoCH_NRT.ipynb                near-real-time run: transfer → interpolate → model → mask → stats → clear sky → maps
  DiscoCH_Retro.ipynb              retrospective run of a finished season, same steps, all-data interpolation
  DiscoCH_Retro_NRTminmax.ipynb    retrospective interpolation with the NRT min/max (reproduces the earlier
                                   "full data" results)
  NRT_vs_Retro.ipynb               point validation and mapping-recall tables, NRT vs Retro time series,
                                   per-region comparison, single-pixel interpolation example
  dep/                             superseded notebooks
src/disco_ch_operational/          current pipeline modules (see below)
src/dep/                           superseded scripts kept for reference (rbf.py, apply_discoch.py, ...)
output/                            run outputs (not tracked)
scratch/                           ad-hoc scripts
```

### Modules (`src/disco_ch_operational/`)

| Module | Main entry points | Purpose |
|---|---|---|
| `transfer.py` | `transfer()` | Resumable copy of files whose names match a string, keeping the folder structure |
| `interpolate.py` | `interpolate_tiles()`, `check_data_currency()`, `run_full_data_season()` | Ensemble RBF interpolation to a 5-day series. NRT mode with `as_of` / `lag_days`, appends only new dates |
| `apply_model.py` | `apply_discoch_tiles()`, `plot_probability()` | Stepwise normalization, model prediction and mosaicking. Resumable through a min/max cache |
| `mask.py` | `mask_and_scale_dir()` | Reproject to the mask grid (EPSG:2056), mask, write scaled int16 |
| `mask_builder.py` | `vi_date_tiles()`, `snapshot_tiles()`, `mosaic_and_mask()`, `difference()` | Single-date VI / min-max layers as masked Swiss mosaics, and year-to-year differences |
| `train_model.py` | `train_discoch_svc()`, `plot_cv()` | Train SVC or logistic regression on labelled polygons, with nested spatial block CV |
| `validate.py` | `validate_threshold()`, `validate_thresholds()` | Confusion matrix and metrics (accuracy, F1, kappa, IoU, ...) against polygons |
| `QAQC.py` | `qaqc_sample()`, `qaqc_assess()`, `polygon_recall()`, `point_validation_table()`, `recall_table()` | Stratified sampling of probability rasters, design-based accuracy and area estimates, polygon recall; the paper's tables for several maps |
| `disco_statistics.py` | `summarize_discoloration()`, `discoloration_stats()` | Per-date discolored share and hectares, total and per zone (chunked, multithreaded) |
| `clear_sky.py` | `count_clear_sky_tiles()`, `mosaic_clear_sky()`, `summarize_clear_sky()`, `plot_clear_sky()` | Cumulative clear-sky observations from the raw EVI tiles on the 5-day NRT dates: uint8 tiles, masked uint8 COG mosaics, mean/SD per date and zone, national + per-region plot |
| `compare.py` | `plot_year_comparison()` | Multi-year seasonal curves: overlaid, one panel per region, or national + per region (`layout="summary"`) |
| `hexmap.py` | `plot_hex_maps()` | Hexagon maps (months × years) over the swisstopo relief, with GeoPackage export |
| `plot_date_grid.py` | `plot_date_grid()` | Every date of a season as a map grid on the swisstopo WMS |
| `visualize_interpolation.py` | `plot_pixel()`, `effective_bandwidth_stats()` | Single-pixel diagnostics of the RBF interpolation |
| `config.py` | `load_config()`, `run_dirs()` | Reads the machine-specific paths from `config.yaml`; the output folders of each run |
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

## Running the pipelines

Each notebook starts with a cell of run parameters; paths come from `config.yaml`. Run the cells in order.

**Near real time — `DiscoCH_NRT.ipynb`.** Set `date_to_run_to` (`date.today()` in operation, never a future date)
and `lag_days` (5); the season is `date_to_run_to.year`. Every step only does what is outstanding: the transfer
copies new or changed files, the interpolation adds the dates with *d* + `lag_days` ≤ `date_to_run_to` that are not
yet written (written dates are frozen), the model processes the new dates, and masking redoes changed mosaics.
So running the notebook daily keeps the season up to date. `check_data_currency()` warns when the newest
acquisition on disk is older than `date_to_run_to − lag_days`, i.e. when dates would be produced before their
data arrived. Before the first model month the notebook stops after Masking.

**Retrospective — `DiscoCH_Retro.ipynb`** (and `DiscoCH_Retro_NRTminmax.ipynb`). Set `year` to a finished season.
The NRT-min/max variant needs the NRT run's `model/minmax_by_date/` and shares the Retro run's interpolation.

**Comparison — `NRT_vs_Retro.ipynb`**, after both runs. The point validation reads the labelled sample in
`point_validation_dir` (config.yaml); the recall uses `validation_30km_gpkg`.

The core calls of the NRT run:

```python
from datetime import date
from src.disco_ch_operational.config import load_config, run_dirs, map_template
from src.disco_ch_operational.interpolate import interpolate_tiles
from src.disco_ch_operational.apply_model import apply_discoch_tiles
from src.disco_ch_operational.mask import mask_and_scale_dir

cfg = load_config()
date_to_run_to, lag_days = date.today(), 5
year = date_to_run_to.year
dirs = run_dirs(cfg["output_dir"], year, "nrt")
vi_keys = {"NDVI": "NDVI", "EVI": "EVI", "NDMI": "NDMI", "CIRE": "CIre", "CCI": "CCI"}

interpolate_tiles(tss_dir=cfg["force_dir"], out_dir=dirs["interpolated"], vi_keys=vi_keys,
                  date_range=(date(year, 1, 1), date(year, 12, 31)),
                  as_of=date_to_run_to, lag_days=lag_days, rbf_sigma=(10, 20, 30),
                  tiles_csv=cfg["tiles_csv"], max_workers=10)

apply_discoch_tiles(rbf_dir=dirs["interpolated"], out_dir=dirs["model"], year=year,
                    model_path=cfg["model_path"], vi_keys=vi_keys, tiles_csv=cfg["tiles_csv"],
                    model_months=(6, 7, 8, 9), export_minmax=True, max_workers=10)

mask_and_scale_dir(input_dir=dirs["mosaic"], mask_path=cfg["mask_path"],
                   output_dir=dirs["masked"], pattern="Disco_Proba_*.tif",
                   name_template=map_template("nrt"))      # -> masked/DiscoCH_NRT_<date>.tif
```

### Outputs

Under `output/DiscoCH/<year>/` (`config.run_dirs()`), one folder per run — `nrt/`, `retro/`, `retro_nrtminmax/`:

```
<run>/interpolated/<tile>/*_<VI>_TSS_TSI.tif   interpolated 5-day VIs (int16, nodata −9999; retro_nrtminmax uses retro's)
<run>/model/<tile>/Disco_Proba_<date>.tif      per-tile probability
<run>/model/mosaic/Disco_Proba_<date>.tif      national mosaic
<run>/model/minmax/, minmax_by_date/, vi_norm/ normalization cache and per-date snapshots
<run>/masked/DiscoCH_<RUN>_<date>.tif          forest-masked, EPSG:2056, the product to use
                                               (DiscoCH_NRT_…, DiscoCH_RETRO_…, DiscoCH_RETRO_NRTMINMAX_…)
<run>/unclipped/DiscoCH_<RUN>_<date>.tif       the same without the forest mask (nrt and retro), to clip later
<run>/clear_sky/masked/ClearSky_<date>.tif     cumulative clear-sky observations (uint8)
<run>/statistics/, maps/                       CSV tables and figures; hexagon maps and date grids
nrt_vs_retro/validation/, statistics/, figures/  NRT_vs_Retro.ipynb
```

The rasters in `masked/`, `unclipped/` and `clear_sky/masked/` are Cloud Optimized GeoTIFFs (`utils.to_cog()`):
one band per file, 512 × 512 tiles, DEFLATE with predictor 2, internal overviews down to about 1/256 (average;
nearest for the clear-sky counts), with nodata set before the overviews are built (int16 −32768 for the
probabilities, band scale 0.0001 so 0–10000 = 0–1; uint8 255 for the clear-sky counts). The notebooks check them
with `utils.validate_cog_dir()` (rio-cogeo plus these settings); by hand: `rio cogeo validate <file>` or `gdalinfo <file>`.

## Data not in the repository

- FORCE level-3 TSS time series (network share, copied locally with `transfer()`)
- Validation data: the labelled stratified point sample (`point_validation_dir`) and the opportunistically mapped
  discolored polygons (`CH30km_Validation.gpkg`); ground-truth polygons for `validate.py` (`Mapping_2026.gpkg`)
- District boundaries (`fgeolib_geolib_g1b20_LV95.shp`, field `bzname`) and biogeographic regions
- 2018 discoloration mosaics (Bloom et al., 2026), used for the year comparisons

Set their locations in `config.yaml` (see [Configure paths](#configure-paths-for-your-machine)).

## Contact

Colin Bloom, WSL, colin.bloom@wsl.ch
