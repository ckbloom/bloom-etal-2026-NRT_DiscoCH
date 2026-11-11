"""
config.py

Machine-specific paths (FORCE data, validation polygons, boundaries, the
2018 mosaics, ...) live in config.yaml at the project root, which is not
committed. config.example.yaml is the committed template: copy it to
config.yaml and fill in your own paths.

  - Relative paths are resolved against the project root, so the defaults
    (data/..., output) work wherever the repository is checked out.
  - ~ and environment variables (%VAR% / $VAR) are expanded.
  - An empty value (or ~ / null) becomes None, for data you don't have;
    asking for it with require() gives a clear error.
  - The environment variable DISCOCH_CONFIG points to another config file
    (e.g. a second machine or an operational server).

Usage (notebook or script):

    from src.disco_ch_operational.config import load_config

    cfg = load_config()
    force_directory = cfg["force_dir"]
    districts = cfg.require("districts")   # error if it isn't set

    dirs = run_dirs(cfg["output_dir"], 2026, "nrt")   # output folders of one run
"""

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
CONFIG_FILE = ROOT / "config.yaml"
EXAMPLE_FILE = ROOT / "config.example.yaml"
ENV_VAR = "DISCOCH_CONFIG"


class Config(dict):
    """dict of resolved paths (str or None) with require() for mandatory keys."""

    def __init__(self, values, source):
        super().__init__(values)
        self.source = source

    def __missing__(self, key):
        raise KeyError(f"'{key}' is not in {self.source} -- add it (see {EXAMPLE_FILE.name})")

    def require(self, key):
        value = self[key]
        if value is None:
            raise ValueError(f"'{key}' is empty in {self.source} -- set it to a path on this machine")
        return value


def _resolve(value):
    if value is None or (isinstance(value, str) and not value.strip()):
        return None
    if not isinstance(value, str):
        return value  # numbers / booleans / lists pass through
    path = Path(os.path.expandvars(os.path.expanduser(value.strip())))
    if not path.is_absolute() and not value.strip().startswith(("\\\\", "//")):
        path = ROOT / path
    return str(path)


# ---------------------------------------------------------------------
# Output folders
# ---------------------------------------------------------------------

RUNS = ("nrt", "retro", "retro_nrtminmax")
COMPARISON = "nrt_vs_retro"

# File names of the final maps (masked/, unclipped/): DiscoCH_<RUN>_<YYYY-MM-DD>.tif,
# e.g. DiscoCH_NRT_2026-08-14.tif. The model's own per-tile files and mosaics keep
# Disco_Proba_<date>.tif. Readers find maps by their date (MAP_GLOB), so either works.
MAP_GLOB = "*_????-??-??.tif"


def map_template(run):
    """Output name template of a run's final maps, for mask_and_scale_dir(name_template=...):
    "DiscoCH_<RUN>_{date}.tif" (run upper-cased, e.g. NRT, RETRO, RETRO_NRTMINMAX)."""
    return f"DiscoCH_{run.upper()}_{{date}}.tif"


def map_name(run, d):
    """File name of a run's final map of date d (date or 'YYYY-MM-DD')."""
    return map_template(run).format(date=str(d))


def run_dirs(output_dir, year, run):
    """The output folders of one run of a season, shared by the notebooks:

        <output_dir>/DiscoCH/<year>/<run>/
            interpolated/   5-day RBF series per tile (*_TSS_TSI.tif)
            model/          per-tile probabilities, min/max cache, logs, mosaic/
            masked/         DiscoCH_<RUN>_<date>.tif on the forest mask (EPSG:2056), COG
            unclipped/      the same without the forest mask (same grid and scaling), COG
            clear_sky/      cumulative clear-sky counts (tiles/, masked/)
            statistics/     tables and time-series plots
            maps/           hexagon maps and date grids

    run: "nrt" (near real time, DiscoCH_NRT.ipynb), "retro" (all-data
    interpolation, own stepwise min/max, DiscoCH_Retro.ipynb) or
    "retro_nrtminmax" (all-data interpolation with the NRT run's min/max,
    DiscoCH_Retro_NRTminmax.ipynb; it shares retro's interpolated/), or
    COMPARISON for NRT_vs_Retro.ipynb (validation/, statistics/, figures/).
    """
    if run not in RUNS + (COMPARISON,):
        raise ValueError(f"run must be one of {RUNS + (COMPARISON,)}, got {run!r}")
    season = os.path.join(output_dir, "DiscoCH", str(year))
    base = os.path.join(season, run)
    if run == COMPARISON:
        names = ("validation", "statistics", "figures")
        return {"root": base, **{n: os.path.join(base, n) for n in names}}
    interpolated_run = "retro" if run == "retro_nrtminmax" else run
    model = os.path.join(base, "model")
    return {
        "root": base,
        "interpolated": os.path.join(season, interpolated_run, "interpolated"),
        "model": model,
        "mosaic": os.path.join(model, "mosaic"),
        "masked": os.path.join(base, "masked"),
        "unclipped": os.path.join(base, "unclipped"),
        "clear_sky": os.path.join(base, "clear_sky"),
        "statistics": os.path.join(base, "statistics"),
        "maps": os.path.join(base, "maps"),
    }


def load_config(path=None):
    """Read config.yaml (or `path`, or $DISCOCH_CONFIG) into a Config of
    resolved paths."""
    import yaml

    path = Path(path or os.environ.get(ENV_VAR) or CONFIG_FILE)
    if not path.exists():
        raise FileNotFoundError(
            f"No config file at {path}.\n"
            f"Copy {EXAMPLE_FILE.name} to {CONFIG_FILE.name} in {ROOT} and set the paths for this machine.")
    with open(path, encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}
    return Config({k: _resolve(v) for k, v in raw.items()}, source=path)
