"""
clear_sky.py

Cumulative clear-sky (valid) Sentinel-2 observations per pixel through the
season, counted from FORCE's raw, uninterpolated EVI time series
(*_EVI_TSS.tif, one band per acquisition), on the same 5-day dates and
with the same near-real-time look-ahead as interpolate.py:

  - an acquisition date counts as one observation for a pixel when at
    least one of that date's EVI bands is valid there (not nodata).
    FORCE has already masked clouds, shadows and snow with the QAI layer,
    so "valid" = clear-sky. Two bands on one date (two satellites) count
    once -- the interpolation averages them into one value too;
  - the count for output date d is the number of such dates from
    start_date (default 1 Jan) up to and including d + lag_days, the data
    the near-real-time interpolation of d can use;
  - output dates are the interpolation's grid (start_date, + int_day, ...)
    and, with as_of, only those with d + lag_days <= as_of;
  - window_days=W (optional) gives a rolling count instead: only the
    acquisition dates in the W days ending at d + lag_days, i.e.
    (d + lag_days - W, d + lag_days]. These rasters are named
    ClearSky<W>d_<date>.tif, so they sit beside the cumulative ones, and
    mosaic_clear_sky(), summarize_clear_sky() and plot_clear_sky() take the
    same window_days to use them.

Steps:
  1. count_clear_sky_tiles(): per tile, one uint8 raster per date
     (<out_dir>/tiles/<tile>/ClearSky_<date>.tif; counts capped at 254,
     255 = nodata). Dates already written are kept (frozen, as in the
     interpolation), so each run only adds new dates.
  2. mosaic_clear_sky(): per date, the tiles merged, reprojected onto the
     forest mask's grid (EPSG:2056, nearest neighbour), blanked outside
     the mask and written as a uint8 Cloud Optimized GeoTIFF
     (<out_dir>/masked/ClearSky_<date>.tif, DEFLATE + predictor 2,
     nearest overviews). The national mosaic is built in memory, so only
     the masked file is written.
  3. summarize_clear_sky(): per date, the mean and standard deviation of
     the count over the masked pixels -- for all of Switzerland and per
     zone (e.g. biogeographic regions) -- as CSV.
  4. plot_clear_sky(): (A) the national mean against month/day with
     +-1 SD across pixels shaded, (B) one panel per region (A4 portrait).

Usage (notebook, after the Interpolate cell; same as_of / lag_days):

    from src.disco_ch_operational.clear_sky import (
        count_clear_sky_tiles, mosaic_clear_sky, summarize_clear_sky, plot_clear_sky)
    from src.disco_ch_operational.compare import BIOGEO_NAMES_EN, BIOGEO_COLORS

    clear_sky_output = os.path.join(discoCH_output, 'clear_sky')
    count_clear_sky_tiles(tss_dir=force_directory, out_dir=clear_sky_output, year=year,
                          as_of=date_to_run_to, lag_days=5, int_day=5,
                          export_start_date=f'{year}-06-01',   # counts still start 1 Jan
                          tiles_csv=force_tiles_to_process, max_workers=10)
    mosaic_clear_sky(clear_sky_output, mask_path=mask_path, threads=8)
    clear_sky = summarize_clear_sky(os.path.join(clear_sky_output, 'masked'),
                                    zones=cfg.require("biogeo_regions"), zone_field="DERegionNa",
                                    year=year, output_dir=stats_output, max_workers=8)
    plot_clear_sky({str(year): clear_sky}, region_names=BIOGEO_NAMES_EN,
                   region_colors=BIOGEO_COLORS, output_dir=stats_output)
"""

import os
import glob
import time
import bisect
import warnings
import contextlib
from datetime import date, timedelta
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed

import numpy as np
import pandas as pd
import rasterio
import shapely
from rasterio.errors import NotGeoreferencedWarning
from rasterio.features import rasterize
from rasterio.merge import merge as rio_merge
from rasterio.warp import reproject, Resampling
from rasterio.windows import Window

from src.disco_ch_operational.utils import (
    discover_tile_dirs, find_vi_file, band_dates, as_date, chunk_windows, to_cog,
    format_duration, write_run_params,
)
from src.disco_ch_operational.interpolate import (
    DEFAULT_TSS_PATTERN, DEFAULT_LAG_DAYS, output_dates_from_range,
)
from src.disco_ch_operational.mask import TARGET_CRS, load_mask
from src.disco_ch_operational.disco_statistics import (
    TOTAL, list_mosaics, load_zones, _aligned_files,
)

PREFIX = "ClearSky_"
NODATA = 255
MAX_COUNT = 254


def file_prefix(window_days=None):
    """File name prefix: ClearSky_ (cumulative) or ClearSky<W>d_ (rolling)."""
    if window_days is None:
        return PREFIX
    if int(window_days) != window_days or window_days < 1:
        raise ValueError(f"window_days must be a positive whole number of days, got {window_days!r}")
    return f"ClearSky{int(window_days)}d_"


def _in_range(d, start, end):
    return (start is None or d >= start) and (end is None or d <= end)


# ---------------------------------------------------------------------
# 1. Per-tile counts
# ---------------------------------------------------------------------

def count_tile(tss_path, out_dir, dates, start_date, lag_days=DEFAULT_LAG_DAYS, as_of=None,
               window_days=None, overwrite=False, chunk_size=512):
    """Writes out_dir/ClearSky_<date>.tif for each of `dates` not yet
    written (all of them with overwrite): the number of acquisition dates
    in tss_path from start_date to date + lag_days (and <= as_of) with a
    valid value, as uint8. With window_days=W only those after
    date + lag_days - W count (out_dir/ClearSky<W>d_<date>.tif).

    :return: (number of dates written, status) -- "written" or "skipped".
    """
    prefix = file_prefix(window_days)
    out_paths = {d: os.path.join(out_dir, f"{prefix}{d.isoformat()}.tif") for d in dates}
    todo = [d for d in dates if overwrite or not os.path.exists(out_paths[d])]
    if not todo:
        return 0, "skipped"
    os.makedirs(out_dir, exist_ok=True)

    last = max(todo) + timedelta(days=lag_days)
    bands_by_date = band_dates(tss_path)
    acq = sorted(a for a in bands_by_date
                 if start_date <= a <= last and (as_of is None or a <= as_of))
    bands = [b for a in acq for b in bands_by_date[a]]
    groups, i = [], 0                   # positions of each date's bands in `bands`
    for a in acq:
        groups.append(list(range(i, i + len(bands_by_date[a]))))
        i += len(bands_by_date[a])
    # Number of acquisition dates <= d + lag_days, per output date
    ends = [bisect.bisect_right(acq, d + timedelta(days=lag_days)) for d in todo]
    # ... and <= d + lag_days - window_days (subtracted for a rolling count)
    begins = ([bisect.bisect_right(acq, d + timedelta(days=lag_days - window_days)) for d in todo]
              if window_days else [0] * len(todo))
    tags = {"CLEAR_SKY_SOURCE": os.path.basename(tss_path),
            "CLEAR_SKY_START": start_date.isoformat(), "CLEAR_SKY_LAG_DAYS": str(lag_days),
            "CLEAR_SKY_AS_OF": as_of.isoformat() if as_of else "none",
            "CLEAR_SKY_WINDOW_DAYS": str(window_days) if window_days else "cumulative"}

    with contextlib.ExitStack() as stack:
        src = stack.enter_context(rasterio.open(tss_path))
        profile = dict(driver="GTiff", height=src.height, width=src.width, count=1, dtype="uint8",
                       crs=src.crs, transform=src.transform, nodata=NODATA, compress="deflate",
                       predictor=2, tiled=True, blockxsize=256, blockysize=256)
        dsts = {}
        for d, j, k in zip(todo, begins, ends):
            dst = stack.enter_context(rasterio.open(out_paths[d] + ".tmp", "w", **profile))
            dst.update_tags(**tags, CLEAR_SKY_DATE=d.isoformat(), CLEAR_SKY_N_DATES=str(k - j))
            dsts[d] = dst

        for window in chunk_windows(src.width, src.height, chunk_size):
            shape = (window.height, window.width)
            if bands:
                valid = ~np.ma.getmaskarray(src.read(bands, window=window, masked=True))
                obs = np.stack([valid[g].any(axis=0) for g in groups])
                cum = np.cumsum(obs, axis=0, dtype="uint16")
            for d, j, k in zip(todo, begins, ends):
                if k > j:
                    n_obs = cum[k - 1] - cum[j - 1] if j else cum[k - 1]
                    count = np.minimum(n_obs, MAX_COUNT).astype("uint8")
                else:
                    count = np.zeros(shape, "uint8")
                dsts[d].write(count, 1, window=window)

    for d in todo:
        os.replace(out_paths[d] + ".tmp", out_paths[d])
    return len(todo), "written"


def _run_job(tss_path, out_dir, kwargs):
    try:
        n, status = count_tile(tss_path, out_dir, **kwargs)
        return f"{status} ({n} date(s))" if status == "written" else status
    except Exception as e:
        return f"failed: {type(e).__name__}: {e}"


def clear_sky_dates(year, as_of, start_date=None, end_date=None, lag_days=DEFAULT_LAG_DAYS,
                    int_day=5, export_start_date=None, export_end_date=None):
    """The output dates: every int_day days from start_date (default 1 Jan
    of `year`) to end_date (default 31 Dec), with d + lag_days <= as_of when
    as_of is given, within the export window."""
    start_date = as_date(start_date) or date(year, 1, 1)
    end_date = as_date(end_date) or date(year, 12, 31)
    as_of = as_date(as_of)
    dates = output_dates_from_range((start_date, end_date), int_day)
    if as_of is not None:
        dates = [d for d in dates if d + timedelta(days=lag_days) <= as_of]
    return [d for d in dates if _in_range(d, as_date(export_start_date), as_date(export_end_date))]


def count_clear_sky_tiles(tss_dir, out_dir, year, as_of, start_date=None, end_date=None,
                          lag_days=DEFAULT_LAG_DAYS, int_day=5, window_days=None,
                          export_start_date=None, export_end_date=None,
                          vi_token="EVI", tss_pattern=DEFAULT_TSS_PATTERN,
                          tiles_csv=None, tiles_column="Tile_ID", tiles_csv_mode="include",
                          overwrite=False, chunk_size=512, max_workers=1):
    """Counts clear-sky observations for every tile under tss_dir into
    out_dir/tiles/<tile>/ClearSky_<date>.tif (see the module docstring).

    :param as_of: date up to which data may be used (as in
        interpolate_tiles(); None = every date of the season, each still
        counted only up to date + lag_days).
    :param start_date: first acquisition counted and first output date
        (default 1 Jan of `year`); end_date: last output date (31 Dec).
    :param lag_days, int_day: as in interpolate_tiles().
    :param window_days: None = cumulative since start_date; W = rolling
        count of the W days ending at date + lag_days (ClearSky<W>d_*.tif).
    :param export_start_date, export_end_date: limit which dates get
        rasters (e.g. f"{year}-06-01"); the counts still start at
        start_date.
    :param vi_token, tss_pattern: the raw EVI file in each tile folder.
    :param overwrite: recount dates already written.
    :param max_workers: tiles processed in parallel.
    :return: {tile_id: status}
    """
    call_params = dict(locals())
    as_of = as_date(as_of)
    start_date = as_date(start_date) or date(year, 1, 1)
    dates = clear_sky_dates(year, as_of, start_date, end_date, lag_days, int_day,
                            export_start_date, export_end_date)
    if not dates:
        print("No output dates (check as_of and the export window)")
        return {}
    span = (f"in the {window_days} days up to date + {lag_days} days" if window_days
            else f"from {start_date} up to date + {lag_days} days")
    print(f"{len(dates)} output date(s), {dates[0]} to {dates[-1]}, counting {span}")

    jobs = []
    for tile_dir in discover_tile_dirs(tss_dir, tiles_csv, tiles_column, tiles_csv_mode):
        tile_id = os.path.basename(tile_dir)
        try:
            jobs.append((tile_id, find_vi_file(tile_dir, vi_token, year, tss_pattern)))
        except (FileNotFoundError, ValueError) as e:
            print(f"  {tile_id}: skipped -- {e}")
    print(f"{len(jobs)} tile(s) with a {vi_token} file under {tss_dir}")

    kwargs = dict(dates=dates, start_date=start_date, lag_days=lag_days, as_of=as_of,
                  window_days=window_days, overwrite=overwrite, chunk_size=chunk_size)
    tiles_dir = os.path.join(out_dir, "tiles")
    results, start = {}, time.monotonic()

    def report(done, tile_id, status):
        results[tile_id] = status
        print(f"  [{done}/{len(jobs)}, {format_duration(time.monotonic() - start)}] {tile_id}: {status}",
              flush=True)

    if max_workers <= 1:
        for i, (tile_id, path) in enumerate(jobs, start=1):
            report(i, tile_id, _run_job(path, os.path.join(tiles_dir, tile_id), kwargs))
    else:
        with ProcessPoolExecutor(max_workers=max_workers) as pool:
            futures = {pool.submit(_run_job, path, os.path.join(tiles_dir, tile_id), kwargs): tile_id
                       for tile_id, path in jobs}
            for i, fut in enumerate(as_completed(futures), start=1):
                try:
                    status = fut.result()
                except Exception as e:
                    status = f"failed: worker crashed: {e}"
                report(i, futures[fut], status)

    summary = {k: sum(s.startswith(k) for s in results.values()) for k in ("written", "skipped", "failed")}
    summary["elapsed_seconds"] = time.monotonic() - start
    print(f"Done: {summary['written']} written, {summary['skipped']} up to date, "
          f"{summary['failed']} failed in {format_duration(summary['elapsed_seconds'])}")
    write_run_params(out_dir, "clear_sky.count_clear_sky_tiles"
                     + (f"_{window_days}d" if window_days else ""), call_params, summary)
    return results


# ---------------------------------------------------------------------
# 2. Mosaic and mask
# ---------------------------------------------------------------------

def _load_keep(mask_path, target_crs=TARGET_CRS, threads=1):
    """(keep, transform, crs) of the mask (True where == 1); read directly
    when it is already in target_crs (no float32 copy of the whole grid)."""
    with rasterio.open(mask_path) as src:
        if src.crs == rasterio.crs.CRS.from_user_input(target_crs):
            return src.read(1) == 1, src.transform, src.crs
    return load_mask(mask_path, target_crs, threads)


def mosaic_clear_sky(out_dir, mask_path, tiles_dir=None, masked_dir=None, target_crs=TARGET_CRS,
                     window_days=None, overwrite=False, threads=4):
    """Merges each date's tiles (out_dir/tiles/*/ClearSky_<date>.tif, or
    ClearSky<W>d_<date>.tif with window_days),
    reprojects the mosaic onto the mask's grid (nearest), blanks it outside
    the mask and writes out_dir/masked/ClearSky_<date>.tif as a uint8 COG.
    A date is redone only if its output is missing or older than one of
    its tiles or the mask.

    :return: {filename: status}
    """
    t0 = time.time()
    tiles_dir = tiles_dir or os.path.join(out_dir, "tiles")
    masked_dir = masked_dir or os.path.join(out_dir, "masked")
    os.makedirs(masked_dir, exist_ok=True)
    prefix = file_prefix(window_days)
    by_name = {}
    for p in glob.glob(os.path.join(tiles_dir, "*", f"{prefix}*.tif")):
        by_name.setdefault(os.path.basename(p), []).append(p)
    if not by_name:
        print(f"No {prefix}*.tif tiles under {tiles_dir}")
        return {}

    mask_time = os.path.getmtime(mask_path)
    todo = {n: ps for n, ps in sorted(by_name.items())
            if overwrite or not os.path.exists(os.path.join(masked_dir, n))
            or os.path.getmtime(os.path.join(masked_dir, n)) < max(mask_time, *map(os.path.getmtime, ps))}
    status = {n: "skipped" for n in by_name if n not in todo}
    if not todo:
        print(f"All {len(by_name)} mosaic(s) up to date in {masked_dir}")
        return status

    print(f"Loading mask {mask_path} ...")
    keep, transform, crs = _load_keep(mask_path, target_crs, threads)
    print(f"  {keep.shape[1]} x {keep.shape[0]} px, {keep.mean():.1%} kept; "
          f"{len(todo)} date(s) to mosaic")
    for name, paths in todo.items():
        out_path = os.path.join(masked_dir, name)
        try:
            with contextlib.ExitStack() as stack:
                srcs = [stack.enter_context(rasterio.open(p)) for p in paths]
                arr, src_transform = rio_merge(srcs, nodata=NODATA)
                src_crs, tags = srcs[0].crs, srcs[0].tags()
            out = np.full(keep.shape, NODATA, dtype="uint8")
            reproject(arr[0], out, src_transform=src_transform, src_crs=src_crs, src_nodata=NODATA,
                      dst_transform=transform, dst_crs=crs, dst_nodata=NODATA,
                      resampling=Resampling.nearest, num_threads=threads)
            del arr
            out[~keep] = NODATA
            tags = {k: v for k, v in tags.items() if k.startswith("CLEAR_SKY_") and k != "CLEAR_SKY_SOURCE"}
            tags.pop("CLEAR_SKY_N_DATES", None)          # differs between tiles
            tmp = out_path + ".tmp"
            with rasterio.open(tmp, "w", driver="GTiff", height=out.shape[0], width=out.shape[1],
                               count=1, dtype="uint8", crs=crs, transform=transform, nodata=NODATA,
                               compress="deflate", predictor=2, tiled=True) as dst:
                dst.write(out, 1)
                dst.update_tags(**tags, CLEAR_SKY_TILES=str(len(paths)))
                dst.set_band_description(1, "clear-sky observations")
            try:
                to_cog(tmp, out_path, overview_resampling="nearest", threads=threads)
            finally:
                os.remove(tmp)
            status[name] = f"written ({len(paths)} tiles)"
        except Exception as e:
            status[name] = f"failed: {type(e).__name__}: {e}"
        print(f"  {name}: {status[name]}", flush=True)
    print(f"Done in {format_duration(time.time() - t0)}")
    return status


# ---------------------------------------------------------------------
# 3. Statistics
# ---------------------------------------------------------------------

def _sum_window(files, window, transform, shapes, tree, n_zones):
    """Pixel count, sum and sum of squares of the counts (dates, 1 +
    n_zones) in one window; column 0 = every valid pixel."""
    n = np.zeros((len(files), 1 + n_zones), dtype="int64")
    s, ss = np.zeros_like(n), np.zeros_like(n)
    zone = None
    if shapes:
        hits = tree.query(shapely.box(*rasterio.windows.bounds(window, transform)))
        if len(hits):
            zone = rasterize([shapes[j] for j in hits], out_shape=(window.height, window.width),
                             transform=rasterio.windows.transform(window, transform),
                             fill=-1, dtype="int32") + 1     # 0 = outside every zone
            if not zone.any():
                zone = None
    for i, (_, path, dc, dr, fw, fh) in enumerate(files):
        with rasterio.open(path) as src:
            full = (dc, dr, fw, fh) == (0, 0, src.width, src.height) and \
                   window.col_off + window.width <= fw and window.row_off + window.height <= fh
            w = Window(window.col_off - dc, window.row_off - dr, window.width, window.height)
            nd = src.nodata
            arr = (src.read(1, window=w) if full
                   else src.read(1, window=w, boundless=True, fill_value=nd))
        valid = arr != nd if nd is not None else np.ones(arr.shape, bool)
        v = arr[valid].astype("int64")
        n[i, 0], s[i, 0], ss[i, 0] = v.size, v.sum(), (v * v).sum()
        if zone is not None:
            z = zone[valid]
            n[i, 1:] = np.bincount(z, minlength=n_zones + 1)[1:]
            s[i, 1:] = np.bincount(z, weights=v, minlength=n_zones + 1)[1:]
            ss[i, 1:] = np.bincount(z, weights=v * v, minlength=n_zones + 1)[1:]
    return n, s, ss


def clear_sky_stats(mosaic_dir, zones=None, zone_field="DERegionNa", pattern=None,
                    year=None, window_days=None, chunk_size=2048, max_workers=4):
    """Long DataFrame of the clear-sky count over the valid (masked) pixels
    per date and zone: columns date, zone, valid_pixels, mean, std
    (population SD across pixels). "All" covers every valid pixel.
    pattern defaults to the cumulative files, or the rolling ones of
    window_days."""
    t0 = time.time()
    pattern = pattern or f"{file_prefix(window_days)}*.tif"
    files, (crs, transform, width, height) = _aligned_files(list_mosaics(mosaic_dir, pattern, year))
    names, shapes = load_zones(zones, crs, zone_field) if zones is not None else ([], [])
    tree = shapely.STRtree([g for g, _ in shapes]) if shapes else None
    print(f"Summarizing {len(files)} clear-sky mosaics in {mosaic_dir} ({width} x {height} px"
          + (f", {len(names)} zones" if names else "") + ") ...")

    n = np.zeros((len(files), 1 + len(names)), dtype="int64")
    s, ss = np.zeros_like(n), np.zeros_like(n)
    windows = list(chunk_windows(width, height, chunk_size))
    with warnings.catch_warnings(), ThreadPoolExecutor(max_workers=max_workers) as pool:
        warnings.simplefilter("ignore", NotGeoreferencedWarning)
        for dn, ds, dss in pool.map(lambda w: _sum_window(files, w, transform, shapes, tree, len(names)),
                                    windows):
            n += dn
            s += ds
            ss += dss

    zone_names = [TOTAL] + names
    with np.errstate(invalid="ignore", divide="ignore"):
        mean = np.where(n > 0, s / n, np.nan)
        std = np.sqrt(np.clip(np.where(n > 0, ss / n, np.nan) - mean ** 2, 0, None))
    df = pd.DataFrame({
        "date": pd.to_datetime(np.repeat([f[0] for f in files], len(zone_names))),
        "zone": np.tile(zone_names, len(files)),
        "valid_pixels": n.ravel(), "mean": mean.ravel(), "std": std.ravel(),
    })
    print(f"Done in {format_duration(time.time() - t0)}")
    return df


def summarize_clear_sky(mosaic_dir, zones=None, zone_field="DERegionNa", pattern=None,
                        year=None, window_days=None, output_dir=None, label=None,
                        chunk_size=2048, max_workers=4):
    """clear_sky_stats(), printed for "All" and, with output_dir, saved
    there as clear_sky_<label>.csv (long) and clear_sky_<label>_mean_by_zone.csv
    (wide date x zone means). label defaults to the year or "all", plus
    _<W>d with window_days."""
    stats = clear_sky_stats(mosaic_dir, zones, zone_field, pattern, year, window_days,
                            chunk_size, max_workers)
    total = stats[stats.zone == TOTAL].set_index("date")
    print(total[["valid_pixels", "mean", "std"]].round(2).to_string())
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        label = label or (("all" if year is None else str(year))
                          + (f"_{window_days}d" if window_days else ""))
        base = os.path.join(output_dir, f"clear_sky_{label}")
        stats.to_csv(base + ".csv", index=False, date_format="%Y-%m-%d", encoding="utf-8-sig")
        wide = stats.pivot(index="date", columns="zone", values="mean")
        wide = wide[[TOTAL] + [c for c in wide.columns if c != TOTAL]]
        wide.to_csv(base + "_mean_by_zone.csv", date_format="%Y-%m-%d", encoding="utf-8-sig")
        print(f"Saved {base}.csv and {base}_mean_by_zone.csv")
    return stats


# ---------------------------------------------------------------------
# 4. Plot
# ---------------------------------------------------------------------

def plot_clear_sky(stats, regions=None, region_names=None, region_colors=None, ncols=3,
                   figsize=(8.27, 11.69), top_ratio=1.8, font_scale=0.85, date_window=None, shade_sd=True,
                   lag_days=DEFAULT_LAG_DAYS, window_days=None, title=None, colors=None, linestyles=None, guides=True,
                   output_dir=None, output_name=None, show=True):
    """(A) mean cumulative clear-sky observations of all of Switzerland
    against month/day, +-1 SD across pixels shaded (lower edge cut at 0);
    (B) the same per region, one panel each. A4 portrait by default.

    :param stats: a clear_sky_stats() table, or {label: table} to draw
        several years on the month/day axis (e.g. {"2018": ..., "2026": ...}).
    :param regions: zones for (B), after region_names; default every zone
        in the tables, in region_colors order. [] = only panel (A).
    :param region_names: {zone: display name}, e.g. compare.BIOGEO_NAMES_EN.
    :param region_colors: {region: colour}, e.g. compare.BIOGEO_COLORS;
        used with one dataset (with several, the colour is the dataset).
    :param figsize: (width, height) in inches; default A4 portrait. None =
        sized by the number of panels (the earlier landscape layout).
    :param top_ratio: height of panel A relative to one row of region panels.
    :param font_scale: text size relative to 12 pt.
    :param date_window: ("MM-DD", "MM-DD") x-axis range; None = the data's.
    :param shade_sd: shade +-1 SD across pixels.
    :param lag_days: only for the subtitle.
    :param window_days: the tables are rolling counts of this many days
        (summarize_clear_sky(window_days=...)); only titles, labels and the
        default output name change.
    :param colors, linestyles: {label: ...} per dataset.
    :param output_dir: saves <output_name>.png (default
        clear_sky_timeseries_<labels>) and the plotted table as .csv.
    :return: the combined table.
    """
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch
    from src.disco_ch_operational.compare import rc_params, national_region_figure, style_season_axes

    datasets = stats if isinstance(stats, dict) else {"": stats}
    tables = []
    for lab, df in datasets.items():
        df = df.drop(columns=["dataset", "season_date"], errors="ignore")
        df["date"] = pd.to_datetime(df["date"])
        df.insert(0, "dataset", str(lab))
        tables.append(df)
    data = pd.concat(tables, ignore_index=True)
    if region_names:
        data["zone"] = data["zone"].replace(region_names)
    data["season_date"] = data["date"].map(lambda d: d.replace(year=2000))
    labels = [str(k) for k in datasets]
    region_colors = dict(region_colors or {})
    if regions is None:
        zones = set(data.zone) - {TOTAL}
        regions = [r for r in region_colors if r in zones] + sorted(zones - set(region_colors))

    palette = rc_params()["axes.prop_cycle"].by_key()["color"]
    one = len(labels) == 1
    ds_color = {lab: (colors or {}).get(lab, "black" if one else palette[i % len(palette)])
                for i, lab in enumerate(labels)}
    ds_style = {lab: (linestyles or {}).get(lab, "-") for lab in labels}
    reg_color = {r: region_colors.get(r, palette[i % len(palette)]) for i, r in enumerate(regions)}

    def draw(ax, zone, color_of):
        for lab in labels:
            d = data[(data.dataset == lab) & (data.zone == zone)].sort_values("season_date")
            if d.empty:
                continue
            c = color_of(lab)
            if shade_sd:
                ax.fill_between(d.season_date, np.clip(d["mean"] - d["std"], 0, None),
                                d["mean"] + d["std"], color=c, alpha=0.3, linewidth=0)
            ax.plot(d.season_date, d["mean"], color=c, linestyle=ds_style[lab], linewidth=2.2)

    with plt.rc_context(rc_params(font_scale)):
        fig, ax_a, axes_b, ncols_used = national_region_figure(regions, ncols, figsize, top_ratio)
        draw(ax_a, TOTAL, lambda lab: ds_color[lab])
        ylabel = (f"Clear-sky observations\n(last {window_days} days)" if window_days
                  else "Clear-sky observations")
        ax_a.set_ylabel(ylabel)
        for i, (ax, r) in enumerate(zip(axes_b, regions)):
            draw(ax, r, (lambda lab, r=r: reg_color[r]) if one else (lambda lab: ds_color[lab]))
            if i % ncols_used == 0:
                ax.set_ylabel(ylabel)
        style_season_axes([ax_a] + axes_b, guides, bottom_only=(axes_b, ncols_used))
        if date_window:
            ax_a.set_xlim(*(pd.Timestamp(f"2000-{md}") for md in date_window))

        handles = [Line2D([], [], color=ds_color[lab], linestyle=ds_style[lab], linewidth=2.2)
                   for lab in labels]
        names = ["Mean" if one else lab for lab in labels]
        if shade_sd:
            handles.append(Patch(color="black" if one else "0.5", alpha=0.3))
            names.append("±1 SD across pixels")
        ax_a.legend(handles, names, loc="upper left")

        if window_days:
            fig.suptitle((title or f"Mean clear-sky observations of forest pixels, rolling "
                                   f"{window_days}-day window")
                         + f"\n(valid EVI observations in the {window_days} days up to each date"
                           f" + {lag_days} days)")
        else:
            fig.suptitle((title or "Mean cumulative clear-sky observations of forest pixels")
                         + f"\n(valid EVI observations up to each date + {lag_days} days)")
        fig.supxlabel("Date (month/day)")
        fig.tight_layout()

        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
            name = output_name or "clear_sky_timeseries_" + ("_vs_".join(l.replace(" ", "") for l in labels)
                                                  if labels != [""] else "season") \
                + (f"_{window_days}d" if window_days else "")
            png = os.path.join(output_dir, name + ".png")
            fig.savefig(png, dpi=200, bbox_inches="tight")
            data.drop(columns="season_date").to_csv(os.path.join(output_dir, name + ".csv"), index=False,
                                                    date_format="%Y-%m-%d", encoding="utf-8-sig")
            print(f"Saved {png} and {name}.csv")
        if show:
            plt.show()
        plt.close(fig)
    return data
