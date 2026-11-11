"""
hexmap.py

Maps of discoloration aggregated to hexagons: columns = years (e.g. 2018,
2026), rows = months (default June-September, each the last mosaic date
of that month, labelled e.g. "June 30th"), over swisstopo's grey shaded
relief clipped to Switzerland, dark green (0 %) -> yellow (100 %), with a
horizontal colour bar below -- the layout of the DiscoCH 2026 overview
figure (years_as="rows" swaps rows and columns). Also exports each year's
hexagons with one proportion field per mosaic date.

  - One flat-top hexagon grid of hex_area_km2 cells, built in EPSG:2056
    over the boundary (e.g. the districts, i.e. Switzerland) or else over
    the union of the datasets' extents, and clipped to the boundary. The
    same grid is used for every year, so the maps and the exported layers
    line up hexagon by hexagon.
  - A hexagon's proportion on a date is (valid pixels >= threshold) /
    (valid pixels) inside it, counted with disco_statistics.discoloration_stats()
    -- the same nodata, threshold and grid handling as the national and
    district statistics, read in chunks. NaN where it has no valid pixel;
    hexagons without data on any date are dropped.
  - Thresholds are on the mosaics' stored scale (4000 = 40 %).

Replaces plot_hex_discoloration_grid.py. Differences:
  - one hexagon grid for all years (it built one per dataset);
  - any number of years (rows); larger text (font_scale, shared with
    compare.py);
  - counting is chunked and multithreaded instead of whole rasters, and
    mosaics with a smaller extent on the same pixel grid are counted
    instead of skipped;
  - one colour scale for every panel, 0-100 % by default (vmax=None =
    the 99th percentile of the values shown); empty months are labelled; the hexagons are written as GeoPackage by default (no
    10-character field limit, but dYYYYMMDD names still fit a shapefile).

Usage (notebook):

    from src.disco_ch_operational.hexmap import plot_hex_maps

    hexes = plot_hex_maps({
        "2018": dict(mosaic_dir=mosaics_2018, pattern="Proba_mosaic_*.tif", year=2018),
        "2026": dict(mosaic_dir=masked_output, year=2026),
    }, threshold=4000, hex_area_km2=3, boundary=districts, output_dir=stats_output)
"""

import os
import time
from datetime import datetime

import numpy as np
import pandas as pd
import rasterio
import shapely
from rasterio.warp import transform_bounds

from src.disco_ch_operational.compare import rc_params
from src.disco_ch_operational.disco_statistics import TOTAL, MAP_GLOB, discoloration_stats, list_mosaics
from src.disco_ch_operational.utils import format_duration, write_run_params

HEX_CRS = "EPSG:2056"


# ---------------------------------------------------------------------
# Hexagon grid
# ---------------------------------------------------------------------

def hex_grid(bounds, hex_area_km2=3, crs=HEX_CRS, boundary=None):
    """GeoDataFrame of flat-top hexagons ("hex_id" from 1) of hex_area_km2
    covering bounds (minx, miny, maxx, maxy in crs), clipped to `boundary`
    (path or GeoDataFrame, any CRS) if given."""
    import geopandas as gpd

    r = np.sqrt(hex_area_km2 * 1e6 / (1.5 * np.sqrt(3)))     # circumradius
    dx, dy = 1.5 * r, np.sqrt(3) * r
    minx, miny, maxx, maxy = bounds
    cols = np.arange(int(np.ceil((maxx - minx) / dx)) + 3)
    rows = np.arange(int(np.ceil((maxy - miny) / dy)) + 3)
    cc, rr = np.meshgrid(cols, rows, indexing="ij")
    cx = minx - dx + cc * dx
    cy = miny - dy + rr * dy + (cc % 2) * dy / 2                # odd columns shifted half a row
    angles = np.radians(np.arange(0, 360, 60))
    ring = np.stack([cx.ravel()[:, None] + r * np.cos(angles),
                     cy.ravel()[:, None] + r * np.sin(angles)], axis=-1)
    hexes = gpd.GeoDataFrame({"hex_id": np.arange(1, len(ring) + 1)},
                             geometry=shapely.polygons(ring), crs=crs)

    if boundary is not None:
        b = gpd.read_file(boundary) if isinstance(boundary, (str, os.PathLike)) else boundary
        shape = b.to_crs(crs).union_all()
        hexes = hexes[hexes.intersects(shape)]
        hexes = hexes.set_geometry(hexes.intersection(shape))
        hexes = hexes[~hexes.is_empty].reset_index(drop=True)
    return hexes


def _dataset_bounds(datasets, crs=HEX_CRS):
    """Union of the first mosaic's extent of every dataset, in crs."""
    boxes = []
    for spec in datasets.values():
        _, path = list_mosaics(spec["mosaic_dir"], spec.get("pattern", MAP_GLOB),
                               spec.get("year"))[0]
        with rasterio.open(path) as src:
            boxes.append(transform_bounds(src.crs, crs, *src.bounds))
    b = np.array(boxes)
    return b[:, 0].min(), b[:, 1].min(), b[:, 2].max(), b[:, 3].max()


# ---------------------------------------------------------------------
# Proportions per hexagon
# ---------------------------------------------------------------------

def date_field(d):
    """'dYYYYMMDD' (fits a shapefile's 10-character field names)."""
    return f"d{d:%Y%m%d}"


def hex_discoloration(mosaic_dir, hexes, threshold=4000, pattern=MAP_GLOB, year=None,
                      nodata=None, output=None, **stats_kw):
    """hexes with one proportion column (0-1, NaN = no valid pixel) per
    mosaic date, named date_field(date); hexagons without data on any date
    are dropped. Saved to `output` (.gpkg, .shp, ...) if given.

    :param stats_kw: chunk_size, max_workers for discoloration_stats().
    """
    stats = discoloration_stats(mosaic_dir, threshold, zones=hexes, zone_field="hex_id",
                                pattern=pattern, year=year, nodata=nodata, **stats_kw)
    stats = stats[stats.zone != TOTAL]
    wide = stats.pivot(index="zone", columns="date", values="proportion")
    wide.index = wide.index.astype(int)
    wide.columns = [date_field(d) for d in wide.columns]
    out = hexes.merge(wide, left_on="hex_id", right_index=True, how="inner")
    out = out[out[wide.columns].notna().any(axis=1)].reset_index(drop=True)
    print(f"  {len(out):,} of {len(hexes):,} hexagons have data")
    if output:
        os.makedirs(os.path.dirname(output) or ".", exist_ok=True)
        out.to_file(output)
        print(f"Saved {len(out):,} hexagons x {len(wide.columns)} dates to {output}")
    return out


def month_end_fields(hexes, months):
    """{month: date field of the latest date in that month} of a
    hex_discoloration() layer; months without a date are left out."""
    dates = [datetime.strptime(c[1:], "%Y%m%d") for c in hexes.columns
             if len(c) == 9 and c[0] == "d" and c[1:].isdigit()]
    out = {}
    for m in months:
        in_month = [d for d in dates if d.month == m]
        if in_month:
            out[m] = date_field(max(in_month))
    return out


# ---------------------------------------------------------------------
# Figure
# ---------------------------------------------------------------------

# Colours sampled from the DiscoCH 2026 overview figure: dark green (0 %) -> yellow (100 %)
HEX_COLORS = ["#244f36", "#2a6737", "#2c8841", "#419e46", "#56a946", "#6cb545",
              "#81c142", "#98ca46", "#b3d343", "#d5e039", "#f3ed27"]
RELIEF_LAYER = "ch.swisstopo.swissalti3d-reliefschattierung"   # swisstopo WMS, grey hillshade
LAKES_LAYER = "ch.bafu.vec25-seen"                              # swisstopo WMS, lakes


def hex_cmap():
    from matplotlib.colors import LinearSegmentedColormap
    return LinearSegmentedColormap.from_list("discoch_hex", HEX_COLORS)


def relief_basemap(bounds, boundary=None, width=2400, layer=RELIEF_LAYER, lakes_layer=LAKES_LAYER,
                   lighten=0.5):
    """Grey shaded relief for bounds (EPSG:2056) from the swisstopo WMS, as
    (rgba float array, lake mask or None, imshow extent). lighten blends
    the relief towards white (0 = as served, 1 = white); `boundary` (a
    shapely geometry in EPSG:2056) blanks everything outside it."""
    from rasterio.features import rasterize
    from rasterio.transform import from_bounds
    from src.disco_ch_operational.plot_date_grid import fetch_basemap

    a = fetch_basemap(bounds, width, layer).astype("float32") / 255
    h, w = a.shape[:2]
    grey = 1 - (1 - a[..., :3].mean(axis=-1)) * (1 - lighten)
    alpha = a[..., 3]
    if boundary is not None:
        inside = rasterize([boundary], out_shape=(h, w), transform=from_bounds(*bounds, w, h),
                           fill=0, default_value=1, dtype="uint8").astype(bool)
        alpha = alpha * inside
    lakes = fetch_basemap(bounds, width, lakes_layer)[..., 3] > 0 if lakes_layer else None
    minx, miny, maxx, maxy = bounds
    return np.dstack([grey, grey, grey, alpha]), lakes, (minx, maxx, miny, maxy)


def _draw_relief(ax, relief):
    rgba, lakes, extent = relief
    ax.imshow(rgba, extent=extent, interpolation="bilinear", zorder=0)
    if lakes is not None:
        white = np.zeros(lakes.shape + (4,), "float32")
        white[lakes] = 1.0
        ax.imshow(white, extent=extent, interpolation="nearest", zorder=1)
        x = np.linspace(extent[0], extent[1], lakes.shape[1])
        y = np.linspace(extent[3], extent[2], lakes.shape[0])
        ax.contour(x, y, lakes.astype("float32"), levels=[0.5], colors="0.6", linewidths=0.5, zorder=1)


def _ordinal(n):
    return f"{n}{'th' if 10 <= n % 100 <= 20 else {1: 'st', 2: 'nd', 3: 'rd'}.get(n % 10, 'th')}"


def date_label(field):
    """'June 30th' of a date field."""
    d = datetime.strptime(field[1:], "%Y%m%d")
    return f"{d:%B} {_ordinal(d.day)}"


def plot_hex_maps(datasets, threshold=4000, hex_area_km2=3, boundary=None, months=(6, 7, 8, 9),
                  cmap=None, vmax=1.0, years_as="columns", basemap=True, basemap_layer=RELIEF_LAYER,
                  lakes_layer=LAKES_LAYER, relief_lighten=0.35, basemap_width=2400, panel_width=5.0,
                  font_scale=1.4, colorbar_label="Proportion of European beech\ndominant forest discolored",
                  regions=None, region_field="DERegionNa", region_color="0.2", region_linewidth=0.6,
                  title=None, output_dir=None, output_name=None, export_hexes=True, show=True,
                  **stats_kw):
    """Grid of hexagon maps: one column per dataset (year) and one row per
    month (that month's last mosaic date), over a grey shaded relief
    clipped to the boundary, with a horizontal 0-100 % colour bar below --
    the layout of the DiscoCH 2026 overview figure.

    :param datasets: {label: dict(mosaic_dir=..., pattern=..., year=...,
        nodata=...)}, e.g. {"2018": ..., "2026": ...}; columns in this order.
    :param threshold: stored value at/above which a pixel is discolored.
    :param hex_area_km2: hexagon size.
    :param boundary: polygons to build the grid over and clip it (and the
        relief) to (path or GeoDataFrame, e.g. the districts); None = the
        datasets' extent, relief not clipped.
    :param months: month numbers, one row each.
    :param cmap: colormap of the proportion; None = HEX_COLORS (dark green
        -> yellow).
    :param vmax: top of the colour scale (proportion, default 1 = 100 %);
        None = the 99th percentile of the values shown.
    :param years_as: "columns" (default) or "rows".
    :param basemap: grey relief under the hexagons (swisstopo WMS
        basemap_layer and lakes_layer, needs internet; skipped with a
        message otherwise). relief_lighten blends it towards white,
        basemap_width is the WMS image width in pixels.
    :param panel_width: width of one map in inches.
    :param font_scale: text size relative to 12 pt.
    :param regions: polygons whose boundaries are drawn over the hexagons
        (path or GeoDataFrame, e.g. the biogeographic regions); None = none.
        Polygons are dissolved by region_field when the layer has it (the
        biogeographic layer holds sub-regions; DERegionNa = the 6 regions).
        region_color / region_linewidth style the lines.
    :param title: optional figure title (none by default).
    :param output_dir: saves <output_name>.png (default
        hex_discoloration_<labels>_t<threshold>.png), the run parameters and,
        with export_hexes, hex_<label>_t<threshold>.gpkg per dataset.
    :param stats_kw: chunk_size, max_workers for the counting.
    :return: {label: hex_discoloration() GeoDataFrame}.
    """
    import matplotlib.pyplot as plt
    import matplotlib.ticker as mtick

    if years_as not in ("columns", "rows"):
        raise ValueError(f"years_as must be 'columns' or 'rows', got {years_as!r}")
    t0 = time.time()
    labels = [str(k) for k in datasets]
    boundary_path = boundary if isinstance(boundary, (str, os.PathLike)) else None
    boundary_geom = None
    if boundary is None:
        bounds = _dataset_bounds(datasets)
    else:
        import geopandas as gpd
        if isinstance(boundary, (str, os.PathLike)):
            boundary = gpd.read_file(boundary)
        boundary_geom = boundary.to_crs(HEX_CRS).union_all()
        bounds = boundary_geom.bounds
    grid = hex_grid(bounds, hex_area_km2, HEX_CRS, boundary)
    print(f"{len(grid):,} hexagons of {hex_area_km2:g} km2")

    hexes, fields = {}, {}
    for label, spec in zip(labels, datasets.values()):
        print(f"\n{label}:")
        out = (os.path.join(output_dir, f"hex_{label}_t{threshold:g}.gpkg")
               if output_dir and export_hexes else None)
        hexes[label] = hex_discoloration(threshold=threshold, hexes=grid, output=out, **spec, **stats_kw)
        fields[label] = month_end_fields(hexes[label], months)
        missing = [m for m in months if m not in fields[label]]
        if missing:
            print(f"  No {label} mosaic in month(s) {missing}")

    shown = np.concatenate([hexes[lab][f].dropna().to_numpy() for lab in labels
                            for f in fields[lab].values()] or [np.zeros(1)])
    if vmax is None:    # 99th percentile: a few tiny border slivers reach 100 %
        vmax = float(np.percentile(shown, 99)) or 1.0
    norm = plt.Normalize(vmin=0, vmax=vmax)
    cmap = cmap or hex_cmap()

    minx, miny, maxx, maxy = grid.total_bounds
    pad = 0.01 * (maxx - minx)
    view = (minx - pad, miny - pad, maxx + pad, maxy + pad)
    relief = None
    if basemap:
        try:
            relief = relief_basemap(view, boundary_geom, basemap_width, basemap_layer, lakes_layer,
                                    relief_lighten)
        except Exception as e:      # no internet / WMS down: plot without
            print(f"  No basemap ({type(e).__name__}: {e})")
    region_lines = None
    if regions is not None:
        import geopandas as gpd
        reg = gpd.read_file(regions) if isinstance(regions, (str, os.PathLike)) else regions
        reg = reg.to_crs(HEX_CRS)
        if region_field and region_field in reg:
            reg = reg[[region_field, "geometry"]].dissolve(by=region_field)
        region_lines = reg.boundary
        if boundary_geom is not None:   # only inside the boundary (no lines beyond the border)
            region_lines = region_lines.intersection(boundary_geom)
            region_lines = region_lines[~region_lines.is_empty]

    # Panel (row, column) of each (dataset i, month j)
    n_lab, n_mon = len(labels), len(months)
    nrows, ncols = (n_mon, n_lab) if years_as == "columns" else (n_lab, n_mon)
    pos = {(i, j): ((j, i) if years_as == "columns" else (i, j))
           for i in range(n_lab) for j in range(n_mon)}
    # Each month is labelled with the first dataset's date; a panel whose date differs gets its own
    month_label = {m: next((date_label(fields[lab][m]) for lab in labels if m in fields[lab]),
                           datetime(2000, m, 1).strftime("%B")) for m in months}

    with plt.rc_context(rc_params(font_scale)):
        aspect = (view[3] - view[1]) / (view[2] - view[0])
        cbar_h = 1.6
        fig_w = panel_width * ncols + 0.8
        fig_h = panel_width * aspect * nrows + 0.6 + cbar_h
        fig, axes = plt.subplots(nrows, ncols, squeeze=False, figsize=(fig_w, fig_h))
        fig.subplots_adjust(left=0.8 / fig_w, right=0.995, top=1 - 0.6 / fig_h, bottom=cbar_h / fig_h,
                            wspace=0.02, hspace=0.02)
        big = plt.rcParams["axes.titlesize"] * 1.1
        for i, lab in enumerate(labels):
            for j, m in enumerate(months):
                r, c = pos[(i, j)]
                ax = axes[r, c]
                ax.set_xlim(view[0], view[2])
                ax.set_ylim(view[1], view[3])
                ax.set_aspect("equal")
                ax.set_axis_off()
                if relief is not None:
                    _draw_relief(ax, relief)
                field = fields[lab].get(m)
                if field is None:
                    ax.text(0.5, 0.5, "no data", transform=ax.transAxes, ha="center", va="center")
                else:
                    d = hexes[lab][hexes[lab][field].notna()]
                    d.plot(column=field, ax=ax, cmap=cmap, norm=norm, edgecolor="face", linewidth=0.3,
                           zorder=2)   # edge in the fill colour: no seams between hexagons
                    if region_lines is not None:
                        region_lines.plot(ax=ax, color=region_color, linewidth=region_linewidth, zorder=3)
                    if date_label(field) != month_label[m]:
                        ax.text(0.98, 0.02, date_label(field), transform=ax.transAxes,
                                ha="right", va="bottom")
                row_text, col_text = ((month_label[m], lab) if years_as == "columns"
                                      else (lab, month_label[m]))
                if r == 0:
                    ax.set_title(col_text, fontsize=big)
                if c == 0:
                    ax.text(-0.02, 0.5, row_text, transform=ax.transAxes, rotation=90,
                            ha="right", va="center", fontsize=big)

        cw = min(0.3, 3.2 / fig_w)
        cax = fig.add_axes([0.5 - cw / 2 + 0.4 / fig_w, 0.45 / fig_h, cw, 0.28 / fig_h])
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        cbar = fig.colorbar(sm, cax=cax, orientation="horizontal",
                            extend="max" if shown.max() > vmax else "neither")
        cbar.set_ticks([0, vmax])
        cbar.ax.xaxis.set_major_formatter(mtick.PercentFormatter(1.0, decimals=0))
        cbar.ax.tick_params(length=0)
        cbar.outline.set_linewidth(1.2)
        cax.set_title(colorbar_label, fontsize=plt.rcParams["font.size"])
        if title:
            fig.suptitle(title)

        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
            name = output_name or f"hex_discoloration_{'_vs_'.join(labels)}_t{threshold:g}"
            png = os.path.join(output_dir, name + ".png")
            fig.savefig(png, dpi=200, bbox_inches="tight", facecolor="white")
            print(f"Saved {png}")
            write_run_params(output_dir, "hexmap.plot_hex_maps",
                             dict(datasets=datasets, threshold=threshold, hex_area_km2=hex_area_km2,
                                  boundary=boundary_path, months=months, vmax=vmax, years_as=years_as,
                                  regions=regions if isinstance(regions, (str, os.PathLike)) else None),
                             {lab: {m: f for m, f in fields[lab].items()} for lab in labels})
        if show:
            plt.show()
        plt.close(fig)
    print(f"Done in {format_duration(time.time() - t0)}")
    return hexes
