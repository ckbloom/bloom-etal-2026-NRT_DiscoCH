"""
train_model.py

Trains an SVC (or, with classifier="logreg", a logistic regression)
discoloration model on labelled polygons (e.g. the Planet
mapping, Discolored 1 / 0, several dates) with the same features the
operational model uses, so apply_model.apply_discoch_tiles() can run it
unchanged:

  1. Features: for every pixel inside a polygon, the five VIs of the 5-day
     interpolated series (interpolate.interpolate_tiles() output) at the
     interpolated date matching the polygon's date, each normalized
     stepwise as apply_model.process_tile() does -- (value - min) / (max -
     min) with the pixel's min/max over the season from start_date up to
     and including that date, nothing later.
  2. Nested spatial cross-validation: the polygons are grouped into blocks
     of a block_size grid (30 km = the FORCE tile grid, in its CRS); the
     outer loop holds out whole blocks (n_outer folds, stratified by label),
     and within each outer training set a randomized search tunes C, gamma
     and class_weight with an inner spatial block CV. The outer test folds
     give the model's accuracy for unseen areas.
  3. Final model: the same randomized search on all data, refit with
     probability=True, pickled as a Pipeline([("classifier", SVC)]) fitted on
     the MODEL_FEATURES column names, like the operational model.

All pixels of a polygon fall in the block of its centroid, so a polygon is
never split between training and test. Large polygons are subsampled to
max_pixels_per_polygon pixels so they do not dominate (and SVC stays fast).

Usage (notebook):

    from src.disco_ch_operational.train_model import train_discoch_svc

    result = train_discoch_svc(
        labels=r"..\\..\\Validation\\Mapping_Planet_Oskar.gpkg",   # all layers, date from the layer name
        rbf_dir=rbf_output, year=2026, vi_keys=vi_keys,
        out_dir=model_output, n_outer=5, block_size=30000, n_iter=30)
    result["model_path"]  # -> apply_discoch_tiles(model_path=...)
"""

import os
import re
import json
import time
import pickle
import warnings
from datetime import datetime, UTC

import numpy as np
import pandas as pd
import rasterio
from rasterio import features as rio_features
from rasterio.windows import from_bounds, Window
from shapely.geometry import box

from src.disco_ch_operational.apply_model import MODEL_FEATURES, DEFAULT_VI_PATTERN, normalize
from src.disco_ch_operational.utils import (
    resolve_vi_keys, discover_tile_dirs, find_vi_file, band_dates, as_date, read_date_mean,
    format_duration, write_run_params,
)

FEATURES = list(MODEL_FEATURES.values())   # column order of the operational model
LABEL_COLUMNS = ("Discolored", "Disco", "Field", "label")


# ---------------------------------------------------------------------
# Labels
# ---------------------------------------------------------------------

def _layer_date(name, regex, fmt):
    m = re.search(regex, name)
    if not m:
        raise ValueError(f"no date in layer name {name!r} (regex {regex!r}); give a date_col instead")
    return datetime.strptime(m.group(1), fmt).date()


def load_labels(path, layers=None, label_cols=LABEL_COLUMNS, date_col="date",
                layer_date_regex=r"(\d{6})(?!\d)", layer_date_format="%d%m%y"):
    """Labelled polygons from every (or the listed) layer of a GeoPackage /
    shapefile as one GeoDataFrame: polygon_id, layer, date, label, geometry.

    The label is the first of label_cols present (0 / 1; empty rows are
    dropped). The date comes from date_col if the layer has it, otherwise
    from the layer name (default "planet_170726" -> 2026-07-17)."""
    import geopandas as gpd
    import pyogrio

    layers = layers or [name for name, _ in pyogrio.list_layers(path)]
    parts = []
    for layer in layers:
        g = gpd.read_file(path, layer=layer)
        col = next((c for c in label_cols if c in g.columns), None)
        if col is None:
            raise ValueError(f"layer {layer!r} has none of the label columns {label_cols}: {list(g.columns)}")
        if date_col in g.columns:
            dates = pd.to_datetime(g[date_col]).dt.date
        else:
            dates = _layer_date(layer, layer_date_regex, layer_date_format)
        g = gpd.GeoDataFrame({"layer": layer, "date": dates,
                              "label": pd.to_numeric(g[col], errors="coerce")},
                             geometry=g.geometry.force_2d(), crs=g.crs)
        parts.append(g)
    labels = pd.concat(parts, ignore_index=True)
    labels = labels[labels["label"].notna() & ~labels.geometry.is_empty & labels.geometry.notna()].copy()
    bad = ~labels["label"].isin([0, 1])
    if bad.any():
        raise ValueError(f"labels must be 0 or 1; found {sorted(labels.loc[bad, 'label'].unique())}")
    labels["label"] = labels["label"].astype(int)
    labels.insert(0, "polygon_id", [f"P{i + 1:05d}" for i in range(len(labels))])
    print(labels.groupby(["layer", "date"])["label"].agg(n="size", discolored="sum").to_string())
    return labels


# ---------------------------------------------------------------------
# Features: stepwise-normalized VIs at the polygon's date
# ---------------------------------------------------------------------

def match_date(d, dates, how="nearest"):
    """The interpolated date for a label date: "nearest" (ties -> earlier)
    or "before" (latest date <= d). None if there is none."""
    dates = sorted(dates)
    if how == "before":
        before = [x for x in dates if x <= d]
        return before[-1] if before else None
    if how != "nearest":
        raise ValueError("how must be 'nearest' or 'before'")
    return min(dates, key=lambda x: (abs((x - d).days), x)) if dates else None


def _polygon_window(geom, transform, shape):
    """Window of the tile covering geom's bounds (None if outside)."""
    win = from_bounds(*geom.bounds, transform=transform).round_offsets().round_lengths()
    win = Window(win.col_off, win.row_off, win.width + 1, win.height + 1)
    try:
        return win.intersection(Window(0, 0, shape[1], shape[0]))
    except Exception:  # rasterio.errors.WindowError: no overlap
        return None


def extract_features(labels, rbf_dir, year, vi_keys=None, vi_pattern=DEFAULT_VI_PATTERN,
                     start_date=None, date_match="nearest", tiles_csv=None, all_touched=False):
    """One row per pixel (centre inside a polygon, or touched with
    all_touched) with the five stepwise-normalized VIs as FEATURES columns.

    The normalization is apply_model.process_tile()'s: per pixel, min/max
    over the interpolated dates from start_date (default 1 Jan) up to and
    including the matched date. Pixels missing any VI are dropped.

    Columns: polygon_id, layer, date, tsi_date, label, tile, row, col, x, y
    (pixel centre), cx, cy (polygon centroid; both in the tiles' CRS),
    FEATURES, and the raw VI values (raw_<VI>)."""
    vi_map = resolve_vi_keys(vi_keys)
    vis = list(MODEL_FEATURES)
    if set(vis) - set(vi_map):
        raise ValueError(f"vi_keys must include {vis}")
    start_date = as_date(start_date) or as_date(f"{year}-01-01")
    tile_dirs = discover_tile_dirs(rbf_dir, tiles_csv)
    rows, skipped = [], []

    for tile_dir in tile_dirs:
        tile_id = os.path.basename(tile_dir)
        paths = {vi: find_vi_file(tile_dir, vi_map[vi], year, vi_pattern) for vi in vis}
        dates_by_vi = {vi: band_dates(p) for vi, p in paths.items()}
        common = sorted(d for d in set.intersection(*(set(v) for v in dates_by_vi.values()))
                        if d.year == year and d >= start_date)
        with rasterio.open(paths[vis[0]]) as src:
            transform, crs, shape, bounds = src.transform, src.crs, src.shape, src.bounds
        lab = labels.to_crs(crs)
        lab = lab[lab.intersects(box(*bounds))]
        if lab.empty or not common:
            continue

        srcs = {vi: rasterio.open(p) for vi, p in paths.items()}
        try:
            for _, poly in lab.iterrows():
                tsi_date = match_date(poly["date"], common, date_match)
                if tsi_date is None:
                    skipped.append((poly["polygon_id"], f"no interpolated date for {poly['date']}"))
                    continue
                win = _polygon_window(poly.geometry, transform, shape)
                if win is None or win.width < 1 or win.height < 1:
                    continue
                wt = rasterio.windows.transform(win, transform)
                inside = rio_features.geometry_mask([poly.geometry], out_shape=(int(win.height), int(win.width)),
                                                    transform=wt, invert=True, all_touched=all_touched)
                if not inside.any():
                    continue
                upto = [d for d in common if d <= tsi_date]
                vals = {}
                for vi in vis:
                    stack = np.stack([read_date_mean(srcs[vi], dates_by_vi[vi][d], win) for d in upto])
                    with warnings.catch_warnings():  # all-NaN pixels
                        warnings.simplefilter("ignore", RuntimeWarning)
                        vmin, vmax = np.nanmin(stack, axis=0), np.nanmax(stack, axis=0)
                    vals[vi] = (stack[-1], normalize(stack[-1].copy(), vmin, vmax))
                r, c = np.nonzero(inside)
                xs, ys = rasterio.transform.xy(wt, r, c)
                centroid = poly.geometry.centroid
                df = pd.DataFrame({
                    "polygon_id": poly["polygon_id"], "layer": poly["layer"], "date": poly["date"],
                    "tsi_date": tsi_date, "label": poly["label"], "tile": tile_id,
                    "row": r + int(win.row_off), "col": c + int(win.col_off),
                    "x": np.asarray(xs), "y": np.asarray(ys), "cx": centroid.x, "cy": centroid.y,
                    **{MODEL_FEATURES[vi]: vals[vi][1][r, c] for vi in vis},
                    **{f"raw_{vi}": vals[vi][0][r, c] for vi in vis},
                })
                rows.append(df)
        finally:
            for s in srcs.values():
                s.close()

    if not rows:
        raise RuntimeError("no pixels extracted: do the labels overlap the tiles in rbf_dir, "
                           "and do their dates have interpolated data?")
    out = pd.concat(rows, ignore_index=True)
    n_all = len(out)
    out = out.dropna(subset=FEATURES).reset_index(drop=True)
    missing = set(labels["polygon_id"]) - set(out["polygon_id"])
    print(f"{len(out):,} pixels from {out['polygon_id'].nunique()} polygons "
          f"({n_all - len(out):,} pixels without all five VIs dropped; "
          f"{len(missing)} polygon(s) without pixels)")
    for pid, why in skipped[:10]:
        print(f"  skipped {pid}: {why}")
    return out


def subsample_pixels(samples, max_pixels_per_polygon=50, seed=42):
    """At most max_pixels_per_polygon random pixels per polygon (None = all)."""
    if not max_pixels_per_polygon:
        return samples
    out = samples.sample(frac=1, random_state=seed).groupby("polygon_id").head(max_pixels_per_polygon)
    print(f"subsampled to {len(out):,} pixels (<= {max_pixels_per_polygon} per polygon)")
    return out.sort_index().reset_index(drop=True)


def assign_blocks(samples, block_size=30000, origin=None, rbf_dir=None):
    """Spatial block (grid cell of block_size m) of each polygon's centroid.
    origin = (x, y) of a grid corner (blocks named "<column>_<row>");
    default a tile corner from rbf_dir, so 30 km blocks are exactly the
    FORCE tiles and are named by tile ID ("X0056_Y0061")."""
    tile_ref = None
    if origin is None:
        if rbf_dir is None:
            origin = (0.0, 0.0)
        else:
            tile_dir = discover_tile_dirs(rbf_dir)[0]
            tif = next(os.path.join(tile_dir, f) for f in sorted(os.listdir(tile_dir)) if f.endswith(".tif"))
            with rasterio.open(tif) as src:
                origin = (src.transform.c, src.transform.f)
                if np.isclose(src.width * src.transform.a, block_size):
                    tile_ref = re.match(r"X(\d{4})_Y(\d{4})", os.path.basename(tile_dir))
    bx = np.floor((samples["cx"] - origin[0]) / block_size).astype(int)
    by = np.floor((origin[1] - samples["cy"]) / block_size).astype(int)
    samples = samples.copy()
    if tile_ref:
        tx, ty = int(tile_ref.group(1)), int(tile_ref.group(2))
        samples["block"] = [f"X{tx + i:04d}_Y{ty + j:04d}" for i, j in zip(bx, by)]
    else:
        samples["block"] = bx.astype(str) + "_" + by.astype(str)
    per = samples.groupby("block").agg(polygons=("polygon_id", "nunique"), pixels=("label", "size"),
                                       share_discolored=("label", "mean"))
    print(f"{len(per)} spatial blocks of {block_size / 1000:g} km with labelled polygons "
          f"(polygons per block: median {per['polygons'].median():g}, max {per['polygons'].max()})")
    return samples


# ---------------------------------------------------------------------
# Model and nested spatial cross-validation
# ---------------------------------------------------------------------

CLASSIFIERS = ("svc", "logreg")


def default_param_distributions(classifier="svc"):
    from scipy.stats import loguniform
    if classifier == "logreg":
        return {"poly__degree": [1, 2, 3],
                "classifier__C": loguniform(1e-3, 1e3),
                "classifier__class_weight": [None, "balanced"]}
    return {"classifier__C": loguniform(1e-2, 1e3),
            "classifier__gamma": loguniform(1e-2, 1e2),
            "classifier__class_weight": [None, "balanced"]}


def make_pipeline(classifier="svc", final=False):
    """svc: Pipeline([("classifier", SVC)]) -- the operational model's
    structure (the features are already normalized 0-1, so no scaler);
    probability=True only for the final fit (the search scores with
    decision_function).
    logreg: Pipeline([("poly", PolynomialFeatures), ("scale", StandardScaler),
    ("classifier", LogisticRegression)]) -- degree 1 is a plain linear
    model, 2/3 add interactions and curvature; at most 55 coefficients,
    so it applies much faster than an SVC with thousands of support vectors."""
    from sklearn.pipeline import Pipeline
    if classifier == "logreg":
        from sklearn.linear_model import LogisticRegression
        from sklearn.preprocessing import PolynomialFeatures, StandardScaler
        return Pipeline([("poly", PolynomialFeatures(include_bias=False)), ("scale", StandardScaler()),
                         ("classifier", LogisticRegression(max_iter=5000))])
    if classifier != "svc":
        raise ValueError(f"classifier must be one of {CLASSIFIERS}, not {classifier!r}")
    from sklearn.svm import SVC
    return Pipeline([("classifier", SVC(kernel="rbf", probability=final))])


def _format_params(params):
    return ", ".join(f"{k.split('__')[-1]}={v:.3g}" if isinstance(v, float) else f"{k.split('__')[-1]}={v}"
                     for k, v in params.items())


def _search(X, y, groups, n_inner, n_iter, param_distributions, scoring, seed, n_jobs, classifier="svc"):
    """Randomized search with a spatial (block) inner CV; SVC without
    probability during the search (scoring uses decision_function)."""
    from sklearn.model_selection import RandomizedSearchCV, StratifiedGroupKFold
    search = RandomizedSearchCV(
        make_pipeline(classifier), param_distributions, n_iter=n_iter, scoring=scoring,
        cv=StratifiedGroupKFold(n_splits=n_inner, shuffle=True, random_state=seed),
        refit=False, random_state=seed, n_jobs=n_jobs, error_score=np.nan)
    search.fit(X, y, groups=groups)
    return search


def _search_scores(search):
    """Mean inner-CV score per parameter set over the folds with a score:
    an inner test fold with a single class has no AUC (nan), which would
    otherwise make the mean_test_score of every parameter set nan."""
    splits = np.column_stack([search.cv_results_[f"split{i}_test_score"] for i in range(search.n_splits_)])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)  # all-nan rows -> nan
        scores = np.nanmean(splits, axis=1)
    if np.isnan(scores).all():
        raise ValueError("no inner CV fold could be scored; use fewer folds or a smaller block_size")
    n_bad = int(np.isnan(splits).all(axis=0).sum())
    if n_bad:
        print(f"    {n_bad} of {search.n_splits_} inner folds not scorable (single class?); ranked on the others")
    return scores


def _fit_final(X, y, best_params, seed, classifier="svc"):
    model = make_pipeline(classifier, final=True).set_params(**best_params)
    if classifier == "svc":
        model.set_params(classifier__random_state=seed)  # probability calibration CV
    return model.fit(X, y)


def binary_metrics(y, proba, threshold=0.5):
    """Threshold-free (AUC, Brier, log loss) and thresholded scores."""
    from sklearn import metrics as m
    y = np.asarray(y)
    pred = (np.asarray(proba) >= threshold).astype(int)
    both = len(np.unique(y)) == 2
    tp, fp = int(((pred == 1) & (y == 1)).sum()), int(((pred == 1) & (y == 0)).sum())
    fn, tn = int(((pred == 0) & (y == 1)).sum()), int(((pred == 0) & (y == 0)).sum())
    return dict(n=len(y), n_discolored=int(y.sum()),
                roc_auc=m.roc_auc_score(y, proba) if both else np.nan,
                pr_auc=m.average_precision_score(y, proba) if both else np.nan,
                brier=m.brier_score_loss(y, proba),
                log_loss=m.log_loss(y, np.clip(proba, 1e-6, 1 - 1e-6), labels=[0, 1]),
                accuracy=m.accuracy_score(y, pred),
                balanced_accuracy=m.balanced_accuracy_score(y, pred) if both else np.nan,
                precision=m.precision_score(y, pred, zero_division=np.nan),
                recall=m.recall_score(y, pred, zero_division=np.nan),
                f1=m.f1_score(y, pred, zero_division=np.nan),
                kappa=m.cohen_kappa_score(y, pred), tp=tp, fp=fp, fn=fn, tn=tn)


def nested_spatial_cv(samples, n_outer=5, n_inner=5, n_iter=30, param_distributions=None,
                      scoring="roc_auc", threshold=0.5, seed=42, n_jobs=-1, classifier="svc"):
    """Outer StratifiedGroupKFold over spatial blocks; in each outer
    training set a randomized search with an inner spatial block CV, then
    the best parameters refit with probability=True and scored on the
    held-out blocks -- per pixel and per polygon (mean probability of its
    pixels vs. its label).

    :return: (folds DataFrame: best params + test scores per fold,
              out-of-fold predictions per pixel)
    """
    from sklearn.model_selection import StratifiedGroupKFold

    param_distributions = param_distributions or default_param_distributions(classifier)
    X, y, groups = samples[FEATURES], samples["label"].to_numpy(), samples["block"].to_numpy()
    n_blocks = len(np.unique(groups))
    if n_blocks < n_outer:
        raise ValueError(f"only {n_blocks} spatial blocks for {n_outer} outer folds; "
                         f"use fewer folds or a smaller block_size")
    outer = StratifiedGroupKFold(n_splits=n_outer, shuffle=True, random_state=seed)
    preds, folds = [], []
    t0 = time.monotonic()
    for k, (tr, te) in enumerate(outer.split(X, y, groups), start=1):
        search = _search(X.iloc[tr], y[tr], groups[tr], n_inner, n_iter, param_distributions,
                         scoring, seed, n_jobs, classifier)
        scores = _search_scores(search)
        best = search.cv_results_["params"][int(np.nanargmax(scores))]
        model = _fit_final(X.iloc[tr], y[tr], best, seed, classifier)
        proba = model.predict_proba(X.iloc[te])[:, 1]
        p = samples.iloc[te][["polygon_id", "layer", "date", "label", "block", "tile", "row", "col"]].copy()
        p["fold"], p["proba"] = k, proba
        preds.append(p)
        poly = p.groupby("polygon_id").agg(label=("label", "first"), proba=("proba", "mean"))
        pix, pol = binary_metrics(y[te], proba, threshold), binary_metrics(poly["label"], poly["proba"], threshold)
        folds.append(dict(fold=k, test_blocks=",".join(sorted(np.unique(groups[te]))),
                          n_train=len(tr), inner_best_score=np.nanmax(scores),
                          **{k2.removeprefix("classifier__"): v for k2, v in best.items()},
                          **{f"pixel_{m}": v for m, v in pix.items()},
                          **{f"polygon_{m}": v for m, v in pol.items()}))
        print(f"  fold {k}/{n_outer} [{format_duration(time.monotonic() - t0)}]: "
              f"{pix['n']} test px / {pol['n']} polygons, AUC {pix['roc_auc']:.3f}, "
              f"F1 {pix['f1']:.3f} (polygon AUC {pol['roc_auc']:.3f}); {_format_params(best)}", flush=True)
    return pd.DataFrame(folds), pd.concat(preds, ignore_index=True)


def summarize_cv(folds, predictions, threshold=0.5):
    """Mean +/- SD of the outer-fold scores, and the scores of all pooled
    out-of-fold predictions, per pixel and per polygon."""
    rows = []
    for level in ("pixel", "polygon"):
        cols = [c for c in folds.columns if c.startswith(f"{level}_")]
        for c in cols:
            rows.append(dict(level=level, metric=c.removeprefix(f"{level}_"),
                             fold_mean=folds[c].mean(), fold_sd=folds[c].std()))
    summary = pd.DataFrame(rows)
    poly = predictions.groupby("polygon_id").agg(label=("label", "first"), proba=("proba", "mean"))
    pooled = {"pixel": binary_metrics(predictions["label"], predictions["proba"], threshold),
              "polygon": binary_metrics(poly["label"], poly["proba"], threshold)}
    summary["pooled"] = [pooled[r.level][r.metric] for r in summary.itertuples()]
    return summary


# ---------------------------------------------------------------------
# All together
# ---------------------------------------------------------------------

def train_discoch_svc(labels, rbf_dir, year, out_dir, vi_keys=None, layers=None,
                      vi_pattern=DEFAULT_VI_PATTERN, start_date=None, date_match="nearest",
                      tiles_csv=None, max_pixels_per_polygon=50, block_size=30000,
                      n_outer=5, n_inner=5, n_iter=30, param_distributions=None,
                      scoring="roc_auc", threshold=0.5, seed=42, n_jobs=-1,
                      model_name=None, samples=None, classifier="svc"):
    """Extracts the features, runs the nested spatial CV and trains the
    final model. Writes to out_dir:
        training_pixels.csv        every extracted pixel (before subsampling)
        cv_folds.csv               per outer fold: held-out blocks, tuned
                                   parameters, pixel and polygon scores
        cv_predictions.csv         out-of-fold probability per pixel
        cv_summary.csv             fold mean +/- SD and pooled scores
        final_search.csv           randomized search on all data
        <model_name>.pkl           final model for apply_discoch_tiles()
        <model_name>.json          its parameters, data and CV summary

    :param labels: GeoDataFrame from load_labels() or a path for it (all
        layers, or `layers`).
    :param rbf_dir: interpolate_tiles() output (one folder per tile).
    :param start_date: first date of the stepwise min/max (default 1 Jan),
        as apply_discoch_tiles(start_date=...).
    :param date_match: "nearest" or "before" (see match_date()).
    :param max_pixels_per_polygon: pixels kept per polygon (None = all).
    :param block_size: spatial block size in m (30000 = FORCE tiles).
    :param n_outer, n_inner, n_iter: outer folds, inner folds, parameter
        sets tried per search.
    :param samples: a training_pixels DataFrame from an earlier run, to
        skip the extraction.
    :param classifier: "svc" (RBF SVC, as the operational model) or
        "logreg" (logistic regression on polynomial features, far fewer
        parameters, so much faster to apply); use a separate out_dir per
        classifier, the CV files have the same names.
    :return: dict with samples, folds, predictions, summary, model, model_path
    """
    call_params = {k: v for k, v in locals().items() if k not in ("labels", "samples")}
    call_params["labels"] = labels if isinstance(labels, (str, os.PathLike)) else "<GeoDataFrame>"
    os.makedirs(out_dir, exist_ok=True)
    model_name = model_name or f"discoch_{classifier}_{datetime.now().strftime('%Y_%m_%d')}"
    t0 = time.monotonic()

    if samples is None:
        if isinstance(labels, (str, os.PathLike)):
            labels = load_labels(labels, layers)
        print("=== Extracting stepwise-normalized VIs")
        samples = extract_features(labels, rbf_dir, year, vi_keys, vi_pattern, start_date, date_match, tiles_csv)
        samples.to_csv(os.path.join(out_dir, "training_pixels.csv"), index=False)
    data = subsample_pixels(samples, max_pixels_per_polygon, seed)
    data = assign_blocks(data, block_size, rbf_dir=rbf_dir)

    print(f"=== Nested spatial CV ({classifier}): {n_outer} outer x {n_inner} inner folds, {n_iter} parameter sets")
    folds, predictions = nested_spatial_cv(data, n_outer, n_inner, n_iter, param_distributions,
                                           scoring, threshold, seed, n_jobs, classifier)
    summary = summarize_cv(folds, predictions, threshold)
    folds.to_csv(os.path.join(out_dir, "cv_folds.csv"), index=False)
    predictions.to_csv(os.path.join(out_dir, "cv_predictions.csv"), index=False)
    summary.to_csv(os.path.join(out_dir, "cv_summary.csv"), index=False)
    print(summary.pivot(index="metric", columns="level", values=["fold_mean", "fold_sd", "pooled"])
          .round(3).to_string())

    print("=== Final model: randomized search on all data")
    search = _search(data[FEATURES], data["label"].to_numpy(), data["block"].to_numpy(), n_inner, n_iter,
                     param_distributions or default_param_distributions(classifier), scoring, seed, n_jobs,
                     classifier)
    res = pd.DataFrame(search.cv_results_)
    res["mean_test_score"] = _search_scores(search)
    res["rank_test_score"] = res["mean_test_score"].rank(ascending=False, method="min")
    res = res.sort_values("mean_test_score", ascending=False, na_position="last")
    res.drop(columns="params").to_csv(os.path.join(out_dir, "final_search.csv"), index=False)
    best = res["params"].iloc[0]
    model = _fit_final(data[FEATURES], data["label"].to_numpy(), best, seed, classifier)
    model_path = os.path.join(out_dir, f"{model_name}.pkl")
    with open(model_path, "wb") as f:
        pickle.dump(model, f)

    import sklearn
    meta = dict(created=datetime.now(UTC).isoformat(), sklearn=sklearn.__version__, features=FEATURES,
                classifier=classifier,
                best_params={k.removeprefix("classifier__"): v for k, v in best.items()},
                inner_cv_score=float(res["mean_test_score"].iloc[0]), scoring=scoring,
                n_pixels=len(data), n_polygons=int(data["polygon_id"].nunique()),
                n_blocks=int(data["block"].nunique()), share_discolored=float(data["label"].mean()),
                dates=sorted({str(d) for d in data["date"]}),
                cv_summary=summary.to_dict(orient="records"), params=call_params)
    with open(os.path.join(out_dir, f"{model_name}.json"), "w") as f:
        json.dump(meta, f, indent=2, default=str)
    print(f"Final model {meta['best_params']} (inner {scoring} {meta['inner_cv_score']:.3f}) -> {model_path} "
          f"[{format_duration(time.monotonic() - t0)}]")
    write_run_params(out_dir, "train_model.train_discoch_svc", call_params,
                     {"model_path": model_path, "elapsed_seconds": time.monotonic() - t0})
    return dict(samples=data, folds=folds, predictions=predictions, summary=summary,
                model=model, model_path=model_path)


def plot_cv(result, output=None, show=True):
    """Out-of-fold results: ROC curves per outer fold, calibration of the
    pooled probabilities, and per-fold pixel/polygon AUC and F1."""
    import matplotlib.pyplot as plt
    from sklearn.metrics import roc_curve
    from sklearn.calibration import calibration_curve

    pred, folds = result["predictions"], result["folds"]
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(17, 5))
    for k, g in pred.groupby("fold"):
        if g["label"].nunique() == 2:
            fpr, tpr, _ = roc_curve(g["label"], g["proba"])
            ax1.plot(fpr, tpr, lw=1, alpha=0.7, label=f"fold {k}")
    fpr, tpr, _ = roc_curve(pred["label"], pred["proba"])
    ax1.plot(fpr, tpr, color="black", lw=2, label="pooled")
    ax1.plot([0, 1], [0, 1], color="0.6", ls="--", lw=1)
    ax1.set(xlabel="False positive rate", ylabel="True positive rate", title="ROC, held-out blocks")
    ax1.legend(fontsize=7, ncol=2, loc="lower right")

    frac, mean_p = calibration_curve(pred["label"], pred["proba"], n_bins=10, strategy="quantile")
    ax2.plot([0, 1], [0, 1], color="0.6", ls="--", lw=1)
    ax2.plot(mean_p, frac, "o-")
    ax2.set(xlim=(0, 1), ylim=(0, 1), xlabel="Mean predicted probability", ylabel="Share discoloured",
            title="Calibration (pooled out-of-fold pixels)")

    x = np.arange(len(folds))
    for i, (col, lab) in enumerate([("pixel_roc_auc", "pixel AUC"), ("polygon_roc_auc", "polygon AUC"),
                                    ("pixel_f1", "pixel F1"), ("polygon_f1", "polygon F1")]):
        ax3.bar(x + (i - 1.5) * 0.2, folds[col], 0.18, label=lab)
    ax3.set_xticks(x, folds["fold"])
    ax3.set(ylim=(0, 1.02), xlabel="Outer fold", ylabel="Score", title="Scores per outer fold")
    ax3.legend(fontsize=8, ncol=4, loc="upper center", bbox_to_anchor=(0.5, -0.13))
    fig.tight_layout()
    if output:
        fig.savefig(output, dpi=200, bbox_inches="tight")
    if show:
        plt.show()
    else:
        plt.close(fig)
    return fig
