"""Notebook-facing benchmark inspection helpers."""

from __future__ import annotations

from collections import Counter
from typing import Any

import numpy as np
import pandas as pd


def _preview(values: Any, limit: int = 6) -> str:
    arr = np.asarray(values, dtype=object).reshape(-1)
    if arr.size == 0:
        return "-"
    vals = []
    for item in arr[:limit]:
        if isinstance(item, float):
            vals.append(f"{item:.4g}" if np.isfinite(item) else str(item))
        else:
            vals.append(str(item))
    suffix = "" if arr.size <= limit else ", ..."
    return ", ".join(vals) + suffix


def _coverage(mask: np.ndarray) -> str:
    total = int(mask.size)
    if total == 0:
        return "0/0"
    count = int(mask.sum())
    return f"{count}/{total} ({count / total:.1%})"


def _top_counts(values: Any, limit: int = 6) -> str:
    arr = np.asarray(values, dtype=object).reshape(-1)
    if arr.size == 0:
        return "-"
    counts = Counter(arr.tolist()).most_common(limit)
    suffix = "" if len(set(arr.tolist())) <= limit else ", ..."
    return ", ".join(f"{key}: {count}" for key, count in counts) + suffix


def _row(field: str, scope: str, dtype_or_type: str, coverage: str, preview: str, role: str) -> dict[str, str]:
    return {
        "field": field,
        "scope": scope,
        "dtype/type": dtype_or_type,
        "coverage": coverage,
        "example / summary": preview,
        "role": role,
    }


def benchmark_metadata_table(bench: Any) -> pd.DataFrame:
    """Return non-sensor benchmark metadata available to notebooks and evaluators.

    The table intentionally separates split/evaluation/context metadata from the model input
    contract. Some fields, such as coordinates or time, may be consumed by specific models; the
    point of this table is that the benchmark exposes them independently of the sensor-band tables.
    """
    rows: list[dict[str, str]] = []

    patches = getattr(bench, "patches", None)
    if patches is not None:
        folds = np.asarray([patch.fold for patch in patches], dtype=np.int64)
        patch_ids = np.asarray([patch.patch_id for patch in patches], dtype=np.int64)
        s2_obs = np.asarray([len(patch.s2_months) for patch in patches], dtype=np.int64)
        s1_obs = np.asarray([len(patch.s1_months) for patch in patches], dtype=np.int64)
        rows.extend(
            [
                _row("patch_id", "patch", str(patch_ids.dtype), _coverage(np.ones_like(patch_ids, dtype=bool)),
                     _preview(patch_ids), "source patch identifier"),
                _row("fold", "patch/tile", str(folds.dtype), _coverage(np.ones_like(folds, dtype=bool)),
                     _top_counts(folds), "spatial split group"),
                _row("s2_months", "patch", "int64 sequence", _coverage(s2_obs > 0),
                     f"observations per patch: min={s2_obs.min()}, median={np.median(s2_obs):.0f}, max={s2_obs.max()}",
                     "calendar metadata for S2 observations"),
                _row("s1_months", "patch", "int64 sequence", _coverage(s1_obs > 0),
                     f"observations per patch: min={s1_obs.min()}, median={np.median(s1_obs):.0f}, max={s1_obs.max()}",
                     "calendar metadata for S1 observations"),
                _row("tile_size", "benchmark", type(getattr(bench, "tile_size", None)).__name__, "1/1",
                     str(getattr(bench, "tile_size", "-")), "dense tiling parameter"),
                _row("ignore_index", "pixel", type(getattr(bench, "ignore_index", None)).__name__, "1/1",
                     str(getattr(bench, "ignore_index", "-")), "void label excluded from metrics"),
            ]
        )
        return pd.DataFrame(rows)

    n = int(getattr(bench, "n_samples", 0))
    labels = np.asarray(getattr(bench, "labels", []))
    if labels.size:
        rows.append(
            _row("labels", "sample", str(labels.dtype), _coverage(np.ones(labels.shape, dtype=bool)),
                 _top_counts(labels), "supervision target before task-specific coarsening")
        )

    groups = np.asarray(getattr(bench, "groups", []), dtype=object)
    if groups.size:
        rows.append(
            _row("groups", "sample", str(groups.dtype), _coverage(np.ones(groups.shape, dtype=bool)),
                 _top_counts(groups), "split / holdout group")
        )

    years = getattr(bench, "years", None)
    if years is not None:
        years_arr = np.asarray(years)
        finite = np.isfinite(years_arr.astype(float, copy=False)) if years_arr.size else np.asarray([], dtype=bool)
        summary = "-"
        if finite.any():
            valid = years_arr[finite].astype(int)
            summary = f"min={valid.min()}, max={valid.max()}, values={_preview(np.unique(valid))}"
        rows.append(_row("years", "sample", str(years_arr.dtype), _coverage(finite), summary, "representative calendar year"))

    latlon = getattr(bench, "latlon", None)
    if latlon is not None:
        ll = np.asarray(latlon, dtype=float)
        finite = np.isfinite(ll).all(axis=1) if ll.ndim == 2 and ll.shape[1] == 2 else np.asarray([], dtype=bool)
        nonzero = finite & np.any(ll != 0.0, axis=1) if finite.size else finite
        summary = "-"
        if nonzero.any():
            valid = ll[nonzero]
            summary = (
                f"lat {valid[:, 0].min():.4g}..{valid[:, 0].max():.4g}; "
                f"lon {valid[:, 1].min():.4g}..{valid[:, 1].max():.4g}"
            )
        rows.append(_row("latlon", "sample", str(ll.dtype), _coverage(nonzero), summary, "geographic coordinate"))

    label_names = getattr(bench, "label_names", None)
    if label_names is not None:
        names = np.asarray(label_names, dtype=object)
        rows.append(
            _row("label_names", "class", "object", _coverage(np.ones(names.shape, dtype=bool)),
                 _preview(names), "class id to name/code mapping")
        )

    native = getattr(bench, "native", None)
    if native is not None:
        for modality in ("s2", "s1", "climate"):
            series = getattr(native, modality)
            bands = list(getattr(series, "bands", []))
            lengths = np.asarray([len(v) for v in getattr(series, "values", [])], dtype=np.int64)
            if not bands:
                rows.append(_row(f"{modality}.bands", "modality", "list[str]", "0/1", "-", "modality unavailable"))
                continue
            summary = f"{len(bands)} bands: {', '.join(bands[:8])}" + (", ..." if len(bands) > 8 else "")
            rows.append(_row(f"{modality}.bands", "modality", "list[str]", "1/1", summary, "native band names"))
            if lengths.size:
                rows.append(
                    _row(f"{modality}.observation_count", "sample", str(lengths.dtype), _coverage(lengths > 0),
                         f"min={lengths.min()}, median={np.median(lengths):.0f}, max={lengths.max()}",
                         "native acquisition count")
                )

    if not rows:
        rows.append(_row("n_samples", "benchmark", "int", "1/1", str(n), "sample count"))
    return pd.DataFrame(rows)


def benchmark_input_contract_table(bench: Any) -> pd.DataFrame:
    """Return the model-facing input contract exposed by a benchmark loader."""
    rows: list[dict[str, str]] = []

    patches = getattr(bench, "patches", None)
    if patches is not None:
        rows.extend(
            [
                _row("s2", "tile", "float32", "1/1", "B2, B3, B4, B5, B6, B7, B8, B8A, B11, B12", "Sentinel-2 image series"),
                _row("s1", "tile", "float32", "1/1", "VV, VH, VV/VH", "Sentinel-1 ascending image series"),
                _row("time", "patch", "int64", "1/1", "S2/S1 calendar months per native observation", "temporal metadata"),
                _row("labels", "pixel", "int64", "1/1", "semantic crop-type class id", "dense supervision"),
            ]
        )
        return pd.DataFrame(rows)

    native = getattr(bench, "native", None)
    if native is not None:
        for modality in ("s2", "s1", "climate"):
            series = getattr(native, modality)
            bands = list(getattr(series, "bands", []))
            coverage = "1/1" if bands else "0/1"
            preview = ", ".join(bands) if bands else "-"
            rows.append(_row(modality, "sample time series", "float32", coverage, preview, "model input modality"))

    ll = np.asarray(getattr(bench, "latlon", []), dtype=float)
    if ll.ndim == 2 and ll.shape[1] == 2:
        finite_nonzero = np.isfinite(ll).all(axis=1) & np.any(ll != 0.0, axis=1)
        rows.append(_row("latlon", "sample", "float32", _coverage(finite_nonzero), "[lat, lon]", "location input where used"))

    years = getattr(bench, "years", None)
    if years is not None:
        y = np.asarray(years)
        rows.append(_row("time", "sample/observation", str(y.dtype), _coverage(np.ones(y.shape, dtype=bool)), "months, day-of-year, years", "temporal input metadata"))

    labels = np.asarray(getattr(bench, "labels", []))
    if labels.size:
        rows.append(_row("labels", "sample", str(labels.dtype), _coverage(np.ones(labels.shape, dtype=bool)), "class id / target value", "supervision"))

    return pd.DataFrame(rows)
