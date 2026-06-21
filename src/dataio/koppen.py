"""Köppen–Geiger climate-zone lookup for the ``climate_ood`` domain.

**This is evaluation metadata, never model input.** We label each sample with a climate
domain (its main Köppen group A–E) by sampling a static global grid at the sample's
lat/lon. The model still consumes exactly the benchmark's ``x``; this only *partitions*
the evaluation, so it is fully faithful to both the model's and the benchmark's input
contracts (see "Designing Deployable Splits").

Expects a staged grid at ``$KOPPEN_GRID`` (default ``data/input/aux/koppen_main.npy``):
a ``(n_lat, n_lon)`` ``uint8`` array on a regular lon/lat grid covering
``[-90,90] x [-180,180]`` with row 0 = +90° (north), col 0 = -180°, and values
``0=unknown/ocean, 1=A, 2=B, 3=C, 4=D, 5=E``. Build it once from the Beck et al. (2018)
Köppen raster by collapsing the 30 classes to their leading letter. If the grid is
absent, :func:`koppen_main_group` raises ``FileNotFoundError`` so the ``climate_ood``
regime skips that benchmark instead of crashing the run.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np

_REPO = Path(__file__).resolve().parents[2]
KOPPEN_GRID = Path(os.environ.get("KOPPEN_GRID", _REPO / "data" / "input" / "aux" / "koppen_main.npy"))
_GROUPS = np.array(["unknown", "A", "B", "C", "D", "E"], dtype=object)


def koppen_main_group(latlon: np.ndarray, grid_path: Path | None = None) -> np.ndarray:
    """Map ``(N, 2)`` ``[lat, lon]`` to ``(N,)`` main Köppen group labels.

    Returns ``'A'..'E'`` per sample, or ``'unknown'`` where coordinates are missing
    (NaN or the (0, 0) sentinel used by coordinate-less benchmarks) or off-grid.
    """
    path = Path(grid_path or KOPPEN_GRID)
    if not path.exists():
        raise FileNotFoundError(
            f"Köppen grid not found at {path}. Stage it (see dataio/koppen.py) or set $KOPPEN_GRID."
        )
    grid = np.load(path)
    n_lat, n_lon = grid.shape
    latlon = np.asarray(latlon, dtype=float)
    lat, lon = latlon[:, 0], latlon[:, 1]
    valid = np.isfinite(lat) & np.isfinite(lon) & ~((lat == 0.0) & (lon == 0.0))
    row = np.clip(((90.0 - lat) / 180.0 * n_lat).astype(int), 0, n_lat - 1)
    col = np.clip(((lon + 180.0) / 360.0 * n_lon).astype(int), 0, n_lon - 1)
    codes = np.where(valid, grid[row, col], 0).astype(int)
    codes = np.clip(codes, 0, len(_GROUPS) - 1)
    return _GROUPS[codes]
