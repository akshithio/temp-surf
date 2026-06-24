"""Native per-model preprocessing (Benchmark.native + view accessors) — input-assembly checks.

These exercise the numpy input-assembly only (no model weights): embeddings are a deterministic
function of these inputs, so input-equivalence implies output-equivalence.
"""

from __future__ import annotations

import numpy as np

from dataio.get_input import (
    Benchmark,
    ModalitySeries,
    NativeSeries,
    _synthetic_month_doy,
    monthly_composite,
)

S2_BANDS = ["B2", "B3", "B4", "B5", "B6", "B7", "B8", "B8A", "B11", "B12", "NDVI"]


def _native_bench(s2_series, s2_months, s2_bands, *, name="eurocropsml") -> Benchmark:
    n = len(s2_series)
    doy = [_synthetic_month_doy(12)[np.asarray(m) % 12].astype(np.float32) for m in s2_months]
    years = [np.full(len(s), 2021, dtype=np.int64) for s in s2_series]
    s2 = ModalitySeries(
        list(s2_series), [np.asarray(m, dtype=np.int64) for m in s2_months], doy, years, s2_bands
    )
    native = NativeSeries(s2=s2, s1=ModalitySeries.absent(n), climate=ModalitySeries.absent(n))
    return Benchmark(
        name=name, label_kind="multiclass", native=native,
        labels=np.zeros(n, np.int64), groups=np.array(["g"] * n, dtype=object),
        latlon=np.zeros((n, 2), np.float32), years=np.full(n, 2021, np.int64),
    )


def test_monthly_composite_means_per_calendar_month() -> None:
    vals = np.array([[1.0], [3.0], [10.0]], dtype=np.float32)  # months 0, 0, 5
    out, mask = monthly_composite(vals, np.array([0, 0, 5]), 12)
    assert out[0, 0] == 2.0  # mean(1, 3)
    assert out[5, 0] == 10.0
    assert mask[0] == 1.0 and mask[5] == 1.0 and mask[1] == 0.0


def test_benchmark_monthly_view_composites_native_series() -> None:
    raw = np.arange(24 * 3, dtype=np.float32).reshape(24, 3)  # 24 sub-monthly acquisitions
    months = np.repeat(np.arange(12), 2)  # 2 per calendar month
    bench = _native_bench([raw], [months], ["A", "B", "C"])
    vals, mask, _doy, bands = bench.monthly("s2")
    expected = np.stack([raw[months == m].mean(0) for m in range(12)])
    assert vals.shape == (1, 12, 3)
    np.testing.assert_allclose(vals[0], expected)
    assert mask.min() == 1.0 and bands == ["A", "B", "C"]


def test_presto_monthly_grid_is_calendar_ordered_january_start() -> None:
    from models.presto import PrestoModel

    rng = np.random.default_rng(0)
    raw = rng.random((12, 11), dtype=np.float32) * 1000 + 2000
    bench = _native_bench([raw], [np.arange(12)], S2_BANDS)
    x, mask, _dw, _latlons, months = PrestoModel().to_presto_inputs(bench)
    assert x.shape[1] == 12  # Presto composites to a 12-month grid
    assert int(months[0]) == 0  # calendar-ordered -> January start


def test_tessera_uses_full_native_series_not_12() -> None:
    from models.tessera import TesseraModel

    rng = np.random.default_rng(2)
    raw = rng.random((40, 11), dtype=np.float32) * 4000  # 40 native acquisitions
    bench = _native_bench([raw], [np.repeat(np.arange(12), 4)[:40]], S2_BANDS)
    streams = TesseraModel()._prepare_streams(bench)
    s2_len = max(s2.shape[1] for _, s2, _ in streams.values())
    assert s2_len >= 40  # full 40-acquisition series (bucketed), not collapsed to <=16


def test_all_native_bands_exposed_and_olmoearth_maps_b1_b9() -> None:
    from models.olmoearth import _BENCH_TO_OLMOEARTH_IDX

    assert "B1" in _BENCH_TO_OLMOEARTH_IDX and "B9" in _BENCH_TO_OLMOEARTH_IDX  # the all-bands gain
    all_bands = ["B1", "B2", "B3", "B4", "B5", "B6", "B7", "B8", "B8A", "B9", "B10", "B11", "B12", "NDVI"]
    bench = _native_bench([np.ones((6, 14), np.float32)], [np.arange(6)], all_bands)
    _vals, _mask, _doy, view_bands = bench.monthly("s2")
    assert "B1" in view_bands and "B9" in view_bands  # carried natively, available for OlmoEarth to map
