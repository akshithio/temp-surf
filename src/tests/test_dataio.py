from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import xarray as xr

from dataio.get_input import (
    Benchmark,
    CH_CLIMATE_BANDS,
    CH_S1_BANDS,
    CH_S2_BANDS,
    YS_CLIMATE_BANDS,
    YS_CLIMATE_SOURCE_BANDS,
    YS_COORD_BANDS,
    YS_COUNTRIES,
    YS_NETCDF_NAME,
    YS_S2_BANDS,
    YS_S2_SOURCE_BANDS,
    _resample_to,
    _ys_doy,
    _ys_latlon_from_unit_xyz,
    corrupt,
    get_input,
    load_yieldsat,
)


def _tiny_benchmark() -> Benchmark:
    n, t = 3, 4
    return Benchmark(
        name="tiny",
        task="regression",
        s2=np.arange(n * t * 11, dtype=np.float32).reshape(n, t, 11),
        s1=np.ones((n, t, 2), dtype=np.float32),
        climate=np.full((n, t, 3), 2.0, dtype=np.float32),
        s2_mask=np.ones((n, t), dtype=np.float32),
        s1_mask=np.ones((n, t), dtype=np.float32),
        climate_mask=np.ones((n, t), dtype=np.float32),
        doy=np.tile(np.arange(1, t + 1, dtype=np.float32), (n, 1)),
        labels=np.arange(n, dtype=np.float32),
        groups=np.array(["a", "b", "c"], dtype=object),
        latlon=np.zeros((n, 2), dtype=np.float32),
        s2_bands=CH_S2_BANDS,
        s1_bands=CH_S1_BANDS,
        climate_bands=CH_CLIMATE_BANDS,
    )


def test_corrupt_sensor_off() -> None:
    bench = _tiny_benchmark()

    out = corrupt(bench, sensor_off="s2")

    assert np.all(out.s2 == 0)
    assert np.all(out.s2_mask == 0)
    np.testing.assert_array_equal(out.s1, bench.s1)
    np.testing.assert_array_equal(out.climate, bench.climate)


def test_corrupt_temporal_drop() -> None:
    bench = _tiny_benchmark()

    out = corrupt(bench, temporal_drop=0.99, seed=3)

    assert np.all(out.s2_mask[:, 0] == 1)
    assert np.all(out.s1_mask[:, 0] == 1)
    assert np.all(out.climate_mask[:, 0] == 1)
    assert np.all(out.s2_mask.sum(axis=1) >= 2)
    assert np.all(out.s1_mask.sum(axis=1) >= 2)
    assert np.all(out.climate_mask.sum(axis=1) >= 2)


def test_resample_to() -> None:
    arr = np.arange(5 * 2, dtype=np.float32).reshape(5, 2)
    dates = np.array(["2020-01-01", "2020-01-11", "2020-01-21", "2020-02-01", "2020-02-11"], dtype="datetime64[D]")

    down, down_mask, down_doy = _resample_to(arr, dates, timesteps=3)
    pad, pad_mask, pad_doy = _resample_to(arr[:2], dates[:2], timesteps=4)

    np.testing.assert_array_equal(down, arr[[0, 2, 4]])
    np.testing.assert_array_equal(down_mask, np.ones(3, dtype=np.float32))
    np.testing.assert_array_equal(down_doy, np.array([1, 21, 42], dtype=np.float32))
    np.testing.assert_array_equal(pad[:2], arr[:2])
    np.testing.assert_array_equal(pad[2:], np.zeros((2, 2), dtype=np.float32))
    np.testing.assert_array_equal(pad_mask, np.array([1, 1, 0, 0], dtype=np.float32))
    np.testing.assert_array_equal(pad_doy, np.array([1, 11, 0, 0], dtype=np.float32))


def test_get_input_rejects_unknown_benchmark() -> None:
    with pytest.raises(KeyError, match="Unknown benchmark"):
        get_input("not-a-benchmark")


def _write_yieldsat_country(path: Path, offset: float) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    bands = YS_S2_SOURCE_BANDS + YS_CLIMATE_SOURCE_BANDS + YS_COORD_BANDS
    n_index, n_time = 3, 24
    sample = np.zeros((n_index, n_time, len(bands)), dtype=np.float32)

    for band_i, band in enumerate(bands):
        sample[:, :, band_i] = offset + band_i
    sample[:, :, bands.index("B04")] = 2.0
    sample[:, :, bands.index("B08")] = 6.0
    sample[:, :, bands.index("temp_mean")] = 280.0 + offset
    sample[:, :, bands.index("total_prec")] = 0.2 + offset
    sample[:, :, bands.index("dem")] = 100.0 + offset
    sample[:, :, bands.index("coord_x")] = 1.0
    sample[:, :, bands.index("coord_y")] = 0.0
    sample[:, :, bands.index("coord_z")] = 0.0

    times = np.tile(
        np.arange("2021-01-01", "2021-01-25", dtype="datetime64[D]"),
        (n_index, 1),
    )
    ds = xr.Dataset(
        data_vars={
            "sample": (("index", "time_step", "band"), sample),
            "target": (("index",), np.array([2.0, 4.0, 10.0], dtype=np.float32) + offset),
            "times": (("index", "time_step"), times),
            "field_shared_name": (("index",), np.array([0, 0, 1], dtype=np.int32)),
        },
        coords={
            "index": np.arange(n_index),
            "time_step": np.arange(n_time),
            "band": np.array(bands, dtype=object),
        },
    )
    ds["field_shared_name"].attrs.update({"0": "field_a", "1": "field_b"})
    ds.to_netcdf(path)


def test_yieldsat_loader_aggregates_pixels_to_fields(tmp_path: Path) -> None:
    for i, country in enumerate(YS_COUNTRIES):
        _write_yieldsat_country(
            tmp_path / "yieldsat" / "preprocessed-24-ts" / country / YS_NETCDF_NAME,
            offset=float(i),
        )

    bench = load_yieldsat(tmp_path, max_samples=2, shuffle=False)

    assert bench.name == "yieldsat"
    assert bench.task == "regression"
    assert bench.n_samples == 2
    assert bench.timesteps == 24
    assert bench.s2.shape == (2, 24, len(YS_S2_BANDS))
    assert bench.s1.shape == (2, 24, 2)
    assert bench.climate.shape == (2, 24, len(YS_CLIMATE_BANDS))
    assert bench.s2_bands == YS_S2_BANDS
    assert bench.climate_bands == YS_CLIMATE_BANDS
    assert set(bench.groups.tolist()) == {"Argentina"}
    np.testing.assert_allclose(bench.labels, np.array([3.0, 10.0], dtype=np.float32))
    np.testing.assert_allclose(bench.s2[:, :, -1], 0.5)
    np.testing.assert_array_equal(bench.s1_mask, np.zeros((2, 24), dtype=np.float32))
    np.testing.assert_allclose(bench.latlon, np.zeros((2, 2), dtype=np.float32), atol=1e-6)


def test_yieldsat_helpers_decode_doy_and_unit_xyz() -> None:
    doy = _ys_doy(np.array(["2020-01-01", "2020-12-31"], dtype="datetime64[D]"))
    assert doy.tolist() == [1.0, 366.0]
    lat, lon = _ys_latlon_from_unit_xyz(0.0, 1.0, 0.0, fallback=(10.0, 20.0))
    assert abs(lat) < 1e-6
    assert abs(lon - 90.0) < 1e-6
    assert _ys_latlon_from_unit_xyz(0.0, 0.0, 0.0, fallback=(10.0, 20.0)) == (10.0, 20.0)
