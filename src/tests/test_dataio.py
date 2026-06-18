from __future__ import annotations

import json

import numpy as np
import pytest

from dataio.get_input import (
    CH_CLIMATE_BANDS,
    CH_S1_BANDS,
    CH_S2_BANDS,
    Benchmark,
    PastisBenchmark,
    _resample_to,
    degrade,
    get_input,
)
from utils import ioutils


def _tiny_benchmark() -> Benchmark:
    n, t = 3, 4
    return Benchmark(
        name="tiny",
        task="regression",
        s2=np.arange(n * t * 11, dtype=np.float32).reshape(n, t, 11),
        s1=np.ones((n, t, 2), dtype=np.float32),
        climate=np.full((n, t, len(CH_CLIMATE_BANDS)), 2.0, dtype=np.float32),
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


def test_degraded_sensor_off() -> None:
    bench = _tiny_benchmark()

    out = degrade(bench, sensor_off="s2")

    assert np.all(out.s2 == 0)
    assert np.all(out.s2_mask == 0)
    np.testing.assert_array_equal(out.s1, bench.s1)
    np.testing.assert_array_equal(out.climate, bench.climate)


def test_degraded_temporal_drop() -> None:
    bench = _tiny_benchmark()

    out = degrade(bench, temporal_drop=0.99, seed=3)

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


def test_pastis_is_lazy_and_yields_64_pixel_tiles(tmp_path) -> None:
    base = tmp_path / "pastis"
    for directory in ("DATA_S2", "DATA_S1A", "ANNOTATIONS"):
        (base / directory).mkdir(parents=True)
    properties = {
        "ID_PATCH": 10000,
        "Fold": 1,
        "dates-S2": {"0": 20190115, "1": 20190215},
        "dates-S1A": {"0": 20190110, "1": 20190210},
    }
    (base / "metadata.geojson").write_text(
        json.dumps({"type": "FeatureCollection", "features": [{"type": "Feature", "properties": properties}]})
    )
    np.save(base / "DATA_S2" / "S2_10000.npy", np.ones((2, 10, 128, 128), dtype=np.int16))
    np.save(base / "DATA_S1A" / "S1A_10000.npy", np.ones((2, 3, 128, 128), dtype=np.float16))
    target = np.zeros((3, 128, 128), dtype=np.uint8)
    target[0, 0, 0] = 19
    np.save(base / "ANNOTATIONS" / "TARGET_10000.npy", target)

    bench = get_input("pastis", root=tmp_path, shuffle=False)
    tiles = list(bench.iter_tiles())

    assert isinstance(bench, PastisBenchmark)
    assert bench.n_samples == 4
    assert len(tiles) == 4
    pixels = tiles[0][2].pixel_benchmark()
    assert pixels.n_samples == 64 * 64 - 1
    assert pixels.s2.shape == (64 * 64 - 1, 12, 11)
    assert pixels.s1.shape == (64 * 64 - 1, 12, 3)
    assert 19 not in tiles[0][3]


def test_summarize_rows_ignores_legacy_rows_missing_grouping_keys() -> None:
    rows = [
        {"encoder": "old", "f1": 0.0},
        {"encoder": "new", "split_regime": "random_id", "f1": 1.0},
    ]

    summary = ioutils.summarize_rows(rows, keys=["encoder", "split_regime"], metrics=["f1"])

    assert len(summary) == 1
    assert summary[0]["encoder"] == "new"
    assert summary[0]["split_regime"] == "random_id"
    assert summary[0]["mean_f1"] == 1.0
