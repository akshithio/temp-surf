from __future__ import annotations

import json
from types import SimpleNamespace

import numpy as np
import pytest

from dataio.get_input import get_input
from evals.benchmarks.pastis_r import PastisBenchmark
from utils import cacheutils, ioutils


def test_get_input_rejects_unknown_benchmark() -> None:
    with pytest.raises(KeyError, match="Unknown benchmark"):
        get_input("not-a-benchmark")


def test_pastis_is_lazy_and_yields_64_pixel_tiles(tmp_path) -> None:
    base = tmp_path / "pastis_r"
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

    bench = get_input("pastis_r", root=tmp_path, shuffle=False)
    tiles = list(bench.iter_tiles())

    assert isinstance(bench, PastisBenchmark)
    assert bench.n_samples == 4
    assert len(tiles) == 4
    pixels = tiles[0][2].pixel_benchmark()
    assert pixels.n_samples == 64 * 64 - 1
    assert pixels.monthly("s2")[0].shape == (64 * 64 - 1, 12, 11)
    assert pixels.monthly("s1")[0].shape == (64 * 64 - 1, 12, 3)
    assert 19 not in tiles[0][3]


def test_summarize_rows_ignores_legacy_rows_missing_grouping_keys() -> None:
    rows = [
        {"model": "old", "f1": 0.0},
        {"model": "new", "split_regime": "random_id", "f1": 1.0},
    ]

    summary = ioutils.summarize_rows(rows, keys=["model", "split_regime"], metrics=["f1"])

    assert len(summary) == 1
    assert summary[0]["model"] == "new"
    assert summary[0]["split_regime"] == "random_id"
    assert summary[0]["mean_f1"] == 1.0


def test_load_cached_embeddings_requires_existing_matrix(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(cacheutils, "EMBEDDINGS_DIR", tmp_path)
    bench = SimpleNamespace(n_samples=2)

    with pytest.raises(FileNotFoundError, match="Embedding cache not found"):
        cacheutils.load_cached_embeddings(bench, "cropharvest", "presto", "tag")


def test_load_cached_embeddings_reads_existing_matrix(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(cacheutils, "EMBEDDINGS_DIR", tmp_path)
    bench = SimpleNamespace(n_samples=2)
    path = cacheutils.embedding_cache_path(bench, "cropharvest", "presto", "tag")
    path.parent.mkdir(parents=True)
    expected = np.arange(6, dtype=np.float16).reshape(2, 3)
    np.save(path, expected)

    actual = cacheutils.load_cached_embeddings(bench, "cropharvest", "presto", "tag")

    np.testing.assert_array_equal(actual, expected.astype(np.float32))


def test_pastis_tile_is_native_cadence_and_models_aggregate_their_own_way() -> None:
    """The PASTIS tile carries the native acquisition cadence; per-pixel models then aggregate it:
    TESSERA uses the full series, Presto/Galileo/OlmoEarth composite to a 12-month grid."""
    from evals.benchmarks.pastis_r import PastisTile, _monthly_patch

    rng = np.random.default_rng(0)
    t_native = 24  # native cadence, 2 acquisitions per calendar month (vs the old fixed 12)
    s2 = rng.random((t_native, 10, 4, 4)).astype(np.float32)
    s1 = rng.random((t_native, 3, 4, 4)).astype(np.float32)
    months = np.repeat(np.arange(12), 2).astype(np.int64)
    tile = PastisTile(
        s2=s2, s1=s1, s2_months=months, s1_months=months,
        s2_mask=np.ones(t_native, np.float32), s1_mask=np.ones(t_native, np.float32),
        labels=np.zeros((4, 4), np.int64), valid=np.ones((4, 4), bool), fold=1,
    )
    bench = tile.pixel_benchmark()
    assert bench.n_samples == 16

    # TESSERA-style native view: the FULL 24-acquisition cadence per pixel (10 bands + NDVI).
    s2_native, _doy, _m, _bands = bench.native_series("s2")
    assert s2_native[0].shape == (t_native, 11)

    # Presto-style monthly view: collapsed onto a 12-calendar-month grid (every month observed here).
    mv, mask, _doy, _b = bench.monthly("s2")
    assert mv.shape == (16, 12, 11) and mask.min() == 1.0

    # Galileo / OlmoEarth composite the native tile to the same 12-month grid in their encode path.
    s2_m, s2_mask = _monthly_patch(s2, months)
    assert s2_m.shape == (12, 10, 4, 4) and s2_mask.min() == 1.0


def test_input_footprint_and_s2_only_common_input_view() -> None:
    """#10: each model's input footprint is declared (stratifiable), and s2_only() yields the
    common-input view so a fairness table can compare every model on S2 alone."""
    from dataio.get_input import Benchmark, ModalitySeries, NativeSeries
    from evals import compat

    n = 4

    def _mod(c, bands):
        return ModalitySeries(
            [np.ones((3, c), np.float32) for _ in range(n)],
            [np.arange(3)] * n, [np.arange(3, dtype=np.float32)] * n, [np.full(3, 2020)] * n, bands,
        )

    bench = Benchmark(
        name="b", label_kind="binary",
        native=NativeSeries(_mod(2, ["B2", "B3"]), _mod(2, ["VV", "VH"]), _mod(1, ["temperature"])),
        labels=np.zeros(n, np.int64), groups=np.array(["a"] * n, dtype=object),
        latlon=np.array([[40.0, -3.0]] * n, np.float32), years=np.full(n, 2020, np.int64),
    )
    assert bench.available_modalities() == {"s2", "s1", "climate", "latlon", "time"}

    s2o = bench.s2_only()
    assert s2o.available_modalities() == {"s2", "time"}  # S1/climate emptied, coordinates zeroed
    assert s2o.native.s1.bands == [] and s2o.native.climate.bands == []
    assert np.all(s2o.latlon == 0.0)
    assert s2o.monthly("s1")[0].shape[2] == 0
    assert bench.native.s1.bands == ["VV", "VH"]  # original benchmark untouched (s2_only returns a copy)

    # Declared footprints expose the modality confound: Presto gets coordinates, OlmoEarth/AgriFM are S2-only.
    assert "latlon" in compat.input_modalities("presto")
    assert "s1" not in compat.input_modalities("olmoearth")
    assert compat.input_modalities("agrifm") == ("s2",)
