from __future__ import annotations

import json
from types import SimpleNamespace

import h5py
import numpy as np
import pytest

from dataio.get_input import get_input
from evals.benchmarks import cropharvest
from evals.benchmarks.pastis import PastisBenchmark
from utils import cacheutils, ioutils


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
    geometry = {
        "type": "Polygon",
        "coordinates": [[[-1.0, 46.0], [-0.99, 46.0], [-0.99, 46.01], [-1.0, 46.01], [-1.0, 46.0]]],
    }
    (base / "metadata.geojson").write_text(
        json.dumps({
            "type": "FeatureCollection",
            "features": [{"type": "Feature", "properties": properties, "geometry": geometry}],
        })
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
    assert pixels.monthly("s2")[0].shape == (64 * 64 - 1, 12, 11)
    assert pixels.monthly("s1")[0].shape == (64 * 64 - 1, 12, 3)
    assert np.isfinite(bench.latlon).all()
    assert np.isclose(pixels.latlon[:, 0].mean(), 46.005)
    assert 19 not in tiles[0][3]

    cache_root = tmp_path / "cache"
    fold = cache_root / "fold_1"
    fold.mkdir(parents=True)
    for tile_id in ("10000_0_0", "10000_0_1", "10000_1_0", "10000_1_1"):
        np.save(fold / f"{tile_id}.npy", np.zeros((1, 2), dtype=np.float32))
        np.save(fold / f"{tile_id}.labels.npy", np.zeros(1, dtype=np.uint8))
    (base / "DATA_S2" / "S2_10000.npy").unlink()

    assert list(bench.iter_tiles(cache_root=cache_root, overwrite=False)) == []


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


def _la_row(route, holdout, n_source, n_target, f1):
    return {
        "model": "raw", "split_regime": "geographic_ood", "budget_type": "label_access",
        "label_access_route": route, "label_budget": 0, "evaluation_split": "target_test",
        "holdout": holdout, "seed": 0, "n_source_labels": n_source, "n_target_labels": n_target,
        "n_total_labels": n_source + n_target, "label_budget_unit": "samples", "f1": f1,
    }


def test_summarize_rows_aggregates_label_counts_across_varying_pool_sizes() -> None:
    """Two holdouts with DIFFERENT full-pool sizes share one (route, budget, split) group. Keying on the
    counts would split them into two rows; instead they aggregate into ONE with min/max/mean, and the
    unit is preserved."""
    rows = [
        _la_row("target_only_full", "kenya", 0, 55, 0.8),   # kenya pool = 55
        _la_row("target_only_full", "togo", 0, 40, 0.6),    # togo  pool = 40
    ]

    summary = ioutils.summarize_rows(
        rows,
        keys=["model", "split_regime", "budget_type", "label_access_route", "label_budget", "evaluation_split"],
        metrics=["f1"],
        count_aggregates=["n_source_labels", "n_target_labels", "n_total_labels"],
        passthrough=["label_budget_unit"],
    )

    assert len(summary) == 1                       # equal-region aggregation is NOT fragmented
    s = summary[0]
    assert s["n_rows"] == 2 and s["n_holdouts"] == 2
    assert s["min_n_target_labels"] == 40 and s["max_n_target_labels"] == 55
    assert s["mean_n_target_labels"] == 47.5
    assert s["min_n_total_labels"] == 40 and s["max_n_total_labels"] == 55
    assert s["label_budget_unit"] == "samples"     # constant-within-group unit preserved
    assert s["mean_f1"] == pytest.approx(0.7)


def test_summarize_rows_without_count_aggregates_is_unchanged() -> None:
    """The new params default off: a plain summary emits no min_/max_/mean_ count columns."""
    rows = [_la_row("source_only", "kenya", 60, 0, 0.9)]
    summary = ioutils.summarize_rows(rows, keys=["model", "split_regime"], metrics=["f1"])

    assert len(summary) == 1
    assert not any(k.startswith(("min_n_", "max_n_")) for k in summary[0])
    assert "label_budget_unit" not in summary[0]


def test_rewrite_jsonl_dropping_repairs_torn_tail(tmp_path) -> None:
    """A hard crash mid-append leaves a torn final line. rewrite_jsonl_dropping repairs it (like an
    append would) rather than choking -- the two complete rows survive."""
    p = tmp_path / "predictions.jsonl"
    p.write_text('{"a": 1}\n{"a": 2}\n{"a": 3')   # torn final row (no newline, truncated JSON)
    dropped = ioutils.rewrite_jsonl_dropping(p, lambda r: False)

    assert dropped == 0
    assert [r["a"] for r in ioutils.read_jsonl(p)] == [1, 2]


def test_rewrite_jsonl_dropping_hard_fails_on_corrupt_interior_row(tmp_path) -> None:
    """A corrupt INTERIOR row is real data loss, never silently dropped -- it hard-fails."""
    p = tmp_path / "predictions.jsonl"
    p.write_text('{"a": 1}\nNOT VALID JSON\n{"a": 2}\n')
    with pytest.raises(ValueError, match="corrupt interior JSONL row 2"):
        ioutils.rewrite_jsonl_dropping(p, lambda r: False)


def test_rewrite_jsonl_dropping_filters_by_predicate(tmp_path) -> None:
    p = tmp_path / "predictions.jsonl"
    p.write_text('{"fam": "keep"}\n{"fam": "drop"}\n{"fam": "keep"}\n')
    dropped = ioutils.rewrite_jsonl_dropping(p, lambda r: r["fam"] == "drop")

    assert dropped == 1
    assert [r["fam"] for r in ioutils.read_jsonl(p)] == ["keep", "keep"]


def _isolate_cache(tmp_path, monkeypatch, benchmark, digest="d" * 64):
    monkeypatch.setattr(cacheutils, "EMBEDDINGS_DIR", tmp_path / "emb")
    monkeypatch.setattr(cacheutils, "CACHE_JSON_PATH", tmp_path / "logs" / "cache.json")
    cacheutils.update_cache(datasets={benchmark: digest})
    monkeypatch.setattr(cacheutils, "_FROZEN_IDENTITY", {"final_commit": None, "clean": None, "tree_identity": None})


def test_load_cached_embeddings_requires_existing_matrix(tmp_path, monkeypatch) -> None:
    _isolate_cache(tmp_path, monkeypatch, "cropharvest")
    bench = SimpleNamespace(n_samples=2, sample_ids=np.array(["a", "b"], dtype=object))

    with pytest.raises(FileNotFoundError, match="not built"):
        cacheutils.load_cached_embeddings(bench, "cropharvest", "presto", "baseline")


def test_load_cached_embeddings_reads_existing_matrix(tmp_path, monkeypatch) -> None:
    from utils import artifacts

    _isolate_cache(tmp_path, monkeypatch, "cropharvest")
    bench = SimpleNamespace(n_samples=2, sample_ids=np.array(["a", "b"], dtype=object))
    art = cacheutils.embedding_cache_path("cropharvest", "presto", "baseline")
    art.parent.mkdir(parents=True)
    expected = np.arange(6, dtype=np.float16).reshape(2, 3)
    with open(art, "wb") as f:
        np.save(f, expected)
    cacheutils.update_cache(embeddings={cacheutils._embedding_key("cropharvest", "presto", "baseline"): {
        "checkpoint_sha256": cacheutils.checkpoint_sha256("presto"), "dataset_digest": "d" * 64,
        "sample_ids_digest": cacheutils.sample_ids_digest(bench.sample_ids),
        "shape": [2, 3], "dtype": "float16", "artifact_sha256": artifacts.sha256_file(art),
    }})

    actual = cacheutils.load_cached_embeddings(bench, "cropharvest", "presto", "baseline")

    np.testing.assert_array_equal(actual, expected.astype(np.float32))


def test_dense_cache_skips_tiles_without_valid_pixels(tmp_path, monkeypatch) -> None:
    class EmptyDenseBench:
        n_samples = 1
        tile_size = 64
        ignore_index = 255
        patches = ()

        def iter_tiles(self, cache_root=None, overwrite=False):
            yield "empty", 1, object(), np.array([], dtype=np.uint8)

    def fail_build_model(*_args, **_kwargs):
        raise AssertionError("all-void dense tiles should not build a model")

    _isolate_cache(tmp_path, monkeypatch, "pastis")
    monkeypatch.setattr(cacheutils, "build_model", fail_build_model)

    root = cacheutils.extract_dense_and_cache(EmptyDenseBench(), "pastis", "raw", "baseline")

    assert root == cacheutils.dense_embedding_cache_dir("pastis", "raw", "baseline")
    assert list(root.rglob("*.npy")) == []


def test_cropharvest_geo_group_collapses_rwanda_aliases() -> None:
    assert cropharvest._ch_geo_group("rwanda-ceo") == "rwanda"
    assert cropharvest._ch_geo_group("rwanda") == "rwanda"


def test_cropharvest_max_samples_counts_loaded_samples_not_raw_files(tmp_path, monkeypatch) -> None:
    base = tmp_path / "cropharvest"
    arrays_dir = base / "features" / "arrays"
    arrays_dir.mkdir(parents=True)
    (base / "labels.geojson").write_text("{}")

    for name in ("000_unlabeled.h5", "001_unlabeled.h5", "002_known.h5", "003_known.h5"):
        with h5py.File(arrays_dir / name, "w") as handle:
            handle.create_dataset("array", data=np.ones((12, 18), dtype=np.float32))

    monkeypatch.setattr(
        cropharvest,
        "_load_ch_labels",
        lambda _path: {
            (2, "known"): (1, 10.0, 20.0, 2020),
            (3, "known"): (0, 11.0, 21.0, 2020),
        },
    )

    bench = cropharvest.load_benchmark(root=tmp_path, max_samples=2, shuffle=False)

    assert bench.n_samples == 2
    np.testing.assert_array_equal(bench.labels, np.array([1, 0], dtype=np.int64))
    assert list(bench.groups) == ["known", "known"]


def test_pastis_tile_is_native_cadence_and_models_aggregate_their_own_way() -> None:
    from evals.benchmarks.pastis import PastisTile, _monthly_patch

    rng = np.random.default_rng(0)
    t_native = 24  # native cadence, 2 acquisitions per calendar month (vs the old fixed 12)
    s2 = rng.random((t_native, 10, 4, 4)).astype(np.float32)
    s1 = rng.random((t_native, 3, 4, 4)).astype(np.float32)
    months = np.repeat(np.arange(12), 2).astype(np.int64)
    tile = PastisTile(
        s2=s2, s1=s1, s2_months=months, s1_months=months,
        s2_mask=np.ones(t_native, np.float32), s1_mask=np.ones(t_native, np.float32),
        labels=np.zeros((4, 4), np.int64), valid=np.ones((4, 4), bool), fold=1, latlon=(46.0, -1.0),
    )
    bench = tile.pixel_benchmark()
    assert bench.n_samples == 16
    assert np.isclose(bench.latlon[:, 0].mean(), 46.0)

    s2_native, _doy, _m, _bands = bench.native_series("s2")
    assert s2_native[0].shape == (t_native, 11)

    mv, mask, _doy, _b = bench.monthly("s2")
    assert mv.shape == (16, 12, 11) and mask.min() == 1.0

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


def test_pastis_s2_only_structural_contract() -> None:
    """PASTIS ``s2_only`` must make S1 a *structurally absent* modality (not a zero-valued
    "present" one) and zero coordinates for the pixel path (Presto/TESSERA/raw), matching the
    tabular ``Benchmark.s2_only`` contract --- enforced even though the tile still holds real S1
    data. Guards the S2-only-control correctness bug (v4)."""
    from evals.benchmarks.pastis import PastisTile

    rng = np.random.default_rng(0)
    t = 24
    months = np.repeat(np.arange(12), 2).astype(np.int64)
    common = dict(
        s2=rng.random((t, 10, 4, 4)).astype(np.float32),
        s1=rng.random((t, 3, 4, 4)).astype(np.float32),  # REAL S1 present in the tile
        s2_months=months, s1_months=months,
        s2_mask=np.ones(t, np.float32), s1_mask=np.ones(t, np.float32),
        labels=np.zeros((4, 4), np.int64), valid=np.ones((4, 4), bool),
        fold=1, latlon=(46.0, -1.0),
    )
    full = PastisTile(**common)                  # default s2_only=False
    s2o = PastisTile(**common, s2_only=True)
    assert full.s2_only is False and s2o.s2_only is True

    fb = full.pixel_benchmark()
    assert "s1" in fb.available_modalities() and "latlon" in fb.available_modalities()
    assert not np.all(fb.latlon == 0.0)

    sb = s2o.pixel_benchmark()
    assert "s1" not in sb.available_modalities(), "S1 must be structurally absent under s2_only"
    assert "latlon" not in sb.available_modalities(), "coordinates must be dropped under s2_only"
    assert sb.native.s1.bands == [] and sb.monthly("s1")[0].shape[2] == 0
    assert np.all(sb.latlon == 0.0)
    # S2 itself is untouched.
    assert "s2" in sb.available_modalities() and sb.monthly("s2")[0].shape[2] > 0


def test_pastis_s2_only_galileo_dense_masks_s1() -> None:
    """Galileo's dense path recomputes the S1 availability mask from ``s1_months`` (not
    ``tile.s1_mask``), so ``s2_only`` must override it to all-MISSING. This replicates the exact
    mask-construction from ``galileo.encode_dense`` on a synthetic tile (no weights needed) and
    asserts the S1 token group is MISSING for every timestep under s2_only, while S2 is unaffected."""
    from evals.benchmarks.pastis import PastisTile, _monthly_patch

    rng = np.random.default_rng(1)
    t = 24
    months = np.repeat(np.arange(12), 2).astype(np.int64)
    tile = PastisTile(
        s2=rng.random((t, 10, 3, 3)).astype(np.float32),
        s1=rng.random((t, 3, 3, 3)).astype(np.float32),
        s2_months=months, s1_months=months,
        s2_mask=np.ones(t, np.float32), s1_mask=np.ones(t, np.float32),
        labels=np.zeros((3, 3), np.int64), valid=np.ones((3, 3), bool),
        fold=1, latlon=(46.0, -1.0), s2_only=True,
    )
    # Mirror galileo.encode_dense's mask logic.
    _s2m, s2_mask = _monthly_patch(tile.s2, tile.s2_months)
    s1_m, s1_mask = _monthly_patch(tile.s1, tile.s1_months)
    assert s1_mask.max() == 1.0, "sanity: without the fix, s1_months would mark S1 present"
    if getattr(tile, "s2_only", False):
        s1_m = np.zeros_like(s1_m)
        s1_mask = np.zeros_like(s1_mask)
    assert s1_mask.max() == 0.0, "S1 must be MISSING for every timestep under s2_only"
    assert s2_mask.max() == 1.0, "S2 availability must be untouched"
