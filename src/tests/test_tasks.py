from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from evals import evals as EV
from evals import probes
from evals.benchmarks import breizhcrops, cropharvest, eurocropsml, pastis
from evals.regimes import base as regime_base
from evals.regimes import spatial_cluster_ood
from utils import cacheutils, gputils
from utils import ioutils as IOU


def test_bin_crop_class_targets() -> None:
    bench = SimpleNamespace(labels=np.array([0.0, 1.0]), groups=np.array(["a", "b"], dtype=object))

    y, groups = cropharvest.make_targets(bench)

    assert cropharvest.BENCHMARK == "cropharvest"
    assert cropharvest.LABEL_KIND == "binary"
    assert y.dtype == np.int64
    np.testing.assert_array_equal(y, np.array([0, 1]))
    np.testing.assert_array_equal(groups, bench.groups)


def test_prune_partial_budgets_prunes_rows_and_preds_without_collision(tmp_path):
    from utils import runstate

    rows_path = tmp_path / "probe_results.jsonl"
    preds_path = tmp_path / "predictions.jsonl"
    row = {
        "seed": 0, "split_regime": "random_id", "holdout": "id", "method": "erm",
        "probe_family": "logistic", "budget_type": "source", "label_budget": 1.0,
        "evaluation_split": "test",
    }
    other = {**row, "seed": 1}  # different budget key -> must be KEPT
    IOU.append_jsonl(rows_path, [row, other])
    IOU.append_jsonl(preds_path, [row, other])

    kept = runstate.prune_partial_budgets([row, other], rows_path, preds_path, {runstate.budget_row_key(row)})

    # the matching (seed 0) entry is pruned from BOTH files; the non-matching (seed 1) survives;
    # rows and preds are rewritten independently (no temp-path collision) and no temp files remain.
    assert [r["seed"] for r in kept] == [1]
    assert [r["seed"] for r in IOU.read_jsonl(rows_path)] == [1]
    assert [p["seed"] for p in IOU.read_jsonl(preds_path)] == [1]
    assert not list(tmp_path.glob("*.tmp")) and not list(tmp_path.glob("*.prune.tmp"))


def test_read_jsonl_only_tolerates_unterminated_final_row(tmp_path) -> None:
    p = tmp_path / "rows.jsonl"
    p.write_text('{"a": 1}\n{"bad":')
    assert IOU.read_jsonl(p) == [{"a": 1}]
    p.write_text('{"a": 1}\n{"bad":\n{"a": 2}\n')
    with pytest.raises(ValueError, match="Corrupt JSONL row 2"):
        IOU.read_jsonl(p)


def test_eurocrops_official_splits_use_release_membership(tmp_path) -> None:
    base = tmp_path / "eurocropsml" / "split" / "latvia_vs_estonia"
    (base / "pretrain").mkdir(parents=True)
    (base / "finetune").mkdir()
    (base / "pretrain" / "region_split.json").write_text('{"train":["LV_a.npz"],"val":["LV_b.npz"]}')
    (base / "finetune" / "region_split_all.json").write_text(
        '{"train":["EE_a.npz"],"val":["EE_b.npz"],"test":["EE_c.npz"]}'
    )
    splits = eurocropsml._official_splits(tmp_path / "eurocropsml", ["LV_a.npz", "LV_b.npz", "EE_c.npz"])
    split = splits["latvia_vs_estonia"]
    assert split["train"].tolist() == [0]
    assert split["val"].tolist() == [1]
    assert split["test"].tolist() == [2]


def test_cropharvest_geo_group_collapses_dataset_aliases() -> None:
    assert cropharvest._ch_geo_group("togo-eval") == "togo"
    assert cropharvest._ch_geo_group("togo") == "togo"
    assert cropharvest._ch_geo_group("lem-brazil") == "lem-brazil"


def test_cropharvest_geographic_domains_are_canonical_source_domains() -> None:
    """geographic_ood must leave out canonical domains, not 2-degree coordinate blocks.

    The block basis silently redefined the holdout universe: it produced hundreds of anonymous
    `block_*` labels instead of the 18 named domains the benchmark actually carries.
    """
    bench = SimpleNamespace(
        groups=np.array(["kenya", "kenya", "togo", "geowiki-landcover-2017"], dtype=object),
        latlon=np.array([[0.0, 0.0], [0.5, 0.5], [5.0, 5.0], [np.nan, 0.0]]),
    )

    domains = cropharvest.geographic_domains(bench)

    assert list(domains) == ["kenya", "kenya", "togo", "geowiki-landcover-2017"]
    assert not any(str(d).startswith("block_") for d in domains)


def test_cropharvest_geographic_split_is_leave_one_domain_out() -> None:
    """The curated five-region subset must not define the geographic_ood universe."""
    spec = cropharvest.GEOGRAPHIC_SPLIT

    assert spec["strategy"] == "leave_one_domain_out"
    assert spec["purge_km"] == 50.0
    # one-class domains are real regions and are evaluated, not discarded
    assert spec["allow_one_class_target"] is True
    assert not hasattr(cropharvest, "GEOGRAPHIC_HOLDOUTS")


def test_cropharvest_retains_the_block_basis_for_historical_artifacts() -> None:
    """The pre-LODO basis still exists and still handles missing coordinates.

    Exact reproduction of the canonical `spatial_block_2deg_purge50km` split -- all 2,496 block
    names and the final per-sample partitions -- is pinned in test_golden_spatial_block_split.py
    against the real artifact; this only covers the no-coordinate edge case that has no canonical
    counterpart (unlocatable samples never appear in the manifest's domain lists).
    """
    bench = SimpleNamespace(latlon=np.array([[0.0, 0.0], [np.nan, 0.0], [1.0, np.nan]]))

    domains = cropharvest.spatial_block_domains(bench)

    assert cropharvest.GEOGRAPHIC_BLOCK_DEGREES == 2.0
    assert domains[0].startswith("block_")
    assert domains[1] == "unknown"
    assert domains[2] == "unknown"


def test_spatial_cluster_ood_uses_coordinate_clusters() -> None:
    centers = np.array([[0.0, 0.0], [0.0, 5.0], [5.0, 0.0], [5.0, 5.0], [10.0, 0.0], [10.0, 5.0]])
    latlon = np.vstack([c + 0.01 * np.array([i, -i]) for c in centers for i in range(6)])
    bench = SimpleNamespace(name="unknown_bench", latlon=latlon)
    y = np.array([0, 1] * (len(latlon) // 2), dtype=np.int64)

    domains = spatial_cluster_ood.assign_domains(bench)
    [split] = list(spatial_cluster_ood.iter_splits(y, domains, seed=0, bench=bench))

    assert len(set(domains.astype(str)) - {"unknown"}) >= 3
    assert set(split.train).isdisjoint(split.val)
    assert set(split.train).isdisjoint(split.test)
    assert set(split.val).isdisjoint(split.test)


def test_spatial_cluster_regime_is_available_for_located_benchmarks() -> None:
    for mod in (cropharvest, eurocropsml, breizhcrops, pastis):
        assert "spatial_cluster_ood" in mod.SPLIT_REGIMES


def test_best_f1_threshold_falls_back_when_grid_scores_zero(capsys) -> None:
    probes._F1_THRESHOLD_WARNED = False
    threshold = probes.best_f1_threshold(np.array([0, 1]), np.array([np.nan, np.nan]))
    assert threshold == 0.5
    assert "degenerate" in capsys.readouterr().out


def test_test_optimal_binary_metrics_are_not_aggregate_metrics() -> None:
    assert "calibrated_f1_target_optimal" not in EV.METRICS_BINARY
    assert "optimal_threshold_test" not in EV.METRICS_BINARY
    assert "diagnostic_calibrated_f1_target_optimal" in EV.METRIC_ROLES["binary"]["diagnostic"]


def test_hf_default_checkpoint_sha256_tracks_local_bytes(tmp_path, monkeypatch) -> None:
    path = tmp_path / "models" / "presto" / "model-f317d103.pth"
    path.parent.mkdir(parents=True)
    path.write_bytes(b"AAAA")
    monkeypatch.setattr(cacheutils, "_INPUT_BASE", tmp_path)
    cacheutils._CHECKPOINT_SHA_CACHE.clear()
    first = cacheutils.checkpoint_sha256("presto")
    path.write_bytes(b"BBBBB")
    cacheutils._CHECKPOINT_SHA_CACHE.clear()
    assert cacheutils.checkpoint_sha256("presto") != first


def test_invalid_shard_config_raises(monkeypatch) -> None:
    monkeypatch.setenv(gputils.SHARD_ENV, "4")
    monkeypatch.setenv(gputils.NUM_SHARDS_ENV, "4")
    with pytest.raises(ValueError):
        gputils.take_shard([1, 2, 3])


def test_dense_training_sample_is_pixel_random_not_tile_balanced(tmp_path) -> None:
    fold_dir = tmp_path / "fold_1"
    fold_dir.mkdir()
    np.save(fold_dir / "100_0_0.labels.npy", np.zeros(100, dtype=np.int64))
    np.save(fold_dir / "100_0_0.npy", np.zeros((100, 2), dtype=np.float32))
    np.save(fold_dir / "999_0_0.labels.npy", np.ones(2, dtype=np.int64))
    np.save(fold_dir / "999_0_0.npy", np.ones((2, 2), dtype=np.float32))
    _x, y, _groups, _tile_ids, patch_ids = cacheutils.load_dense_samples(tmp_path, {1}, 20, seed=0)
    assert len(y) == 20
    assert 999 not in set(patch_ids.tolist())


def test_source_budget_probe_scores_source_diagnostics() -> None:
    rng = np.random.default_rng(3)
    x_train = rng.normal(size=(40, 4))
    y_train = np.array([0, 1] * 20)
    x_test = rng.normal(size=(6, 4))
    y_test = np.array([0, 1, 0, 1, 0, 1])
    x_diag = rng.normal(size=(8, 4))
    y_diag = np.array([0, 1] * 4)
    rows: list[dict] = []
    preds: list[dict] = []

    EV.run_probes(
        rows,
        x_train,
        x_test,
        y_train,
        y_test,
        seed=0,
        budgets=[1.0],
        meta={"model": "m", "benchmark": "b", "method": "erm", "split_regime": "geographic_ood"},
        predictions=preds,
        extra_evals={
            "source_validation": (x_diag[:4], y_diag[:4], np.arange(10, 14), np.array(["src"] * 4)),
            "source_test": (x_diag[4:], y_diag[4:], np.arange(20, 24), np.array(["src"] * 4)),
        },
    )

    assert {r["evaluation_split"] for r in rows} == {"test", "source_validation", "source_test"}
    assert {p["evaluation_split"] for p in preds} == {"test", "source_validation", "source_test"}


def test_spatial_cluster_pastis_split_is_patch_level(tmp_path) -> None:
    patch_latlon: dict[int, tuple[float, float]] = {}
    for i in range(15):
        fold = (i % 5) + 1
        patch = 10_000 + i
        fold_dir = tmp_path / f"fold_{fold}"
        fold_dir.mkdir(exist_ok=True)
        np.save(fold_dir / f"{patch}_0_0.labels.npy", np.array([0, 1], dtype=np.int64))
        patch_latlon[patch] = (40.0 + i, -5.0 + (i % 3))
    bench = SimpleNamespace(name="pastis", patch_latlon=patch_latlon)

    [(regime, cfg)] = list(
        regime_base.segmentation_fold_configs(
            pastis,
            ["spatial_cluster_ood"],
            seed=0,
            emb_dir=tmp_path,
            strict_mode=True,
            bench=bench,
        )
    )

    assert regime == "spatial_cluster_ood"
    assert cfg.label == "spatial_cluster_purge2km"
    assert cfg.train_patches and cfg.val_patches and cfg.test_patches
    assert cfg.train_patches.isdisjoint(cfg.val_patches)
    assert cfg.train_patches.isdisjoint(cfg.test_patches)
    assert cfg.val_patches.isdisjoint(cfg.test_patches)
    assert cfg.has_target is True


def test_crop_class() -> None:
    label_names = ["3301010500", "3301019999", "3302000000"]
    bench = SimpleNamespace(labels=np.array([0, 1, 2, 0]), groups=np.array(["LV", "LV", "EE", "PT"], dtype=object), label_names=label_names)

    y, groups = eurocropsml.make_targets(bench)

    assert eurocropsml.BENCHMARK == "eurocropsml"
    assert eurocropsml.LABEL_KIND == "multiclass"
    np.testing.assert_array_equal(y, np.array([0, 0, 1, 0]))
    np.testing.assert_array_equal(groups, bench.groups)


def test_pastis_crop_seg_protocol() -> None:
    bench = SimpleNamespace(groups=np.array([1, 2, 3, 4, 5]))

    y, groups = pastis.make_targets(bench)

    assert pastis.BENCHMARK == "pastis"
    assert pastis.LABEL_KIND == "segmentation"
    assert pastis.TRAIN_FOLDS == {1, 2, 3}
    assert pastis.VAL_FOLDS == {4}
    assert pastis.TEST_FOLDS == {5}
    assert y.size == 0
    np.testing.assert_array_equal(groups, bench.groups)
