from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from evals import evals as EV
from evals.benchmarks import breizhcrops, cropharvest, eurocropsml, pastis
from evals.regimes import base as regime_base
from evals.regimes import spatial_cluster_ood


def test_bin_crop_class_targets() -> None:
    bench = SimpleNamespace(labels=np.array([0.0, 1.0]), groups=np.array(["a", "b"], dtype=object))

    y, groups = cropharvest.make_targets(bench)

    assert cropharvest.BENCHMARK == "cropharvest"
    assert cropharvest.LABEL_KIND == "binary"
    assert y.dtype == np.int64
    np.testing.assert_array_equal(y, np.array([0, 1]))
    np.testing.assert_array_equal(groups, bench.groups)


def test_cropharvest_geo_group_collapses_dataset_aliases() -> None:
    assert cropharvest._ch_geo_group("togo-eval") == "togo"
    assert cropharvest._ch_geo_group("togo") == "togo"
    assert cropharvest._ch_geo_group("lem-brazil") == "lem-brazil"


def test_cropharvest_geographic_domains_use_spatial_blocks() -> None:
    bench = SimpleNamespace(latlon=np.array([[0.0, 0.0], [0.5, 0.5], [5.0, 5.0], [np.nan, 0.0]]))

    domains = cropharvest.geographic_domains(bench)

    assert domains[0] == domains[1]
    assert domains[0].startswith("block_")
    assert domains[2] != domains[0]
    assert domains[3] == "unknown"


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
            overwrite_mode=True,
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
