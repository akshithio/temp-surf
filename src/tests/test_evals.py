from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from evals import evals as EV
from evals.metrics import expected_calibration_error, score_segmentation, score_segmentation_streamed
from evals.regimes import geographic_ood, official, random_id

# The old geographic_ood tabular tests were removed with the schema-v2 rewrite: geographic_ood now
# emits SourceTargetSplit via true LODO + purge + partition_source, covered by test_regime_geographic.py.


def test_random_id_v2_split_is_exact_with_singleton_classes() -> None:
    """The schema-v2 successor to the retired make_splits fallback: a distribution with singleton
    classes yields an EXACT disjoint+complete source split (partition_source, no fallback)."""
    y = np.array([0, 0, 0, 1, 1, 2, 3, 4, 5, 6], dtype=np.int64)  # classes 2..6 are singletons
    groups = np.array(["A"] * len(y), dtype=object)
    bench = SimpleNamespace(labels=y, groups=groups, sample_ids=np.arange(len(y)))
    bench_mod = SimpleNamespace(BENCHMARK="toy", make_targets=lambda b: (b.labels, b.groups))

    split = next(iter(random_id.iter_source_target_splits(bench, bench_mod, 0)))
    train, val, test = split.source_train, split.source_val, split.source_test
    assert set(train.tolist()).isdisjoint(val.tolist())
    assert set(train.tolist()).isdisjoint(test.tolist())
    assert set(val.tolist()).isdisjoint(test.tolist())
    assert sorted(np.concatenate([train, val, test]).tolist()) == list(range(len(y)))
    assert split.has_target is False and split.target_test.size == 0


def test_official_pastis_split_uses_published_folds() -> None:
    """schema v2: official emits a patch-level DenseSourceTargetSplit -- folds 1-3 source_train,
    fold 4 source_val, fold 5 target_test -- has_target=True, supports_target_labels=False."""
    bench_mod = SimpleNamespace(BENCHMARK="pastis", TRAIN_FOLDS={1, 2, 3}, VAL_FOLDS={4}, TEST_FOLDS={5})
    patches = [SimpleNamespace(patch_id=100 + i, fold=(i % 5) + 1, tile=f"T3{i % 4}") for i in range(20)]
    fold_of = {p.patch_id: p.fold for p in patches}
    bench = SimpleNamespace(patches=patches)

    [d] = list(official.iter_dense_source_target_splits(bench, bench_mod, 0))

    assert d.label == "fold_5"
    assert d.has_target is True and d.supports_target_labels is False
    assert d.source_train_patches == frozenset(p for p, f in fold_of.items() if f in {1, 2, 3})
    assert d.source_val_patches == frozenset(p for p, f in fold_of.items() if f == 4)
    assert d.target_test_patches == frozenset(p for p, f in fold_of.items() if f == 5)
    assert not d.source_test_patches and not d.target_label_pool_patches


def test_geographic_pastis_split_is_leave_one_tile_out() -> None:
    """schema v2: geographic_ood dense LODO is over Sentinel TILES (never the published folds),
    patch-level, has_target=True/supports_target_labels=True."""
    from evals import split_spec

    tiles = list(split_spec.PASTIS.geographic_targets)
    centers = {t: (45.0 + 3 * i, -1.0 + 3 * i) for i, t in enumerate(tiles)}  # tiles spatially separated
    patches, pid = [], 100
    for tile in tiles:
        la, lo = centers[tile]
        for k in range(6):
            patches.append(SimpleNamespace(patch_id=pid, fold=(pid % 5) + 1, tile=tile, latlon=(la + k * 0.01, lo + k * 0.01)))
            pid += 1
    pids = [p.patch_id for p in patches]
    tile_of = {p.patch_id: p.tile for p in patches}
    bench = SimpleNamespace(
        patches=patches,
        patch_ids=lambda folds=None, _p=pids: list(_p),
        patch_class_sets=lambda ids=None, _p=pids: {int(p): {p % 4, 10 + p % 3} for p in (ids if ids is not None else _p)},
        patch_tiles={p.patch_id: p.tile for p in patches},
        patch_latlon={p.patch_id: p.latlon for p in patches},
    )
    bench_mod = SimpleNamespace(BENCHMARK="pastis")

    splits = list(geographic_ood.iter_dense_source_target_splits(bench, bench_mod, 0))
    assert sorted(d.label for d in splits) == sorted(str(t) for t in tiles)
    for d in splits:
        target = d.target_label_pool_patches | d.target_test_patches
        assert {tile_of[p] for p in target} == {d.label}          # held out is exactly its tile
        source = d.source_train_patches | d.source_val_patches | d.source_test_patches
        assert d.label not in {tile_of[p] for p in source}        # source never contains the target tile
        assert d.has_target is True and d.supports_target_labels is True


def test_random_pastis_split_is_patch_level() -> None:
    """schema v2: random_id emits a source-only DenseSourceTargetSplit over patch IDs (cache-free --
    the regime reads bench.patch_ids/patch_class_sets, never fold dirs). Patches, never folds/pixels,
    are the split unit."""
    pids = list(range(100, 120))  # 20 patches spanning all folds
    patches = [SimpleNamespace(patch_id=p, fold=(p % 5) + 1, tile=f"T3{p % 4}") for p in pids]
    bench = SimpleNamespace(
        patches=patches,
        patch_ids=lambda folds=None, _p=pids: list(_p),
        patch_class_sets=lambda ids=None, _p=pids: {int(p): {0, 1} for p in (ids if ids is not None else _p)},
        patch_tiles={p.patch_id: p.tile for p in patches},
        patch_latlon={},
    )
    bench_mod = SimpleNamespace(BENCHMARK="pastis")

    [d] = list(random_id.iter_dense_source_target_splits(bench, bench_mod, 0))
    train, val, test = d.source_train_patches, d.source_val_patches, d.source_test_patches

    assert d.label == "random_patch"
    assert d.has_target is False and d.supports_target_labels is False
    assert train and val and test
    assert train.isdisjoint(val)
    assert train.isdisjoint(test)
    assert val.isdisjoint(test)
    assert len(train | val | test) == 20
    assert not d.target_label_pool_patches and not d.target_test_patches


def test_subset_indices() -> None:
    y = np.array([0, 1, 0, 1, 0])

    tiny = EV.subset_indices(y, budget=0.01, seed=4, stratify=True)
    full = EV.subset_indices(y, budget=1.0, seed=4, stratify=True)

    assert len(tiny) == 2
    np.testing.assert_array_equal(full, np.arange(len(y)))


def test_expected_calibration_error() -> None:
    y = np.array([0, 0, 1, 1])
    prob = np.array([0.0, 0.0, 1.0, 1.0])

    assert expected_calibration_error(y, prob, n_bins=2) == 0.0


def test_probe_family_modules_build_expected_estimators() -> None:
    from evals.probes import _build_knn, _build_logistic, _build_mlp

    linear_probe = _build_logistic(1.0, solver="liblinear", seed=0, n_fit=8)
    mlp_probe = _build_mlp(1e-3, solver="unused", seed=0, n_fit=8)
    knn_probe = _build_knn(20, solver="unused", seed=0, n_fit=8)

    assert linear_probe.steps[-1][1].__class__.__name__ == "LogisticRegression"
    assert mlp_probe.steps[-1][1].__class__.__name__ == "MLPClassifier"
    assert knn_probe.steps[-1][1].__class__.__name__ == "KNeighborsClassifier"


def test_metric_roles_label_deployment_and_diagnostic_metrics() -> None:
    assert "calibrated_f1" in EV.METRIC_ROLES["binary"]["deployment"]
    assert "diagnostic_calibrated_f1_target_optimal" in EV.METRIC_ROLES["binary"]["diagnostic"]
    assert "worst_group_macro_f1" in EV.METRIC_ROLES["multiclass"]["deployment"]
    assert "worst_tile_miou" in EV.METRIC_ROLES["segmentation"]["deployment"]


def test_binary_probe_writes_predictions_for_each_source_budget() -> None:
    rng = np.random.default_rng(0)
    x_train = rng.normal(size=(40, 4))
    y_train = np.array([0, 1] * 20)
    x_train[y_train == 1, 0] += 2.0
    x_test = rng.normal(size=(6, 4))
    y_test = np.array([0, 1, 0, 1, 0, 1])
    rows: list[dict] = []
    preds: list[dict] = []

    EV.run_probes(
        rows,
        x_train,
        x_test,
        y_train,
        y_test,
        seed=0,
        budgets=[0.5, 1.0],
        meta={"model": "e", "benchmark": "t", "method": "erm", "split_regime": "random_id"},
        predictions=preds,
        sample_ids_test=np.arange(100, 106),
        groups_test=np.array(["g"] * 6),
    )

    assert [r["label_budget"] for r in rows] == [0.5, 1.0]
    assert {r["evaluation_split"] for r in rows} == {"test"}
    assert len(preds) == 12
    assert {p["label_budget"] for p in preds} == {0.5, 1.0}
    assert {p["evaluation_split"] for p in preds} == {"test"}
    assert {p["sample_id"] for p in preds} == set(range(100, 106))
    assert {"prob", "pred_default", "pred_calibrated"}.issubset(preds[0])
    assert {"worst_group_calibrated_f1", "worst_group_score", "n_groups_scored"}.issubset(rows[0])
    assert rows[0]["n_groups_scored"] == 1


def test_source_budget_zero_is_rejected() -> None:
    rng = np.random.default_rng(9)
    x_train = rng.normal(size=(40, 4))
    y_train = np.array([0, 1] * 20)
    x_train[y_train == 1, 0] += 2.0
    rows: list[dict] = []

    with pytest.raises(ValueError, match="source budgets"):
        EV.run_probes(
            rows,
            x_train,
            x_train,
            y_train,
            y_train,
            seed=0,
            budgets=[0],
            meta={"model": "e", "benchmark": "t", "method": "erm", "split_regime": "random_id"},
        )


def test_binary_probe_reports_worst_group_on_mixed_domain_test_set() -> None:
    rng = np.random.default_rng(2)
    x_train = rng.normal(size=(50, 4))
    y_train = np.array([0, 1] * 25)
    x_train[y_train == 1, 0] += 2.0
    x_test = rng.normal(size=(8, 4))
    y_test = np.array([0, 1, 0, 1, 0, 1, 0, 1])
    groups_test = np.array(["a", "a", "a", "a", "b", "b", "b", "b"], dtype=object)
    rows: list[dict] = []

    EV.run_probes(
        rows,
        x_train,
        x_test,
        y_train,
        y_test,
        seed=0,
        budgets=[1.0],
        meta={"model": "e", "benchmark": "t", "method": "erm", "split_regime": "random_id"},
        groups_test=groups_test,
    )

    assert rows[0]["n_groups_scored"] == 2
    assert rows[0]["worst_group"] in {"a", "b"}
    assert rows[0]["worst_group_metric"] == "calibrated_f1"


def test_multiclass_probe_writes_prediction_vectors() -> None:
    rng = np.random.default_rng(1)
    y_train = np.array([0, 1, 2] * 12)
    x_train = rng.normal(size=(36, 5))
    x_train[np.arange(len(y_train)), y_train] += 2.0
    y_test = np.array([0, 1, 2, 0, 1, 2])
    x_test = rng.normal(size=(6, 5))
    rows: list[dict] = []
    preds: list[dict] = []

    EV.run_probes_multiclass(
        rows,
        x_train,
        x_test,
        y_train,
        y_test,
        seed=0,
        budgets=[1.0],
        meta={"model": "e", "benchmark": "t", "method": "erm", "split_regime": "random_id"},
        predictions=preds,
        sample_ids_test=np.arange(200, 206),
        groups_test=np.array(["g"] * 6),
    )

    assert len(rows) == 1
    assert len(preds) == 6
    assert {"pred", "prob_true", "prob_pred", "classes", "probs"}.issubset(preds[0])
    assert {"worst_group_macro_f1", "worst_group_score", "n_groups_scored"}.issubset(rows[0])


def test_target_zero_reuses_fitted_probe_for_both_evaluation_scopes(monkeypatch) -> None:
    rng = np.random.default_rng(8)
    y_source = np.tile(np.arange(3), 20)
    y_target = np.tile(np.arange(3), 10)
    y_val = np.tile(np.arange(3), 5)

    def features(labels):
        values = rng.normal(size=(len(labels), 5))
        values[np.arange(len(labels)), labels] += 3.0
        return values

    original = EV.fit_probe_multiclass
    fit_calls = 0

    def counted_fit(*args, **kwargs):
        nonlocal fit_calls
        fit_calls += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(EV, "fit_probe_multiclass", counted_fit)
    rows: list[dict] = []
    EV.run_probes_multiclass_target(
        rows,
        features(y_source),
        features(y_target),
        y_source,
        y_target,
        seed=0,
        budgets=[0],
        x_val=features(y_val),
        y_val=y_val,
    )

    assert fit_calls == 1
    assert {row["evaluation_split"] for row in rows} == {"held_out", "full"}

    binary_original = EV.fit_probe_with_calibration
    binary_fit_calls = 0

    def counted_binary_fit(*args, **kwargs):
        nonlocal binary_fit_calls
        binary_fit_calls += 1
        return binary_original(*args, **kwargs)

    monkeypatch.setattr(EV, "fit_probe_with_calibration", counted_binary_fit)
    binary_source = np.array([0, 1] * 30)
    binary_target = np.array([0, 1] * 15)
    binary_val = np.array([0, 1] * 8)
    binary_rows: list[dict] = []
    EV.run_probes_target(
        binary_rows,
        features(binary_source),
        features(binary_target),
        binary_source,
        binary_target,
        seed=0,
        budgets=[0],
        x_val=features(binary_val),
        y_val=binary_val,
    )

    assert binary_fit_calls == 1
    assert {row["evaluation_split"] for row in binary_rows} == {"held_out", "full"}


def test_fractional_target_budget_uses_fraction_of_target_train_pool() -> None:
    rng = np.random.default_rng(12)
    y_source = np.array([0, 1] * 30)
    y_target = np.array([0, 1] * 50)
    y_val = np.array([0, 1] * 10)

    def features(labels):
        values = rng.normal(size=(len(labels), 4))
        values[np.arange(len(labels)), labels] += 2.0
        return values

    rows: list[dict] = []
    EV.run_probes_target(
        rows,
        features(y_source),
        features(y_target),
        y_source,
        y_target,
        seed=0,
        budgets=[0.1],
        x_val=features(y_val),
        y_val=y_val,
    )

    assert rows[0]["label_budget"] == 0.1
    assert rows[0]["n_train_sub"] == len(y_source) + 8  # 10% of the fixed 80-sample target train pool


def test_segmentation_probe_reports_official_validation_and_test_splits() -> None:
    rng = np.random.default_rng(7)
    y_train = np.tile(np.arange(3), 30)
    y_val = np.tile(np.arange(3), 8)
    y_test = np.tile(np.arange(3), 8)

    def features(labels):
        values = rng.normal(size=(len(labels), 6))
        values[np.arange(len(labels)), labels] += 4.0
        return values

    x_train, x_val, x_test = features(y_train), features(y_val), features(y_test)
    rows: list[dict] = []
    EV.run_probes_segmentation(
        rows,
        x_train,
        x_val,
        y_train,
        y_val,
        seed=0,
        eval_streams={
            "validation": lambda: iter([(x_val, y_val)]),
            "test": lambda: iter([(x_test, y_test)]),
        },
        budgets=[1.0],
        meta={"benchmark": "pastis"},
    )

    assert {row["evaluation_split"] for row in rows} == {"validation", "test"}
    assert all({"miou", "pixel_accuracy", "macro_f1", "weighted_f1"}.issubset(row) for row in rows)
    assert all(row["n_test"] == len(y_val) for row in rows if row["evaluation_split"] == "validation")


def test_streamed_segmentation_matches_whole_array_scoring() -> None:
    eval_classes = np.arange(19, dtype=np.int64)
    rng = np.random.default_rng(0)

    class _Clf:
        classes_ = np.arange(19)

        def predict(self, x):
            return (np.abs(x[:, 0]) * 100 % 19).astype(int)

    clf = _Clf()
    t1x, t1y = rng.normal(size=(50, 4)), rng.integers(0, 19, 50)
    t2x, t2y = rng.normal(size=(70, 4)), rng.integers(0, 19, 70)
    whole = score_segmentation(clf, np.concatenate([t1x, t2x]), np.concatenate([t1y, t2y]), eval_classes=eval_classes)
    streamed = score_segmentation_streamed(clf, iter([(t1x, t1y), (t2x, t2y)]), eval_classes)

    assert abs(streamed["miou"] - whole["miou"]) < 1e-9
    assert abs(streamed["pixel_accuracy"] - whole["pixel_accuracy"]) < 1e-9
    assert streamed["n_test"] == 120 and streamed["n_tiles_scored"] == 2


def test_streamed_segmentation_rejects_invalid_labels() -> None:
    class _Clf:
        def predict(self, x):
            return np.array([0, 99])

    with pytest.raises(ValueError, match="predictions"):
        score_segmentation_streamed(_Clf(), iter([(np.zeros((2, 3)), np.array([0, 1]))]), np.arange(3))

    class _GoodClf:
        def predict(self, x):
            return np.array([0, 1])

    with pytest.raises(ValueError, match="labels"):
        score_segmentation_streamed(_GoodClf(), iter([(np.zeros((2, 3)), np.array([0, 99]))]), np.arange(3))


def test_score_multiclass_reports_shared_and_unseen_class_decomposition() -> None:
    """#7: a source-only probe can't predict target-only classes, so report shared-class metrics +
    the unseen-class prevalence separately from the full-label metrics."""
    from evals.metrics import score_multiclass

    class _Clf:
        classes_ = np.array([0, 1, 2])  # trained only on classes 0,1,2

        def predict(self, x):
            return np.zeros(len(x), dtype=int)

        def predict_proba(self, x):
            return np.tile([0.6, 0.3, 0.1], (len(x), 1))

    y = np.array([0, 1, 2, 3, 0, 1, 2, 3, 0, 1])  # class 3 is target-only (absent from training)
    s = score_multiclass(_Clf(), np.zeros((len(y), 4)), y)

    assert s["n_classes_seen"] == 3
    assert s["n_classes_unseen"] == 1
    assert abs(s["unseen_prevalence"] - 0.2) < 1e-9  # 2 of 10 samples are the unseen class
    assert not np.isnan(s["shared_macro_f1"])  # representation-only metric on the seen-class subset


def test_domain_confound_report_flags_geography_entanglement() -> None:
    """Cross-tab the domain bases so a confounded axis is visible."""
    from evals.confounds import domain_confound_report

    geography = np.array(["PT", "PT", "EE", "EE", "LV", "LV"])
    year = np.array([2020, 2021, 2020, 2021, 2020, 2021])
    rep = domain_confound_report(
        {"geography": geography, "year": year, "class": np.zeros(6)}
    )

    assert rep["pairs"]["geography__vs__year"]["nmi"] < 0.2  # geography/year not entangled
    assert "class" not in rep["axis_cardinality"]  # single-valued axis dropped
