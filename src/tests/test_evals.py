from __future__ import annotations

import numpy as np
import pytest

from evals import evals as EV


def test_make_strict_holdout_splits() -> None:
    y = np.array([0, 1, 0, 1, 0, 1, 0, 1])
    groups = np.array(["src1", "src1", "src2", "src2", "hold", "hold", "src3", "src3"], dtype=object)

    train, val, test, train_val = EV.make_strict_holdout_splits(y, groups, "hold", seed=0)

    assert set(test.tolist()) == {4, 5}
    assert set(train).isdisjoint(test)
    assert set(val).isdisjoint(test)
    assert set(train_val.tolist()) == {0, 1, 2, 3, 6, 7}
    assert np.all(groups[train] != "hold")
    assert np.all(groups[val] != "hold")


def test_make_splits_falls_back_when_a_class_is_singleton() -> None:
    y = np.array([0, 0, 0, 1, 1, 2, 3, 4, 5, 6])

    train, val, test = EV.make_splits(y, seed=0)

    assert len(set(train) & set(val)) == 0
    assert len(set(train) & set(test)) == 0
    assert len(set(val) & set(test)) == 0
    assert sorted(np.concatenate([train, val, test]).tolist()) == list(range(len(y)))


def test_subset_indices() -> None:
    y = np.array([0, 1, 0, 1, 0])

    tiny = EV.subset_indices(y, budget=0.01, seed=4, stratify=True)
    full = EV.subset_indices(y, budget=1.0, seed=4, stratify=True)

    assert len(tiny) == 2
    np.testing.assert_array_equal(full, np.arange(len(y)))


def test_expected_calibration_error() -> None:
    y = np.array([0, 0, 1, 1])
    prob = np.array([0.0, 0.0, 1.0, 1.0])

    assert EV.expected_calibration_error(y, prob, n_bins=2) == 0.0


def test_metric_roles_label_deployment_and_diagnostic_metrics() -> None:
    assert "calibrated_f1" in EV.METRIC_ROLES["binary"]["deployment"]
    assert "calibrated_f1_target_optimal" in EV.METRIC_ROLES["binary"]["diagnostic"]
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
        # full-fold streaming eval: each split presented as a single in-memory "tile"
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
    """Streaming tiles into the confusion-based scorer gives the exact same mIoU / pixel-accuracy as
    scoring the concatenation in one shot -- so full-fold streaming is a faithful (not approximate)
    replacement for the old capped-sample scoring."""
    from evals.probes import score_segmentation, score_segmentation_streamed

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
    from evals.probes import score_segmentation_streamed

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
    from evals.probes import score_multiclass

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
    year = np.array([2020, 2021, 2020, 2021, 2020, 2021])  # each country spans both years -> independent
    rep = domain_confound_report(
        {"geography": geography, "year": year, "class": np.zeros(6)}
    )

    assert rep["pairs"]["geography__vs__year"]["nmi"] < 0.2  # geography/year not entangled
    assert "class" not in rep["axis_cardinality"]  # single-valued axis dropped
