from __future__ import annotations

import numpy as np

from evals import evals as EV


def test_filter_conditions_by_axes() -> None:
    conditions = [
        ("baseline", "none", 0.0),
        ("sensor_off_s2", "s2", 0.0),
        ("temporal_drop_50", "none", 0.5),
        ("s2_off_tdrop50", "s2", 0.5),
    ]

    sensorial = EV.filter_conditions_by_axes(conditions, ["sensorial"])
    both = EV.filter_conditions_by_axes(conditions, ["sensorial", "temporal"])

    assert [c[0] for c in sensorial] == ["baseline", "sensor_off_s2"]
    assert [c[0] for c in both] == ["baseline", "sensor_off_s2", "temporal_drop_50", "s2_off_tdrop50"]


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
        meta={"encoder": "e", "task": "t", "method": "erm", "split_regime": "random_id", "condition": "baseline"},
        predictions=preds,
        sample_ids_test=np.arange(100, 106),
        groups_test=np.array(["g"] * 6),
    )

    assert [r["label_budget"] for r in rows] == [0.5, 1.0]
    assert len(preds) == 12
    assert {p["label_budget"] for p in preds} == {0.5, 1.0}
    assert {p["sample_id"] for p in preds} == set(range(100, 106))
    assert {"prob", "pred_default", "pred_calibrated"}.issubset(preds[0])


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
        meta={"encoder": "e", "task": "t", "method": "erm", "split_regime": "random_id", "condition": "baseline"},
        predictions=preds,
        sample_ids_test=np.arange(200, 206),
        groups_test=np.array(["g"] * 6),
    )

    assert len(rows) == 1
    assert len(preds) == 6
    assert {"pred", "prob_true", "prob_pred", "classes", "probs"}.issubset(preds[0])


def test_segmentation_probe_reports_official_validation_and_test_splits() -> None:
    rng = np.random.default_rng(7)
    y_train = np.tile(np.arange(3), 30)
    y_val = np.tile(np.arange(3), 8)
    y_test = np.tile(np.arange(3), 8)

    def features(labels):
        values = rng.normal(size=(len(labels), 6))
        values[np.arange(len(labels)), labels] += 4.0
        return values

    rows: list[dict] = []
    EV.run_probes_segmentation(
        rows,
        features(y_train),
        features(y_val),
        features(y_test),
        y_train,
        y_val,
        y_test,
        seed=0,
        budgets=[1.0],
        meta={"task": "pastis-crop-seg"},
    )

    assert {row["evaluation_split"] for row in rows} == {"validation", "test"}
    assert all({"miou", "pixel_accuracy", "macro_f1", "weighted_f1"}.issubset(row) for row in rows)
