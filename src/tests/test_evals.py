from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from evals import evals as EV
from evals.regimes import phenology_ood
from evals.regimes.phenology_ood import phenology_domains


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


def test_phenology_domains_use_ndvi_peak_timing_and_low_amplitude() -> None:
    bench = SimpleNamespace(
        name="toy",
        s2=np.array(
            [
                [[0.1], [0.8], [0.2], [0.1]],  # early peak
                [[0.1], [0.2], [0.3], [0.8]],  # late peak
                [[0.4], [0.4], [0.4], [0.4]],  # low amplitude
            ],
            dtype=np.float32,
        ),
        s2_mask=np.ones((3, 4), dtype=np.float32),
        s2_bands=["NDVI"],
    )

    domains = phenology_domains(bench)

    np.testing.assert_array_equal(
        domains,
        np.array(["phenology_early_peak", "phenology_late_peak", "phenology_low_amplitude"], dtype=object),
    )


def test_phenology_ood_splits_by_assigned_domains() -> None:
    y = np.array([0, 1, 0, 1, 0, 1])
    domains = np.array(["early", "early", "mid", "mid", "late", "late"], dtype=object)

    splits = list(phenology_ood.iter_splits(y, domains, seed=0))

    assert {s.label for s in splits} == {"early", "mid", "late"}
    for s in splits:
        assert set(domains[s.test]) == {s.label}
        assert set(s.train).isdisjoint(s.test)
        assert set(s.val).isdisjoint(s.test)  # val is held out of test too


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
    assert len(preds) == 12
    assert {p["label_budget"] for p in preds} == {0.5, 1.0}
    assert {p["sample_id"] for p in preds} == set(range(100, 106))
    assert {"prob", "pred_default", "pred_calibrated"}.issubset(preds[0])
    assert {"worst_group_calibrated_f1", "worst_group_score", "n_groups_scored"}.issubset(rows[0])
    assert rows[0]["n_groups_scored"] == 1


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
        meta={"benchmark": "pastis_r"},
    )

    assert {row["evaluation_split"] for row in rows} == {"validation", "test"}
    assert all({"miou", "pixel_accuracy", "macro_f1", "weighted_f1"}.issubset(row) for row in rows)
