"""P0-1 regression tests: calibration when target-only (unsupported) classes exist.

A class present in the test labels but absent from the probe's training classes has NO
probability column, so the model necessarily assigns it p=0. The old implementation built an
all-zero one-hot row and then (a) clipped the true-class probability to 1e-12 for NLL, which
manufactured a finite value set entirely by the arbitrary constant, and (b) computed Brier
against the all-zero vector, silently dropping the unavoidable (0-1)^2 = 1 penalty. Proper
scoring rules only hold over the correct outcome space (Gneiting & Raftery 2007).
"""

from __future__ import annotations

import json

import numpy as np

from evals.metrics import CALIBRATION_KEYS, multiclass_calibration, score_segmentation_streamed


def test_unseen_true_class_gets_unit_brier_penalty() -> None:
    """An unsupported example must pay 1 + sum_j p_j^2, not just sum_j p_j^2."""
    classes = np.array([0, 1])
    proba = np.array([[0.7, 0.3]])
    y_true = np.array([2])  # class 2 was never trained on -> no column

    cal = multiclass_calibration(y_true, proba, classes)

    sq = 0.7**2 + 0.3**2
    assert np.isclose(cal["union_brier"], 1.0 + sq)
    assert np.isclose(cal["brier"], 1.0 + sq)
    # the old, buggy convention would have reported just sq
    assert not np.isclose(cal["brier"], sq)


def test_unseen_true_class_makes_full_nll_infinite() -> None:
    """-log(0) is +inf; it must not be a finite artefact of the clip constant."""
    classes = np.array([0, 1])
    proba = np.array([[0.7, 0.3], [0.4, 0.6]])
    y_true = np.array([0, 5])  # second row unsupported

    cal = multiclass_calibration(y_true, proba, classes)

    assert np.isinf(cal["nll"])
    # the old code clipped to 1e-12 -> ~27.6/2; assert we are not reproducing that
    assert not np.isfinite(cal["nll"])


def test_top_label_ece_includes_unseen_examples() -> None:
    """Unsupported examples are always wrong, so they must still count toward top-label ECE:
    a confident-but-impossible prediction should be penalised, not dropped."""
    classes = np.array([0, 1])
    # every row is predicted with 1.0 confidence; the unsupported rows are all wrong
    proba = np.array([[1.0, 0.0], [1.0, 0.0], [1.0, 0.0], [1.0, 0.0]])
    y_true = np.array([0, 0, 9, 9])  # half unsupported

    cal = multiclass_calibration(y_true, proba, classes)

    # confidence 1.0, accuracy 0.5 over all examples -> ECE = 0.5
    assert np.isclose(cal["top_label_ece_all"], 0.5)
    assert np.isclose(cal["ece"], cal["top_label_ece_all"])
    # restricted to supported examples the probe is perfectly calibrated
    assert np.isclose(cal["shared_ece"], 0.0)


def test_shared_metrics_exclude_only_unsupported_examples() -> None:
    """shared_* must equal the metrics computed on the supported subset alone."""
    classes = np.array([0, 1])
    proba = np.array([[0.8, 0.2], [0.3, 0.7], [0.6, 0.4]])
    y_true = np.array([0, 1, 7])  # last unsupported

    cal = multiclass_calibration(y_true, proba, classes)
    only_shared = multiclass_calibration(y_true[:2], proba[:2], classes)

    assert np.isclose(cal["shared_nll"], only_shared["nll"])
    assert np.isclose(cal["shared_brier"], only_shared["brier"])
    assert np.isclose(cal["shared_ece"], only_shared["ece"])


def test_shared_and_full_coincide_when_all_classes_seen() -> None:
    """With full label support the distinction is vacuous and the scores must agree."""
    classes = np.array([0, 1, 2])
    proba = np.array([[0.7, 0.2, 0.1], [0.1, 0.8, 0.1], [0.2, 0.2, 0.6]])
    y_true = np.array([0, 1, 2])

    cal = multiclass_calibration(y_true, proba, classes)

    assert cal["unseen_prevalence"] == 0.0
    assert np.isfinite(cal["nll"])
    assert np.isclose(cal["shared_nll"], cal["nll"])
    assert np.isclose(cal["shared_brier"], cal["brier"])
    assert np.isclose(cal["shared_ece"], cal["ece"])


def test_shape_mismatch_returns_controlled_missing_values() -> None:
    """Malformed inputs must yield NaNs across every column, never a partial dict."""
    classes = np.array([0, 1])
    y_true = np.array([0, 1])

    bad_cases = [
        (None, classes),                               # no predict_proba
        (np.array([[0.5, 0.5]]), classes),             # rows != len(y_true)
        (np.array([[0.3, 0.3, 0.4], [0.3, 0.3, 0.4]]), classes),  # cols != len(classes)
        (np.zeros((2, 0)), np.array([])),              # zero columns
    ]
    for proba, cls in bad_cases:
        cal = multiclass_calibration(y_true, proba, cls)
        assert set(cal) == set(CALIBRATION_KEYS)
        assert all(np.isnan(v) for v in cal.values()), cal

    # empty input
    cal = multiclass_calibration(np.array([]), np.zeros((0, 2)), classes)
    assert all(np.isnan(v) for v in cal.values())


def test_mixed_batch_reports_unseen_prevalence() -> None:
    """unseen_prevalence is the label-support diagnostic that explains the other columns."""
    classes = np.array([0, 1])
    proba = np.tile(np.array([[0.6, 0.4]]), (4, 1))
    y_true = np.array([0, 1, 4, 5])  # 2 of 4 unsupported

    cal = multiclass_calibration(y_true, proba, classes)

    assert np.isclose(cal["unseen_prevalence"], 0.5)
    assert np.isinf(cal["nll"])          # degenerate over the full label space
    assert np.isfinite(cal["shared_nll"])  # but well posed on supported examples

    # all-unsupported: shared metrics undefined, full Brier still scores the unit penalty
    cal_none = multiclass_calibration(np.array([8, 9]), proba[:2], classes)
    assert cal_none["unseen_prevalence"] == 1.0
    assert np.isnan(cal_none["shared_nll"])
    assert np.isnan(cal_none["shared_brier"])
    assert np.isclose(cal_none["union_brier"], 1.0 + 0.6**2 + 0.4**2)


def test_streamed_calibration_matches_tabular_with_unsupported_classes() -> None:
    """PASTIS scores calibration in a streaming pass with its own inline accumulators, which
    carried the same unsupported-class bug. It must agree with the tabular implementation
    column-for-column, including when the tiles contain a class the probe never trained on."""
    rng = np.random.default_rng(0)

    class _Clf:
        classes_ = np.array([0, 1, 2])  # probe saw 3 classes...

        def predict_proba(self, x):
            z = np.asarray(x)[:, :3]
            e = np.exp(z - z.max(axis=1, keepdims=True))
            return e / e.sum(axis=1, keepdims=True)

        def predict(self, x):
            return self.classes_[self.predict_proba(x).argmax(axis=1)]

    clf = _Clf()
    eval_classes = np.arange(4, dtype=np.int64)  # ...but class 3 appears in the tiles
    x1, y1 = rng.normal(size=(60, 3)), rng.integers(0, 4, 60)
    x2, y2 = rng.normal(size=(40, 3)), rng.integers(0, 4, 40)

    streamed = score_segmentation_streamed(clf, iter([(x1, y1), (x2, y2)]), eval_classes)
    tabular = multiclass_calibration(
        np.concatenate([y1, y2]),
        clf.predict_proba(np.concatenate([x1, x2])),
        clf.classes_,
    )

    assert streamed["unseen_prevalence"] > 0  # the fixture is actually exercising the case
    for key in CALIBRATION_KEYS:
        s, t = streamed[key], tabular[key]
        if np.isinf(t):
            assert np.isinf(s), f"{key}: streamed={s} tabular={t}"
        else:
            # rtol=1e-6: the streamed path accumulates bin/Brier sums tile-by-tile, so float
            # summation order differs from the tabular single-pass. That shifts shared_ece by
            # ~1e-8; anything larger would be a genuine logic divergence, not associativity.
            assert np.isclose(s, t, rtol=1e-6, atol=1e-9), f"{key}: streamed={s} tabular={t}"


def test_rescorer_accumulator_matches_reference_implementation() -> None:
    """rescore_calibration re-implements the math as streaming accumulators (predictions.jsonl
    is ~10GB/cell, too big to hold in memory). Pin it to evals.metrics so the two can't drift,
    and check it reproduces the OLD buggy values it claims to report as the delta."""
    from evals.rescore_calibration import _Acc

    rng = np.random.default_rng(7)
    classes = [0, 1, 2]
    n = 200
    logits = rng.normal(size=(n, 3))
    proba = np.exp(logits) / np.exp(logits).sum(axis=1, keepdims=True)
    y_true = rng.integers(0, 5, n)  # classes 3,4 are unsupported

    acc = _Acc()
    for i in range(n):
        acc.add(int(y_true[i]), proba[i].tolist(), classes)
    got = acc.finalize()

    ref = multiclass_calibration(y_true, proba, np.asarray(classes))

    assert got["unseen_prevalence"] > 0
    for key in ("ece", "top_label_ece_all", "brier", "union_brier",
                "shared_ece", "shared_nll", "shared_brier", "unseen_prevalence"):
        assert np.isclose(got[key], ref[key], rtol=1e-6, atol=1e-9), f"{key}: {got[key]} vs {ref[key]}"
    assert np.isinf(got["nll"]) and np.isinf(ref["nll"])

    # the reported OLD values must reproduce the pre-fix convention: a finite NLL created by
    # the 1e-12 clip, and a Brier missing the unit penalty on unsupported rows.
    assert np.isfinite(got["old_nll"])
    assert got["old_brier"] < got["union_brier"]  # the omitted penalty made it look better


def test_epsilon_sweep_reproduces_old_nll_and_exposes_floor_sensitivity() -> None:
    """The pre-fix NLL clipped every row at 1e-12, so it must be exactly the 1e-12 sweep point.
    The sweep exists to show when an NLL is an artefact of the floor rather than the model."""
    from evals.rescore_calibration import EPS_SWEEP, _Acc

    classes = [0, 1]
    acc = _Acc()
    # one supported row with a *vanishing* true-class probability, one unsupported row --
    # both are floor-dominated, which is exactly the EuroCropsML situation.
    acc.add(0, [1e-20, 1.0 - 1e-20], classes)  # supported but p_true ~ 0
    acc.add(9, [0.5, 0.5], classes)            # unsupported
    got = acc.finalize()

    # identity: the old convention IS the 1e-12 sweep point
    assert np.isclose(got["old_nll"], got["nll_eps_1e-12"])

    # a floor-dominated cell moves a lot across the sweep -> the number is the floor, not the model
    lo, hi = got["nll_eps_1e-06"], got["nll_eps_1e-15"]
    assert hi > lo * 2, f"expected strong floor sensitivity, got {lo} -> {hi}"
    for e in EPS_SWEEP:
        assert f"nll_eps_{e:.0e}" in got and f"shared_nll_eps_{e:.0e}" in got


def test_epsilon_sweep_is_flat_when_probabilities_are_healthy() -> None:
    """Contrast: with sane probabilities the floor never binds, so the sweep is flat and the
    reported NLL is a real property of the model."""
    from evals.rescore_calibration import _Acc

    classes = [0, 1]
    acc = _Acc()
    for _ in range(10):
        acc.add(0, [0.7, 0.3], classes)
    got = acc.finalize()

    assert np.isclose(got["nll_eps_1e-06"], got["nll_eps_1e-15"])  # floor irrelevant
    assert np.isclose(got["nll"], -np.log(0.7))


def test_rescorer_fails_loudly_on_malformed_rows(tmp_path) -> None:
    """Silently skipping rows would break the n_test reconciliation that proves nothing was
    dropped, so bad input must raise with file:line context."""
    import pytest

    from evals.rescore_calibration import rescore

    good = json.dumps({"y_true": 0, "probs": [0.6, 0.4], "classes": [0, 1], "model": "m",
                       "benchmark": "b", "probe_family": "logistic", "split_regime": "r",
                       "evaluation_split": "test", "label_budget": 0, "holdout": "h",
                       "seed": 0, "method": "erm", "budget_type": "source"})

    bad_json = tmp_path / "bad.jsonl"
    bad_json.write_text(good + "\n{ not json\n")
    with pytest.raises(ValueError, match="malformed JSON"):
        list(rescore([str(bad_json)]))

    missing = tmp_path / "missing.jsonl"
    missing.write_text(good + "\n" + json.dumps({"y_true": 0, "model": "m"}) + "\n")
    with pytest.raises(ValueError, match="missing"):
        list(rescore([str(missing)]))

    # a well-formed file reconciles and yields one cell
    ok = tmp_path / "ok.jsonl"
    ok.write_text(good + "\n" + good + "\n")
    rows = list(rescore([str(ok)]))
    assert len(rows) == 1 and rows[0]["n_test"] == 2
