"""Scoring metrics for tabular probes and dense segmentation."""

from __future__ import annotations

import json
import warnings
from typing import Any

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    brier_score_loss,
    f1_score,
    log_loss,
    roc_auc_score,
)

from utils import perfutils as perf

_F1_THRESHOLD_WARNED = False


def best_f1_threshold(y_true: np.ndarray, prob: np.ndarray) -> float:
    global _F1_THRESHOLD_WARNED
    if len(np.unique(y_true)) < 2:
        return 0.5
    finite = np.isfinite(prob)
    if not finite.any():
        best_score = 0.0
    else:
        y = np.asarray(y_true, dtype=np.int64)[finite]
        p = np.asarray(prob, dtype=np.float64)[finite]
        order = np.argsort(p, kind="mergesort")
        p_sorted, y_sorted = p[order], y[order]
        thresholds, starts = np.unique(p_sorted, return_index=True)
        tp = np.cumsum(y_sorted[::-1])[::-1][starts].astype(float)
        pred_pos = (len(y_sorted) - starts).astype(float)
        fp = pred_pos - tp
        fn = float(y_sorted.sum()) - tp
        denom = 2 * tp + fp + fn
        scores = np.divide(2 * tp, denom, out=np.zeros_like(tp), where=denom > 0)
        best = int(np.argmax(scores))
        best_score = float(scores[best])
        best_threshold = float(thresholds[best])
    if best_score <= 0.0:
        if not _F1_THRESHOLD_WARNED:
            print("   !! F1 threshold calibration is degenerate; using threshold=0.5", flush=True)
            _F1_THRESHOLD_WARNED = True
        return 0.5
    return best_threshold


def expected_calibration_error(y_true: np.ndarray, prob: np.ndarray, n_bins: int = 10) -> float:
    if len(y_true) == 0:
        return float("nan")
    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    bin_indices = np.digitize(prob, bin_edges[1:-1])
    ece = 0.0
    for b in range(n_bins):
        in_bin = bin_indices == b
        if not in_bin.any():
            continue
        acc = y_true[in_bin].mean()
        conf = prob[in_bin].mean()
        ece += in_bin.sum() * abs(acc - conf)
    return float(ece / len(y_true))


#: Floor used only for ordinary numerical stability on SUPPORTED classes (their true-class
#: column exists, so this never manufactures a score the way clipping an absent class would).
_NUM_EPS = 1e-15

CALIBRATION_KEYS = (
    "ece",
    "top_label_ece_all",
    "nll",
    "brier",
    "union_brier",
    "shared_ece",
    "shared_nll",
    "shared_brier",
    "unseen_prevalence",
)


def _class_index(y_true: np.ndarray, classes: np.ndarray) -> np.ndarray:
    """Column index of each true label within ``classes``; -1 where the label is unsupported
    (i.e. the probe never saw that class, so it has no probability column at all).

    Vectorized: this runs once per scoring call over the full test set, which reaches 706k rows on
    EuroCropsML and is multiplied by every budget x seed x regime. Unlike ``_as_eval_indices``,
    an unsupported label is NOT an error here -- returning -1 for it is the whole point, since
    target-only classes are exactly what the calibration correction has to account for.
    """
    y = np.asarray(y_true)
    classes = np.asarray(classes)
    if classes.size == 0:
        return np.full(len(y), -1, dtype=np.int64)
    order = np.argsort(classes)
    sorted_classes = classes[order]
    pos = np.searchsorted(sorted_classes, y)
    safe = np.minimum(pos, len(sorted_classes) - 1)
    found = (pos < len(sorted_classes)) & (sorted_classes[safe] == y)
    return np.where(found, order[safe], -1).astype(np.int64)


def multiclass_calibration(
    y_true: np.ndarray, proba: np.ndarray | None, classes: np.ndarray, n_bins: int = 10
) -> dict[str, float]:
    """Calibration from a class-probability matrix whose columns are ``classes`` (same order).

    Target-only classes (present in the test labels, absent from the probe's training classes)
    have **no probability column**, so the model necessarily assigns them p=0. That makes the
    full-label scores degenerate, and they must not be papered over:

    * ``nll`` (full label) is ``-log(0) = +inf`` whenever any unsupported example is present.
      Previously this was clipped to 1e-12, which manufactured a *finite* value determined
      entirely by the arbitrary constant -- changing 1e-12 to 1e-6 would have moved the reported
      NLL without changing the model. It is reported as ``inf`` rather than silently smoothed.
    * ``brier``/``union_brier`` score over the UNION of train and test labels, padding a zero
      column for each target-only class. An unsupported example therefore pays the unavoidable
      ``(0-1)^2 = 1`` on its true class plus ``sum_j p_j^2`` on the rest. The previous code
      omitted that unit penalty, reporting only ``sum_j p_j^2``. Brier/log scores are proper
      only over the correct outcome space (Gneiting & Raftery 2007).
    * ``shared_*`` restrict to supported examples: ordinary, well-posed calibration.
    * ``ece``/``top_label_ece_all`` stay defined over *all* examples: an unsupported true class
      is simply always wrong (correctness 0), so top-label ECE still measures whether the probe
      remains overconfident on impossible-to-predict examples.
    * ``unseen_prevalence`` is the label-support diagnostic that explains the rest.

    Returns NaNs when probabilities are unavailable (no ``predict_proba``, or a shape mismatch).
    """
    y_true = np.asarray(y_true)
    classes = np.asarray(classes)
    nan = dict.fromkeys(CALIBRATION_KEYS, float("nan"))
    if proba is None:
        return nan
    proba = np.asarray(proba, dtype=np.float64)
    if (
        len(y_true) == 0
        or proba.ndim != 2
        or proba.shape[0] != len(y_true)
        or proba.shape[1] != len(classes)
        or proba.shape[1] == 0
    ):
        return nan

    idx = _class_index(y_true, classes)
    seen = idx >= 0
    n = len(y_true)

    # Top-label reliability over ALL examples (Guo et al. 2017).
    conf = proba.max(axis=1)
    pred_label = classes[proba.argmax(axis=1)]
    correct = (pred_label == y_true).astype(np.float64)
    ece_all = float(expected_calibration_error(correct, conf, n_bins=n_bins))

    # True-class probability: exactly 0 for unsupported classes (no column exists).
    p_true = np.where(seen, proba[np.arange(n), np.clip(idx, 0, None)], 0.0)
    sq = (proba**2).sum(axis=1)

    # Full-label NLL: honestly infinite if any true class has no column.
    nll_all = float(-np.mean(np.log(np.clip(p_true, _NUM_EPS, 1.0)))) if seen.all() else float("inf")

    # Union-space Brier: every row pays sum(oh^2)=1 on its true class, supported or not.
    union_brier = float(np.mean(sq - 2.0 * p_true + 1.0))

    out = {
        "ece": ece_all,
        "top_label_ece_all": ece_all,
        "nll": nll_all,
        "brier": union_brier,
        "union_brier": union_brier,
        "unseen_prevalence": float(1.0 - seen.mean()),
    }
    if seen.any():
        out["shared_ece"] = float(expected_calibration_error(correct[seen], conf[seen], n_bins=n_bins))
        out["shared_nll"] = float(-np.mean(np.log(np.clip(p_true[seen], _NUM_EPS, 1.0))))
        out["shared_brier"] = float(np.mean(sq[seen] - 2.0 * p_true[seen] + 1.0))
    else:
        out["shared_ece"] = out["shared_nll"] = out["shared_brier"] = float("nan")
    return out


def score_binary(
    clf: Any,
    threshold: float,
    x_test: np.ndarray,
    y_test: np.ndarray,
    return_per_sample: bool = False,
) -> dict[str, float] | tuple[dict[str, float], dict[str, np.ndarray]]:
    with perf.measure("probe.score/binary", n_samples=len(y_test), n_features=x_test.shape[1]):
        pred = clf.predict(x_test)
        prob = clf.predict_proba(x_test)[:, 1]
        calibrated_pred = (prob >= threshold).astype(np.int64)
    two_class = len(np.unique(y_test)) == 2
    test_optimal_threshold = best_f1_threshold(y_test, prob)
    calibrated_pred_target_optimal = (prob >= test_optimal_threshold).astype(np.int64)
    scores = {
        "f1": float(f1_score(y_test, pred, zero_division=0)),
        "auc": float(roc_auc_score(y_test, prob)) if two_class else float("nan"),
        "balanced_accuracy": float(balanced_accuracy_score(y_test, pred)),
        "calibrated_f1": float(f1_score(y_test, calibrated_pred, zero_division=0)),
        "calibrated_balanced_accuracy": float(balanced_accuracy_score(y_test, calibrated_pred)),
        "diagnostic_calibrated_f1_target_optimal": float(
            f1_score(y_test, calibrated_pred_target_optimal, zero_division=0)
        ),
        "diagnostic_optimal_threshold_test": float(test_optimal_threshold),
        "ece": expected_calibration_error(y_test, prob),
        "brier": float(brier_score_loss(y_test, prob)),
        "nll": float(log_loss(y_test, prob, labels=[0, 1])),
        "test_pos_rate": float(np.mean(y_test)),
    }
    n_classes = len(np.unique(y_test))
    perf.log_static("probe.macs/binary", macs=x_test.shape[1] * n_classes, n_samples=len(y_test))
    if return_per_sample:
        per_sample = {
            "y_true": np.asarray(y_test, dtype=np.int64),
            "prob": prob.astype(np.float64),
            "pred_default": np.asarray(pred, dtype=np.int64),
            "pred_calibrated": calibrated_pred,
        }
        return scores, per_sample
    return scores


def score_multiclass(
    clf: Any,
    x_test: np.ndarray,
    y_test: np.ndarray,
    return_per_sample: bool = False,
) -> dict[str, float] | tuple[dict[str, float], dict[str, np.ndarray]]:
    proba = None
    with perf.measure("probe.score/multiclass", n_samples=len(y_test), n_features=x_test.shape[1]):
        pred = clf.predict(x_test)
        try:
            proba = clf.predict_proba(x_test)
            macro_auc = float(
                roc_auc_score(y_test, proba, multi_class="ovr", average="macro", labels=clf.classes_)
            )
        except (ValueError, AttributeError):
            macro_auc = float("nan")
    n_classes = len(getattr(clf, "classes_", []))
    perf.log_static("probe.macs/multiclass", macs=x_test.shape[1] * max(n_classes, 1), n_samples=len(y_test))
    values, counts = np.unique(y_test, return_counts=True)
    seen = np.asarray(getattr(clf, "classes_", []))
    seen_mask = np.isin(y_test, seen) if seen.size else np.zeros(len(y_test), dtype=bool)
    n_unseen = int(len(set(np.unique(y_test).tolist()) - set(seen.tolist())))
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="y_pred contains classes not in y_true")
        scores = {
            "macro_f1": float(f1_score(y_test, pred, average="macro", zero_division=0)),
            "weighted_f1": float(f1_score(y_test, pred, average="weighted", zero_division=0)),
            "balanced_accuracy": float(balanced_accuracy_score(y_test, pred)),
            "accuracy": float(accuracy_score(y_test, pred)),
            "macro_auc": macro_auc,
            "test_n_classes": int(len(values)),
            "test_majority_rate": float(counts.max() / len(y_test)),
            "n_classes_seen": int(seen.size),
            "n_classes_unseen": n_unseen,
            "unseen_prevalence": float(1.0 - seen_mask.mean()) if len(y_test) else float("nan"),
            "shared_macro_f1": (
                float(f1_score(y_test[seen_mask], pred[seen_mask], average="macro", zero_division=0))
                if seen_mask.any()
                else float("nan")
            ),
            "shared_balanced_accuracy": (
                float(balanced_accuracy_score(y_test[seen_mask], pred[seen_mask]))
                if seen_mask.any()
                else float("nan")
            ),
            "shared_accuracy": (
                float(accuracy_score(y_test[seen_mask], pred[seen_mask])) if seen_mask.any() else float("nan")
            ),
        }
    # Carry every calibration column (full-label + shared-class + label-support diagnostic).
    # NOTE: cal["unseen_prevalence"] is computed from the same classes_ as the block above,
    # so merging is consistent rather than conflicting.
    cal = multiclass_calibration(y_test, proba, np.asarray(getattr(clf, "classes_", [])))
    scores.update(cal)
    if return_per_sample:
        per_sample = {
            "y_true": np.asarray(y_test, dtype=np.int64),
            "pred": np.asarray(pred, dtype=np.int64),
            "classes": np.asarray(getattr(clf, "classes_", []), dtype=np.int64),
            "proba": np.asarray(proba, dtype=np.float64) if proba is not None else np.zeros((len(y_test), 0)),
        }
        return scores, per_sample
    return scores


def per_class_iou(y_true: np.ndarray, y_pred: np.ndarray, classes: np.ndarray) -> dict[int, float]:
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    out: dict[int, float] = {}
    for c in classes:
        c = int(c)
        t = y_true == c
        if not t.any():
            out[c] = float("nan")
            continue
        p = y_pred == c
        union = int(np.logical_or(t, p).sum())
        out[c] = float(int(np.logical_and(t, p).sum()) / union) if union > 0 else 0.0
    return out


def score_segmentation(
    clf: Any,
    x_test: np.ndarray,
    y_test: np.ndarray,
    *,
    eval_classes: np.ndarray | None = None,
    return_per_sample: bool = False,
) -> dict[str, float] | tuple[dict[str, float], dict[str, np.ndarray]]:
    with perf.measure("probe.score/segmentation", n_samples=len(y_test), n_features=x_test.shape[1]):
        pred = clf.predict(x_test)
        try:
            proba = clf.predict_proba(x_test)
        except (ValueError, AttributeError):
            proba = None
    classes = np.asarray(eval_classes if eval_classes is not None else getattr(clf, "classes_", np.unique(y_test)))
    proba_classes = np.asarray(getattr(clf, "classes_", classes))
    ious = per_class_iou(y_test, pred, classes)
    present = [v for v in ious.values() if not np.isnan(v)]
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="y_pred contains classes not in y_true")
        scores = {
            "miou": float(np.mean(present)) if present else float("nan"),
            "pixel_accuracy": float(accuracy_score(y_test, pred)),
            "macro_f1": float(f1_score(y_test, pred, average="macro", zero_division=0)),
            "weighted_f1": float(f1_score(y_test, pred, average="weighted", zero_division=0)),
            "n_eval_classes": int(len(classes)),
            "n_present_classes": int(len(present)),
        }
    # Per-pixel calibration (columns aligned to the probe's own classes_, not eval_classes).
    cal = multiclass_calibration(y_test, proba, proba_classes)
    scores.update(cal)
    if return_per_sample:
        per_sample = {
            "y_true": np.asarray(y_test, dtype=np.int64),
            "pred": np.asarray(pred, dtype=np.int64),
            "classes": proba_classes.astype(np.int64),
            "proba": np.asarray(proba, dtype=np.float64) if proba is not None else np.zeros((len(y_test), 0)),
        }
        return scores, per_sample
    return scores


def _miou_from_confusion(conf: np.ndarray) -> float:
    conf = conf.astype(np.float64)
    tp = np.diag(conf)
    row, col = conf.sum(1), conf.sum(0)
    present = row > 0
    union = row + col - tp
    iou = np.divide(tp, union, out=np.zeros_like(tp), where=union > 0)
    return float(iou[present].mean()) if present.any() else float("nan")


def _as_eval_indices(values: np.ndarray, classes: np.ndarray, name: str) -> np.ndarray:
    values = np.asarray(values, dtype=np.int64)
    classes = np.asarray(classes, dtype=np.int64)
    if classes.size == 0:
        raise ValueError("eval_classes is empty")
    if np.array_equal(classes, np.arange(classes.size)):
        bad = (values < 0) | (values >= classes.size)
        if bad.any():
            raise ValueError(f"{name} contains values outside eval_classes: {np.unique(values[bad])[:10].tolist()}")
        return values
    order = np.argsort(classes)
    sorted_classes = classes[order]
    pos = np.searchsorted(sorted_classes, values)
    valid = pos < len(sorted_classes)
    safe_pos = np.minimum(pos, len(sorted_classes) - 1)
    valid &= sorted_classes[safe_pos] == values
    if not valid.all():
        raise ValueError(f"{name} contains values outside eval_classes: {np.unique(values[~valid])[:10].tolist()}")
    return order[pos]


def _segmentation_metrics_from_confusion(conf: np.ndarray, tile_mious: list[float]) -> dict[str, float]:
    c = conf.astype(np.float64)
    tp = np.diag(c)
    row, col, total = c.sum(1), c.sum(0), c.sum()
    present = row > 0
    union = row + col - tp
    iou = np.divide(tp, union, out=np.zeros_like(tp), where=union > 0)
    prec = np.divide(tp, col, out=np.zeros_like(tp), where=col > 0)
    rec = np.divide(tp, row, out=np.zeros_like(tp), where=row > 0)
    f1 = np.divide(2 * prec * rec, prec + rec, out=np.zeros_like(tp), where=(prec + rec) > 0)
    labelset = (row > 0) | (col > 0)
    out = {
        "miou": float(iou[present].mean()) if present.any() else float("nan"),
        "pixel_accuracy": float(tp.sum() / total) if total > 0 else float("nan"),
        "macro_f1": float(f1[labelset].mean()) if labelset.any() else 0.0,
        "weighted_f1": float((row * f1).sum() / row.sum()) if row.sum() > 0 else 0.0,
        "n_eval_classes": int(conf.shape[0]),
        "n_present_classes": int(present.sum()),
    }
    if tile_mious:
        arr = np.asarray(tile_mious)
        out.update(
            {
                "mean_per_tile_miou": float(arr.mean()),
                "worst_tile_miou": float(arr.min()),
                "n_tiles_scored": len(arr),
            }
        )
    else:
        out.update({"mean_per_tile_miou": float("nan"), "worst_tile_miou": float("nan"), "n_tiles_scored": 0})
    return out


def score_segmentation_streamed(
    clf: Any,
    tiles: Any,
    eval_classes: np.ndarray,
    *,
    predict_sink: Any = None,
) -> dict[str, float]:
    """Stream tiles ONCE, accumulating segmentation metrics. When ``predict_sink`` is given, the SAME
    per-tile inference also feeds predictions: ``tiles`` must then yield ``(tile_key, features, labels)``
    and ``predict_sink(tile_key, labels, pred)`` is called once per tile (a single inference pass -- never
    a second one). With no sink, ``tiles`` yields the usual ``(features, labels)`` pairs."""
    k = len(eval_classes)
    conf = np.zeros((k, k), dtype=np.int64)
    tile_mious: list[float] = []
    n_pixels = 0
    # Streaming, memory-safe pixel-level calibration, on the same convention as
    # multiclass_calibration (see its docstring). Skipped if the probe has no predict_proba.
    # proba columns are the probe's own classes_ (constant across tiles for one fitted clf).
    # Pixels whose true class has no column are tracked separately: they make the full-label
    # NLL infinite, and in union space they still owe the (0-1)^2 = 1 Brier penalty.
    n_bins = 10
    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    bin_count = np.zeros(n_bins)
    bin_conf = np.zeros(n_bins)
    bin_correct = np.zeros(n_bins)
    bin_conf_s = np.zeros(n_bins)      # shared-class (supported pixels only)
    bin_correct_s = np.zeros(n_bins)
    nll_sum_seen = 0.0                 # accumulated over SUPPORTED pixels only
    sq_sum = 0.0                       # sum over all pixels of sum_j p_j^2
    sq_sum_seen = 0.0
    pt_sum_seen = 0.0
    n_seen = 0
    cal_n = 0
    cal_ok = True
    proba_classes: np.ndarray | None = None
    with perf.measure("probe.score/segmentation_streamed", n_features=-1):
        for item in tiles:
            if predict_sink is not None:
                tile_key, features, labels = item
            else:
                tile_key, (features, labels) = None, item
            labels = np.asarray(labels)
            if labels.size == 0:
                continue
            features = np.asarray(features, dtype=np.float32)
            scoring_clf = clf
            proba = None
            if cal_ok:
                try:
                    proba = np.asarray(scoring_clf.predict_proba(features), dtype=np.float64)
                    if proba_classes is None:
                        proba_classes = np.asarray(getattr(scoring_clf, "classes_", []), dtype=np.int64)
                except (AttributeError, ValueError):
                    proba, cal_ok = None, False
            if proba is not None and proba_classes is not None and proba_classes.size and proba.shape[1] == proba_classes.size:
                pred = proba_classes[proba.argmax(axis=1)]
                conf_px = proba.max(axis=1)
                correct_px = (pred == labels).astype(np.float64)
                idx = np.clip(np.digitize(conf_px, bin_edges[1:-1]), 0, n_bins - 1)
                np.add.at(bin_count, idx, 1.0)
                np.add.at(bin_conf, idx, conf_px)
                np.add.at(bin_correct, idx, correct_px)
                pos = np.clip(np.searchsorted(proba_classes, labels), 0, proba_classes.size - 1)
                seen = proba_classes[pos] == labels
                pt = np.where(seen, proba[np.arange(len(labels)), pos], 0.0)
                sq_px = (proba**2).sum(axis=1)
                # supported pixels: ordinary NLL / Brier contributions
                nll_sum_seen += float(-np.log(np.clip(pt[seen], _NUM_EPS, 1.0)).sum())
                pt_sum_seen += float(pt[seen].sum())
                sq_sum_seen += float(sq_px[seen].sum())
                sq_sum += float(sq_px.sum())
                n_seen += int(seen.sum())
                # shared-class ECE bins (supported pixels only)
                np.add.at(bin_conf_s, idx[seen], conf_px[seen])
                np.add.at(bin_correct_s, idx[seen], correct_px[seen])
                cal_n += int(labels.size)
            else:
                pred = np.asarray(scoring_clf.predict(features))
            if predict_sink is not None:  # SAME inference feeds predictions -- no second pass
                predict_sink(tile_key, labels, pred)
            lab = _as_eval_indices(labels, eval_classes, "segmentation labels")
            prd = _as_eval_indices(pred, eval_classes, "segmentation predictions")
            tile_conf = np.bincount(lab * k + prd, minlength=k * k).reshape(k, k)
            conf += tile_conf
            miou_t = _miou_from_confusion(tile_conf)
            if not np.isnan(miou_t):
                tile_mious.append(miou_t)
            n_pixels += int(labels.size)
    metrics = _segmentation_metrics_from_confusion(conf, tile_mious)
    metrics["n_test"] = n_pixels
    if cal_n > 0:
        ece_all = float(np.abs(bin_correct - bin_conf).sum() / cal_n)
        # Union-space Brier: EVERY pixel owes sum(oh^2)=1 on its true class, supported or not.
        union_brier = float((sq_sum - 2.0 * pt_sum_seen + cal_n) / cal_n)
        metrics["ece"] = ece_all
        metrics["top_label_ece_all"] = ece_all
        # Full-label NLL is degenerate (-log 0) if any pixel's true class had no column.
        metrics["nll"] = float(nll_sum_seen / cal_n) if n_seen == cal_n else float("inf")
        metrics["brier"] = union_brier
        metrics["union_brier"] = union_brier
        metrics["unseen_prevalence"] = float(1.0 - n_seen / cal_n)
        if n_seen > 0:
            metrics["shared_ece"] = float(np.abs(bin_correct_s - bin_conf_s).sum() / n_seen)
            metrics["shared_nll"] = float(nll_sum_seen / n_seen)
            metrics["shared_brier"] = float((sq_sum_seen - 2.0 * pt_sum_seen + n_seen) / n_seen)
        else:
            metrics["shared_ece"] = metrics["shared_nll"] = metrics["shared_brier"] = float("nan")
    else:
        metrics.update(dict.fromkeys(CALIBRATION_KEYS, float("nan")))
    # Persist the K×K confusion (+ the eval-class order, the probe's SEEN classes, and the
    # per-tile mIoUs) as JSON-string fields so PASTIS per-class IoU, shared-class (restrict to
    # seen classes), worst-region (per-holdout rows), and worst-tile bootstrap CIs are all
    # derivable post-hoc without re-running the dense benchmark. csv.DictWriter quotes these.
    metrics["seg_confusion"] = json.dumps(conf.tolist())
    metrics["seg_eval_classes"] = json.dumps(np.asarray(eval_classes, dtype=np.int64).tolist())
    metrics["seg_train_classes"] = json.dumps(np.asarray(getattr(clf, "classes_", []), dtype=np.int64).tolist())
    metrics["seg_tile_mious"] = json.dumps([round(float(m), 6) for m in tile_mious])
    return metrics
