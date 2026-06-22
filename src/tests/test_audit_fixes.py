"""Regression tests for the correctness-audit fixes (cache identity, resume, strict regimes,
bootstrap, target-only calibration, Galileo metadata). Pure numpy/sklearn where possible so
they run without the heavy model deps; the Galileo test skips if torch is unavailable."""

from __future__ import annotations

import os

import numpy as np
import pytest
from sklearn.metrics import roc_auc_score

from utils import cacheutils as C
from utils import ioutils as IOU


# --------------------------------------------------------------------------- #
# AUC ties + seed-collapse + bootstrap
# --------------------------------------------------------------------------- #
def test_auc_midrank_matches_sklearn_on_ties():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, 300)
    p = rng.integers(0, 5, 300) / 4.0  # heavy ties (KNN-like)
    assert abs(IOU._auc(y, p.astype(float)) - roc_auc_score(y, p)) < 1e-9
    pc = rng.random(300)
    assert abs(IOU._auc(y, pc) - roc_auc_score(y, pc)) < 1e-9


def test_by_seed_arrays_dedups_within_seed_no_collapse():
    # 3 seeds, each predicts the same 5 sample_ids -> grouped per seed (NOT collapsed across seeds);
    # an exact (seed, holdout, sample_id) duplicate is de-duped so no observation is double-counted.
    preds = []
    for seed in range(3):
        for sid in range(5):
            preds.append({"holdout": "r1", "sample_id": sid, "seed": seed, "y_true": sid % 2,
                          "prob": 0.5 + 0.1 * seed, "pred_default": sid % 2, "pred_calibrated": sid % 2})
    preds.append(dict(preds[0]))  # crash-recovery duplicate of (seed0, r1, sample0)
    by_seed = IOU._by_seed_arrays(preds)
    assert sorted(by_seed) == [0, 1, 2]                 # seeds kept separate, not collapsed
    assert all(len(by_seed[s][0]) == 5 for s in by_seed)  # duplicate removed within seed 0


def test_cluster_resample_is_two_stage():
    rng = np.random.default_rng(0)
    clusters = np.array(["A"] * 6 + ["B"] * 6)
    draws = [IOU._cluster_resample(rng, clusters) for _ in range(20)]
    # within-cluster resampling => some draw has a repeated index (not a clean partition)
    assert any(len(set(d.tolist())) < len(clusters) for d in draws)
    # single cluster falls back to a plain sample resample of the right length
    one = IOU._cluster_resample(rng, np.array(["X"] * 8))
    assert len(one) == 8


def _delta_rows(extra_regimes=()):
    rows = []

    def add(regime, bt, lb, val, seed, holdout):
        rows.append({"model": "m", "benchmark": "b", "method": "erm", "probe_family": "logistic",
                     "split_regime": regime, "budget_type": bt, "label_budget": lb, "seed": seed,
                     "holdout": holdout, "auc": val, "test_pos_rate": 0.5, "test_n_classes": 2,
                     "test_majority_rate": 0.5})
    for s in range(3):
        add("random_id", "source", 1.0, 0.9, s, "random_id")
        for reg, base in [("regA", 0.7), ("regB", 0.6), ("regC", 0.65)]:
            add("geographic_ood", "target", 0.0, base + 0.01 * s, s, reg)
        for reg in extra_regimes:
            add(reg, "target", 0.0, 0.63, s, f"{reg}_dom")
    return rows


def test_compute_deltas_surfaces_secondary_ood_regimes():
    rows = _delta_rows(extra_regimes=("climate_ood", "temporal_ood"))
    out = [r for r in IOU.compute_deltas(rows, ["auc"], n_boot=100, seed=0) if r["metric"] == "auc"]
    r = out[0]
    for axis in ("climate", "temporal"):
        assert f"ood_{axis}" in r and f"delta_{axis}" in r
    # primary geographic CI is present and brackets the point estimate (non-degenerate)
    assert r["delta_ci_lo"] <= r["delta"] <= r["delta_ci_hi"]


def test_sample_ci_filters_to_anchor_budgets():
    rows = _delta_rows()
    preds = []

    def addp(regime, bt, lb, holdout, prob, n, seed):
        for i in range(n):
            preds.append({"model": "m", "benchmark": "b", "method": "erm", "probe_family": "logistic",
                          "split_regime": regime, "budget_type": bt, "label_budget": lb, "seed": seed,
                          "holdout": holdout, "group": holdout, "sample_id": i, "y_true": i % 2,
                          "prob": prob, "pred_default": int(prob > 0.5), "pred_calibrated": int(prob > 0.5)})
    for s in range(3):
        addp("random_id", "source", 1.0, "random_id", 0.8, 40, s)
        addp("random_id", "source", 0.05, "random_id", 0.2, 40, s)            # distractor budget
        addp("geographic_ood", "target", 0.0, "regA", 0.6, 30, s)
        addp("geographic_ood", "target", 0.0, "regB", 0.65, 30, s)
        addp("geographic_ood", "target", 50, "regA", 0.99, 30, s)             # distractor budget
    out = [r for r in IOU.compute_deltas(rows, ["auc"], predictions=preds, n_boot=50,
                                         n_boot_sample=50, seed=0) if r["metric"] == "auc"][0]
    # rows are kept per-seed (no collapse): 3 seeds x 40 id, 3 seeds x (30+30) ood; distractor
    # budgets (source 0.05, target 50) are excluded by the anchor-budget filter.
    assert out["n_id_samples"] == 120 and out["n_ood_samples"] == 180
    assert "delta_sample_pt" in out                                   # explicit CI centre present


# --------------------------------------------------------------------------- #
# Cache identity
# --------------------------------------------------------------------------- #
def test_hf_checkpoint_fingerprint_is_download_independent_and_revision_pinned():
    # HF fingerprint doesn't depend on any local file, and changing the pinned revision changes it.
    fp_presto = C._checkpoint_fingerprint("presto")
    assert fp_presto and fp_presto == C._checkpoint_fingerprint("presto")  # stable
    saved = C._CHECKPOINT_SPECS["presto"]
    try:
        C._CHECKPOINT_SPECS["presto"] = ("hf", None, None, "torchgeo/presto@DIFFERENT:model.pth")
        assert C._checkpoint_fingerprint("presto") != fp_presto           # revision baked into key
    finally:
        C._CHECKPOINT_SPECS["presto"] = saved
    # raw has no checkpoint; its identity is the RAW_MODE featurization (so it changes with mode)
    import os as _os
    _saved = _os.environ.get("RAW_MODE")
    try:
        _os.environ["RAW_MODE"] = "flatten"
        fp_flat = C._checkpoint_fingerprint("raw")
        _os.environ["RAW_MODE"] = "stats"
        assert C._checkpoint_fingerprint("raw") != fp_flat
    finally:
        if _saved is None:
            _os.environ.pop("RAW_MODE", None)
        else:
            _os.environ["RAW_MODE"] = _saved


def test_local_checkpoint_fingerprint_tracks_content(tmp_path, monkeypatch):
    wp = tmp_path / "AgriFM.pth"
    wp.write_bytes(b"weights-v1" + b"\0" * 1000)
    monkeypatch.setenv("AGRIFM_WEIGHTS", str(wp))
    fp1 = C._checkpoint_fingerprint("agrifm")
    # same size, mtime preserved, but different content -> fingerprint must still change
    mtime = wp.stat().st_mtime
    wp.write_bytes(b"weights-v2" + b"\0" * 1000)
    os.utime(wp, (mtime, mtime))
    assert C._checkpoint_fingerprint("agrifm") != fp1


def test_embedding_key_changes_with_checkpoint():
    class B:
        n_samples = 10
    a = C.embedding_cache_path(B(), "cropharvest", "presto", "tag")
    assert "_w" in str(a)                       # checkpoint folded into the key
    assert "_w" not in C.embedding_cache_path(B(), "cropharvest", "raw", "tag").name


def test_input_fingerprint_recursive_catches_deep_edit(tmp_path, monkeypatch):
    sub = tmp_path / "preprocess"
    sub.mkdir()
    (sub / "a.npz").write_bytes(b"x" * 10)
    monkeypatch.setenv("DATA_FINGERPRINT", "deep")
    fp1 = C._input_fingerprint(tmp_path)
    (sub / "a.npz").write_bytes(b"y" * 20)      # deep in-place edit
    assert C._input_fingerprint(tmp_path) != fp1
    monkeypatch.setenv("DATA_FINGERPRINT", "top")
    assert C._input_fingerprint(tmp_path)       # top mode still produces a value


# --------------------------------------------------------------------------- #
# Atomic append (resume)
# --------------------------------------------------------------------------- #
def test_append_jsonl_roundtrips_batch(tmp_path):
    p = tmp_path / "rows.jsonl"
    IOU.append_jsonl(p, [{"a": 1}, {"a": 2}])
    IOU.append_jsonl(p, [])                      # empty batch is a no-op
    IOU.append_jsonl(p, [{"a": 3}])
    assert [r["a"] for r in IOU.read_jsonl(p)] == [1, 2, 3]


# --------------------------------------------------------------------------- #
# Strict regimes + target-only calibration + subset guard
# --------------------------------------------------------------------------- #
def test_dropped_curated_holdout_is_surfaced_and_strict_raises():
    import main
    groups = np.array(["A"] * 20 + ["B"] * 20 + ["C"] * 5, dtype=object)
    y = np.array([0, 1] * 20 + [0] * 5)          # C is one-class -> dropped
    bench = type("FB", (), {"name": "fb", "groups": groups})()

    main._REGIME_PROBLEMS.clear()
    saved = main.STRICT_REGIMES
    try:
        main.STRICT_REGIMES = False
        list(main._iter_splits("geographic_ood", bench, y, holdouts=["A", "B", "C"], seed=0))
        assert any("C" in reason for _, reg, reason in main._REGIME_PROBLEMS if reg == "geographic_ood")

        main.STRICT_REGIMES = True
        with pytest.raises(RuntimeError):
            list(main._iter_splits("geographic_ood", bench, y, holdouts=["A", "B", "C"], seed=0))
    finally:
        main.STRICT_REGIMES = saved


def test_strict_regimes_defaults_true():
    import importlib

    import main
    saved = os.environ.pop("STRICT_REGIMES", None)
    try:
        importlib.reload(main)
        assert main.STRICT_REGIMES is True       # in-file default is strict
    finally:
        if saved is not None:
            os.environ["STRICT_REGIMES"] = saved
        importlib.reload(main)


def test_subset_indices_guards_tiny_pool():
    from evals.evals import subset_indices
    assert list(subset_indices(np.array([1]), budget=0.5, seed=0)) == [0]   # no crash on <2 pool


# --------------------------------------------------------------------------- #
# Galileo metadata (masks + months) -- needs torch
# --------------------------------------------------------------------------- #
def test_galileo_point_path_uses_real_masks_and_months():
    pytest.importorskip("torch")
    try:
        from models.galileo import _GALILEO_S2_GROUP_IDS, GalileoModel
    except Exception as exc:  # galileo package / weights stub unavailable in this env
        pytest.skip(f"galileo import unavailable: {exc}")
    n, t = 2, 12
    doy = np.tile(np.array([15, 46, 75, 106, 136, 167, 197, 228, 259, 289, 320, 350]), (n, 1))
    s2_mask = np.ones((n, t), np.float32)
    s2_mask[:, 3] = 0.0  # timestep 3 unobserved
    bench = type("B", (), {
        "s2": np.ones((n, t, 11), np.float32), "s1": np.zeros((n, t, 2), np.float32),
        "s2_bands": ["B2", "B3", "B4", "B5", "B6", "B7", "B8", "B8A", "B11", "B12", "NDVI"],
        "s1_bands": ["VV", "VH"], "s2_mask": s2_mask, "s1_mask": np.zeros((n, t), np.float32),
        "doy": doy,
    })()
    *_, s_t_m, _, _, _, months = GalileoModel()._bench_to_galileo(bench)
    # months follow the calendar (0..11), not a constant July(6)
    assert months[0].tolist() == list(range(12))
    # S2 groups are masked (1) exactly at the unobserved timestep 3, available (0) elsewhere
    gid = _GALILEO_S2_GROUP_IDS[0]
    assert s_t_m[0, 0, 0, 3, gid] == 1.0 and s_t_m[0, 0, 0, 0, gid] == 0.0


# --------------------------------------------------------------------------- #
# Target-sweep redesign (#4/#5/#6): shared fixed test set, oracle fits 80%, nested few-shot
# --------------------------------------------------------------------------- #
def test_target_sweep_shared_test_and_full_pool_oracle():
    from evals import evals as EV
    rng = np.random.default_rng(0)

    def make(n, shift):
        y = rng.integers(0, 2, n)
        x = rng.normal(size=(n, 8)) + y[:, None] * 1.2 + shift
        return x.astype(np.float32), y.astype(np.int64)
    xs, ys = make(300, 0.0)
    xt, yt = make(200, 0.4)
    xv, yv = make(60, 0.0)
    rows = []
    EV.run_probes_target(
        rows, xs, xt, ys, yt, seed=0,
        meta={"holdout": "t", "method": "erm", "benchmark": "syn"},
        groups_source=np.array(["s"] * len(ys)), x_val=xv, y_val=yv, family="logistic",
    )
    # held-out rows (every budget) share ONE fixed 20% test set
    held = {r["label_budget"]: r["n_test"] for r in rows if r.get("evaluation_split") == "held_out"}
    assert len(set(held.values())) == 1 and next(iter(held.values())) == 40
    # budget 0 additionally emits a FULL-target zero-shot anchor (the primary OOD estimand)
    full = [r for r in rows if r["label_budget"] == 0 and r.get("evaluation_split") == "full"]
    assert len(full) == 1 and full[0]["n_test"] == 200
    oracle = next(r for r in rows if r["label_budget"] == EV.TARGET_ID_UPPER_BOUND)
    assert oracle["n_train_sub"] >= int(0.75 * 200)            # oracle trains on ~80% (not 64%)
    assert oracle["threshold_source"] == "target_internal_tuned_oof"  # source-free tuned + OOF threshold


# --------------------------------------------------------------------------- #
# Run-signature guard (#1)
# --------------------------------------------------------------------------- #
def test_run_signature_guard_rejects_mismatch(tmp_path, monkeypatch):
    import main
    d = tmp_path / "results"
    d.mkdir()
    monkeypatch.setattr(main, "OVERWRITE_MODE", "skip")
    main._check_run_signature(d, "sigAAAA")                    # empty dir is fine
    main._publish_run_signature(d, "sigAAAA")
    main._check_run_signature(d, "sigAAAA")                    # same signature resumes fine
    with pytest.raises(RuntimeError):
        main._check_run_signature(d, "sigDIFFERENT")           # different config refuses to mix
    # results present but UNSIGNED -> also refused (stale/foreign adoption)
    d2 = tmp_path / "results2"
    d2.mkdir()
    (d2 / "probe_results.jsonl").write_text('{"a": 1}\n')
    with pytest.raises(RuntimeError):
        main._check_run_signature(d2, "sigAAAA")


# --------------------------------------------------------------------------- #
# Worst-region direction for error metrics (#13)
# --------------------------------------------------------------------------- #
def test_worst_region_uses_max_for_error_metrics():
    rows = []

    def add(metric, val, seed, holdout):
        rows.append({"model": "m", "benchmark": "b", "method": "erm", "probe_family": "logistic",
                     "split_regime": "geographic_ood", "budget_type": "target", "label_budget": 0.0,
                     "seed": seed, "holdout": holdout, metric: val,
                     "test_pos_rate": 0.5, "test_n_classes": 2, "test_majority_rate": 0.5})
    # ID anchor needed for compute_deltas to emit a row
    for s in range(2):
        rows.append({"model": "m", "benchmark": "b", "method": "erm", "probe_family": "logistic",
                     "split_regime": "random_id", "budget_type": "source", "label_budget": 1.0,
                     "seed": s, "holdout": "random_id", "brier": 0.1,
                     "test_pos_rate": 0.5, "test_n_classes": 2, "test_majority_rate": 0.5})
        add("brier", 0.2, s, "regA")     # better (lower)
        add("brier", 0.4, s, "regB")     # WORST region for an error metric = the MAX
    out = [r for r in IOU.compute_deltas(rows, ["brier"], n_boot=0, seed=0) if r["metric"] == "brier"][0]
    assert abs(out["ood_worst_region"] - 0.4) < 1e-9   # max, not min (0.2)
