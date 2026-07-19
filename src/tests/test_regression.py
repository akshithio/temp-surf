from __future__ import annotations

import os

import numpy as np
import pytest
from sklearn.metrics import roc_auc_score

from utils import cacheutils as C
from utils import ioutils as IOU


def test_auc_midrank_matches_sklearn_on_ties():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, 300)
    p = rng.integers(0, 5, 300) / 4.0  # heavy ties (KNN-like)
    assert abs(IOU._auc(y, p.astype(float)) - roc_auc_score(y, p)) < 1e-9
    pc = rng.random(300)
    assert abs(IOU._auc(y, pc) - roc_auc_score(y, pc)) < 1e-9


def test_by_seed_arrays_dedups_within_seed_no_collapse():
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
    assert any(len(set(d.tolist())) < len(clusters) for d in draws)
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
    rows = _delta_rows(extra_regimes=("official",))
    out = [r for r in IOU.compute_deltas(rows, ["auc"], n_boot=100, seed=0) if r["metric"] == "auc"]
    by_regime = {r["ood_regime"]: r for r in out}
    assert {"official", "geographic_ood"} <= set(by_regime)
    assert by_regime["geographic_ood"]["delta_ci_lo"] <= by_regime["geographic_ood"]["delta"]
    assert by_regime["official"]["ood"] == 0.63


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
    assert out["n_id_samples"] == 120 and out["n_ood_samples"] == 180
    assert "delta_sample_pt" in out                                   # explicit CI centre present


def test_checkpoint_sha256_is_stable_and_raw_tracks_mode():
    fp = C.checkpoint_sha256("presto")
    assert fp and fp == C.checkpoint_sha256("presto") and len(fp) == 64  # stable full SHA
    import os as _os
    _saved = _os.environ.get("RAW_MODE")
    try:
        _os.environ["RAW_MODE"] = "flatten"
        flat = C.checkpoint_sha256("raw")
        _os.environ["RAW_MODE"] = "stats"
        assert C.checkpoint_sha256("raw") != flat        # raw identity tracks the featurization mode
    finally:
        if _saved is None:
            _os.environ.pop("RAW_MODE", None)
        else:
            _os.environ["RAW_MODE"] = _saved


def test_checkpoint_sha256_tracks_local_content(tmp_path, monkeypatch):
    wp = tmp_path / "AgriFM.pth"
    wp.write_bytes(b"weights-v1" + b"\0" * 1000)
    monkeypatch.setenv("AGRIFM_WEIGHTS", str(wp))
    C._CHECKPOINT_SHA_CACHE.clear()
    fp1 = C.checkpoint_sha256("agrifm")
    mtime = wp.stat().st_mtime
    wp.write_bytes(b"weights-v2" + b"\0" * 1000)
    os.utime(wp, (mtime, mtime))  # SAME mtime -> content hash must still move
    C._CHECKPOINT_SHA_CACHE.clear()
    assert C.checkpoint_sha256("agrifm") != fp1


def test_embedding_identity_carries_checkpoint_in_manifest_not_path():
    # Fixed readable path -- the checkpoint no longer perturbs it.
    a = C.embedding_cache_path("cropharvest", "presto", "baseline")
    assert a.name == "baseline.npy" and "_w" not in str(a)
    # Checkpoint identity is a full SHA-256 recorded in the manifest, and distinct per model.
    assert len(C.checkpoint_sha256("presto")) == 64
    assert C.checkpoint_sha256("presto") != C.checkpoint_sha256("raw")


def test_append_jsonl_roundtrips_batch(tmp_path):
    p = tmp_path / "rows.jsonl"
    IOU.append_jsonl(p, [{"a": 1}, {"a": 2}])
    IOU.append_jsonl(p, [])                      # empty batch is a no-op
    IOU.append_jsonl(p, [{"a": 3}])
    assert [r["a"] for r in IOU.read_jsonl(p)] == [1, 2, 3]


def test_geographic_absent_target_is_recorded_as_dropped_not_silently_missing():
    """schema v2: a frozen LODO target that is absent from the data yields NO split, but the drop is
    recorded as an explicit dropped_holdout audit event rather than a silent gap."""
    from types import SimpleNamespace

    from evals.regimes import base as RB
    from evals.regimes import geographic_ood as geo

    groups = np.array(["frh01"] * 20 + ["frh02"] * 20 + ["frh03"] * 20, dtype=object)  # frh04 absent
    bench = SimpleNamespace(
        name="breizhcrops", groups=groups, labels=np.arange(60) % 3,
        latlon=np.column_stack([np.linspace(48, 49, 60), np.linspace(-4, -2, 60)]), sample_ids=np.arange(60),
    )
    bench_mod = SimpleNamespace(BENCHMARK="breizhcrops", make_targets=lambda b: (b.labels, b.groups))

    RB.clear_split_audit_events()
    labels = [s.label for s in geo.iter_source_target_splits(bench, bench_mod, 0)]
    assert "frh04" not in labels  # the absent target produces no split
    dropped = [e for e in RB.SPLIT_AUDIT_EVENTS if e["kind"] == "dropped_holdout" and e.get("holdout") == "frh04"]
    assert dropped and dropped[0]["reason"] == "absent_from_data"


def test_subset_indices_guards_tiny_pool():
    from evals.evals import subset_indices
    assert list(subset_indices(np.array([1]), budget=0.5, seed=0)) == [0]   # no crash on <2 pool


def test_galileo_point_path_uses_real_masks_and_months():
    pytest.importorskip("torch")
    try:
        from models.galileo import _GALILEO_S2_GROUP_IDS, GalileoModel
    except Exception as exc:  # galileo package / weights stub unavailable in this env
        pytest.skip(f"galileo import unavailable: {exc}")
    from dataio.get_input import Benchmark, ModalitySeries, NativeSeries, _synthetic_month_doy

    n = 2
    obs_months = np.array([0, 1, 2, 4, 5, 6, 7, 8, 9, 10, 11], dtype=np.int64)  # month 3 (April) unobserved
    doy = _synthetic_month_doy(12)[obs_months].astype(np.float32)
    s2_bands = ["B2", "B3", "B4", "B5", "B6", "B7", "B8", "B8A", "B11", "B12", "NDVI"]
    s2 = ModalitySeries(
        [np.ones((len(obs_months), 11), np.float32) for _ in range(n)],
        [obs_months] * n, [doy] * n, [np.full(len(obs_months), 2019, np.int64)] * n, s2_bands,
    )
    native = NativeSeries(s2=s2, s1=ModalitySeries.absent(n), climate=ModalitySeries.absent(n))
    bench = Benchmark(
        name="pastis", label_kind="segmentation", native=native,
        labels=np.zeros(n, np.int64), groups=np.array(["a", "b"], dtype=object),
        latlon=np.zeros((n, 2), np.float32), years=np.full(n, 2019, np.int64),
    )
    *_, s_t_m, _, _, _, months = GalileoModel()._bench_to_galileo(bench)
    assert months[0].tolist() == list(range(12))
    gid = _GALILEO_S2_GROUP_IDS[0]
    assert s_t_m[0, 0, 0, 3, gid] == 1.0 and s_t_m[0, 0, 0, 0, gid] == 0.0


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
    held = {r["label_budget"]: r["n_test"] for r in rows if r.get("evaluation_split") == "held_out"}
    assert len(set(held.values())) == 1 and next(iter(held.values())) == 40
    full = [r for r in rows if r["label_budget"] == 0 and r.get("evaluation_split") == "full"]
    assert len(full) == 1 and full[0]["n_test"] == 200
    oracle = next(r for r in rows if r["label_budget"] == EV.TARGET_ID_UPPER_BOUND)
    assert oracle["n_train_sub"] >= int(0.75 * 200)
    assert oracle["threshold_source"] == "target_internal_tuned_oof"


def test_run_manifest_guard_rejects_mismatch(tmp_path, monkeypatch):
    from utils import runstate
    d = tmp_path / "results"
    d.mkdir()
    m = {"schema": 1, "seeds": [0]}
    runstate.check_run_manifest(d, m, overwrite_mode=False)                             # empty dir is fine
    runstate.publish_run_manifest(d, m)
    runstate.check_run_manifest(d, m, overwrite_mode=False)                             # exact match resumes fine
    with pytest.raises(RuntimeError):
        runstate.check_run_manifest(d, {"schema": 1, "seeds": [1]}, overwrite_mode=False)  # different config refuses
    d2 = tmp_path / "results2"
    d2.mkdir()
    (d2 / "probe_results.jsonl").write_text('{"a": 1}\n')
    with pytest.raises(RuntimeError):
        runstate.check_run_manifest(d2, m, overwrite_mode=False)                        # rows but no manifest refuses


def test_worst_region_uses_max_for_error_metrics():
    rows = []

    def add(metric, val, seed, holdout):
        rows.append({"model": "m", "benchmark": "b", "method": "erm", "probe_family": "logistic",
                     "split_regime": "geographic_ood", "budget_type": "target", "label_budget": 0.0,
                     "seed": seed, "holdout": holdout, metric: val,
                     "test_pos_rate": 0.5, "test_n_classes": 2, "test_majority_rate": 0.5})
    for s in range(2):
        rows.append({"model": "m", "benchmark": "b", "method": "erm", "probe_family": "logistic",
                     "split_regime": "random_id", "budget_type": "source", "label_budget": 1.0,
                     "seed": s, "holdout": "random_id", "brier": 0.1,
                     "test_pos_rate": 0.5, "test_n_classes": 2, "test_majority_rate": 0.5})
        add("brier", 0.2, s, "regA")     # better (lower)
        add("brier", 0.4, s, "regB")     # WORST region for an error metric = the MAX
    out = [r for r in IOU.compute_deltas(rows, ["brier"], n_boot=0, seed=0) if r["metric"] == "brier"][0]
    assert abs(out["ood_worst_region"] - 0.4) < 1e-9   # max, not min (0.2)


def _scoped_delta_rows():
    """geographic, each with BOTH a full-target (val 0.6) and a held_out (val 0.1)
    budget-0 row per seed, plus a held_out oracle (budget -1, val 0.5)."""
    rows = []

    def add(regime, lb, es, val, seed, holdout):
        rows.append({"model": "m", "benchmark": "b", "method": "erm", "probe_family": "logistic",
                     "split_regime": regime, "budget_type": "target", "label_budget": lb,
                     "evaluation_split": es, "seed": seed, "holdout": holdout, "auc": val,
                     "test_pos_rate": 0.5, "test_n_classes": 2, "test_majority_rate": 0.5})
    for s in range(2):
        rows.append({"model": "m", "benchmark": "b", "method": "erm", "probe_family": "logistic",
                     "split_regime": "random_id", "budget_type": "source", "label_budget": 1.0,
                     "seed": s, "holdout": "random_id", "auc": 0.9,
                     "test_pos_rate": 0.5, "test_n_classes": 2, "test_majority_rate": 0.5})
        for reg in ("regA", "regB"):
            add("geographic_ood", 0.0, "full", 0.6, s, reg)        # deployment scope (full target)
            add("geographic_ood", 0.0, "held_out", 0.1, s, reg)    # noisy 20% (decomposition only)
            add("geographic_ood", -1.0, "held_out", 0.5, s, reg)   # oracle on the held-out 20%
    return rows


def test_compute_deltas_separates_full_and_held_out_scopes():
    out = [r for r in IOU.compute_deltas(_scoped_delta_rows(), ["auc"], n_boot=0, seed=0)
           if r["metric"] == "auc"][0]
    assert abs(out["ood"] - 0.6) < 1e-9              # primary OOD = full-target, NOT mixed with 0.1
    assert abs(out["ood_worst_region"] - 0.6) < 1e-9  # worst region from full scope (not 0.1)
    assert abs(out["ood_matched"] - 0.1) < 1e-9
    assert abs(out["adjusted_delta"] - (0.5 - 0.1)) < 1e-9


def test_compute_deltas_accepts_dense_test_evaluation_split():
    rows = [
        {"model": "m", "benchmark": "pastis", "method": "erm", "probe_family": "logistic",
         "split_regime": "random_id", "budget_type": "source", "label_budget": 1.0,
         "evaluation_split": "validation", "seed": 0, "holdout": "fold_5", "miou": 0.1},
        {"model": "m", "benchmark": "pastis", "method": "erm", "probe_family": "logistic",
         "split_regime": "random_id", "budget_type": "source", "label_budget": 1.0,
         "evaluation_split": "test", "seed": 0, "holdout": "fold_5", "miou": 0.8},
        {"model": "m", "benchmark": "pastis", "method": "erm", "probe_family": "logistic",
         "split_regime": "geographic_ood", "budget_type": "target", "label_budget": 0.0,
         "evaluation_split": "test", "seed": 0, "holdout": "fold_1", "miou": 0.5},
        {"model": "m", "benchmark": "pastis", "method": "erm", "probe_family": "logistic",
         "split_regime": "geographic_ood", "budget_type": "target", "label_budget": -1.0,
         "evaluation_split": "test", "seed": 0, "holdout": "fold_1", "miou": 0.7},
    ]
    out = [r for r in IOU.compute_deltas(rows, ["miou"], n_boot=0, seed=0) if r["metric"] == "miou"][0]
    assert abs(out["id"] - 0.8) < 1e-9
    assert abs(out["ood"] - 0.5) < 1e-9
    assert abs(out["target_id"] - 0.7) < 1e-9
    assert abs(out["ood_matched"] - 0.5) < 1e-9


def test_summarize_rows_keeps_scopes_separate():
    rows = [
        {"model": "m", "evaluation_split": "full", "miou": 0.6},
        {"model": "m", "evaluation_split": "held_out", "miou": 0.1},
    ]
    summ = IOU.summarize_rows(rows, keys=["model", "evaluation_split"], metrics=["miou"])
    by_es = {r["evaluation_split"]: r for r in summ}
    assert abs(by_es["full"]["mean_miou"] - 0.6) < 1e-9 and abs(by_es["held_out"]["mean_miou"] - 0.1) < 1e-9


def test_dense_expected_rels_full_grid(tmp_path):
    """Every subtile of a non-void patch is expected (feature + matching labels)."""
    from types import SimpleNamespace

    from utils import cacheutils as C

    patch = SimpleNamespace(patch_id=7, fold=1, target_path=None)  # no target -> nothing is void
    bench = SimpleNamespace(patches=(patch,), tile_size=64, ignore_index=255)  # 2x2 grid
    feat, lab = C._dense_expected_rels(bench)
    assert feat == ["fold_1/7_0_0.npy", "fold_1/7_0_1.npy", "fold_1/7_1_0.npy", "fold_1/7_1_1.npy"]
    assert lab == [f"fold_1/7_{r}_{c}.labels.npy" for r in range(2) for c in range(2)]


def test_dense_expected_rels_skips_void_tiles(tmp_path):
    """A subtile whose labels are all ignore_index is skipped exactly as extraction skips it."""
    from types import SimpleNamespace

    from utils import cacheutils as C

    target = np.zeros((1, 128, 128), dtype=np.uint8)
    target[0, 64:, 64:] = 19  # only the bottom-right 64x64 tile is all-void
    target_path = tmp_path / "target.npy"
    np.save(target_path, target)
    patch = SimpleNamespace(patch_id=7, fold=1, target_path=target_path)
    bench = SimpleNamespace(patches=(patch,), tile_size=64, ignore_index=19)
    feat, _lab = C._dense_expected_rels(bench)
    assert feat == ["fold_1/7_0_0.npy", "fold_1/7_0_1.npy", "fold_1/7_1_0.npy"]  # 7_1_1 dropped


def test_dense_loaders_return_cached_features_unaltered(tmp_path):
    """Both loaders hand back the frozen embedding as cached -- same width, no extra columns."""
    fold = tmp_path / "fold_1"
    fold.mkdir()
    np.save(fold / "7_0_0.npy", np.ones((2, 3), dtype=np.float32))
    np.save(fold / "7_0_0.labels.npy", np.array([1, 2], dtype=np.uint8))

    x, y, groups, _tiles, patch_ids = C.load_dense_samples(tmp_path, {1}, 10, 0)
    assert x.shape == (2, 3)
    np.testing.assert_array_equal(y, np.array([1, 2]))
    np.testing.assert_array_equal(groups, np.array([1, 1]))
    np.testing.assert_array_equal(patch_ids, np.array([7, 7]))

    tiles = list(C.iter_dense_tiles(tmp_path, {1}))
    assert len(tiles) == 1
    np.testing.assert_allclose(tiles[0][0], x)


def test_run_manifest_rejects_corrupt(tmp_path, monkeypatch):
    from utils import runstate
    d = tmp_path / "r"
    d.mkdir()
    (d / "run_manifest.json").write_text("{not json")   # crashed/corrupt publish
    with pytest.raises(RuntimeError):
        runstate.check_run_manifest(d, {"schema": 1}, overwrite_mode=False)


# A dropped leave-one-domain-out holdout being surfaced (never silent) is now a schema-v2 concern
# covered by test_split_parity.test_generation_records_dropped_holdout (an absent geographic target is
# recorded in dropped_holdouts) and test_phase_b_refuses_zero_yield_requested_regime (a requested
# regime that yields zero leaves is refused at consumption). The v1 base.iter_splits LODO census check
# it used to pin is removed.


def test_run_manifest_includes_seeds_and_enc_kwargs():
    from utils import runstate

    def sig(seeds, enc):
        m = runstate.build_run_manifest(
            "raw", "cropharvest", "baseline", "emb", ["random_id"], seeds, enc,
            active_probes=["logistic"], budget_regimes={"source": [1.0], "target": [0, 0.1]},
            max_dense_pixels=50_000, write_predictions=True,
        )
        return runstate.run_manifest_digest(m)

    base = sig([0, 1, 2], {"device": "cpu"})
    assert sig([0, 1], {"device": "cpu"}) != base                            # seed set is result-defining
    assert sig([0, 1, 2], {"device": "cpu", "weights_path": "/x"}) != base   # enc kwargs are recorded
    from evals import probes
    prev = probes.PROBE_TUNING
    try:
        probes.PROBE_TUNING = True
        assert sig([0, 1, 2], {"device": "cpu"}) != base                     # probe tuning moves the manifest
    finally:
        probes.PROBE_TUNING = prev
    assert sig([0, 1, 2], {"device": "cuda"}) == base                        # device is not result-defining


def test_prune_partial_budgets_removes_surviving_scope(tmp_path):
    from utils import runstate
    rows_path = tmp_path / "probe_results.jsonl"
    preds_path = tmp_path / "predictions.jsonl"

    def row(lb, es):
        return {"seed": 0, "split_regime": "geographic_ood", "holdout": "t", "method": "erm",
                "probe_family": "logistic", "budget_type": "target", "label_budget": lb,
                "evaluation_split": es, "auc": 0.5}
    rows = [row(0, "held_out"), row(5, "held_out")]
    IOU.append_jsonl(rows_path, rows)
    IOU.append_jsonl(preds_path, [{**row(0, "held_out"), "sample_id": 1}])
    base = (0, "geographic_ood", "t", "erm", "logistic")
    # budget_row_key is 9-field (label_access_route last); these non-label-access rows key with "".
    rerun = {(*base, "target", 0, "held_out", ""), (*base, "target", 0, "full", "")}                # budget 0 regenerated (both scopes)
    kept = runstate.prune_partial_budgets(rows, rows_path, preds_path, rerun)
    assert [r["label_budget"] for r in kept] == [5]      # the partial budget-0 row was pruned
    assert [r["label_budget"] for r in IOU.read_jsonl(rows_path)] == [5]   # jsonl rewritten
    assert IOU.read_jsonl(preds_path) == []              # its prediction pruned too


def test_custom_weights_path_changes_checkpoint_identity(tmp_path):
    wa, wb = tmp_path / "a.pth", tmp_path / "b.pth"   # tiny overrides (avoid reading real weights)
    wa.write_bytes(b"AAAA")
    wb.write_bytes(b"BBBB")
    C._CHECKPOINT_SHA_CACHE.clear()
    ka = C.checkpoint_sha256("agrifm", weights_override=str(wa))
    kb = C.checkpoint_sha256("agrifm", weights_override=str(wb))
    assert ka != kb and len(ka) == 64                 # override checkpoint content is the identity
    # The on-disk cache path is fixed regardless of checkpoint (identity lives in the manifest).
    assert C.embedding_cache_path("cropharvest", "agrifm", "baseline").name == "baseline.npy"


def test_run_manifest_digest_is_deterministic():
    from utils import runstate

    def sig():
        m = runstate.build_run_manifest(
            "raw", "cropharvest", "baseline", "emb", ["random_id"], [0], {"device": "cpu"},
            active_probes=["logistic"], budget_regimes={"source": [1.0], "target": [0, 0.1]},
            max_dense_pixels=50_000, write_predictions=True,
        )
        return runstate.run_manifest_digest(m)

    s = sig()
    assert s and s == sig()


def test_checkpoint_sha_reflects_override_content_not_mtime(tmp_path):
    wp = tmp_path / "agrifm.pth"
    wp.write_bytes(b"weights-v1" + b"\0" * 64)
    C._CHECKPOINT_SHA_CACHE.clear()
    sig1 = C.checkpoint_sha256("agrifm", weights_override=str(wp))
    mtime = wp.stat().st_mtime
    wp.write_bytes(b"weights-v2" + b"\0" * 64)
    os.utime(wp, (mtime, mtime))  # SAME mtime -> a content hash must still move
    C._CHECKPOINT_SHA_CACHE.clear()
    assert C.checkpoint_sha256("agrifm", weights_override=str(wp)) != sig1 and len(sig1) == 64


def test_supplementary_stress_targets_excluded_from_headline_geographic_aggregation():
    """Defect 3: CropHarvest one-class supplementary stress targets stay VISIBLE as source-only stress
    rows, but never enter the headline geographic equal-region mean or worst-region F1. Marked by the
    machine-readable target_role (supports_target_labels=False alone is insufficient -- official shares
    that capability), so a stress region with the worst F1 must NOT dominate worst-region."""
    from evals import confounds

    base = {"model": "raw", "benchmark": "cropharvest", "method": "erm", "probe_family": "logistic"}
    rows = [
        # ID baseline (random_id source @ 1.0, eval test)
        {**base, "split_regime": "random_id", "budget_type": "source", "label_budget": 1.0,
         "evaluation_split": "test", "seed": 0, "f1": 0.9, "target_role": "headline"},
        # headline OOD target regions (zero-shot, eval full)
        {**base, "split_regime": "geographic_ood", "budget_type": "target", "label_budget": 0,
         "evaluation_split": "full", "holdout": "kenya", "seed": 0, "f1": 0.8, "target_role": "headline"},
        {**base, "split_regime": "geographic_ood", "budget_type": "target", "label_budget": 0,
         "evaluation_split": "full", "holdout": "togo", "seed": 0, "f1": 0.7, "target_role": "headline"},
        # a supplementary STRESS region with the WORST F1 -- would dominate worst-region if it leaked in
        {**base, "split_regime": "geographic_ood", "budget_type": "target", "label_budget": 0,
         "evaluation_split": "full", "holdout": "central-asia", "seed": 0, "f1": 0.1,
         "target_role": "supplementary_stress"},
    ]

    deltas = confounds.compute_deltas(rows, ["f1"], n_boot=0)
    d = next(r for r in deltas if r["metric"] == "f1")
    # equal-region OOD mean over HEADLINE regions only (0.8, 0.7) -> 0.75; central-asia's 0.1 excluded
    assert d["ood"] == pytest.approx(0.75)
    assert d["ood_min"] == pytest.approx(0.7)              # togo, NOT central-asia's 0.1
    assert d["ood_worst_region"] == pytest.approx(0.7)     # worst headline region, stress excluded
    # the stress region is still present (visible source-only stress evidence)
    assert any(r.get("holdout") == "central-asia" for r in rows)
