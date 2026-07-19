"""Stage 4 -- the dense (PASTIS) geographic label-access suite, at PATCH granularity.

The 13-route contract is the same as tabular, but every selection/removal is over WHOLE patches and the
unit is ``patches``. These tests pin the patch-level runtime with recording fakes for the fit + score
(so the assertions are exact and encoder-free) AND drive the real capped cacheutils loader for the
patch-first / paired-pixel guarantees: patch atomicity, frozen-order mapping, target_test non-leakage,
calibration routing, the shared base-pool cap, run-seed probe init, deterministic paired pixel sampling,
streamed predictions, resume/completeness identity, the dense loader round-trip, and the manifest unit +
prediction-honesty contract. Patch ids live in feature 0."""

from __future__ import annotations

import numpy as np
import pytest

from evals import split_artifacts as SA
from evals.benchmarks import pastis
from utils import artifacts, cacheutils, runstate

COUNTS = (5, 10, 25, 50)
ROUTE_ORDER = [
    SA.ROUTE_SOURCE_ONLY, *[SA.ROUTE_SOURCE_PLUS_TARGET] * 4, SA.ROUTE_TARGET_ONLY_FULL,
    SA.ROUTE_SOURCE_PLUS_TARGET_FULL, SA.ROUTE_MATCHED_SOURCE, SA.ROUTE_MATCHED_TARGET,
    *[SA.ROUTE_FIXED_TOTAL_MIXED] * 4,
]
INTERNAL_CALLS = {5, 7, 8}   # target_only_full, matched_source, matched_target tune internally
SRC_PX, TGT_PX = 4, 3        # per-patch pixel counts (source vs target) -- used for atomicity
POOL_BASE, TEST_BASE = 1000, 2000
RUN_SEED = 7


def _block(patch_id, n_px, rng, n_features=4):
    y = np.array([i % 3 for i in range(n_px)], dtype=np.int64)
    x = rng.normal(size=(n_px, n_features)).astype(np.float32)
    x[:, 0] = patch_id                      # feature 0 == patch id, so a fake can recover membership
    return x, y


class _FakeClf:
    def predict(self, X):
        return np.zeros(len(X), dtype=np.int64)


def _make_dense(S=60, P=55, T=8):
    rng = np.random.default_rng(0)
    source_ids = list(range(S))
    pool_ids = list(range(POOL_BASE, POOL_BASE + P))
    test_ids = list(range(TEST_BASE, TEST_BASE + T))
    store = {p: _block(p, SRC_PX, rng) for p in source_ids}
    store.update({p: _block(p, TGT_PX, rng) for p in pool_ids + test_ids})

    def load_pixels(patch_ids):
        pids = sorted(int(p) for p in patch_ids)
        if not pids:
            return np.zeros((0, 4), np.float32), np.zeros(0, np.int64), np.zeros(0, np.int64), np.zeros(0, np.int64)
        x = np.concatenate([store[p][0] for p in pids])
        y = np.concatenate([store[p][1] for p in pids])
        pid = np.concatenate([np.full(len(store[p][1]), p, np.int64) for p in pids])
        return x, y, np.zeros(len(y), np.int64), pid

    def stream_eval(patch_ids):
        # one synthetic tile per patch: tile_key = (patch_id, row, col) with row=col=0
        for p in sorted(int(q) for q in patch_ids):
            x, y = store[p]
            yield (p, 0, 0), x, y

    order_rng = np.random.default_rng(1)
    return {
        "source_patches": frozenset(source_ids), "pool_patches": frozenset(pool_ids),
        "test_patches": frozenset(test_ids),
        "matched": order_rng.permutation(source_ids).astype(np.int64),
        "fixed": order_rng.permutation(source_ids).astype(np.int64),
        "target": order_rng.permutation(pool_ids).astype(np.int64),
        "x_val": store[0][0], "y_val": store[0][1],
        "load_pixels": load_pixels, "stream_eval": stream_eval,
        "pool_ids": pool_ids, "test_ids": test_ids, "source_ids": source_ids,
    }


class _RecFit:
    """Recording fake fit_probe_multiclass: captures the patch-id -> pixel-count map, the probe seed, and
    whether the route tuned internally, then returns a predicting dummy clf."""

    def __init__(self):
        self.calls = []

    def __call__(self, x_tr, y_tr, seed, x_val=None, y_val=None, family="logistic", tune_internal=False):
        pids = np.asarray(x_tr)[:, 0].astype(np.int64)
        uniq, counts = np.unique(pids, return_counts=True)
        self.calls.append({
            "patch_px": {int(p): int(c) for p, c in zip(uniq, counts, strict=True)},
            "seed": int(seed), "tune_internal": bool(tune_internal), "n_train": int(len(y_tr)),
            "has_cal": x_val is not None and len(x_val) > 0,
        })
        return _FakeClf(), {"probe_converged": 1}


class _RecScore:
    """Recording fake score_segmentation_streamed: mirrors the real one-pass contract -- unpacks 3-tuples
    (tile_key, features, labels) when predict_sink is set (calling it once per tile from the SAME
    inference) and 2-tuples otherwise -- capturing streamed patches + the pixel-level n_test."""

    def __init__(self):
        self.calls = []

    def __call__(self, clf, tiles, eval_classes, *, predict_sink=None):
        pids, n_px = [], 0
        for item in tiles:
            if predict_sink is not None:
                tile_key, feats, labels = item
            else:
                feats, labels = item
                tile_key = None
            feats, labels = np.asarray(feats), np.asarray(labels)
            pids.extend(np.unique(feats[:, 0].astype(np.int64)).tolist())
            n_px += int(len(labels))
            if predict_sink is not None:
                predict_sink(tile_key, labels, clf.predict(feats))
        self.calls.append({"patches": sorted(set(pids)), "n_pixels": n_px})
        return {"miou": 0.5, "n_test": n_px}


def _run(monkeypatch, *, cap_patches=None, counts=COUNTS, meta_extra=None, predictions_sink=None):
    data = _make_dense()
    fit, score = _RecFit(), _RecScore()
    monkeypatch.setattr(pastis, "fit_probe_multiclass", fit)
    monkeypatch.setattr(pastis, "score_segmentation_streamed", score)
    meta = {"model": "raw", "benchmark": "pastis", "method": "erm", "probe_family": "logistic",
            "split_regime": SA.LABEL_ACCESS_REGIME, "holdout": "T1", "budget_type": "label_access"}
    meta.update(meta_extra or {})
    rows: list[dict] = []
    pastis.run_probes_segmentation_label_access(
        rows, RUN_SEED,
        source_patches=data["source_patches"], pool_patches=data["pool_patches"],
        target_test_patches=data["test_patches"], matched_source_order=data["matched"],
        fixed_removal_order=data["fixed"], target_order=data["target"], counts=counts,
        load_pixels=data["load_pixels"], stream_eval=data["stream_eval"],
        x_val=data["x_val"], y_val=data["y_val"], meta=meta, family="logistic",
        cap_patches=cap_patches, predictions_sink=predictions_sink,
    )
    return rows, fit, score, data


def _src(call):
    return {p for p in call["patch_px"] if p < POOL_BASE}


def _tgt(call):
    return {p for p in call["patch_px"] if p >= POOL_BASE}


# --------------------------------------------------------------------------- #
# structure + patch atomicity
# --------------------------------------------------------------------------- #
def test_thirteen_patch_fits_and_source_only_reuse(monkeypatch):
    rows, fit, _score, _ = _run(monkeypatch)
    assert len(fit.calls) == 13
    tt = [r for r in rows if r["evaluation_split"] == SA.EVAL_TARGET_TEST]
    diag = [r for r in rows if r["evaluation_split"] == SA.EVAL_COMPLETE_TARGET]
    assert len(tt) == 13 and len(diag) == 1
    assert {(r["label_access_route"], r["label_budget"], r["evaluation_split"]) for r in rows} == set(
        SA.label_access_expected_rows()
    )
    assert {r["label_budget_unit"] for r in rows} == {"patches"}


def test_patch_atomicity_every_selected_patch_is_whole(monkeypatch):
    _rows, fit, _score, _ = _run(monkeypatch)
    for call in fit.calls:
        for pid, px in call["patch_px"].items():
            assert px == (SRC_PX if pid < POOL_BASE else TGT_PX), f"patch {pid} partially included"


# --------------------------------------------------------------------------- #
# frozen-order mapping + independent orders
# --------------------------------------------------------------------------- #
def test_frozen_order_mapping_and_counts(monkeypatch):
    _rows, fit, _score, data = _run(monkeypatch)
    src_all = set(data["source_ids"])
    matched, fixed, target = data["matched"].tolist(), data["fixed"].tolist(), data["target"].tolist()
    m = min(len(src_all), len(target))
    calls = fit.calls
    assert _src(calls[0]) == src_all and _tgt(calls[0]) == set()
    for idx, k in zip(range(1, 5), COUNTS, strict=True):
        assert _src(calls[idx]) == src_all and _tgt(calls[idx]) == set(target[:k])
    assert _src(calls[5]) == set() and _tgt(calls[5]) == set(target)
    assert _src(calls[6]) == src_all and _tgt(calls[6]) == set(target)
    assert _src(calls[7]) == set(matched[:m]) and _tgt(calls[7]) == set()
    assert _src(calls[8]) == set() and _tgt(calls[8]) == set(target[:m])
    for idx, k in zip(range(9, 13), COUNTS, strict=True):
        assert _src(calls[idx]) == set(fixed[k:]) and _tgt(calls[idx]) == set(target[:k])


def test_matched_and_fixed_consume_separate_orders(monkeypatch):
    _rows, fit, _score, data = _run(monkeypatch)
    assert data["matched"].tolist() != data["fixed"].tolist()
    m = min(len(data["source_ids"]), len(data["target"]))
    assert _src(fit.calls[7]) == set(data["matched"].tolist()[:m])
    assert _src(fit.calls[9]) == set(data["fixed"].tolist()[COUNTS[0]:])


# --------------------------------------------------------------------------- #
# leakage + calibration routing + seeds (Issue 2)
# --------------------------------------------------------------------------- #
def test_no_target_test_patch_enters_any_training_set(monkeypatch):
    _rows, fit, _score, data = _run(monkeypatch)
    test_set = set(data["test_ids"])
    for call in fit.calls:
        assert not (set(call["patch_px"]) & test_set)


def test_calibration_routing_matches_the_contract(monkeypatch):
    _rows, fit, _score, _ = _run(monkeypatch)
    for i, call in enumerate(fit.calls):
        assert call["tune_internal"] == (i in INTERNAL_CALLS)
        assert call["has_cal"] == (i not in INTERNAL_CALLS)


def test_every_route_initializes_the_probe_with_the_run_seed(monkeypatch):
    """No per-budget seed: every one of the 13 routes seeds its probe with the pipeline RUN seed, so
    changing k never injects an unrelated random draw at probe-init time."""
    _rows, fit, _score, _ = _run(monkeypatch)
    assert len(fit.calls) == 13
    assert {c["seed"] for c in fit.calls} == {RUN_SEED}


def test_sweep_never_derives_a_budget_seed(monkeypatch):
    """The sweep must not call perf._budget_seed at all -- that was the source of the k-dependent draw."""
    seen = []
    orig = pastis.perf._budget_seed
    monkeypatch.setattr(pastis.perf, "_budget_seed", lambda *a, **k: seen.append(a) or orig(*a, **k))
    _run(monkeypatch)
    assert seen == []


# --------------------------------------------------------------------------- #
# scoring targets + n_test/n_eval_patches (Issue 5)
# --------------------------------------------------------------------------- #
def test_scored_targets_and_pixel_vs_patch_counts(monkeypatch):
    rows, _fit, score, data = _run(monkeypatch)
    test_set = set(data["test_ids"])
    complete = set(data["pool_ids"]) | test_set
    assert sum(set(c["patches"]) == test_set for c in score.calls) == 13   # 13 routes on target_test
    assert sum(set(c["patches"]) == complete for c in score.calls) == 1    # 1 diagnostic on the region
    tt = [r for r in rows if r["evaluation_split"] == SA.EVAL_TARGET_TEST]
    diag = next(r for r in rows if r["evaluation_split"] == SA.EVAL_COMPLETE_TARGET)
    # n_test stays the evaluated PIXEL count (like every seg row); n_eval_patches is the patch count.
    assert {r["n_test"] for r in tt} == {len(test_set) * TGT_PX}
    assert {r["n_eval_patches"] for r in tt} == {len(test_set)}
    assert diag["n_test"] == len(complete) * TGT_PX and diag["n_eval_patches"] == len(complete)


def test_semantic_validation_uses_patch_counts_and_passes(monkeypatch):
    """The generalized validator derives the realized pool P from n_eval_patches (patches), not n_test
    (pixels), so the dense suite validates cleanly."""
    rows, _fit, _score, _ = _run(monkeypatch)
    full = [{**r, "seed": RUN_SEED, "split_regime": SA.LABEL_ACCESS_REGIME, "holdout": "T1",
             "method": "erm", "probe_family": "logistic"} for r in rows]
    assert artifacts._validate_label_access_semantics(full) == []


# --------------------------------------------------------------------------- #
# predictions: streamed with full stable identity (Issue 4)
# --------------------------------------------------------------------------- #
def test_predictions_stream_with_full_stable_identity(monkeypatch):
    recs: list[dict] = []
    rows, _fit, _score, data = _run(monkeypatch, predictions_sink=recs.extend)
    assert recs
    required = {"patch_id", "tile_row", "tile_col", "pixel_index", "sample_id", "label_access_route",
                "evaluation_split", "label_budget", "seed", "budget_type", "n_source_labels",
                "n_target_labels", "n_total_labels", "label_budget_unit"}
    for r in recs:
        assert required <= set(r)
        # sample_id carries patch, tile row, tile col, and the within-tile valid-pixel index
        assert r["sample_id"] == f'{r["patch_id"]}:{r["tile_row"]}:{r["tile_col"]}:{r["pixel_index"]}'
        assert r["budget_type"] == "label_access" and r["label_budget_unit"] == "patches"
        assert r["seed"] == RUN_SEED
    # target_test predictions cover exactly the frozen target_test patches; the diagnostic adds the pool
    tt_patches = {r["patch_id"] for r in recs if r["evaluation_split"] == SA.EVAL_TARGET_TEST}
    assert tt_patches == set(data["test_ids"])
    diag_patches = {r["patch_id"] for r in recs if r["evaluation_split"] == SA.EVAL_COMPLETE_TARGET}
    assert diag_patches == set(data["pool_ids"]) | set(data["test_ids"])
    assert {r["label_access_route"] for r in recs if r["evaluation_split"] == SA.EVAL_COMPLETE_TARGET} == {
        SA.ROUTE_SOURCE_ONLY
    }
    # UNIQUENESS within each fit (route + budget + evaluation_split): no sample_id repeats. (The same
    # route name spans budget arms that score the same patches, so budget is part of the fit identity.)
    by_group: dict[tuple, list[str]] = {}
    for r in recs:
        by_group.setdefault((r["label_access_route"], r["label_budget"], r["evaluation_split"]), []).append(r["sample_id"])
    for key, ids in by_group.items():
        assert len(ids) == len(set(ids)), f"duplicate sample_id within {key}"


def test_no_sink_writes_no_predictions(monkeypatch):
    rows, _fit, _score, _ = _run(monkeypatch, predictions_sink=None)
    assert rows  # rows still produced; predictions simply not emitted


# --------------------------------------------------------------------------- #
# cap: the shared source base pool, at patch granularity
# --------------------------------------------------------------------------- #
def test_uncapped_base_is_the_whole_source(monkeypatch):
    rows, fit, _score, data = _run(monkeypatch, cap_patches=None)
    assert _src(fit.calls[0]) == set(data["source_ids"])
    assert {r["n_source_base"] for r in rows} == {len(data["source_ids"])}
    assert {r["probe_capped"] for r in rows} == {0}


def test_cap_restricts_the_shared_base_and_both_orders(monkeypatch):
    rows, fit, _score, data = _run(monkeypatch, cap_patches=52)
    base = _src(fit.calls[0])
    assert len(base) == 52 and base <= set(data["source_ids"])
    assert _src(fit.calls[6]) == base
    assert _src(fit.calls[7]) <= base
    assert _src(fit.calls[9]) <= base
    assert {r["n_source_base"] for r in rows} == {52} and {r["probe_capped"] for r in rows} == {1}


def test_cap_is_deterministic_and_encoder_independent(monkeypatch):
    _a, fita, _sa, _da = _run(monkeypatch, cap_patches=52)
    _b, fitb, _sb, _db = _run(monkeypatch, cap_patches=52, meta_extra={"model": "different-encoder"})
    assert _src(fita.calls[0]) == _src(fitb.calls[0])


def test_cap_below_max_count_hard_fails(monkeypatch):
    with pytest.raises(ValueError, match="infeasible"):
        _run(monkeypatch, cap_patches=40)


def test_order_rejects_non_permutation_of_source(monkeypatch):
    data = _make_dense()
    monkeypatch.setattr(pastis, "fit_probe_multiclass", _RecFit())
    monkeypatch.setattr(pastis, "score_segmentation_streamed", _RecScore())
    bad = data["matched"].copy()
    bad[0] = 999999
    with pytest.raises(ValueError, match="matched_source_order"):
        pastis.run_probes_segmentation_label_access(
            [], RUN_SEED, source_patches=data["source_patches"], pool_patches=data["pool_patches"],
            target_test_patches=data["test_patches"], matched_source_order=bad,
            fixed_removal_order=data["fixed"], target_order=data["target"], counts=COUNTS,
            load_pixels=data["load_pixels"], stream_eval=data["stream_eval"],
            x_val=data["x_val"], y_val=data["y_val"], meta={"benchmark": "pastis"},
        )


# --------------------------------------------------------------------------- #
# real capped cacheutils loader: patch-first, paired, MAX_DENSE_PIXELS respected (Issue 3)
# --------------------------------------------------------------------------- #
def _write_patches(fold_dir, patch_ids, n_px, *, tiles=((0, 0),), n_feat=6, seed=0):
    """Write `n_px`-pixel cache tiles for each patch. `tiles` is the list of (row, col) quadrants -- the
    real PASTIS cache stores up to four `{patch}_{row}_{col}` files per patch."""
    fold_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    for p in patch_ids:
        for r, c in tiles:
            x = rng.normal(size=(n_px, n_feat)).astype(np.float32)
            y = (np.arange(n_px) % 3).astype(np.int64)
            np.save(fold_dir / f"{p}_{r}_{c}.npy", x)
            np.save(fold_dir / f"{p}_{r}_{c}.labels.npy", y)


def test_real_loader_is_patch_paired_deterministic_and_bounded(tmp_path):
    emb = tmp_path / "emb"
    _write_patches(emb / "fold_1", [10, 11, 12, 13], n_px=100)
    folds, cap = {1}, 20
    xA, _yA, _gA, pA = cacheutils.load_dense_patch_pixels(emb, folds, {10, 11}, run_seed=7, per_patch_cap=cap)
    xB, yB, _gB, pB = cacheutils.load_dense_patch_pixels(emb, folds, {10, 11, 12, 13}, run_seed=7, per_patch_cap=cap)
    for p in (10, 11):
        assert int((pA == p).sum()) == cap and int((pB == p).sum()) == cap
    assert np.array_equal(xA[pA == 10], xB[pB == 10])
    assert np.array_equal(xA[pA == 11], xB[pB == 11])
    assert len(yB) == 4 * cap
    xC, *_ = cacheutils.load_dense_patch_pixels(emb, folds, {10}, run_seed=99, per_patch_cap=cap)
    assert not np.array_equal(xA[pA == 10], xC)


def test_real_loader_caps_per_patch_across_all_four_tiles(tmp_path):
    """A PASTIS patch is up to four {patch}_{r}_{c} files. The cap is a per-PATCH budget across ALL its
    tiles -- a 4-tile patch must contribute `cap` pixels total, never 4 * cap."""
    emb = tmp_path / "emb"
    _write_patches(emb / "fold_1", [10, 11], n_px=30, tiles=[(0, 0), (0, 1), (1, 0), (1, 1)])  # 120 px/patch
    folds, cap = {1}, 50
    x, y, _g, pid = cacheutils.load_dense_patch_pixels(emb, folds, {10, 11}, run_seed=7, per_patch_cap=cap)
    for p in (10, 11):
        assert int((pid == p).sum()) == cap                 # across all four tiles, exactly `cap`
    assert len(y) == 2 * cap                                 # largest load bounded at n_patches * cap
    # PAIRING across the combined tiles: patch 10's sample is identical whether or not 11 is co-loaded
    xA, _yA, _gA, pA = cacheutils.load_dense_patch_pixels(emb, folds, {10}, run_seed=7, per_patch_cap=cap)
    assert np.array_equal(x[pid == 10], xA[pA == 10])


def test_real_loader_returns_all_pixels_when_under_cap(tmp_path):
    emb = tmp_path / "emb"
    _write_patches(emb / "fold_1", [10, 11], n_px=4, tiles=[(0, 0), (0, 1)])  # 8 combined px/patch
    x, y, _g, pid = cacheutils.load_dense_patch_pixels(emb, {1}, {10, 11}, run_seed=7, per_patch_cap=50)
    assert len(y) == 16 and set(pid.tolist()) == {10, 11}   # no subsampling when the patch fits under cap


def test_real_loader_hard_fails_on_missing_patch(tmp_path):
    emb = tmp_path / "emb"
    _write_patches(emb / "fold_1", [10], n_px=8)
    with pytest.raises(FileNotFoundError, match="patch 99 has no usable cached pixels"):
        cacheutils.load_dense_patch_pixels(emb, {1}, {10, 99}, run_seed=7, per_patch_cap=50)


# --------------------------------------------------------------------------- #
# resume / completeness identity (route-aware) + dense loader round-trip
# --------------------------------------------------------------------------- #
def test_rows_carry_route_aware_identity_and_pass_completeness(monkeypatch):
    rows, _fit, _score, _ = _run(monkeypatch)
    base_key = (RUN_SEED, SA.LABEL_ACCESS_REGIME, "T1", "erm", "logistic")
    expected = {(*base_key, "label_access", b, es, route) for (route, b, es) in SA.label_access_expected_rows()}
    assert {artifacts.cell_key(r) for r in rows} == expected
    for r in rows:
        assert artifacts.cell_key(r) == runstate.budget_row_key(r)
    assert artifacts.completeness(expected, rows)["ok"]
    spt = next(r for r in rows if r["label_access_route"] == SA.ROUTE_SOURCE_PLUS_TARGET and r["label_budget"] == 25)
    ftm = next(r for r in rows if r["label_access_route"] == SA.ROUTE_FIXED_TOTAL_MIXED and r["label_budget"] == 25)
    assert runstate.budget_row_key(spt) != runstate.budget_row_key(ftm)


def _dense_split(source, pool, test, label="T1"):
    from evals.regimes.base import DenseSourceTargetSplit

    return DenseSourceTargetSplit(
        label=label, source_train_patches=frozenset(source),
        source_val_patches=frozenset(), source_test_patches=frozenset(),
        target_label_pool_patches=frozenset(pool), target_test_patches=frozenset(test),
        has_target=True, supports_target_labels=True,
    )


def test_load_dense_label_access_resolves_patch_orders(tmp_path):
    source, pool, test = list(range(60)), list(range(1000, 1055)), list(range(2000, 2008))
    la_rows = SA.build_label_access_rows(
        seed=0, source_ids=[str(p) for p in source], target_pool_ids=[str(p) for p in pool],
        target_test_ids=[str(p) for p in test],
    )
    SA.write_label_access(tmp_path / "splits", "pastis", 0, "T1", la_rows)
    loaded = SA.load_dense_label_access(tmp_path / "splits", "pastis", 0, _dense_split(source, pool, test))
    assert set(loaded.matched_source_ranked_patches.tolist()) == set(source)
    assert set(loaded.fixed_source_removal_ranked_patches.tolist()) == set(source)
    assert set(loaded.target_ranked_patches.tolist()) == set(pool)
    assert loaded.matched_source_ranked_patches.tolist() != loaded.fixed_source_removal_ranked_patches.tolist()


def test_load_dense_label_access_missing_file_hard_errors(tmp_path):
    with pytest.raises(SA.SplitArtifactError, match="missing label_access.csv"):
        SA.load_dense_label_access(tmp_path / "splits", "pastis", 0, _dense_split(range(60), range(1000, 1055), range(2000, 2008)))


def test_load_dense_label_access_rejects_stale_population(tmp_path):
    source, pool, test = list(range(60)), list(range(1000, 1055)), list(range(2000, 2008))
    la_rows = SA.build_label_access_rows(
        seed=0, source_ids=[str(p) for p in source], target_pool_ids=[str(p) for p in pool],
        target_test_ids=[str(p) for p in test],
    )
    SA.write_label_access(tmp_path / "splits", "pastis", 0, "T1", la_rows)
    with pytest.raises(SA.SplitArtifactError):
        SA.load_dense_label_access(tmp_path / "splits", "pastis", 0, _dense_split(list(range(1, 61)), pool, test))


# --------------------------------------------------------------------------- #
# manifest: patches unit + prediction honesty (Issue 4)
# --------------------------------------------------------------------------- #
def _manifest(benchmark, regimes, write_predictions=True):
    return runstate.build_run_manifest(
        "raw", benchmark, "artifact", "digest", regimes, [0], {},
        active_probes=["logistic"], budget_regimes={"source": [1.0]}, max_dense_pixels=10_000,
        write_predictions=write_predictions,
    )


def test_manifest_contract_unit_is_patches_for_pastis():
    contract = SA.label_access_contract(enabled=True, benchmark="pastis")
    assert contract["unit"] == "patches" and contract["enabled"] is True
    man = _manifest("pastis", ["geographic_ood", "official"])
    assert man["label_access"]["enabled"] is True and man["label_access"]["unit"] == "patches"


def test_manifest_write_predictions_never_over_claims():
    # dense WITHOUT geographic_ood writes no predictions -> must not claim enabled
    assert _manifest("pastis", ["official", "random_id"], write_predictions=True)["write_predictions"] is False
    # dense WITH geographic_ood does write them
    assert _manifest("pastis", ["geographic_ood"], write_predictions=True)["write_predictions"] is True
    # tabular always writes when enabled; disabled stays disabled
    assert _manifest("cropharvest", ["random_id"], write_predictions=True)["write_predictions"] is True
    assert _manifest("pastis", ["geographic_ood"], write_predictions=False)["write_predictions"] is False
