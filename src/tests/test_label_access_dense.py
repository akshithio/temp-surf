"""Stage 4 -- the dense (PASTIS) geographic label-access suite, at PATCH granularity.

The route contract is the same as tabular, but every selection is over WHOLE patches and the unit is
``patches``: the fixed-budget ALLOCATION curve (``source_order[:B-k] + target_order[:k]`` with
``k = round_half_up(f% * B)`` and ``B = min(B_d, controlled_cap)``), the additive appendix routes (the
COMPLETE source pool plus ``target_order[:k]``), and the two full references. There is no ``source_only``
route and no ``complete_target`` evaluation -- both retired.

These tests pin the patch-level runtime with recording fakes for the fit + score (so the assertions are
exact and encoder-free) AND drive the real capped cacheutils loader for the patch-first / paired-pixel
guarantees: patch atomicity, strictly nested prefixes, half-up allocation in patch units, target_test
non-leakage, calibration routing, the controlled budget cap, run-seed probe init, deterministic paired
pixel sampling, streamed predictions, resume/completeness identity, the dense loader round-trip, and the
manifest unit + prediction-honesty contract. Patch ids live in feature 0."""

from __future__ import annotations

import numpy as np
import pytest

from evals import split_artifacts as SA
from evals.benchmarks import pastis
from utils import artifacts, cacheutils, runstate

PERCENTS = SA.ALLOCATION_PERCENTS          # (0, 25, 50, 75, 100)
COUNTS = SA.LABEL_ACCESS_COUNTS            # (5, 10, 25, 50)
#: B_d in PATCH units. 50 is chosen so 25% and 75% land exactly on .5 (12.5 / 37.5) -- the only way to
#: tell half-up rounding apart from Python's banker's rounding.
BUDGET = 50
#: Emission order of route_specs: the five allocation points, the four additive points, then the two
#: full references. The allocation points and target_only_full tune internally (no source calibration).
INTERNAL_CALLS = {0, 1, 2, 3, 4, 9}
SRC_PX, TGT_PX = 4, 3        # per-patch pixel counts (source vs target) -- used for atomicity
POOL_BASE, TEST_BASE = 1000, 2000
RUN_SEED = 7


def _k(f, budget=BUDGET):
    return SA.allocation_target_count(f, budget)


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
        "source": order_rng.permutation(source_ids).astype(np.int64),
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


def _run(monkeypatch, *, budget=BUDGET, controlled_cap=None, percents=PERCENTS, counts=COUNTS,
         meta_extra=None, predictions_sink=None, data=None):
    data = data or _make_dense()
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
        target_test_patches=data["test_patches"], source_order=data["source"],
        target_order=data["target"], budget=budget, percents=percents, counts=counts,
        controlled_cap=controlled_cap,
        load_pixels=data["load_pixels"], stream_eval=data["stream_eval"],
        x_val=data["x_val"], y_val=data["y_val"], meta=meta, family="logistic",
        label_budget_unit=SA.LABEL_ACCESS_DENSE_UNIT, predictions_sink=predictions_sink,
    )
    return rows, fit, score, data


def _src(call):
    return {p for p in call["patch_px"] if p < POOL_BASE}


def _tgt(call):
    return {p for p in call["patch_px"] if p >= POOL_BASE}


def _e1(**over):
    """The ordinary full-source geographic (E1) row the label-access cell reuses instead of refitting a
    source_only probe. The semantic validator requires exactly one per cell."""
    return {"model": "raw", "benchmark": "pastis", "method": "erm", "probe_family": "logistic",
            "split_regime": SA.LABEL_ACCESS_REGIME, "holdout": "T1", "seed": RUN_SEED,
            "budget_type": "source", "label_budget": 1.0, "evaluation_split": "test", **over}


# --------------------------------------------------------------------------- #
# structure + patch atomicity
# --------------------------------------------------------------------------- #
def test_exactly_the_configured_allocation_additive_and_reference_patch_fits(monkeypatch):
    """One fit per planned row and nothing more: 5 allocation points + 4 additive points + the two full
    references. The retired source_only route and the complete_target diagnostic are both gone."""
    rows, fit, _score, _ = _run(monkeypatch)
    expected = SA.label_access_expected_rows()
    assert len(fit.calls) == len(expected) == len(PERCENTS) + len(COUNTS) + 2
    assert len(rows) == len(expected)
    assert {r["evaluation_split"] for r in rows} == {SA.EVAL_TARGET_TEST}
    assert not [r for r in rows if r["evaluation_split"] == SA.EVAL_COMPLETE_TARGET]
    assert {(r["label_access_route"], r["label_budget"], r["evaluation_split"]) for r in rows} == set(expected)
    assert {r["label_budget_unit"] for r in rows} == {SA.LABEL_ACCESS_DENSE_UNIT}
    assert {r["label_access_route"] for r in rows} <= set(SA.LABEL_ACCESS_ROUTES)


def test_no_retired_route_is_emitted(monkeypatch):
    rows, _fit, _score, _ = _run(monkeypatch)
    retired = {"source_only", "matched_source", "matched_target", "fixed_total_mixed"}
    assert not ({r["label_access_route"] for r in rows} & retired)


def test_patch_atomicity_every_selected_patch_is_whole(monkeypatch):
    _rows, fit, _score, _ = _run(monkeypatch)
    for call in fit.calls:
        for pid, px in call["patch_px"].items():
            assert px == (SRC_PX if pid < POOL_BASE else TGT_PX), f"patch {pid} partially included"


# --------------------------------------------------------------------------- #
# the fixed-budget allocation curve: half-up rounding, exact totals, strict nesting
# --------------------------------------------------------------------------- #
def test_allocation_uses_half_up_rounding_in_patch_units(monkeypatch):
    """B=50 puts 25% and 75% exactly on .5. Half-up gives k=13 and k=38; Python's banker's rounding
    would give 12 at 25%, so the two conventions are distinguishable here."""
    _rows, fit, _score, _ = _run(monkeypatch)
    assert [_k(f) for f in PERCENTS] == [0, 13, 25, 38, 50]
    assert round(0.25 * BUDGET) == 12 != _k(25)          # banker's rounding is NOT what is used
    for idx, f in enumerate(PERCENTS):
        assert len(_tgt(fit.calls[idx])) == _k(f), f"f={f}%"


def test_allocation_source_plus_target_equals_the_budget_at_every_fraction(monkeypatch):
    rows, fit, _score, _ = _run(monkeypatch)
    for idx, f in enumerate(PERCENTS):
        call = fit.calls[idx]
        assert len(_src(call)) == BUDGET - _k(f)
        assert len(_src(call)) + len(_tgt(call)) == BUDGET
    alloc = [r for r in rows if r["label_access_route"] == SA.ROUTE_FIXED_BUDGET_ALLOCATION]
    assert len(alloc) == len(PERCENTS)
    for r in alloc:
        k = _k(r["label_budget"])
        assert (r["n_source_labels"], r["n_target_labels"]) == (BUDGET - k, k)
        assert r["n_total_labels"] == r["n_source_labels"] + r["n_target_labels"] == BUDGET
        assert r["allocation_total_budget"] == BUDGET
    assert {r["allocation_total_budget"] for r in alloc} == {BUDGET}   # ONE realized budget


def test_allocation_points_are_strictly_nested_prefixes_of_the_frozen_orders(monkeypatch):
    """The single nested source order replaced the old independent matched/fixed draws: as f grows the
    source patch sets SHRINK as prefixes and the target patch sets GROW as prefixes -- never two
    unrelated permutations."""
    _rows, fit, _score, data = _run(monkeypatch)
    src, tgt = data["source"].tolist(), data["target"].tolist()
    srcs = [_src(fit.calls[i]) for i in range(len(PERCENTS))]
    tgts = [_tgt(fit.calls[i]) for i in range(len(PERCENTS))]
    for idx, f in enumerate(PERCENTS):
        assert srcs[idx] == set(src[: BUDGET - _k(f)])
        assert tgts[idx] == set(tgt[: _k(f)])
    for a, b in zip(srcs, srcs[1:], strict=False):
        assert b < a          # strictly shrinking nested source prefixes
    for a, b in zip(tgts, tgts[1:], strict=False):
        assert a < b          # strictly growing nested target prefixes


def test_additive_routes_hold_the_complete_source_pool(monkeypatch):
    rows, fit, _score, data = _run(monkeypatch)
    src_all, tgt = set(data["source_ids"]), data["target"].tolist()
    for idx, k in enumerate(COUNTS, start=len(PERCENTS)):
        assert _src(fit.calls[idx]) == src_all
        assert _tgt(fit.calls[idx]) == set(tgt[:k])
    for r in [r for r in rows if r["label_access_route"] == SA.ROUTE_SOURCE_PLUS_TARGET]:
        assert r["n_source_labels"] == r["n_source_pool"] == len(src_all)
        assert r["n_target_labels"] == r["label_budget"]


def test_full_reference_routes(monkeypatch):
    rows, fit, _score, data = _run(monkeypatch)
    src_all, pool_all = set(data["source_ids"]), set(data["pool_ids"])
    assert _src(fit.calls[9]) == set() and _tgt(fit.calls[9]) == pool_all
    assert _src(fit.calls[10]) == src_all and _tgt(fit.calls[10]) == pool_all
    tof = next(r for r in rows if r["label_access_route"] == SA.ROUTE_TARGET_ONLY_FULL)
    spf = next(r for r in rows if r["label_access_route"] == SA.ROUTE_SOURCE_PLUS_TARGET_FULL)
    assert tof["n_source_labels"] == 0 and tof["n_target_labels"] == len(pool_all)
    assert spf["n_source_labels"] == spf["n_source_pool"] == len(src_all)
    assert spf["n_target_labels"] == spf["n_target_pool"] == len(pool_all)


# --------------------------------------------------------------------------- #
# leakage + calibration routing + seeds
# --------------------------------------------------------------------------- #
def test_no_target_test_patch_enters_any_training_set(monkeypatch):
    _rows, fit, _score, data = _run(monkeypatch)
    test_set = set(data["test_ids"])
    for call in fit.calls:
        assert not (set(call["patch_px"]) & test_set)


def test_calibration_routing_matches_the_contract(monkeypatch):
    """All five allocation points and target_only_full tune internally (f=1.0 is target-only and must
    not inherit source calibration while its siblings do); the additive routes and the combined full
    reference keep the frozen source validation set."""
    _rows, fit, _score, _ = _run(monkeypatch)
    for i, call in enumerate(fit.calls):
        assert call["tune_internal"] == (i in INTERNAL_CALLS), f"call {i}"
        assert call["has_cal"] == (i not in INTERNAL_CALLS), f"call {i}"


def test_every_route_initializes_the_probe_with_the_run_seed(monkeypatch):
    """No per-budget seed: every route seeds its probe with the pipeline RUN seed, so changing f or k
    never injects an unrelated random draw at probe-init time."""
    _rows, fit, _score, _ = _run(monkeypatch)
    assert len(fit.calls) == len(SA.label_access_expected_rows())
    assert {c["seed"] for c in fit.calls} == {RUN_SEED}


def test_sweep_never_derives_a_budget_seed(monkeypatch):
    """The sweep must not call perf._budget_seed at all -- that was the source of the k-dependent draw."""
    seen = []
    orig = pastis.perf._budget_seed
    monkeypatch.setattr(pastis.perf, "_budget_seed", lambda *a, **k: seen.append(a) or orig(*a, **k))
    _run(monkeypatch)
    assert seen == []


# --------------------------------------------------------------------------- #
# scoring targets + n_test/n_eval_patches (pixels vs patches)
# --------------------------------------------------------------------------- #
def test_scored_targets_and_pixel_vs_patch_counts(monkeypatch):
    """Every route is scored on the SAME frozen target_test patch stream and nothing else -- there is no
    complete_target pass any more. n_test stays the evaluated PIXEL count (like every seg row);
    n_eval_patches is the patch count."""
    rows, _fit, score, data = _run(monkeypatch)
    test_set = set(data["test_ids"])
    n_routes = len(SA.label_access_expected_rows())
    assert len(score.calls) == n_routes
    assert all(set(c["patches"]) == test_set for c in score.calls)
    assert {r["n_test"] for r in rows} == {len(test_set) * TGT_PX}
    assert {r["n_eval_patches"] for r in rows} == {len(test_set)}


def test_semantic_validation_uses_patch_counts_and_passes(monkeypatch):
    """The dense rows carry the ``patches`` unit and the allocation arithmetic the generalized validator
    re-derives, so the dense suite validates cleanly beside its E1 row."""
    rows, _fit, _score, _ = _run(monkeypatch)
    full = [{**r, "seed": RUN_SEED, "split_regime": SA.LABEL_ACCESS_REGIME, "holdout": "T1",
             "method": "erm", "probe_family": "logistic"} for r in rows]
    assert artifacts._validate_label_access_semantics([*full, _e1()]) == []


def test_dense_rows_carry_the_patch_unit_not_samples(monkeypatch):
    rows, _fit, _score, _ = _run(monkeypatch)
    tampered = [{**r, "seed": RUN_SEED, "split_regime": SA.LABEL_ACCESS_REGIME, "holdout": "T1",
                 "method": "erm", "probe_family": "logistic",
                 "label_budget_unit": SA.LABEL_ACCESS_TABULAR_UNIT} for r in rows]
    assert any("unit" in p for p in artifacts._validate_label_access_semantics([*tampered, _e1()]))


# --------------------------------------------------------------------------- #
# predictions: streamed with full stable identity
# --------------------------------------------------------------------------- #
def test_predictions_stream_with_full_stable_identity(monkeypatch):
    recs: list[dict] = []
    _rows, _fit, _score, data = _run(monkeypatch, predictions_sink=recs.extend)
    assert recs
    required = {"patch_id", "tile_row", "tile_col", "pixel_index", "sample_id", "label_access_route",
                "evaluation_split", "label_budget", "seed", "budget_type", "n_source_labels",
                "n_target_labels", "n_total_labels", "label_budget_unit"}
    for r in recs:
        assert required <= set(r)
        # sample_id carries patch, tile row, tile col, and the within-tile valid-pixel index
        assert r["sample_id"] == f'{r["patch_id"]}:{r["tile_row"]}:{r["tile_col"]}:{r["pixel_index"]}'
        assert r["budget_type"] == "label_access"
        assert r["label_budget_unit"] == SA.LABEL_ACCESS_DENSE_UNIT
        assert r["seed"] == RUN_SEED
    # every prediction is on the frozen target_test population -- no complete_target pass exists
    assert {r["evaluation_split"] for r in recs} == {SA.EVAL_TARGET_TEST}
    assert {r["patch_id"] for r in recs} == set(data["test_ids"])
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
# the controlled cap: applied to the BUDGET, before the mixture is built
# --------------------------------------------------------------------------- #
def test_uncapped_budget_is_the_benchmark_budget(monkeypatch):
    rows, fit, _score, _ = _run(monkeypatch, controlled_cap=None)
    assert len(_src(fit.calls[0])) == BUDGET
    assert {r["allocation_total_budget"] for r in rows} == {BUDGET}
    assert {r["benchmark_budget"] for r in rows} == {BUDGET}
    assert {r["controlled_budget_cap"] for r in rows} == {0}


def test_cap_shrinks_the_realized_budget_for_every_allocation_point(monkeypatch):
    """The cap is a TOTAL-budget cap applied before the mixture is built -- never a post-hoc subsample of
    an already-constructed route -- so B, k and B-k all move together and the fixed-budget claim holds."""
    cap = 20
    rows, fit, _score, data = _run(monkeypatch, controlled_cap=cap)
    src, tgt = data["source"].tolist(), data["target"].tolist()
    for idx, f in enumerate(PERCENTS):
        k = _k(f, cap)
        assert _src(fit.calls[idx]) == set(src[: cap - k])
        assert _tgt(fit.calls[idx]) == set(tgt[:k])
        assert len(_src(fit.calls[idx])) + len(_tgt(fit.calls[idx])) == cap
    assert {r["allocation_total_budget"] for r in rows} == {cap}
    assert {r["controlled_budget_cap"] for r in rows} == {cap}
    assert {r["benchmark_budget"] for r in rows} == {BUDGET}
    # the additive routes are NOT capped -- they are a separate operational question on the whole pool
    add = next(r for r in rows if r["label_access_route"] == SA.ROUTE_SOURCE_PLUS_TARGET)
    assert add["n_source_labels"] == len(data["source_ids"])


def test_cap_above_the_budget_never_raises_it(monkeypatch):
    rows, _fit, _score, _ = _run(monkeypatch, controlled_cap=BUDGET + 25)
    assert {r["allocation_total_budget"] for r in rows} == {BUDGET}


def test_cap_is_deterministic_and_encoder_independent(monkeypatch):
    _a, fita, _sa, _da = _run(monkeypatch, controlled_cap=20)
    _b, fitb, _sb, _db = _run(monkeypatch, controlled_cap=20, meta_extra={"model": "different-encoder"})
    assert [_src(c) for c in fita.calls] == [_src(c) for c in fitb.calls]
    assert [_tgt(c) for c in fita.calls] == [_tgt(c) for c in fitb.calls]


def test_budget_larger_than_a_pool_hard_fails(monkeypatch):
    """B is never clamped: a budget the source or target pool cannot supply is a hard error."""
    with pytest.raises(ValueError, match="infeasible"):
        _run(monkeypatch, budget=56)     # target pool holds 55 patches


def test_cap_that_empties_the_budget_hard_fails(monkeypatch):
    with pytest.raises(ValueError, match="realized budget"):
        _run(monkeypatch, controlled_cap=0)


def test_additive_count_above_the_target_pool_hard_fails(monkeypatch):
    with pytest.raises(ValueError, match="infeasible"):
        _run(monkeypatch, counts=(5, 60))


def test_order_rejects_non_permutation_of_source(monkeypatch):
    data = _make_dense()
    bad = data["source"].copy()
    bad[0] = 999999
    data = {**data, "source": bad}
    with pytest.raises(ValueError, match="source_order"):
        _run(monkeypatch, data=data)


def test_order_rejects_non_permutation_of_target(monkeypatch):
    data = _make_dense()
    bad = data["target"].copy()
    bad[0] = 999999
    data = {**data, "target": bad}
    with pytest.raises(ValueError, match="target_order"):
        _run(monkeypatch, data=data)


def test_retired_keyword_arguments_are_gone(monkeypatch):
    """``matched_source_order`` / ``fixed_removal_order`` / ``cap_patches`` belonged to the retired
    two-order contract; passing one must be a hard TypeError, never silently ignored."""
    data = _make_dense()
    monkeypatch.setattr(pastis, "fit_probe_multiclass", _RecFit())
    monkeypatch.setattr(pastis, "score_segmentation_streamed", _RecScore())
    for dead in ("matched_source_order", "fixed_removal_order", "cap_patches"):
        with pytest.raises(TypeError):
            pastis.run_probes_segmentation_label_access(
                [], RUN_SEED, source_patches=data["source_patches"], pool_patches=data["pool_patches"],
                target_test_patches=data["test_patches"], source_order=data["source"],
                target_order=data["target"], budget=BUDGET, percents=PERCENTS, counts=COUNTS,
                load_pixels=data["load_pixels"], stream_eval=data["stream_eval"],
                x_val=data["x_val"], y_val=data["y_val"], meta={"benchmark": "pastis"},
                **{dead: data["source"]},
            )


# --------------------------------------------------------------------------- #
# real capped cacheutils loader: patch-first, paired, MAX_DENSE_PIXELS respected
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
    # the numeric collision between allocation@25 (a PERCENT) and source_plus_target@25 (a COUNT) is
    # resolved by the route field, so the two never share a resume key.
    alloc = next(r for r in rows if r["label_access_route"] == SA.ROUTE_FIXED_BUDGET_ALLOCATION
                 and r["label_budget"] == 25)
    add = next(r for r in rows if r["label_access_route"] == SA.ROUTE_SOURCE_PLUS_TARGET
               and r["label_budget"] == 25)
    assert runstate.budget_row_key(alloc) != runstate.budget_row_key(add)


def _dense_split(source, pool, test, label="T1"):
    from evals.regimes.base import DenseSourceTargetSplit

    return DenseSourceTargetSplit(
        label=label, source_train_patches=frozenset(source),
        source_val_patches=frozenset(), source_test_patches=frozenset(),
        target_label_pool_patches=frozenset(pool), target_test_patches=frozenset(test),
        has_target=True, supports_target_labels=True,
    )


def test_load_dense_label_access_resolves_one_nested_source_order(tmp_path):
    source, pool, test = list(range(60)), list(range(1000, 1055)), list(range(2000, 2008))
    la_rows = SA.build_label_access_rows(
        seed=0, source_ids=[str(p) for p in source], target_pool_ids=[str(p) for p in pool],
        target_test_ids=[str(p) for p in test],
    )
    _path, sha = SA.write_label_access(tmp_path / "splits", "pastis", 0, "T1", la_rows)
    loaded = SA.load_dense_label_access(
        tmp_path / "splits", "pastis", 0, _dense_split(source, pool, test), sha, benchmark_budget=BUDGET,
    )
    assert isinstance(loaded, SA.LoadedDenseLabelAccess)
    assert loaded._fields == ("holdout", "source_ranked_patches", "target_ranked_patches", "benchmark_budget")
    assert loaded.holdout == "T1" and loaded.benchmark_budget == BUDGET
    assert sorted(loaded.source_ranked_patches.tolist()) == source     # exact permutation, one order
    assert sorted(loaded.target_ranked_patches.tolist()) == pool
    # every allocation point slices a PREFIX of this single order, so the curve is nested by construction
    prefixes = [set(loaded.source_ranked_patches[: BUDGET - _k(f)].tolist()) for f in PERCENTS]
    for a, b in zip(prefixes, prefixes[1:], strict=False):
        assert b < a


def test_load_dense_label_access_missing_file_hard_errors(tmp_path):
    with pytest.raises(SA.SplitArtifactError, match="missing label_access.csv"):
        SA.load_dense_label_access(
            tmp_path / "splits", "pastis", 0,
            _dense_split(range(60), range(1000, 1055), range(2000, 2008)), "0" * 64,
            benchmark_budget=BUDGET,
        )


def test_load_dense_label_access_rejects_stale_population(tmp_path):
    source, pool, test = list(range(60)), list(range(1000, 1055)), list(range(2000, 2008))
    la_rows = SA.build_label_access_rows(
        seed=0, source_ids=[str(p) for p in source], target_pool_ids=[str(p) for p in pool],
        target_test_ids=[str(p) for p in test],
    )
    _p, stale_sha = SA.write_label_access(tmp_path / "splits", "pastis", 0, "T1", la_rows)
    with pytest.raises(SA.SplitArtifactError):
        SA.load_dense_label_access(
            tmp_path / "splits", "pastis", 0, _dense_split(list(range(1, 61)), pool, test), stale_sha,
            benchmark_budget=BUDGET,
        )


def test_load_dense_label_access_requires_the_frozen_benchmark_budget(tmp_path):
    source, pool, test = list(range(60)), list(range(1000, 1055)), list(range(2000, 2008))
    la_rows = SA.build_label_access_rows(
        seed=0, source_ids=[str(p) for p in source], target_pool_ids=[str(p) for p in pool],
        target_test_ids=[str(p) for p in test],
    )
    _p, sha = SA.write_label_access(tmp_path / "splits", "pastis", 0, "T1", la_rows)
    with pytest.raises(SA.SplitArtifactError, match="benchmark_budget"):
        SA.load_dense_label_access(
            tmp_path / "splits", "pastis", 0, _dense_split(source, pool, test), sha, benchmark_budget=None,
        )


# --------------------------------------------------------------------------- #
# manifest: patches unit + prediction honesty
# --------------------------------------------------------------------------- #
def _manifest(benchmark, regimes, write_predictions=True):
    return runstate.build_run_manifest(
        "raw", benchmark, "artifact", "digest", regimes, [0], {},
        active_probes=["logistic"], budget_regimes={"source": [1.0]}, max_dense_pixels=10_000,
        write_predictions=write_predictions,
    )


def test_manifest_contract_unit_is_patches_for_pastis():
    contract = SA.label_access_contract(enabled=True, benchmark="pastis")
    assert contract["unit"] == SA.LABEL_ACCESS_DENSE_UNIT and contract["enabled"] is True
    assert contract["allocation_percents"] == list(PERCENTS)
    assert contract["additive_counts"] == list(COUNTS)
    assert contract["evaluation_splits"] == [SA.EVAL_TARGET_TEST]
    man = _manifest("pastis", ["geographic_ood", "official"])
    assert man["label_access"]["enabled"] is True
    assert man["label_access"]["unit"] == SA.LABEL_ACCESS_DENSE_UNIT


def test_manifest_write_predictions_never_over_claims():
    # dense WITHOUT geographic_ood writes no predictions -> must not claim enabled
    assert _manifest("pastis", ["official", "random_id"], write_predictions=True)["write_predictions"] is False
    # dense WITH geographic_ood does write them
    assert _manifest("pastis", ["geographic_ood"], write_predictions=True)["write_predictions"] is True
    # tabular always writes when enabled; disabled stays disabled
    assert _manifest("cropharvest", ["random_id"], write_predictions=True)["write_predictions"] is True
    assert _manifest("pastis", ["geographic_ood"], write_predictions=False)["write_predictions"] is False
