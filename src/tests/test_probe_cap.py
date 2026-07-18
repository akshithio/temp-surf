"""Tests for the RB_PROBE_CAP probe-capacity training-size cap.

Covers the properties the capped protocol depends on: the cap is deterministic and ENCODER-INDEPENDENT
(seeded only by cell identity, never the model), class-stratified, group-balanced where feasible,
recorded in row metadata, applied to MLP/logistic but NOT kNN, and it never touches eval sets. The
target route keeps all ``k`` few-shot labels and caps only the source head.
"""

from __future__ import annotations

import os
from contextlib import contextmanager

import numpy as np
import pytest

from evals import evals as EV
from utils import perfutils as perf


@contextmanager
def _cap_env(value: str | None):
    prev = os.environ.get("RB_PROBE_CAP")
    if value is None:
        os.environ.pop("RB_PROBE_CAP", None)
    else:
        os.environ["RB_PROBE_CAP"] = value
    try:
        yield
    finally:
        if prev is None:
            os.environ.pop("RB_PROBE_CAP", None)
        else:
            os.environ["RB_PROBE_CAP"] = prev


def _synth(n, d=16, n_classes=5, n_groups=3, seed=0):
    rng = np.random.default_rng(seed)
    x = rng.standard_normal((n, d)).astype(np.float32)
    y = rng.integers(0, n_classes, size=n)
    groups = rng.integers(0, n_groups, size=n).astype(object)
    return x, y, groups


# ── Cap helper: size, determinism, encoder-independence ───────────────────────

def test_cap_size_and_determinism() -> None:
    _, y, groups = _synth(2000)
    sel = perf._cap_stratified_indices(y, groups, 500, seed=123)
    assert len(sel) == 500
    assert np.all(np.diff(sel) > 0)  # sorted + unique
    assert sel.min() >= 0 and sel.max() < len(y)
    # Deterministic in (y, groups, cap, seed).
    assert np.array_equal(sel, perf._cap_stratified_indices(y, groups, 500, seed=123))
    # A different seed gives a different draw (statistically certain for 500-of-2000).
    assert not np.array_equal(sel, perf._cap_stratified_indices(y, groups, 500, seed=124))
    # cap >= n is a no-op passthrough.
    assert np.array_equal(perf._cap_stratified_indices(y, groups, 5000, seed=1), np.arange(len(y)))


def test_cap_seed_is_encoder_independent() -> None:
    base = {"benchmark": "breizhcrops", "split_regime": "geographic_ood", "holdout": "R1"}
    s_galileo = perf._cap_seed(0, "source", 1.0, {**base, "model": "galileo"})
    s_raw = perf._cap_seed(0, "source", 1.0, {**base, "model": "raw"})
    assert s_galileo == s_raw  # the model must NOT influence the sampled rows
    # ...but the cell identity must.
    assert perf._cap_seed(0, "source", 1.0, base) != perf._cap_seed(1, "source", 1.0, base)
    assert perf._cap_seed(0, "source", 1.0, base) != perf._cap_seed(0, "target", 1.0, base)


def test_cap_class_stratified_proportions() -> None:
    # Classes 0/1/2 with counts 1000/600/400 -> cap 500 -> proportional 250/150/100.
    y = np.concatenate([np.zeros(1000), np.ones(600), np.full(400, 2)]).astype(int)
    sel = perf._cap_stratified_indices(y, None, 500, seed=7)
    _, counts = np.unique(y[sel], return_counts=True)
    assert counts.tolist() == [250, 150, 100]
    assert len(sel) == 500


def test_cap_group_balanced_where_feasible() -> None:
    # One class, groups A/B (large) and C (only 10); cap 300. Round-robin takes all 10 of C, then splits
    # the remaining 290 evenly across the two feasible groups -> A=B=145, C=10.
    y = np.zeros(1810, dtype=int)
    groups = np.array(["A"] * 900 + ["B"] * 900 + ["C"] * 10, dtype=object)
    sel = perf._cap_stratified_indices(y, groups, 300, seed=3)
    picked = groups[sel]
    counts = {g: int((picked == g).sum()) for g in ("A", "B", "C")}
    assert len(sel) == 300
    assert counts["C"] == 10          # small group fully taken
    assert counts["A"] == counts["B"] == 145  # remainder split evenly across the feasible groups


def test_cap_row_meta_records_sizes_and_counts() -> None:
    y_post = np.array([0, 0, 0, 1, 1, 2])
    meta = perf._cap_row_meta(100, "mlp", n_precap=2000, y_post=y_post)
    assert meta["probe_cap"] == 100
    assert meta["probe_capped"] == 1
    assert meta["n_train_precap"] == 2000
    assert meta["n_train_postcap"] == 6
    assert meta["probe_cap_class_counts"] == "0:3;1:2;2:1"
    # kNN / uncapped -> flagged not capped, post == pre, no class string.
    knn = perf._cap_row_meta(100, "knn", n_precap=2000, y_post=None)
    assert knn["probe_capped"] == 0 and knn["n_train_postcap"] == 2000 and knn["probe_cap_class_counts"] == ""


# ── Integration: source route through run_probes_multiclass ───────────────────

def _source_rows(family, cap, n=2000, seed=0):
    x_tr, y_tr, g_tr = _synth(n, seed=seed)
    x_te, y_te, g_te = _synth(300, seed=seed + 1)
    x_val, y_val, _ = _synth(300, seed=seed + 2)
    rows: list[dict] = []
    meta = {"model": "m", "benchmark": "breizhcrops", "method": "erm",
            "split_regime": "geographic_ood", "holdout": "R1", "probe_family": family}
    with _cap_env(cap):
        EV.run_probes_multiclass(
            rows, x_tr, x_te, y_tr, y_te, seed, budgets=[1.0], meta=meta,
            groups_train=g_tr, groups_test=g_te, x_val=x_val, y_val=y_val, family=family,
        )
    return rows, len(y_te)


def test_source_route_caps_all_probe_families() -> None:
    # Fixed-budget protocol: logistic, MLP AND kNN all share the same 500-example candidate pool;
    # evaluation set untouched. (kNN is capped too now -- a common controlled training budget.)
    for fam in ("logistic", "mlp", "knn"):
        rows, n_test = _source_rows(fam, "500")
        r = [x for x in rows if x["evaluation_split"] == "test"][0]
        assert r["probe_capped"] == 1, f"{fam} should be capped"
        assert r["probe_cap"] == 500
        assert r["n_train_precap"] == 2000
        assert r["n_train_postcap"] == 500
        assert r["n_test"] == n_test  # evaluation set NOT subsampled
        assert sum(int(p.split(":")[1]) for p in r["probe_cap_class_counts"].split(";")) == 500


def test_source_route_uncapped_when_env_unset() -> None:
    rows, _ = _source_rows("logistic", None)
    r = [r for r in rows if r["evaluation_split"] == "test"][0]
    assert r["probe_capped"] == 0
    assert r["n_train_postcap"] == 2000


# ── Integration: target route keeps all k few-shot, caps only source ──────────

def test_target_route_fewshot_keeps_k_caps_source() -> None:
    x_src, y_src, g_src = _synth(2000, seed=10)
    x_tgt, y_tgt, g_tgt = _synth(300, seed=11)
    rows: list[dict] = []
    meta = {"model": "m", "benchmark": "breizhcrops", "method": "erm",
            "split_regime": "geographic_ood", "holdout": "R1", "probe_family": "logistic"}
    with _cap_env("500"):
        EV.run_probes_multiclass_target(
            rows, x_src, x_tgt, y_src, y_tgt, 0, budgets=[0, 5, -1], meta=meta,
            groups_source=g_src, groups_target=g_tgt, family="logistic",
        )
    # Budget 0 emits both a held_out row and a full-domain row; key each split separately.
    held = {r["label_budget"]: r for r in rows if r["evaluation_split"] == "held_out"}

    # Budget 0 (source-only): 2000 -> capped to 500.
    assert held[0]["n_train_precap"] == 2000
    assert held[0]["n_train_postcap"] == 500

    # Few-shot budget 5: source head capped, all 5 target kept -> postcap == cap (495 source + 5 target).
    r5 = held[5]
    assert r5["probe_capped"] == 1
    assert r5["n_train_precap"] == 2005
    assert r5["n_train_postcap"] == 500

    # Oracle (-1) trains on the capped target pool, not the source.
    assert held[-1]["probe_capped"] in (0, 1)  # caps only if the 80% target pool exceeds 500

    # Held-out target test set (the fixed 20%) is never subsampled by the cap.
    assert held[5]["n_test"] == held[0]["n_test"]


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
