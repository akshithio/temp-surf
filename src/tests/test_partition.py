"""Tests for the deterministic exact-size constrained partitioners (evals.partition).

Covers: exact sizes; determinism (same seed -> same membership, new seed -> same sizes new
membership); the target guarantees (singleton class -> target_test, >=2-example class in BOTH target
partitions); INDEPENDENT class + region marginals for the source split (with an adversarial sparse
fixture that the joint-stratum approach fails); patch-level multilabel capacity + balance; and hard
failure (never a silent fallback) on infeasible requests.
"""

from __future__ import annotations

import math
from types import SimpleNamespace

import numpy as np
import pytest

from evals import partition as P
from evals.split_spec import source_partition_sizes, target_partition_sizes


def _is_partition(result, n):
    seen = np.concatenate([v for v in result.values()]) if result else np.empty(0, int)
    assert len(seen) == n
    assert set(seen.tolist()) == set(range(n)), "not a partition of range(n)"
    for name, arr in result.items():
        assert arr.dtype == np.int64
        assert list(arr) == sorted(arr), f"{name} not sorted"


def _assert_marginal_contract(out, values, n):
    """Every value's count in every partition is floor/ceil of its proportional quota."""
    values = np.asarray(values)
    counts = {v: int((values == v).sum()) for v in set(values.tolist())}
    n_total = len(values)
    assert n_total == n
    for name, idx in out.items():
        cap = len(idx)
        got = {v: int((values[idx] == v).sum()) for v in counts}
        for v, nv in counts.items():
            q = nv * cap / n_total
            lo, hi = math.floor(q), math.ceil(q)
            assert lo <= got[v] <= hi, (name, v, got[v], f"[{lo},{hi}]", f"quota={q:.3f}")


# --------------------------------------------------------------------------- #
# Target 80/20 split
# --------------------------------------------------------------------------- #
def test_target_exact_sizes_and_partition():
    rng = np.random.default_rng(0)
    classes = rng.integers(0, 6, size=200).tolist()
    pool, test = target_partition_sizes(200)
    out = P.partition_target(classes, pool, test, seed=0)
    assert {k: len(v) for k, v in out.items()} == {P.TARGET_LABEL_POOL: pool, P.TARGET_TEST: test}
    _is_partition(out, 200)


def test_target_singleton_class_goes_to_test():
    # classes 2 and 5 appear once each -> must both be in target_test, never the pool
    classes = ([0] * 60) + ([1] * 38) + [2] + [3] * 60 + [4] * 39 + [5]  # N=199
    pool, test = target_partition_sizes(len(classes))
    for seed in range(6):
        out = P.partition_target(classes, pool, test, seed=seed)
        s = np.asarray(classes)
        for singleton in (2, 5):
            assert (s[out[P.TARGET_TEST]] == singleton).sum() == 1, (seed, singleton)
            assert (s[out[P.TARGET_LABEL_POOL]] == singleton).sum() == 0, (seed, singleton)
        _is_partition(out, len(classes))


def test_target_two_example_class_in_both_partitions():
    classes = ([0] * 60) + ([1] * 38) + [2, 2]  # class 2 has exactly two examples, N=100
    pool, test = target_partition_sizes(100)
    for seed in range(8):
        out = P.partition_target(classes, pool, test, seed=seed)
        s = np.asarray(classes)
        assert (s[out[P.TARGET_LABEL_POOL]] == 2).sum() >= 1, seed
        assert (s[out[P.TARGET_TEST]] == 2).sum() >= 1, seed


def test_target_same_seed_identical_new_seed_new_membership():
    rng = np.random.default_rng(3)
    classes = rng.integers(0, 5, size=300).tolist()
    pool, test = target_partition_sizes(300)
    a = P.partition_target(classes, pool, test, seed=0)
    a2 = P.partition_target(classes, pool, test, seed=0)
    b = P.partition_target(classes, pool, test, seed=1)
    for k in a:
        assert np.array_equal(a[k], a2[k])
        assert len(a[k]) == len(b[k])
    assert not all(np.array_equal(a[k], b[k]) for k in a)


def test_target_hard_fails_when_singletons_exceed_test_size():
    # 10 singleton classes but test_size is small -> infeasible, must raise (never silently relax)
    classes = list(range(10)) + [99] * 2  # 10 singletons + one 2-example class, N=12
    pool, test = target_partition_sizes(12)  # 80/20 -> pool 10, test 2 ; 10 singletons > 2
    with pytest.raises(P.PartitionError, match="singleton target classes"):
        P.partition_target(classes, pool, test, seed=0)


# --------------------------------------------------------------------------- #
# Source 80/10/10 split -- INDEPENDENT class + region marginals
# --------------------------------------------------------------------------- #
def test_source_exact_sizes_and_partition():
    rng = np.random.default_rng(1)
    classes = rng.integers(0, 4, size=600).tolist()
    regions = rng.integers(0, 3, size=600).tolist()
    train, val, test = source_partition_sizes(600)
    out = P.partition_source(classes, regions, [("train", train), ("val", val), ("test", test)], seed=0)
    assert {k: len(v) for k, v in out.items()} == {"train": train, "val": val, "test": test}
    _is_partition(out, 600)


def test_source_same_seed_identical_new_seed_new_membership():
    rng = np.random.default_rng(2)
    classes = rng.integers(0, 5, size=400).tolist()
    regions = rng.integers(0, 4, size=400).tolist()
    sizes = list(zip(("train", "val", "test"), source_partition_sizes(400), strict=True))
    a = P.partition_source(classes, regions, sizes, seed=0)
    a2 = P.partition_source(classes, regions, sizes, seed=0)
    b = P.partition_source(classes, regions, sizes, seed=1)
    for k in a:
        assert np.array_equal(a[k], a2[k])
        assert len(a[k]) == len(b[k])
    assert not all(np.array_equal(a[k], b[k]) for k in a)


def _region_counts(regions, idx):
    r = np.asarray(regions)[idx]
    return {reg: int((r == reg).sum()) for reg in set(np.asarray(regions).tolist())}


def test_source_marginal_contract_holds_for_both_class_and_region():
    rng = np.random.default_rng(5)
    classes = rng.integers(0, 5, size=500).tolist()
    regions = [chr(ord("A") + r) for r in rng.integers(0, 4, size=500)]
    sizes = list(zip(("train", "val", "test"), source_partition_sizes(500), strict=True))
    for seed in range(4):
        out = P.partition_source(classes, regions, sizes, seed=seed)
        _is_partition(out, 500)
        _assert_marginal_contract(out, classes, 500)   # class marginal contract in every partition
        _assert_marginal_contract(out, regions, 500)   # region marginal contract in every partition


@pytest.mark.parametrize("cells", [
    {("0", "A"): 5, ("0", "B"): 5, ("1", "A"): 5, ("1", "B"): 5},   # balanced joint
    {("0", "A"): 8, ("0", "B"): 2, ("1", "A"): 2, ("1", "B"): 8},   # correlated joint
    {("0", "A"): 9, ("0", "B"): 1, ("1", "A"): 1, ("1", "B"): 9},   # strongly correlated joint
])
def test_source_n20_two_class_two_region_16_2_2(cells):
    """The required feasible adversarial: N=20, 2 classes x 2 regions, capacities 16/2/2. Every
    feasible class and region satisfies the marginal contract in EVERY partition -- here exactly
    1 of each class and each region in val and in test, 8 of each in train (integer quotas)."""
    classes, regions = [], []
    for (c, r), k in cells.items():
        classes += [c] * k
        regions += [r] * k
    assert len(classes) == 20
    sizes = [("train", 16), ("val", 2), ("test", 2)]
    for seed in range(6):
        out = P.partition_source(classes, regions, sizes, seed=seed)
        assert {k: len(v) for k, v in out.items()} == {"train": 16, "val": 2, "test": 2}
        _is_partition(out, 20)
        _assert_marginal_contract(out, classes, 20)
        _assert_marginal_contract(out, regions, 20)
        # exact quotas here: val/test have exactly one of each class and each region
        for part in ("val", "test"):
            assert _region_counts(regions, out[part]) == {"A": 1, "B": 1}, (seed, part)
            assert _region_counts(classes, out[part]) == {"0": 1, "1": 1}, (seed, part)


def test_source_regression_n10_integer_quota_not_violated():
    """P1 regression: N=10, caps 8/1/1, class counts 3/5/2, one region. The five-example class has an
    exactly-integer train quota of 4; the old independent rounding (+1 on an already-integer quota)
    assigned 5 and self-rejected. The joint MILP gives exactly 4 in train."""
    classes = [0] * 3 + [1] * 5 + [2] * 2
    regions = ["R"] * 10
    for seed in range(4):
        out = P.partition_source(classes, regions, [("train", 8), ("val", 1), ("test", 1)], seed=seed)
        assert {k: len(v) for k, v in out.items()} == {"train": 8, "val": 1, "test": 1}
        _is_partition(out, 10)
        assert int((np.asarray(classes)[out["train"]] == 1).sum()) == 4, seed  # exactly four, not five
        _assert_marginal_contract(out, classes, 10)


def test_source_regression_n11_sequential_fill_would_reject_feasible():
    """P1 regression: N=11, caps 7/2/2, joint cells 0/A=2, 0/B=4, 1/A=4, 1/B=1. A globally feasible
    split exists; the old val-then-test sequential fill consumed cells test needed and raised. The
    joint MILP accepts it."""
    classes, regions = [], []
    for (cl, rg), k in {("0", "A"): 2, ("0", "B"): 4, ("1", "A"): 4, ("1", "B"): 1}.items():
        classes += [cl] * k
        regions += [rg] * k
    assert len(classes) == 11
    for seed in range(6):
        out = P.partition_source(classes, regions, [("train", 7), ("val", 2), ("test", 2)], seed=seed)
        assert {k: len(v) for k, v in out.items()} == {"train": 7, "val": 2, "test": 2}
        _is_partition(out, 11)
        _assert_marginal_contract(out, classes, 11)
        _assert_marginal_contract(out, regions, 11)


def _source_feasible(alloc, cells, cell_count, caps, n_c, n_r, n_total):
    """Does a per-cell count allocation satisfy every source constraint?"""
    nparts = len(caps)
    if any(sum(alloc[cell]) != cell_count[cell] for cell in cells):
        return False
    if any(sum(alloc[cell][p] for cell in cells) != caps[p] for p in range(nparts)):
        return False
    for values, key in ((n_c, 0), (n_r, 1)):
        for v, nv in values.items():
            for p in range(nparts):
                got = sum(alloc[cell][p] for cell in cells if cell[key] == v)
                if not (math.floor(nv * caps[p] / n_total) <= got <= math.ceil(nv * caps[p] / n_total)):
                    return False
    return True


def test_source_secondary_tiebreak_is_non_separable():
    """The secondary tie-break must genuinely distinguish feasible allocations. N=6, cells (0,A)=2,
    (0,B)=1,(1,A)=1,(1,B)=2, caps 3/3: two feasible allocations X, Y have DIFFERENT k*p values, while
    a separable variable-index objective gives them the SAME value (constant over the feasible set --
    which is why the index objective provides no tie-break)."""
    cells = [("0", "A"), ("0", "B"), ("1", "A"), ("1", "B")]
    cell_count = {("0", "A"): 2, ("0", "B"): 1, ("1", "A"): 1, ("1", "B"): 2}
    caps, n_c, n_r, n = [3, 3], {"0": 3, "1": 3}, {"A": 3, "B": 3}, 6
    x = {("0", "A"): [2, 0], ("0", "B"): [0, 1], ("1", "A"): [0, 1], ("1", "B"): [1, 1]}
    y = {("0", "A"): [1, 1], ("0", "B"): [1, 0], ("1", "A"): [0, 1], ("1", "B"): [1, 1]}
    assert _source_feasible(x, cells, cell_count, caps, n_c, n_r, n)
    assert _source_feasible(y, cells, cell_count, caps, n_c, n_r, n)

    ncell, nparts = 4, 2
    coeffs = P._source_tiebreak_coeffs(ncell, nparts)

    def val(coef, alloc):
        return float(np.dot(coef, [alloc[cells[k]][p] for k in range(ncell) for p in range(nparts)]))

    assert val(coeffs, x) != val(coeffs, y), "non-separable tie-break must differ between allocations"
    # a separable variable-index objective is constant over the feasible set -> no tie-break
    assert val(np.arange(ncell * nparts, dtype=float), x) == val(np.arange(ncell * nparts, dtype=float), y)


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_source_balanced_2x2_spreads_every_cell_not_segregated(seed):
    """P-quality regression: N=100, cells 0/A=0/B=1/A=1/B=25, caps 80/10/10. The deviation-minimizing
    primary objective must give val and test floor/ceil of 2.5 (i.e. 2 or 3) from EVERY joint cell,
    not drop two cells entirely (the segregation the old k*p-as-primary objective produced:
    val/test = 0/B x5, 1/A x5, with zero 0/A and zero 1/B)."""
    classes, regions = [], []
    for (c, r), k in {("0", "A"): 25, ("0", "B"): 25, ("1", "A"): 25, ("1", "B"): 25}.items():
        classes += [c] * k
        regions += [r] * k
    out = P.partition_source(classes, regions, [("train", 80), ("val", 10), ("test", 10)], seed=seed)
    _is_partition(out, 100)
    _assert_marginal_contract(out, classes, 100)
    _assert_marginal_contract(out, regions, 100)
    all_cells = {("0", "A"), ("0", "B"), ("1", "A"), ("1", "B")}
    for part in ("val", "test"):
        cc = {}
        for i in out[part].tolist():
            cc[(classes[i], regions[i])] = cc.get((classes[i], regions[i]), 0) + 1
        assert set(cc) == all_cells, (seed, part, cc)                    # every cell present
        assert all(2 <= v <= 3 for v in cc.values()), (seed, part, cc)  # floor/ceil of 2.5 each


def test_source_raises_when_stage_two_tiebreak_fails_no_silent_fallback(monkeypatch):
    """Stage one proves the deviation optimum but gives no deterministic secondary selection, so a
    stage-two failure must RAISE (with the solver status/message), never silently fall back to the
    stage-one solution. Stub milp: first call (stage 1) succeeds; second call (stage 2) fails."""
    real_milp = P.milp
    calls = {"n": 0}

    def fake_milp(*a, **k):
        calls["n"] += 1
        if calls["n"] == 1:
            return real_milp(*a, **k)  # stage 1 solves for real
        return SimpleNamespace(success=False, x=None, status=2, message="stubbed stage-two failure")

    monkeypatch.setattr(P, "milp", fake_milp)
    classes = ["0"] * 50 + ["1"] * 50
    regions = ["A"] * 50 + ["B"] * 50
    with pytest.raises(P.PartitionError, match="stage-two"):
        P.partition_source(classes, regions, [("train", 80), ("val", 10), ("test", 10)], seed=0)
    assert calls["n"] == 2, "both stages should have been attempted"


def _cell_counts(classes, regions, out):
    cc = {}
    for name, idx in out.items():
        for i in idx.tolist():
            cc[(str(classes[i]), str(regions[i]), name)] = cc.get((str(classes[i]), str(regions[i]), name), 0) + 1
    return cc


def test_source_counts_reproducible_and_seed_independent_membership_varies():
    rng = np.random.default_rng(7)
    classes = rng.integers(0, 5, size=300).tolist()
    regions = [chr(ord("A") + r) for r in rng.integers(0, 4, size=300)]
    sizes = list(zip(("train", "val", "test"), source_partition_sizes(300), strict=True))
    a = P.partition_source(classes, regions, sizes, seed=0)
    a2 = P.partition_source(classes, regions, sizes, seed=0)
    b = P.partition_source(classes, regions, sizes, seed=1)
    for k in a:                                              # repeated call, same seed: identical
        assert np.array_equal(a[k], a2[k])
    assert _cell_counts(classes, regions, a) == _cell_counts(classes, regions, b)  # counts seed-independent
    assert not all(np.array_equal(a[k], b[k]) for k in a)   # membership varies across seeds


def test_source_independent_marginals_where_joint_strata_fail():
    """Adversarial sparse fixture: 2 regions x 30 shared classes, every (region,class) cell size 1.

    Independent marginals keep each region ~proportional in every partition. The joint-stratum
    approach (systematic over a single (region,class) key) clusters a whole region into contiguous
    blocks, so val/test end up region-pure -- the failure this rewrite fixes.
    """
    classes, regions = [], []
    for reg in ("A", "B"):
        for c in range(30):
            classes.append(c)
            regions.append(reg)
    n = 60
    sizes = [("train", 48), ("val", 6), ("test", 6)]  # proportional region share in val/test = 3 each

    out = P.partition_source(classes, regions, sizes, seed=0)
    _is_partition(out, n)
    for name, sz in sizes:
        rc = _region_counts(regions, out[name])
        for reg in ("A", "B"):
            assert abs(rc[reg] - sz / 2) <= 1, ("independent imbalanced", name, reg, rc)

    # joint-stratum reference: it puts a region entirely on one side -> some partition has 0 of a
    # region that should hold ~3. Proves the fixture discriminates the two approaches.
    joint_keys = [f"{r}|{c}" for c, r in zip(classes, regions, strict=True)]
    order = P._systematic_order(joint_keys, seed=0, op=P._OP_SYSTEMATIC_SLOT)
    joint, start = {}, 0
    for name, sz in sizes:
        joint[name] = order[start:start + sz]
        start += sz
    worst_zero = any(
        _region_counts(regions, joint[name]).get(reg, 0) == 0
        for name, sz in sizes if sz >= 4 for reg in ("A", "B")
    )
    assert worst_zero, "adversarial fixture did not expose the joint-stratum failure"


def test_source_length_mismatch_raises():
    with pytest.raises(P.PartitionError, match="length mismatch"):
        P.partition_source([0, 1, 2], [0, 1], [("train", 3)], seed=0)


# --------------------------------------------------------------------------- #
# Multilabel (PASTIS patches)
# --------------------------------------------------------------------------- #
def _multilabel_fixture(n=200, n_labels=6, seed=0):
    rng = np.random.default_rng(seed)
    return [sorted(set(rng.integers(0, n_labels, size=rng.integers(1, 4)).tolist())) for _ in range(n)]


def test_multilabel_hits_exact_capacities_and_partitions():
    sets = _multilabel_fixture(200)
    out = P.multilabel_assign(sets, [("train", 160), ("val", 20), ("test", 20)], seed=0)
    assert {k: len(v) for k, v in out.items()} == {"train": 160, "val": 20, "test": 20}
    _is_partition(out, 200)


def test_multilabel_deterministic_in_seed():
    sets = _multilabel_fixture(150)
    sizes = [("train", 120), ("val", 15), ("test", 15)]
    a = P.multilabel_assign(sets, sizes, seed=0)
    a2 = P.multilabel_assign(sets, sizes, seed=0)
    b = P.multilabel_assign(sets, sizes, seed=1)
    for k in a:
        assert np.array_equal(a[k], a2[k])
        assert len(a[k]) == len(b[k])
    assert not all(np.array_equal(a[k], b[k]) for k in a)


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_multilabel_single_label_meets_exact_floor_ceil_quota(seed):
    """Single-label items reduce to single-attribute stratification: after the rare-label-first fix,
    every label's count is EXACTLY floor/ceil of its quota in every partition (train, val, AND test)
    for all final seeds -- no +/-1 slack."""
    rng = np.random.default_rng(4)
    labels = rng.integers(0, 6, size=600).tolist()
    out = P.multilabel_assign([[lab] for lab in labels], [("train", 480), ("val", 60), ("test", 60)], seed=seed)
    _is_partition(out, 600)
    lab_arr = np.asarray(labels)
    counts = {v: int((lab_arr == v).sum()) for v in set(labels)}
    for name, idx in out.items():
        cap = len(idx)
        for v, nv in counts.items():
            got = int((lab_arr[idx] == v).sum())
            q = nv * cap / 600
            assert math.floor(q) <= got <= math.ceil(q), (seed, name, v, got, round(q, 2))


def test_multilabel_overlapping_random_fixture_balanced_all_seeds():
    """Realistic OVERLAPPING multilabel regression: the existing random class-set fixture (400 items,
    8 labels, 1-3 labels each) built once, with the assignment run for all final seeds 0, 1, 2. The
    corrected rare-label-first rule locks in: every label present in every partition; achieved/
    expected ratio >= 0.75; max absolute per-label deviation <= 4. (The pre-fix implementation --
    prioritizing the sum over all the item's labels -- violated these.)"""
    from collections import Counter

    sets = _multilabel_fixture(400, n_labels=8, seed=2)  # built once
    sizes = [("train", 320), ("val", 40), ("test", 40)]
    glob = Counter(lab for s in sets for lab in s)
    for seed in (0, 1, 2):
        out = P.multilabel_assign(sets, sizes, seed=seed)
        assert {k: len(v) for k, v in out.items()} == {"train": 320, "val": 40, "test": 40}
        _is_partition(out, 400)
        for name, cap in sizes:
            cnt = Counter(lab for i in out[name] for lab in sets[i])
            for lab, g in glob.items():
                exp = g * cap / 400
                assert cnt[lab] >= 1, (seed, name, lab, "absent")            # present in every partition
                assert cnt[lab] / exp >= 0.75, (seed, name, lab, cnt[lab], exp)  # achieved/expected >= 0.75
                assert abs(cnt[lab] - exp) <= 4, (seed, name, lab, cnt[lab], exp)  # max abs deviation <= 4


def test_multilabel_deliberately_feasible_fixture_is_balanced_in_all_partitions():
    """A deliberately-feasible class-SET fixture with a known-balanced allocation: three disjoint
    label-pairs, 60 items each (N=180 -> 144/18/18), so every label's quota is the integer 48/6/6.
    Iterative stratification must hit it within a meaningful per-label deviation of 1 in train,
    validation AND test, across all final seeds -- with exact capacities, complete assignment,
    determinism, and seed-varying membership."""
    from collections import Counter

    sets = [[0, 1]] * 60 + [[2, 3]] * 60 + [[4, 5]] * 60  # N=180, quotas 48/6/6 per label
    sizes = [("train", 144), ("val", 18), ("test", 18)]
    glob = Counter(lab for s in sets for lab in s)
    memberships = []
    for seed in (0, 1, 2):
        out = P.multilabel_assign(sets, sizes, seed=seed)
        assert {k: len(v) for k, v in out.items()} == {"train": 144, "val": 18, "test": 18}  # exact cap
        _is_partition(out, 180)  # complete, disjoint patch assignment
        for name, cap in sizes:
            cnt = Counter(lab for i in out[name] for lab in sets[i])
            for lab, g in glob.items():
                assert abs(cnt[lab] - g * cap / 180) <= 1, (seed, name, lab, cnt[lab])  # meaningful bound
        out2 = P.multilabel_assign(sets, sizes, seed=seed)  # determinism
        assert all(np.array_equal(out[k], out2[k]) for k in out)
        memberships.append(tuple(tuple(out[k].tolist()) for k in ("train", "val", "test")))
    assert len(set(memberships)) > 1, "membership did not vary across run seeds"


def test_multilabel_handles_label_free_items():
    sets = [[] for _ in range(10)] + [[0, 1]] * 40
    out = P.multilabel_assign(sets, [("train", 40), ("val", 5), ("test", 5)], seed=0)
    _is_partition(out, 50)


# --------------------------------------------------------------------------- #
# Hard failure, never a silent fallback
# --------------------------------------------------------------------------- #
def test_wrong_total_size_raises():
    with pytest.raises(P.PartitionError, match="sum to"):
        P.multilabel_assign([[0], [1], [0]], [("train", 2), ("test", 2)], seed=0)


def test_duplicate_partition_name_raises():
    with pytest.raises(P.PartitionError, match="duplicate"):
        P.multilabel_assign([[0], [1]], [("a", 1), ("a", 1)], seed=0)
