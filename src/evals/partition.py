"""Deterministic, exact-size constrained partitioners for TERRA split construction.

Three primitives, all deterministic (seeded by :class:`numpy.random.SeedSequence` over explicit
integer operation codes -- no hashing) and all hitting EXACT partition sizes with NO unstratified
fallback:

* :func:`partition_target` -- the 80/20 target split. A class with a single target example goes to
  ``target_test`` (unsupported by a locally-trained probe); every class with >=2 examples occurs in
  BOTH ``target_label_pool`` and ``target_test``.

* :func:`partition_source` -- the 80/10/10 source split. Class and region marginals are constrained
  INDEPENDENTLY via ONE joint integer program (``scipy.optimize.milp``) over ``x[class, region,
  partition]`` -- exact cell totals and capacities, with every class-by-partition and
  region-by-partition count held to floor/ceil of its proportional quota. A single joint solve
  avoids both the greedy-multilabel starvation of a marginal and the rejection of globally feasible
  splits that independent rounding + sequential fills produce.

* :func:`multilabel_assign` -- iterative multilabel stratification (rare-label first) over per-item
  label-presence sets, for PASTIS-R patches (a patch, and all its pixels, land wholly in one
  partition). Used only where an item carries a SET of labels; the single-attribute source split
  uses the exact MILP above.

Membership varies with ``seed``; sizes never do. Nothing here reads data, hashes, or touches disk.
"""

from __future__ import annotations

import math
from collections import Counter, defaultdict
from collections.abc import Sequence
from typing import Any

import numpy as np
from scipy.optimize import Bounds, LinearConstraint, milp  # type: ignore[import-untyped]  # no stubs

#: Stable operation codes -- entropy inputs to SeedSequence. NEVER renumber or reuse a value; a new
#: randomized operation gets a new code. This is the transparent replacement for hashed sub-seeds.
_OP_SYSTEMATIC_SLOT = 1   # per-stratum slot shuffle in the systematic single-attribute order
_OP_ML_ITEM = 2           # per-item partition-priority vector (multilabel tie-breaks + item order)
_OP_SOURCE_CELL = 3       # per (class,region) cell item shuffle in the source split

#: Declared numerical tolerance for holding the stage-one joint-cell deviation optimum in stage two.
_SOURCE_DEV_TOL = 1e-6

TARGET_LABEL_POOL = "target_label_pool"
TARGET_TEST = "target_test"


class PartitionError(RuntimeError):
    """A requested constrained partition is infeasible (raised instead of any silent fallback)."""


def _rng(seed: int, op: int, *key_ints: int) -> np.random.Generator:
    """Deterministic Generator seeded transparently by (run seed, op code, integer keys)."""
    return np.random.default_rng(np.random.SeedSequence([int(seed), int(op), *(int(k) for k in key_ints)]))


def _check_sizes(n: int, sizes: Sequence[tuple[str, int]]) -> None:
    names = [name for name, _ in sizes]
    if len(names) != len(set(names)):
        raise PartitionError(f"duplicate partition name in sizes: {names}")
    if any(sz < 0 for _, sz in sizes):
        raise PartitionError(f"negative partition size in {sizes}")
    total = sum(int(sz) for _, sz in sizes)
    if total != n:
        raise PartitionError(f"partition sizes sum to {total} but there are {n} items")


def _systematic_order(keys: Sequence[str], seed: int, op: int) -> list[int]:
    """Global item order that spreads every stratum uniformly (systematic fractional-slot sampling).

    Each stratum's items are shuffled (seed-dependent) then placed at positions ``(slot+0.5)/count``
    in ``[0,1)``; sorting by that interleaves strata evenly, so any contiguous block samples each
    stratum within +/-1 of its proportional share. Stratum shuffle order is keyed by the stratum's
    integer RANK in sorted order (no hashing).
    """
    n = len(keys)
    by: dict[str, list[int]] = defaultdict(list)
    for i, k in enumerate(keys):
        by[k].append(i)
    order_key: dict[int, tuple[float, int, int]] = {}
    for rank, k in enumerate(sorted(by)):
        items = by[k]
        c = len(items)
        perm = _rng(seed, op, rank).permutation(c)
        for slot, p in enumerate(perm.tolist()):
            order_key[items[p]] = ((slot + 0.5) / c, rank, items[p])
    return sorted(range(n), key=lambda i: order_key[i])


# --------------------------------------------------------------------------- #
# Target 80/20 split (single class attribute)
# --------------------------------------------------------------------------- #
def partition_target(
    classes: Sequence[Any], pool_size: int, test_size: int, seed: int,
) -> dict[str, np.ndarray]:
    """Split target examples into ``target_label_pool`` (size ``pool_size``) and ``target_test``.

    Guarantees, at exact sizes: (1) a class with exactly one target example is in ``target_test``;
    (2) every class with >=2 examples occurs in BOTH partitions. Raises :class:`PartitionError` if
    that is infeasible (e.g. more singleton classes than ``test_size``).
    """
    classes = [str(c) for c in classes]
    n = len(classes)
    _check_sizes(n, [(TARGET_LABEL_POOL, pool_size), (TARGET_TEST, test_size)])

    order = _systematic_order(classes, seed, _OP_SYSTEMATIC_SLOT)
    pool = set(order[:pool_size])
    test = set(order[pool_size:])
    counts = Counter(classes)
    class_of = {i: classes[i] for i in range(n)}

    def in_test_of(c: str) -> set[int]:
        return {i for i in test if class_of[i] == c}

    def in_pool_of(c: str) -> set[int]:
        return {i for i in pool if class_of[i] == c}

    def swap(i_pool: int, j_test: int) -> None:
        pool.discard(i_pool)
        pool.add(j_test)
        test.discard(j_test)
        test.add(i_pool)

    # (1) singletons -> test. Move each misplaced singleton pool->test, returning a non-singleton
    # test item (whose class keeps >=1 in test) to the pool. Size-neutral.
    singleton_classes = {c for c, k in counts.items() if k == 1}
    if len(singleton_classes) > test_size:
        raise PartitionError(
            f"{len(singleton_classes)} singleton target classes cannot all fit in target_test (size {test_size})"
        )
    for i in sorted(i for i in list(pool) if class_of[i] in singleton_classes):
        donors = sorted(
            j for j in test
            if class_of[j] not in singleton_classes and len(in_test_of(class_of[j])) >= 2
        )
        if not donors:
            raise PartitionError("cannot move a singleton class into target_test without displacing a supported class")
        swap(i, donors[0])

    # (2) every >=2 class present in both partitions.
    for c, k in sorted(counts.items()):
        if k < 2:
            continue
        if not in_pool_of(c):  # move one c-item test->pool
            movers = sorted(in_test_of(c))
            donors = sorted(j for j in pool if len(in_pool_of(class_of[j])) >= 2 and class_of[j] != c)
            if len(in_test_of(c)) < 2 or not donors:
                raise PartitionError(f"cannot place class {c!r} into target_label_pool at exact size")
            swap(donors[0], movers[0])
        if not in_test_of(c):  # move one c-item pool->test
            movers = sorted(in_pool_of(c))
            donors = sorted(
                j for j in test
                if class_of[j] not in singleton_classes and len(in_test_of(class_of[j])) >= 2 and class_of[j] != c
            )
            if len(in_pool_of(c)) < 2 or not donors:
                raise PartitionError(f"cannot place class {c!r} into target_test at exact size")
            swap(movers[0], donors[0])

    return {
        TARGET_LABEL_POOL: np.sort(np.fromiter(pool, dtype=np.int64, count=len(pool))),
        TARGET_TEST: np.sort(np.fromiter(test, dtype=np.int64, count=len(test))),
    }


# --------------------------------------------------------------------------- #
# Iterative multilabel stratification
# --------------------------------------------------------------------------- #
def multilabel_assign(
    label_sets: Sequence[Sequence[Any]],
    sizes: Sequence[tuple[str, int]],
    seed: int,
) -> dict[str, np.ndarray]:
    """Iterative multilabel stratification over per-item label-presence sets (Sechidis et al.).

    Hits each partition's EXACT capacity while balancing every label's marginal across partitions.
    Deterministic in ``seed`` (which sets a per-item partition-priority vector for tie-breaks and
    item ordering). Labels may be any hashable; their integer RANK in sorted order feeds the seeding,
    so nothing is hashed.
    """
    n = len(label_sets)
    _check_sizes(n, sizes)
    names = [name for name, _ in sizes]
    part_index = {name: k for k, name in enumerate(names)}
    cap = {name: int(sz) for name, sz in sizes}
    remaining_cap = dict(cap)
    sets = [set(ls) for ls in label_sets]

    label_items: dict[Any, set[int]] = defaultdict(set)
    for i, s in enumerate(sets):
        for lab in s:
            label_items[lab].add(i)
    label_rank = {lab: r for r, lab in enumerate(sorted(label_items, key=str))}
    desired: dict[str, dict[Any, float]] = {
        name: {lab: len(items) * cap[name] / n for lab, items in label_items.items()}
        for name in names
    }
    #: seed-varying but deterministic per-item priority over partitions (tie-breaks + item order).
    prio = {i: _rng(seed, _OP_ML_ITEM, i).random(len(names)) for i in range(n)}

    assignment: dict[int, str] = {}
    unassigned: set[int] = set(range(n))
    label_remaining: dict[Any, set[int]] = {lab: set(items) for lab, items in label_items.items()}

    def place(item: int, part: str) -> None:
        assignment[item] = part
        unassigned.discard(item)
        remaining_cap[part] -= 1
        for lab in sets[item]:
            label_remaining[lab].discard(item)
            desired[part][lab] -= 1

    def best_partition(item: int, current: Any) -> str:
        feasible = [p for p in names if remaining_cap[p] > 0]
        if not feasible:
            raise PartitionError("multilabel assignment ran out of capacity before placing all items")
        # Sechidis rule: the partition that most needs the CURRENT rare label (its remaining desired
        # count for THIS label) -- NOT the sum of desired counts over every label on the item, which
        # dilutes the rare label's placement. Ties: overall desire, spare capacity, seeded tie-break.
        return max(feasible, key=lambda p: (
            desired[p][current],
            sum(desired[p][lab] for lab in sets[item]),
            remaining_cap[p],
            prio[item][part_index[p]],
        ))

    # rarest label first (tie-break by stable label rank); place its items (seed-varying order).
    while any(label_remaining[lab] for lab in label_remaining):
        lab = min(
            (lb for lb in label_remaining if label_remaining[lb]),
            key=lambda lb: (len(label_remaining[lb]), label_rank[lb]),
        )
        for item in sorted(label_remaining[lab], key=lambda i: (prio[i][0], i)):
            if item in unassigned:
                place(item, best_partition(item, lab))

    for item in sorted(unassigned, key=lambda i: (prio[i][0], i)):  # label-free leftovers
        feasible = [p for p in names if remaining_cap[p] > 0]
        place(item, max(feasible, key=lambda p: (remaining_cap[p], prio[item][part_index[p]])))

    out: dict[str, list[int]] = {name: [] for name in names}
    for item, part in assignment.items():
        out[part].append(item)
    for name in names:
        if len(out[name]) != cap[name]:
            raise PartitionError(f"multilabel partition {name!r} got {len(out[name])} != capacity {cap[name]}")
    return {name: np.sort(np.asarray(idx, dtype=np.int64)) for name, idx in out.items()}


# --------------------------------------------------------------------------- #
# Source 80/10/10 split (class + region marginals, constrained INDEPENDENTLY)
# --------------------------------------------------------------------------- #
def _quota_bounds(n_v: int, cap: int, n_total: int) -> tuple[int, int]:
    """The marginal contract: a value with global count ``n_v`` gets ``floor`` or ``ceil`` of its
    proportional quota ``n_v*cap/n_total`` in a partition of size ``cap``."""
    q = n_v * cap / n_total
    return math.floor(q), math.ceil(q)


def _assert_marginal_contract(
    assign: dict[str, list[int]], value_of: dict[int, Any], counts: dict[Any, int], n_total: int, attr: str,
) -> None:
    for p, idx in assign.items():
        cap = len(idx)
        cnt = Counter(value_of[i] for i in idx)
        for v, nv in counts.items():
            lo, hi = _quota_bounds(nv, cap, n_total)
            got = cnt.get(v, 0)
            if not (lo <= got <= hi):
                raise PartitionError(
                    f"{attr} marginal contract violated: {v!r} count {got} in {p!r} outside [{lo},{hi}]"
                )


def _source_tiebreak_coeffs(ncell: int, nparts: int) -> np.ndarray:
    """SECONDARY non-separable tie-break ``c[k*nparts+p] = k*p`` over ``x``.

    Applied ONLY in stage 2, after the joint-cell deviation is minimized, to pick a deterministic
    allocation among the deviation-optimal set. It is deliberately non-separable: a separable
    ``f(k)+g(p)`` (e.g. the variable index) is CONSTANT over the feasible set (it collapses to fixed
    cell totals and capacities) and gives no tie-break, whereas ``k*p`` depends on how each cell
    splits across partitions, so it genuinely distinguishes allocations. It is NOT the primary
    objective -- as a primary it would segregate joint cells (push whole cells into single
    partitions); deviation minimization is the primary and prevents that.
    """
    return np.array([float(k * p) for k in range(ncell) for p in range(nparts)], dtype=float)


def _source_hard_rows(
    cells: list[tuple[str, str]], cell_count: dict[tuple[str, str], int],
    caps: list[int], n_c: dict[str, int], n_r: dict[str, int], n_total: int, nvar: int, xi,
) -> tuple[list[list[float]], list[float], list[float]]:
    """The hard-constraint rows (on ``x`` variables): cell completeness, exact capacity, and
    floor/ceil class and region marginals."""
    nparts = len(caps)
    ncell = len(cells)
    rows: list[list[float]] = []
    lo: list[float] = []
    hi: list[float] = []

    def add(row: list[float], lb: float, ub: float) -> None:
        rows.append(row)
        lo.append(lb)
        hi.append(ub)

    for k, cell in enumerate(cells):  # cell completeness (equality)
        row = [0.0] * nvar
        for p in range(nparts):
            row[xi(k, p)] = 1.0
        add(row, cell_count[cell], cell_count[cell])
    for p in range(nparts):  # exact partition capacity (equality)
        row = [0.0] * nvar
        for k in range(ncell):
            row[xi(k, p)] = 1.0
        add(row, caps[p], caps[p])
    for values, key in ((n_c, 0), (n_r, 1)):  # class / region floor/ceil quotas
        for v in sorted(values):
            for p in range(nparts):
                row = [0.0] * nvar
                for k, cell in enumerate(cells):
                    if cell[key] == v:
                        row[xi(k, p)] = 1.0
                f, cl = _quota_bounds(values[v], caps[p], n_total)
                add(row, f, cl)
    return rows, lo, hi


def _solve_source_allocation(
    cells: list[tuple[str, str]],
    cell_count: dict[tuple[str, str], int],
    part_names: list[str],
    caps: list[int],
    n_c: dict[str, int],
    n_r: dict[str, int],
    n_total: int,
) -> dict[tuple[str, str], list[int]]:
    """Joint integer allocation ``x[cell, partition]`` via a two-stage ``scipy.optimize.milp`` solve.

    HARD constraints (both stages): every cell allocated completely; every partition its exact
    capacity; every class-by-partition and region-by-partition count within floor/ceil of its
    proportional quota; ``0 <= x <= cell_count`` integer. Joint-cell proportionality is a SOFT
    objective (so a sparse cell cannot make an otherwise-valid split infeasible):

      stage 1 -- minimize total absolute deviation ``sum |x[cell,p] - cell_count[cell]*cap_p/N|``,
                 linearized with continuous deviation variables ``d >= |x - target|``;
      stage 2 -- hold that deviation optimum fixed and minimize a deterministic non-separable
                 secondary tie-break (:func:`_source_tiebreak_coeffs`) to pick one allocation.

    This spreads every joint cell proportionally across partitions instead of segregating cells into
    single partitions (the failure mode of using the ``k*p`` tie-break as the primary objective).
    Returns ``x[cell] = [count per partition]``. Raises :class:`PartitionError` only when the complete
    joint system is genuinely infeasible.
    """
    nparts = len(part_names)
    ncell = len(cells)
    nx = ncell * nparts
    nvar = 2 * nx  # x (integer) then d (continuous deviation)

    def xi(k: int, p: int) -> int:
        return k * nparts + p

    def di(k: int, p: int) -> int:
        return nx + k * nparts + p

    rows, lo, hi = _source_hard_rows(cells, cell_count, caps, n_c, n_r, n_total, nvar, xi)
    # deviation linearization: d >= x - target  and  d >= target - x
    for k, cell in enumerate(cells):
        for p in range(nparts):
            target = cell_count[cell] * caps[p] / n_total
            r1 = [0.0] * nvar  # d - x >= -target
            r1[di(k, p)] = 1.0
            r1[xi(k, p)] = -1.0
            rows.append(r1)
            lo.append(-target)
            hi.append(np.inf)
            r2 = [0.0] * nvar  # d + x >= target
            r2[di(k, p)] = 1.0
            r2[xi(k, p)] = 1.0
            rows.append(r2)
            lo.append(target)
            hi.append(np.inf)

    hard = LinearConstraint(np.asarray(rows, dtype=float), np.asarray(lo, float), np.asarray(hi, float))
    ub = np.zeros(nvar)
    for k, cell in enumerate(cells):
        for p in range(nparts):
            ub[xi(k, p)] = cell_count[cell]
            ub[di(k, p)] = cell_count[cell]
    bounds = Bounds(lb=np.zeros(nvar), ub=ub)
    integrality = np.zeros(nvar)
    for k in range(ncell):
        for p in range(nparts):
            integrality[xi(k, p)] = 1  # x integer; d continuous

    # STAGE 1: minimize total joint-cell deviation.
    c1 = np.zeros(nvar)
    for k in range(ncell):
        for p in range(nparts):
            c1[di(k, p)] = 1.0
    res1 = milp(c=c1, constraints=hard, integrality=integrality, bounds=bounds)
    if not res1.success or res1.x is None:
        raise PartitionError(
            f"source split is infeasible under the joint class+region marginal contract ({res1.message})"
        )

    # STAGE 2: hold deviation at its optimum, then apply the non-separable secondary tie-break.
    dev_row = [0.0] * nvar
    for k in range(ncell):
        for p in range(nparts):
            dev_row[di(k, p)] = 1.0
    dev_bound = float(res1.fun) + _SOURCE_DEV_TOL
    dev_cap = LinearConstraint(np.asarray([dev_row], float), np.asarray([-np.inf]), np.asarray([dev_bound]))
    c2 = np.zeros(nvar)
    tb = _source_tiebreak_coeffs(ncell, nparts)
    for k in range(ncell):
        for p in range(nparts):
            c2[xi(k, p)] = tb[k * nparts + p]
    res2 = milp(c=c2, constraints=[hard, dev_cap], integrality=integrality, bounds=bounds)
    if not res2.success or res2.x is None:
        # Stage one proves the quality optimum but gives no deterministic secondary selection;
        # silently returning it would violate the deterministic split contract. Fail loudly instead.
        raise PartitionError(
            f"source split stage-two deterministic tie-break failed (status {res2.status}): {res2.message}"
        )
    x = np.rint(res2.x[:nx]).astype(int)
    alloc = {cell: [int(x[xi(k, p)]) for p in range(nparts)] for k, cell in enumerate(cells)}
    _validate_source_allocation(alloc, cells, cell_count, caps, n_c, n_r, n_total, dev_bound)
    return alloc


def _validate_source_allocation(
    alloc: dict[tuple[str, str], list[int]],
    cells: list[tuple[str, str]],
    cell_count: dict[tuple[str, str], int],
    caps: list[int],
    n_c: dict[str, int],
    n_r: dict[str, int],
    n_total: int,
    dev_bound: float,
) -> None:
    """Post-validate the stage-two allocation against every hard constraint plus the stage-one
    deviation bound. Raises :class:`PartitionError` on any violation (never a silent accept)."""
    nparts = len(caps)
    for cell in cells:  # complete joint-cell allocation
        if sum(alloc[cell]) != cell_count[cell]:
            raise PartitionError(f"source allocation: cell {cell!r} sums to {sum(alloc[cell])} != {cell_count[cell]}")
    for p in range(nparts):  # exact partition capacity
        got = sum(alloc[cell][p] for cell in cells)
        if got != caps[p]:
            raise PartitionError(f"source allocation: partition {p} holds {got} != capacity {caps[p]}")
    for values, key in ((n_c, 0), (n_r, 1)):  # class and region marginal bounds
        for v, nv in values.items():
            for p in range(nparts):
                lo, hi = _quota_bounds(nv, caps[p], n_total)
                got = sum(alloc[cell][p] for cell in cells if cell[key] == v)
                if not (lo <= got <= hi):
                    raise PartitionError(f"source allocation: {v!r} count {got} in partition {p} outside [{lo},{hi}]")
    total_dev = sum(
        abs(alloc[cell][p] - cell_count[cell] * caps[p] / n_total)
        for cell in cells for p in range(nparts)
    )
    if total_dev > dev_bound:
        raise PartitionError(
            f"source allocation: total joint-cell deviation {total_dev} exceeds stage-one optimum bound {dev_bound}"
        )


def partition_source(
    classes: Sequence[Any],
    regions: Sequence[Any],
    sizes: Sequence[tuple[str, int]],
    seed: int,
) -> dict[str, np.ndarray]:
    """Exact-size source split enforcing the class AND region marginal contracts INDEPENDENTLY.

    A single joint integer program (:func:`_solve_source_allocation`) decides the per-``(class,
    region)``-cell count for each partition, so both marginals are constrained to floor/ceil of their
    proportional quota at exact capacities in ONE system -- neither the +1-over-integer-quota bug of
    independent rounding nor the val-then-test starvation of a sequential fill can occur, and a
    globally feasible split is never rejected. The count solution is deterministic; ``seed`` only
    shuffles WHICH ids within a cell take each partition's slots (membership varies, marginals do
    not). Every constraint is POST-VALIDATED; a genuinely infeasible system raises
    :class:`PartitionError`.
    """
    classes = [str(c) for c in classes]
    regions = [str(r) for r in regions]
    if len(classes) != len(regions):
        raise PartitionError(f"classes ({len(classes)}) and regions ({len(regions)}) length mismatch")
    n = len(classes)
    _check_sizes(n, sizes)
    part_names = [name for name, _ in sizes]
    caps = [int(sz) for _, sz in sizes]
    n_c = dict(Counter(classes))
    n_r = dict(Counter(regions))

    cell_items: dict[tuple[str, str], list[int]] = defaultdict(list)
    for i, (c, r) in enumerate(zip(classes, regions, strict=True)):
        cell_items[(c, r)].append(i)
    cells = sorted(cell_items)
    cell_count = {cell: len(cell_items[cell]) for cell in cells}

    x = _solve_source_allocation(cells, cell_count, part_names, caps, n_c, n_r, n)

    assign: dict[str, list[int]] = {name: [] for name in part_names}
    for k, cell in enumerate(cells):
        items = cell_items[cell]
        perm = _rng(seed, _OP_SOURCE_CELL, k).permutation(len(items))  # seed-dependent membership
        shuffled = [items[j] for j in perm.tolist()]
        pos = 0
        for p, name in enumerate(part_names):
            take = x[cell][p]
            assign[name].extend(shuffled[pos:pos + take])
            pos += take
        if pos != len(items):
            raise PartitionError(f"cell {cell!r} allocation summed to {pos} != {len(items)}")

    for name, cap in zip(part_names, caps, strict=True):
        if len(assign[name]) != cap:
            raise PartitionError(f"source partition {name!r} got {len(assign[name])} != capacity {cap}")
    class_of = {i: classes[i] for i in range(n)}
    region_of = {i: regions[i] for i in range(n)}
    _assert_marginal_contract(assign, class_of, n_c, n, "class")
    _assert_marginal_contract(assign, region_of, n_r, n, "region")
    return {name: np.sort(np.asarray(idx, dtype=np.int64)) for name, idx in assign.items()}
