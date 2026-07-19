"""ERM characterization oracle -- the contract that removing post-hoc adaptation must preserve.

Post-hoc adaptation is threaded through the experimental core: the transform hook lives inside
`_sweep_budgets` / `_sweep_target_budgets` and the dense cell, which is where every paper number
is produced. ERM rides through that hook as a no-op (`erm` is the `cls=None` sentinel -> a `None`
transform -> an early return), so removing the hook SHOULD be invisible to ERM. Nothing tested
that, and "should be invisible" is not a property you take on faith when the thing it might move
is the paper's results.

So: six cases through the REAL sweep and dense paths -- both budget types x binary/multiclass
tabular + dense -- on small synthetic fixtures. No datasets, no encoders, no GPU.

WHAT THIS DOES AND DOES NOT PROVE. It pins the emitted CONTRACT -- the complete cell-key set,
exact row counts, evaluation scopes, method="erm", probe family, and the ERM metadata constants --
plus METRIC VALUES to an explicit tolerance. It does NOT compare serialized rows, so it is not a
byte-identity check: a change that added a new column, or altered a field this file does not name,
would pass. Contract + metric preservation is the claim; nothing stronger.

Pinned as literal expectations rather than a golden file, so a diff shows what changed rather than
only that something did.
"""

from __future__ import annotations

import numpy as np
import pytest

from evals import evals as EV

TOL = 1e-9  # sklearn is version-pinned, so ERM metrics are reproducible to ~float noise

#: Pinned deterministic metrics of the official dense cell (fit folds 1-3, zero-shot eval fold 5) on
#: the fixed synthetic cache -- characterization values (filled from the reproducible run below).
_OFFICIAL_DENSE_MIOU = 0.8382716049382717
_OFFICIAL_DENSE_PIXACC = 0.9125

#: Every field evals.erm_metadata() emits. Passed into meta by the orchestrator.
ERM_METADATA = {
    "method_class": "erm",
    "method_module": "",
    "method_kwargs": "{}",
    "method_uses_target": 0,
    "method_requires_coords": 0,
    "method_subset_only": 0,
    "method_fit_source_only": 0,
    "method_has_sample_weight": 0,
}

#: Fields the SWEEP injects on every row, from the method-variant selector. ERM has exactly one
#: variant and never tunes, so these are constants -- but they are on all 14,773 canonical rows,
#: so the selector's removal must keep emitting them verbatim or the schema changes underneath
#: the artifacts. (Not part of ERM_METADATA: the sweep adds these, the caller does not.)
ERM_TUNING_METADATA = {
    "method_tuned": 0,
    "method_n_variants": 1,
    "method_selected_variant": "default",
    "method_selected_kwargs": "{}",
    "method_selection_metric": "",
    "method_selection_scope": "none",
}

#: ERM never weights samples. The machinery goes; these columns must not.
ERM_WEIGHT_METADATA = ("probe_weight_ess", "probe_weight_max", "probe_weight_min", "probe_weight_std")


def _xy(n=80, d=6, *, classes=2, seed=0):
    """Linearly separable-ish synthetic data: probes converge, metrics are stable."""
    rng = np.random.default_rng(seed)
    y = np.tile(np.arange(classes), n // classes + 1)[:n]
    x = rng.normal(size=(n, d)) + y[:, None] * 1.5
    return x.astype(np.float64), y.astype(np.int64)


def _meta(budget_type: str) -> dict:
    return {
        "model": "raw", "benchmark": "fake", "method": "erm", **ERM_METADATA,
        "split_regime": "geographic_ood", "domain_basis": "geography",
        "holdout": "kenya", "probe_family": "logistic", "budget_type": budget_type,
    }


K = (0, "geographic_ood", "kenya", "erm", "logistic")  # the fixed prefix of every cell key


def _cell_keys(rows):
    return sorted(
        (r["seed"], r["split_regime"], r["holdout"], r["method"], r["probe_family"],
         r["budget_type"], r["label_budget"], r["evaluation_split"])
        for r in rows
    )


def _assert_erm_contract(rows, *, budget_type, family="logistic", tuning=True):
    """Invariants every ERM row must carry.

    `tuning` is False for the dense path: the variant-selector fields are injected by the TABULAR
    sweeps only, so dense rows have never carried them. That asymmetry is part of the schema the
    canonical artifacts already have, so the removal must reproduce it rather than tidy it.
    """
    assert rows, "the sweep emitted no rows"
    expected = {**ERM_METADATA, **ERM_TUNING_METADATA} if tuning else dict(ERM_METADATA)
    for r in rows:
        assert r["method"] == "erm"
        assert r["budget_type"] == budget_type
        assert r["probe_family"] == family
        for k, v in expected.items():
            assert k in r, f"ERM row lost the {k!r} column"
            assert r[k] == v, f"ERM metadata drifted: {k}={r[k]!r} != {v!r}"
        if tuning:
            assert np.isnan(r["method_selection_score"]), "method_selection_score must stay NaN"
        else:
            assert "method_tuned" not in r, "dense rows must not gain the tabular tuning fields"
        # ERM never weights samples; the zero/NaN companions must survive the removal even though
        # the weighting machinery does not.
        assert r["probe_sample_weighted"] == 0
        for k in ERM_WEIGHT_METADATA:
            assert k in r, f"ERM row lost the {k!r} column"
            assert np.isnan(r[k]), f"{k}={r[k]!r} -- must stay NaN for ERM"


# --- tabular binary ---------------------------------------------------------


@pytest.fixture(scope="module")
def binary_source():
    x_tr, y_tr = _xy(seed=0)
    x_te, y_te = _xy(n=40, seed=1)
    rows: list[dict] = []
    EV.run_probes(
        rows, x_tr, x_te, y_tr, y_te, seed=0, budgets=[0.5, 1.0], meta=_meta("source"),
        x_val=x_te, y_val=y_te, family="logistic",
    )
    return rows


def test_binary_source_budget_contract(binary_source) -> None:
    rows = binary_source
    _assert_erm_contract(rows, budget_type="source")

    assert len(rows) == 2
    assert _cell_keys(rows) == [(*K, "source", 0.5, "test"), (*K, "source", 1.0, "test")]


def test_binary_source_metrics_are_deterministic(binary_source) -> None:
    by_budget = {r["label_budget"]: r for r in binary_source}

    for budget in (0.5, 1.0):
        r = by_budget[budget]
        assert r["f1"] == pytest.approx(1.0, abs=TOL)
        assert r["auc"] == pytest.approx(1.0, abs=TOL)
        assert r["balanced_accuracy"] == pytest.approx(1.0, abs=TOL)
        assert 0.0 <= r["ece"] <= 1.0
        assert r["n_test"] == 40
        assert r["probe_family"] == "logistic"


@pytest.fixture(scope="module")
def binary_target():
    x_src, y_src = _xy(seed=0)
    x_tgt, y_tgt = _xy(n=60, seed=2)
    rows: list[dict] = []
    EV.run_probes_target(
        rows, x_src, x_tgt, y_src, y_tgt, seed=0, budgets=[0, 5, EV.TARGET_ID_UPPER_BOUND],
        meta=_meta("target"), family="logistic",
    )
    return rows


def test_binary_target_budget_contract(binary_target) -> None:
    rows = binary_target
    _assert_erm_contract(rows, budget_type="target")

    # The COMPLETE set, not a subset: budget 0 is scored on BOTH the whole target domain and the
    # held-out subset; every other budget only on held_out. That asymmetry is the deployment-OOD
    # estimand, and asserting the whole set is what catches a scope being silently added or lost.
    assert len(rows) == 4
    assert _cell_keys(rows) == [
        (*K, "target", -1, "held_out"),
        (*K, "target", 0, "full"),
        (*K, "target", 0, "held_out"),
        (*K, "target", 5, "held_out"),
    ]


def test_binary_target_metrics_are_deterministic(binary_target) -> None:
    full = [r for r in binary_target if r["label_budget"] == 0 and r["evaluation_split"] == "full"]

    assert len(full) == 1
    # RECORDED from the current code, not derived -- the point is exact reproduction after the
    # transform hook is removed, whatever the value happens to be.
    assert full[0]["f1"] == pytest.approx(0.9655172413793104, abs=TOL)
    assert full[0]["n_test"] == 60


# --- tabular multiclass -----------------------------------------------------


@pytest.fixture(scope="module")
def multiclass_source():
    x_tr, y_tr = _xy(n=90, classes=3, seed=0)
    x_te, y_te = _xy(n=45, classes=3, seed=1)
    rows: list[dict] = []
    EV.run_probes_multiclass(
        rows, x_tr, x_te, y_tr, y_te, seed=0, budgets=[1.0], meta=_meta("source"),
        x_val=x_te, y_val=y_te, family="logistic",
    )
    return rows


def test_multiclass_source_budget_contract(multiclass_source) -> None:
    _assert_erm_contract(multiclass_source, budget_type="source")

    assert len(multiclass_source) == 1
    assert _cell_keys(multiclass_source) == [(*K, "source", 1.0, "test")]


def test_multiclass_source_metrics_are_deterministic(multiclass_source) -> None:
    r = multiclass_source[0]

    assert r["macro_f1"] == pytest.approx(1.0, abs=TOL)
    assert r["accuracy"] == pytest.approx(1.0, abs=TOL)
    assert r["n_test"] == 45
    assert r["n_classes_seen"] == 3
    # the calibration correction's support diagnostic
    assert r["unseen_prevalence"] == pytest.approx(0.0, abs=TOL)


@pytest.fixture(scope="module")
def multiclass_target():
    x_src, y_src = _xy(n=90, classes=3, seed=0)
    x_tgt, y_tgt = _xy(n=60, classes=3, seed=2)
    rows: list[dict] = []
    EV.run_probes_multiclass_target(
        rows, x_src, x_tgt, y_src, y_tgt, seed=0, budgets=[0, 10],
        meta=_meta("target"), family="logistic",
    )
    return rows


def test_multiclass_target_budget_contract(multiclass_target) -> None:
    _assert_erm_contract(multiclass_target, budget_type="target")

    assert len(multiclass_target) == 3
    assert _cell_keys(multiclass_target) == [
        (*K, "target", 0, "full"),
        (*K, "target", 0, "held_out"),
        (*K, "target", 10, "held_out"),
    ]


def test_multiclass_target_metrics_are_deterministic(multiclass_target) -> None:
    full = [r for r in multiclass_target if r["label_budget"] == 0 and r["evaluation_split"] == "full"]

    assert len(full) == 1
    assert full[0]["macro_f1"] == pytest.approx(0.9665831244778613, abs=TOL)
    assert full[0]["n_test"] == 60


# --- dense / PASTIS ---------------------------------------------------------


def _dense_cache(root, *, folds=(1, 2, 3, 4, 5), patches_per_fold=2, pixels=40, dim=5, classes=3):
    """A minimal on-disk dense cache with DISTINCT patch ids per fold (as real PASTIS -- every patch
    lives in exactly one fold). Returns ``(root, {fold: [patch_id, ...]})``."""
    rng = np.random.default_rng(0)
    fold_patches: dict[int, list[int]] = {}
    for fold in folds:
        d = root / f"fold_{fold}"
        d.mkdir(parents=True, exist_ok=True)
        pids = []
        for k in range(patches_per_fold):
            pid = fold * 10 + k
            y = np.tile(np.arange(classes), pixels // classes + 1)[:pixels].astype(np.int64)
            x = (rng.normal(size=(pixels, dim)) + y[:, None] * 1.5).astype(np.float32)
            np.save(d / f"{pid}_0_0.npy", x)
            np.save(d / f"{pid}_0_0.labels.npy", y)
            pids.append(pid)
        fold_patches[fold] = pids
    return root, fold_patches


@pytest.fixture(scope="module")
def official_dense_rows(tmp_path_factory):
    """The schema-v2 OFFICIAL dense cell (has_target=True, supports_target_labels=False): fit on the
    fold-1-3 source_train patches, calibrate on the fold-4 source_val patches, and evaluate ZERO-SHOT
    on the fold-5 target_test patches -- source budgets only, NO target sweep. Executable now."""
    from evals.regimes.base import DenseSourceTargetSplit
    from utils import runstate as RS

    emb_dir, fp = _dense_cache(tmp_path_factory.mktemp("emb_official"))

    class FakeBenchMod:
        BENCHMARK = "pastis"
        LABEL_KIND = "segmentation"

    cfg = DenseSourceTargetSplit(
        label="fold_5",
        source_train_patches=frozenset(fp[1] + fp[2] + fp[3]),
        source_val_patches=frozenset(fp[4]),
        source_test_patches=frozenset(),
        target_label_pool_patches=frozenset(),
        target_test_patches=frozenset(fp[5]),
        has_target=True, supports_target_labels=False,
    )
    return RS._run_segmentation_cell(
        FakeBenchMod, emb_dir, cfg, 0, "logistic", [1.0], [0], 1000, _meta("source"),
        all_folds={1, 2, 3, 4, 5},
    )


@pytest.fixture(scope="module")
def target_dense_rows(tmp_path_factory):
    """The supports_target_labels=True dense target-budget route (geographic/spatial): the target
    budget draws few-shot patches ONLY from target_label_pool and is scored on the fixed target_test
    -- fold-3 tile held out here, fold-5 pool/test. Executable now that the route is wired."""
    from evals.regimes.base import DenseSourceTargetSplit
    from utils import runstate as RS

    emb_dir, fp = _dense_cache(tmp_path_factory.mktemp("emb_target"))

    class FakeBenchMod:
        BENCHMARK = "pastis"
        LABEL_KIND = "segmentation"

    # target_label_pool (fp[5][:1]) and target_test (fp[5][1:]) are the FROZEN 80/20 split; the sweep
    # must draw few-shot ONLY from the pool and score on target_test -- never labels from target_test.
    cfg = DenseSourceTargetSplit(
        label="fold_5",
        source_train_patches=frozenset(fp[1] + fp[2] + fp[3]), source_val_patches=frozenset(fp[4]),
        source_test_patches=frozenset(),
        target_label_pool_patches=frozenset(fp[5][:1]), target_test_patches=frozenset(fp[5][1:]),
        has_target=True, supports_target_labels=True,
    )
    return RS._run_segmentation_cell(
        FakeBenchMod, emb_dir, cfg, 0, "logistic", [1.0], [0], 1000, _meta("target"),
        all_folds={1, 2, 3, 4, 5},
    )


def test_dense_source_budget_contract(official_dense_rows) -> None:
    source = [r for r in official_dense_rows if r["budget_type"] == "source"]
    _assert_erm_contract(source, budget_type="source", tuning=False)

    # the dense source sweep scores BOTH validation and test (here test == the zero-shot target_test)
    # -- unlike the tabular source sweep, which scores test only
    assert len(source) == 2
    assert _cell_keys(source) == [(*K, "source", 1.0, "test"), (*K, "source", 1.0, "validation")]


def test_dense_source_metrics_are_deterministic(official_dense_rows) -> None:
    test_rows = [r for r in official_dense_rows
                 if r["budget_type"] == "source" and r["evaluation_split"] == "test"]

    assert len(test_rows) == 1
    r = test_rows[0]
    assert r["miou"] == pytest.approx(_OFFICIAL_DENSE_MIOU, abs=TOL)
    assert r["pixel_accuracy"] == pytest.approx(_OFFICIAL_DENSE_PIXACC, abs=TOL)
    # PASTIS scores against its declared 19-class space, not the classes present in the fixture
    assert r["n_eval_classes"] == 19


def test_dense_official_is_zero_shot_no_target_budget(official_dense_rows) -> None:
    """official (supports_target_labels=False) emits ONLY source budgets -- no target sweep."""
    assert {r["budget_type"] for r in official_dense_rows} == {"source"}
    assert all(r["method"] == "erm" for r in official_dense_rows)


def test_dense_target_budget_contract(target_dense_rows) -> None:
    target = [r for r in target_dense_rows if r["budget_type"] == "target"]
    _assert_erm_contract(target, budget_type="target", tuning=False)

    assert len(target) == 2
    assert _cell_keys(target) == [(*K, "target", 0, "full"), (*K, "target", 0, "held_out")]


def test_dense_target_metrics_are_deterministic(target_dense_rows) -> None:
    full = [r for r in target_dense_rows
            if r["budget_type"] == "target" and r["evaluation_split"] == "full"]

    assert len(full) == 1
    assert full[0]["miou"] == pytest.approx(_OFFICIAL_DENSE_MIOU, abs=TOL)


# --- Defect 2: source_test is the untouched within-source reference (evaluated, never trained on) ---


def test_tabular_source_test_scope_uses_exact_partition_ids_not_trained_on() -> None:
    """The 80/10/10 source_test partition is evaluated as its OWN 'source_test' scope, on EXACTLY its
    manifest IDs, and is never part of the training/calibration sets."""
    from utils import runstate as RS

    n = 24
    y = np.array([i % 2 for i in range(n)], dtype=np.int64)
    emb = np.zeros((n, 4), dtype=np.float32)
    emb[y == 1] = 1.0
    groups = np.array(["s"] * n, dtype=object)
    train = np.arange(0, 12)
    val = np.arange(12, 16)
    target_test = np.arange(16, 20)   # the primary (OOD) eval -> "test" scope
    source_test = np.arange(20, 24)   # the untouched within-source reference -> "source_test" scope

    rows, preds = RS._probe_cell(
        EV.run_probes, emb, train, val, target_test, y, groups,
        {"benchmark": "t", "method": "erm", "holdout": "kenya", "seed": 0}, 0, "logistic", [1.0],
        None, source_test, write_predictions=True,
    )
    src_test_rows = [r for r in rows if r["evaluation_split"] == "source_test"]
    assert len(src_test_rows) == 1 and src_test_rows[0]["n_test"] == len(source_test)
    # predictions on the source_test scope cover EXACTLY the source_test partition IDs
    st_pred_ids = {int(p["sample_id"]) for p in preds if p.get("evaluation_split") == "source_test"}
    assert st_pred_ids == set(source_test.tolist())
    # and it is NOT the target_test set (the primary OOD eval)
    assert st_pred_ids.isdisjoint(target_test.tolist())


def test_dense_source_test_scope_streams_exactly_the_source_test_patches(monkeypatch, tmp_path) -> None:
    """The dense source_test scope streams EXACTLY the manifest source_test patches (never trained on)."""
    from evals.regimes.base import DenseSourceTargetSplit
    from utils import cacheutils
    from utils import runstate as RS

    emb_dir, fp = _dense_cache(tmp_path / "emb")
    # geographic-style cell: source folds 1-3 (train), fold 4 source_val, fold 5 source_test AND target_test
    cfg = DenseSourceTargetSplit(
        label="T31TFM",
        source_train_patches=frozenset(fp[1] + fp[2]),
        source_val_patches=frozenset(fp[3]),
        source_test_patches=frozenset(fp[4]),          # the within-source reference
        target_label_pool_patches=frozenset(fp[5][:1]),
        target_test_patches=frozenset(fp[5][1:]),
        has_target=True, supports_target_labels=True,
    )

    streamed: list[frozenset] = []
    real = cacheutils.iter_dense_tiles

    def spy(emb, folds, patch_ids=None):
        streamed.append(frozenset(int(p) for p in (patch_ids or ())))
        return real(emb, folds, patch_ids=patch_ids)

    monkeypatch.setattr(cacheutils, "iter_dense_tiles", spy)

    class FakeBenchMod:
        BENCHMARK = "pastis"
        LABEL_KIND = "segmentation"

    rows = RS._run_segmentation_cell(
        FakeBenchMod, emb_dir, cfg, 0, "logistic", [1.0], [0], 1000, _meta("source"),
        all_folds={1, 2, 3, 4, 5},
    )
    # a source_test-scope row was emitted, and its stream got EXACTLY the source_test patches
    assert any(r["budget_type"] == "source" and r["evaluation_split"] == "source_test" for r in rows)
    assert frozenset(fp[4]) in streamed
