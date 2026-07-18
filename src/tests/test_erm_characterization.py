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


def _dense_cache(root, *, folds=(1, 2, 3), patches=2, pixels=40, dim=5, classes=3):
    """A minimal on-disk dense cache: fold_<f>/<patch>_<r>_<c>{,.labels}.npy."""
    rng = np.random.default_rng(0)
    for fold in folds:
        d = root / f"fold_{fold}"
        d.mkdir(parents=True, exist_ok=True)
        for patch in range(patches):
            y = np.tile(np.arange(classes), pixels // classes + 1)[:pixels].astype(np.int64)
            x = (rng.normal(size=(pixels, dim)) + y[:, None] * 1.5).astype(np.float32)
            np.save(d / f"{patch}_0_0.npy", x)
            np.save(d / f"{patch}_0_0.labels.npy", y)
    return root


@pytest.fixture(scope="module")
def dense_rows(tmp_path_factory):
    from evals.regimes.base import DenseSplit
    from utils import runstate as RS

    emb_dir = _dense_cache(tmp_path_factory.mktemp("emb"))

    class FakeBenchMod:
        BENCHMARK = "pastis"
        LABEL_KIND = "segmentation"

    cfg = DenseSplit("fold_3", {1}, {2}, {3}, has_target=True, group_kind="geography")
    return RS._run_segmentation_cell(
        FakeBenchMod, emb_dir, cfg, 0, "logistic",
        [1.0], [0], 1000, _meta("source"),
    )


def test_dense_source_budget_contract(dense_rows) -> None:
    source = [r for r in dense_rows if r["budget_type"] == "source"]
    _assert_erm_contract(source, budget_type="source", tuning=False)

    # the dense source sweep scores BOTH validation and test -- unlike the tabular source sweep,
    # which scores test only
    assert len(source) == 2
    assert _cell_keys(source) == [(*K, "source", 1.0, "test"), (*K, "source", 1.0, "validation")]


def test_dense_source_metrics_are_deterministic(dense_rows) -> None:
    test_rows = [r for r in dense_rows
                 if r["budget_type"] == "source" and r["evaluation_split"] == "test"]

    assert len(test_rows) == 1
    r = test_rows[0]
    assert r["miou"] == pytest.approx(0.8432123432123432, abs=TOL)
    assert r["pixel_accuracy"] == pytest.approx(0.9125, abs=TOL)
    # PASTIS scores against its declared 19-class space, not the classes present in the fixture
    assert r["n_eval_classes"] == 19


def test_dense_target_budget_contract(dense_rows) -> None:
    target = [r for r in dense_rows if r["budget_type"] == "target"]
    _assert_erm_contract(target, budget_type="target", tuning=False)

    assert len(target) == 2
    assert _cell_keys(target) == [(*K, "target", 0, "full"), (*K, "target", 0, "held_out")]


def test_dense_target_metrics_are_deterministic(dense_rows) -> None:
    full = [r for r in dense_rows
            if r["budget_type"] == "target" and r["evaluation_split"] == "full"]

    assert len(full) == 1
    assert full[0]["miou"] == pytest.approx(0.8432123432123432, abs=TOL)


def test_dense_emits_both_budget_types_from_one_cell(dense_rows) -> None:
    """One dense cell runs the whole source+target sweep; the tabular path splits them."""
    assert len(dense_rows) == 4
    assert {r["budget_type"] for r in dense_rows} == {"source", "target"}
    assert all(r["method"] == "erm" for r in dense_rows)
