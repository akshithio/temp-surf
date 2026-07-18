"""P0-5: leave-one-domain-out over the full canonical domain census.

The regime previously evaluated CropHarvest geographically with a SINGLE spatial-block split,
while a hard-coded five-region `HOLDOUTS` list sat unused for `geographic_ood`. Both understated
the domain universe: the benchmark carries 18 canonical domains, and the curated five cover only
15% of its samples. These tests pin the replacement:

  * the universe comes from the data, not from a literal in the benchmark module;
  * one-class domains are evaluated rather than silently dropped, and lose only the metrics that
    mathematically require both classes;
  * validation is one different valid WHOLE domain, and source training is everything remaining;
  * a domain the census declared valid can never vanish from the results without failing the run.
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from evals.regimes import base as RB
from evals.regimes import geographic_ood as GEO

LODO_SPEC = {
    "strategy": "leave_one_domain_out",
    "label": "lodo_test",
    "purge_km": 0.0,
    "min_target_n": 10,
    "allow_one_class_target": True,
}


def _toy():
    """Four domains: A/B/C are two-class, D is one-class (all positives)."""
    groups, y = [], []
    for domain, n_pos, n_neg in (("A", 20, 20), ("B", 20, 20), ("C", 20, 20), ("D", 20, 0)):
        groups += [domain] * (n_pos + n_neg)
        y += [1] * n_pos + [0] * n_neg
    return np.asarray(groups, dtype=object), np.asarray(y, dtype=np.int64)


@pytest.fixture(autouse=True)
def _clean_census():
    RB.clear_domain_census()
    RB.clear_regime_problems()
    yield
    RB.clear_domain_census()
    RB.clear_regime_problems()


# --- census -----------------------------------------------------------------


def test_census_enumerates_every_domain_with_support() -> None:
    groups, y = _toy()

    census = {r["domain"]: r for r in GEO.domain_census(y, groups, LODO_SPEC)}

    assert set(census) == {"A", "B", "C", "D"}
    assert census["A"]["n"] == 40
    assert census["A"]["n_classes"] == 2
    assert census["D"]["n"] == 20
    assert census["D"]["n_classes"] == 1


def test_one_class_domain_is_a_valid_target_but_never_the_validation_region() -> None:
    """A one-class domain is a real deployment region: evaluate it, but it cannot calibrate a
    decision threshold, so it must not be selected as validation."""
    groups, y = _toy()

    census = {r["domain"]: r for r in GEO.domain_census(y, groups, LODO_SPEC)}

    assert census["D"]["one_class"] is True
    assert census["D"]["valid_target"] is True
    assert census["D"]["valid_val"] is False
    assert census["D"]["excluded_because"] == []
    # excluded only from the metrics that mathematically need both classes
    assert census["D"]["metrics_excluded"] == ["auc"]
    assert census["A"]["valid_val"] is True
    assert census["A"]["metrics_excluded"] == []


def test_census_records_undersized_domains_as_excluded_with_a_reason() -> None:
    groups = np.asarray(["A"] * 40 + ["B"] * 40 + ["tiny"] * 3, dtype=object)
    y = np.asarray([1] * 20 + [0] * 20 + [1] * 20 + [0] * 20 + [1, 0, 1], dtype=np.int64)

    census = {r["domain"]: r for r in GEO.domain_census(y, groups, LODO_SPEC)}

    assert census["tiny"]["valid_target"] is False
    assert census["tiny"]["excluded_because"] == ["n<10"]
    # an excluded domain is still PRESENT in the census -- never a silent gap
    assert census["tiny"]["n"] == 3


def test_census_is_empty_for_non_lodo_holdouts() -> None:
    groups, y = _toy()

    assert GEO.domain_census(y, groups, ["A", "B"]) == []
    assert GEO.expected_domains(y, groups, ["A", "B"]) is None
    assert GEO.expected_domains(y, groups, {"strategy": "spatial_blocks"}) is None


def test_expected_domains_declares_every_valid_target() -> None:
    groups, y = _toy()

    assert sorted(GEO.expected_domains(y, groups, LODO_SPEC)) == ["A", "B", "C", "D"]


# --- fold construction ------------------------------------------------------


def _folds(groups, y, *, seed=0, spec=LODO_SPEC, bench=None):
    return list(GEO.iter_splits(y, groups, seed=seed, holdouts=spec, bench=bench))


def test_lodo_yields_one_fold_per_domain_including_the_one_class_domain() -> None:
    groups, y = _toy()

    folds = _folds(groups, y)

    assert sorted(f.label for f in folds) == ["A", "B", "C", "D"]
    assert sorted(f.domain for f in folds) == ["A", "B", "C", "D"]


def test_each_fold_targets_exactly_its_whole_domain() -> None:
    groups, y = _toy()

    for fold in _folds(groups, y):
        assert set(groups[fold.test]) == {fold.label}
        assert len(fold.test) == int((groups == fold.label).sum())


def test_validation_is_one_different_valid_whole_domain_and_train_is_the_rest() -> None:
    groups, y = _toy()

    for fold in _folds(groups, y):
        val_domains = set(groups[fold.val])
        assert len(val_domains) == 1, "validation must be exactly one whole domain"
        val_domain = val_domains.pop()
        assert val_domain != fold.label
        # the validation region must carry both classes -- the threshold is calibrated on it
        assert len(np.unique(y[fold.val])) == 2
        assert val_domain != "D", "the one-class domain must never be chosen as validation"
        # source training is drawn from every remaining domain, and never the target or val
        train_domains = set(groups[fold.train])
        assert fold.label not in train_domains
        assert val_domain not in train_domains
        assert train_domains <= {"A", "B", "C", "D"} - {fold.label, val_domain}


def test_folds_are_identical_across_seeds() -> None:
    """The split is a property of the data, so seeds must not move the fold boundaries."""
    groups, y = _toy()

    a = {f.label: (set(f.test), set(f.val)) for f in _folds(groups, y, seed=0)}
    b = {f.label: (set(f.test), set(f.val)) for f in _folds(groups, y, seed=7)}

    assert a == b


def test_purge_removes_training_samples_near_the_held_out_target() -> None:
    """A domain label is provenance, not a polygon: CropHarvest's global collections put samples
    inside every held-out region, and the purge is what removes them."""
    groups = np.asarray(["A"] * 20 + ["B"] * 20 + ["C"] * 20 + ["D"] * 20, dtype=object)
    y = np.asarray(([1] * 10 + [0] * 10) * 4, dtype=np.int64)
    # C sits on top of A; B and D are far away. Holding out A (val=B) leaves C and D in training,
    # of which only C is co-located with the target.
    latlon = np.concatenate([
        np.tile([0.0, 0.0], (20, 1)),
        np.tile([40.0, 40.0], (20, 1)),
        np.tile([0.01, 0.01], (20, 1)),
        np.tile([-40.0, -40.0], (20, 1)),
    ])
    bench = SimpleNamespace(name="toy", latlon=latlon, groups=groups)
    spec = {**LODO_SPEC, "purge_km": 50.0}

    fold_a = next(f for f in _folds(groups, y, spec=spec, bench=bench) if f.label == "A")

    train_domains = set(groups[fold_a.train])
    assert "C" not in train_domains, "co-located domain must be purged from training"
    assert "D" in train_domains, "the purge must not remove distant training domains"


def test_undersized_domain_is_excluded_without_failing_the_run() -> None:
    groups = np.asarray(["A"] * 40 + ["B"] * 40 + ["C"] * 40 + ["tiny"] * 3, dtype=object)
    y = np.asarray(([1] * 20 + [0] * 20) * 3 + [1, 0, 1], dtype=np.int64)

    labels = sorted(f.label for f in _folds(groups, y))

    assert labels == ["A", "B", "C"], "declared-invalid domain must not produce a fold"


# --- completeness enforcement ----------------------------------------------


def test_declared_valid_domain_that_produces_no_fold_fails_the_run(monkeypatch) -> None:
    """The census promised a fold for every valid domain; a missing one must be loud."""
    groups, y = _toy()
    bench = SimpleNamespace(name="cropharvest", groups=groups, latlon=None)

    real_iter = GEO.iter_splits

    def drop_c(*args, **kwargs):
        for split in real_iter(*args, **kwargs):
            if split.label != "C":       # silently lose a declared-valid domain
                yield split

    monkeypatch.setattr(GEO, "iter_splits", drop_c)
    monkeypatch.setattr(GEO, "assign_domains", lambda b, h=None: groups)

    with pytest.raises(RuntimeError, match=r"declared-valid domain\(s\) produced no split: \['C'\]"):
        list(RB.iter_splits("geographic_ood", bench, y, LODO_SPEC, 0, strict_mode=True))


def test_census_artifact_records_excluded_domains(tmp_path) -> None:
    import json

    groups = np.asarray(["A"] * 40 + ["B"] * 40 + ["C"] * 40 + ["tiny"] * 3, dtype=object)
    y = np.asarray(([1] * 20 + [0] * 20) * 3 + [1, 0, 1], dtype=np.int64)
    bench = SimpleNamespace(name="cropharvest", latlon=None, groups=groups)

    # the census is accumulated as folds are built, then written beside the probe outputs
    list(GEO.iter_splits(y, groups, seed=0, holdouts=LODO_SPEC, bench=bench))
    RB._write_domain_census(tmp_path)

    census = json.loads((tmp_path / "domain_census.json").read_text())
    assert census["n_domains"] == 4
    assert census["n_valid_targets"] == 3
    assert census["excluded"] == ["tiny"]
    assert {r["domain"] for r in census["domains"]} == {"A", "B", "C", "tiny"}


def test_census_is_deduplicated_across_seeds(tmp_path) -> None:
    import json

    groups, y = _toy()
    bench = SimpleNamespace(name="cropharvest", latlon=None, groups=groups)

    for seed in (0, 1, 2):
        list(GEO.iter_splits(y, groups, seed=seed, holdouts=LODO_SPEC, bench=bench))
    RB._write_domain_census(tmp_path)

    census = json.loads((tmp_path / "domain_census.json").read_text())
    assert census["n_domains"] == 4, "census is a property of the data, not of the seed"
    assert len(census["domains"]) == 4


def test_census_does_not_leak_across_benchmarks(tmp_path) -> None:
    """DOMAIN_CENSUS is process-global: a pair's artifact must describe only that pair's data."""
    import json

    groups, y = _toy()
    RB.DOMAIN_CENSUS.append({"benchmark": "someotherbench", "regime": "geographic_ood",
                             "domain": "FOREIGN", "n": 1, "n_classes": 2, "one_class": False,
                             "valid_target": True, "valid_val": True, "excluded_because": [],
                             "metrics_excluded": []})
    bench = SimpleNamespace(name="cropharvest", latlon=None, groups=groups)
    list(GEO.iter_splits(y, groups, seed=0, holdouts=LODO_SPEC, bench=bench))

    RB._write_domain_census(tmp_path, "cropharvest")

    census = json.loads((tmp_path / "domain_census.json").read_text())
    domains = {r["domain"] for r in census["domains"]}
    assert "FOREIGN" not in domains, "another benchmark's domains leaked into this artifact"
    assert domains == {"A", "B", "C", "D"}
    assert census["n_domains"] == 4


# --- historical spatial_blocks reproducibility ---------------------------------
#
# 20 result files across five runs carry the label `spatial_block_2deg_purge50km`, four of them
# in the canonical output-erm-full-20260711 tree. Switching geographic_ood to LODO must not make
# those unreproducible, and must never emit their label over a different partition.


def test_spatial_blocks_still_uses_the_two_degree_block_basis() -> None:
    from evals.benchmarks import cropharvest

    bench = SimpleNamespace(
        name="cropharvest",
        groups=np.array(["kenya", "kenya", "togo", "geowiki-landcover-2017"], dtype=object),
        latlon=np.array([[0.0, 0.0], [0.5, 0.5], [5.0, 5.0], [np.nan, 0.0]]),
    )
    spec = {"strategy": "spatial_blocks", "label": "spatial_block_2deg_purge50km"}

    domains = GEO.assign_domains(bench, spec)

    # exactly the pre-LODO basis: co-located samples share a block, distant ones do not,
    # and a sample without coordinates has no domain
    assert domains[0] == domains[1]
    assert str(domains[0]).startswith("block_")
    assert domains[2] != domains[0]
    assert domains[3] == "unknown"
    assert list(domains) == list(cropharvest.spatial_block_domains(bench))


def test_lodo_uses_canonical_domains_not_blocks() -> None:
    bench = SimpleNamespace(
        name="cropharvest",
        groups=np.array(["kenya", "kenya", "togo", "geowiki-landcover-2017"], dtype=object),
        latlon=np.array([[0.0, 0.0], [0.5, 0.5], [5.0, 5.0], [np.nan, 0.0]]),
    )

    domains = GEO.assign_domains(bench, LODO_SPEC)

    assert list(domains) == ["kenya", "kenya", "togo", "geowiki-landcover-2017"]


def test_spatial_blocks_refuses_a_basis_it_cannot_reproduce() -> None:
    """Rather than silently emitting the historical label over a different partition."""
    bench = SimpleNamespace(name="breizhcrops", groups=np.array(["a", "b"], dtype=object),
                            latlon=np.array([[0.0, 0.0], [1.0, 1.0]]))

    with pytest.raises(ValueError, match="requires a spatial_block_domains"):
        GEO.assign_domains(bench, {"strategy": "spatial_blocks"})
