"""Regressions for run provenance, completion marking, and JSONL durability.

Each group here pins a failure mode that was silent by construction -- the artifact looked fine,
the run exited 0, and a downstream reader trusted it.
"""

from __future__ import annotations

import json
import os
from types import SimpleNamespace

import numpy as np
import pytest

from evals import probes
from evals.metrics import _class_index
from tests import splitfix
from utils import artifacts, runstate
from utils import ioutils as IOU
from utils import perfutils as perf

# --- torn JSONL tail --------------------------------------------------------


def test_repair_leaves_a_cleanly_terminated_file_alone(tmp_path) -> None:
    p = tmp_path / "rows.jsonl"
    IOU.append_jsonl(p, [{"v": 1}, {"v": 2}])
    before = p.read_bytes()

    assert IOU.repair_jsonl_tail(p) == 0
    assert p.read_bytes() == before


def test_repair_is_a_noop_on_missing_or_empty_files(tmp_path) -> None:
    assert IOU.repair_jsonl_tail(tmp_path / "absent.jsonl") == 0
    empty = tmp_path / "empty.jsonl"
    empty.touch()
    assert IOU.repair_jsonl_tail(empty) == 0


def test_repair_truncates_an_unterminated_tail(tmp_path) -> None:
    p = tmp_path / "rows.jsonl"
    IOU.append_jsonl(p, [{"v": 1}])
    with p.open("a") as f:  # a hard crash mid-write
        f.write('{"v": 2, "partial"')

    removed = IOU.repair_jsonl_tail(p)

    assert removed == len('{"v": 2, "partial"')
    assert IOU.read_jsonl(p) == [{"v": 1}]
    assert p.read_bytes().endswith(b"\n")


def test_repair_empties_a_file_that_is_one_unterminated_row(tmp_path) -> None:
    p = tmp_path / "rows.jsonl"
    p.write_text('{"v": 1, "torn"')

    IOU.repair_jsonl_tail(p)

    assert p.read_bytes() == b""
    assert IOU.read_jsonl(p) == []


def test_append_after_a_torn_tail_does_not_brick_the_file(tmp_path) -> None:
    """THE regression: without repair, the append glues onto the torn bytes, promoting a
    droppable trailing fragment into a corrupt INTERIOR row that read_jsonl raises on forever."""
    p = tmp_path / "rows.jsonl"
    IOU.append_jsonl(p, [{"v": 1}])
    with p.open("a") as f:
        f.write('{"seed": 0, "budget_type": "target", "label_bud')  # killed here

    IOU.append_jsonl(p, [{"v": 2}])  # the resume

    rows = IOU.read_jsonl(p)
    assert rows == [{"v": 1}, {"v": 2}]  # torn row gone, resume row intact, file readable


def test_read_jsonl_still_raises_on_a_corrupt_interior_row(tmp_path) -> None:
    """Repair must not become a licence to swallow real corruption."""
    p = tmp_path / "rows.jsonl"
    p.write_text('{"v": 1}\n{"v": BROKEN}\n{"v": 3}\n')

    with pytest.raises(ValueError, match="Corrupt JSONL row 2"):
        IOU.read_jsonl(p)


# --- _class_index vectorization ---------------------------------------------


def _reference_class_index(y_true, classes):
    """The pre-vectorization implementation, kept as the oracle."""
    col = {int(c): j for j, c in enumerate(classes)}
    return np.asarray([col.get(int(y), -1) for y in y_true], dtype=np.int64)


def test_class_index_returns_minus_one_for_unsupported_classes() -> None:
    """The contract that rules out _as_eval_indices, which raises instead."""
    out = _class_index(np.array([5, 9, 7]), np.array([5, 7]))

    assert out.tolist() == [0, -1, 1]


def test_class_index_handles_empty_class_lists() -> None:
    assert _class_index(np.array([1, 2]), np.array([])).tolist() == [-1, -1]


def test_class_index_matches_the_reference_implementation() -> None:
    rng = np.random.default_rng(0)
    for _ in range(20):
        classes = np.unique(rng.integers(0, 40, size=rng.integers(1, 25)))
        y = rng.integers(0, 40, size=200)
        np.testing.assert_array_equal(_class_index(y, classes), _reference_class_index(y, classes))


def test_class_index_is_correct_for_unsorted_classes() -> None:
    """`classes` is not guaranteed sorted; searchsorted needs the argsort to be honoured."""
    classes = np.array([9, 3, 7])
    y = np.array([7, 9, 3, 4])

    np.testing.assert_array_equal(_class_index(y, classes), _reference_class_index(y, classes))
    assert _class_index(y, classes).tolist() == [2, 0, 1, -1]


# --- signature knobs (PROBE_CAP / PROBE_TUNING) -----------------------------


# A stable stub run manifest shared by the monkeypatch stubs and the "finished dir" seeder, so the
# manifest resume check passes and it is the ENVIRONMENT gate that decides refused-resume tests.
_STUB_MANIFEST = {"stub": "run", "schema": 1}


def _sig(**over):
    kwargs = dict(
        model_name="raw", benchmark="cropharvest", artifact="baseline",
        embedding_manifest_sha256="emb", split_regimes=["random_id"], seeds=[0],
        enc_kwargs={}, active_probes=["logistic"],
        budget_regimes={"source": [1.0]}, max_dense_pixels=None,
    )
    kwargs.update(over)
    man = runstate.build_run_manifest(
        kwargs["model_name"], kwargs["benchmark"], kwargs["artifact"], kwargs["embedding_manifest_sha256"],
        kwargs["split_regimes"], kwargs["seeds"], kwargs["enc_kwargs"],
        active_probes=kwargs["active_probes"], budget_regimes=kwargs["budget_regimes"],
        max_dense_pixels=kwargs["max_dense_pixels"], write_predictions=True,
    )
    return runstate.run_manifest_digest(man)


def test_signature_changes_when_the_probe_cap_is_set(monkeypatch) -> None:
    baseline = _sig()

    monkeypatch.setattr(perf, "PROBE_CAP", 50000)
    capped = _sig()
    monkeypatch.setattr(perf, "PROBE_CAP", 100000)
    capped_bigger = _sig()

    assert capped != baseline, "a capped run must not share the uncapped run's signature"
    assert capped != capped_bigger, "cap=50000 and cap=100000 must not collide"


def test_signature_changes_when_probe_tuning_is_enabled(monkeypatch) -> None:
    baseline = _sig()

    monkeypatch.setattr(probes, "PROBE_TUNING", True)

    assert _sig() != baseline


# --- environment.json -------------------------------------------------------


def test_environment_records_the_numerical_core_and_git_identity(tmp_path) -> None:
    env = artifacts.write_environment(tmp_path)

    written = json.loads((tmp_path / "environment.json").read_text())
    assert written == env
    for pkg in artifacts.NUMERICAL_CORE:
        assert env["numerical_core"][pkg], f"{pkg} version not captured"
    assert env["python"]
    assert "commit" in env["git"] and "dirty" in env["git"]


def test_environment_mismatches_flags_a_sklearn_drift() -> None:
    a = {"python": "3.11.15", "numerical_core": {"scikit-learn": "1.9.0", "numpy": "1.26.4"}}
    b = {"python": "3.11.15", "numerical_core": {"scikit-learn": "1.7.2", "numpy": "1.26.4"}}

    problems = artifacts.environment_mismatches(a, b)

    assert problems == ["scikit-learn: 1.9.0 != 1.7.2"]
    assert artifacts.environment_mismatches(a, a) == []




# --- environment: never attach the current env to rows it did not produce ----


def _env(sklearn="1.9.0", python="3.11.15"):
    return {"schema": 1, "captured_at": "2026-07-11T00:00:00+00:00", "python": python,
            "numerical_core": {"numpy": "1.26.4", "scipy": "1.17.1", "scikit-learn": sklearn,
                               "torch": "2.7.1"},
            "encoder_packages": {}, "cuda": {}, "git": {"commit": "abc", "dirty": False}}


def _with_rows(tmp_path):
    IOU.append_jsonl(tmp_path / "probe_results.jsonl", [_row()])


def test_environment_is_recorded_freely_when_there_are_no_rows(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(artifacts, "capture_environment", lambda repo=None: _env())

    env = artifacts.write_environment(tmp_path)

    assert json.loads((tmp_path / "environment.json").read_text()) == env


def test_environment_state_distinguishes_absent_malformed_and_present(tmp_path) -> None:
    assert artifacts.environment_state(tmp_path) == ("absent", None)
    (tmp_path / "environment.json").write_text("{not json")
    assert artifacts.environment_state(tmp_path)[0] == "malformed"
    (tmp_path / "environment.json").write_text('{"schema": 1}')  # no numerical_core
    assert artifacts.environment_state(tmp_path)[0] == "malformed"
    IOU.write_json(tmp_path / "environment.json", _env())
    state, rec = artifacts.environment_state(tmp_path)
    assert state == "present" and rec["numerical_core"]["scikit-learn"] == "1.9.0"


def test_rows_with_no_environment_record_refuse_resume(tmp_path, monkeypatch) -> None:
    """Stamping the current env on rows it did not produce would FABRICATE provenance, and
    nothing else would catch it -- sklearn is not in the run signature."""
    monkeypatch.setattr(artifacts, "capture_environment", lambda repo=None: _env())
    _with_rows(tmp_path)

    with pytest.raises(artifacts.EnvironmentProvenanceError, match="no environment.json"):
        artifacts.write_environment(tmp_path)

    assert not (tmp_path / "environment.json").exists(), "must not relabel the existing rows"


def test_rows_with_a_malformed_environment_record_refuse_resume(tmp_path, monkeypatch) -> None:
    """Unknown provenance must not silently become 'the current environment'."""
    monkeypatch.setattr(artifacts, "capture_environment", lambda repo=None: _env())
    _with_rows(tmp_path)
    (tmp_path / "environment.json").write_text("{truncated")

    with pytest.raises(artifacts.EnvironmentProvenanceError, match="unreadable"):
        artifacts.write_environment(tmp_path)

    assert (tmp_path / "environment.json").read_text() == "{truncated", "must not overwrite it"


def test_compatible_resume_preserves_the_original_record(tmp_path, monkeypatch) -> None:
    original = _env()
    IOU.write_json(tmp_path / "environment.json", original)
    _with_rows(tmp_path)
    later = {**_env(), "captured_at": "2026-07-16T23:00:00+00:00", "machine": "a-different-box"}
    monkeypatch.setattr(artifacts, "capture_environment", lambda repo=None: later)

    returned = artifacts.write_environment(tmp_path)

    on_disk = json.loads((tmp_path / "environment.json").read_text())
    assert on_disk == original and on_disk["captured_at"] == "2026-07-11T00:00:00+00:00"
    assert returned == original


def test_resume_is_refused_on_a_numerical_core_drift(tmp_path, monkeypatch) -> None:
    IOU.write_json(tmp_path / "environment.json", _env(sklearn="1.9.0"))
    _with_rows(tmp_path)
    monkeypatch.setattr(artifacts, "capture_environment", lambda repo=None: _env(sklearn="1.7.2"))

    with pytest.raises(artifacts.EnvironmentMismatchError, match="scikit-learn: 1.9.0 != 1.7.2"):
        artifacts.write_environment(tmp_path)

    assert json.loads((tmp_path / "environment.json").read_text())["numerical_core"]["scikit-learn"] == "1.9.0"


def test_resume_is_refused_on_a_python_drift(tmp_path, monkeypatch) -> None:
    IOU.write_json(tmp_path / "environment.json", _env(python="3.11.15"))
    _with_rows(tmp_path)
    monkeypatch.setattr(artifacts, "capture_environment", lambda repo=None: _env(python="3.11.13"))

    with pytest.raises(artifacts.EnvironmentMismatchError, match="python"):
        artifacts.write_environment(tmp_path)


def test_overwrite_mode_replaces_rows_and_environment_safely(tmp_path, monkeypatch) -> None:
    """Sound only because the caller discards the rows first."""
    IOU.write_json(tmp_path / "environment.json", _env(sklearn="1.7.2"))
    monkeypatch.setattr(artifacts, "capture_environment", lambda repo=None: _env(sklearn="1.9.0"))

    env = artifacts.write_environment(tmp_path, overwrite_mode=True)

    assert env["numerical_core"]["scikit-learn"] == "1.9.0"
    assert env["superseded_environment"]["previous"]["numerical_core"]["scikit-learn"] == "1.7.2"
    # reads recorded != current, so the OLD version comes first
    assert "scikit-learn: 1.7.2 != 1.9.0" in env["superseded_environment"]["reason"]


def test_overwrite_mode_replaces_a_malformed_record(tmp_path, monkeypatch) -> None:
    (tmp_path / "environment.json").write_text("{broken")
    monkeypatch.setattr(artifacts, "capture_environment", lambda repo=None: _env())

    env = artifacts.write_environment(tmp_path, overwrite_mode=True)

    assert env["numerical_core"]["scikit-learn"] == "1.9.0"
    assert env["superseded_environment"]["reason"].endswith("replaced an unreadable record")


def test_encoder_package_drift_does_not_block_a_resume(tmp_path, monkeypatch) -> None:
    """They move embeddings (covered by the checkpoint fingerprint), not probe arithmetic."""
    IOU.write_json(tmp_path / "environment.json", {**_env(), "encoder_packages": {"timm": "1.0.27"}})
    _with_rows(tmp_path)
    monkeypatch.setattr(
        artifacts, "capture_environment",
        lambda repo=None: {**_env(), "encoder_packages": {"timm": "9.9.9"}},
    )

    artifacts.write_environment(tmp_path)  # must not raise


def test_backfill_environment_is_the_only_sanctioned_way_to_label_historical_rows(tmp_path) -> None:
    _with_rows(tmp_path)
    attributed = _env(sklearn="1.9.0")

    marker = artifacts.backfill_environment(
        tmp_path, verified_by="akshith", note="from the Jul-11 launch record", environment=attributed
    )

    assert marker["backfilled"] is True and marker["verified_by"] == "akshith"
    assert marker["numerical_core"]["scikit-learn"] == "1.9.0"
    assert artifacts.environment_state(tmp_path)[0] == "present"


def test_backfill_environment_refuses_to_be_anonymous_or_to_capture(tmp_path) -> None:
    """Capturing it here would be exactly the fabrication write_environment refuses."""
    _with_rows(tmp_path)
    with pytest.raises(ValueError, match="requires verified_by and note"):
        artifacts.backfill_environment(tmp_path, verified_by="", note="x", environment=_env())
    with pytest.raises(ValueError, match="requires a COMPLETE attributed record"):
        artifacts.backfill_environment(tmp_path, verified_by="a", note="b", environment={"no": "core"})


def test_backfill_environment_requires_rows_to_attribute(tmp_path) -> None:
    """A run that is about to produce rows records its own environment by observation."""
    with pytest.raises(ValueError, match="holds no probe rows"):
        artifacts.backfill_environment(tmp_path, verified_by="a", note="b", environment=_env())


def test_backfill_environment_refuses_to_overwrite_a_readable_record(tmp_path) -> None:
    """That record is the run's own evidence; a human assertion must not replace it."""
    _with_rows(tmp_path)
    IOU.write_json(tmp_path / "environment.json", _env(sklearn="1.9.0"))

    with pytest.raises(ValueError, match="already has a readable"):
        artifacts.backfill_environment(
            tmp_path, verified_by="a", note="b", environment=_env(sklearn="1.7.2")
        )

    assert json.loads((tmp_path / "environment.json").read_text())["numerical_core"]["scikit-learn"] == "1.9.0"


def test_backfill_environment_fills_a_malformed_record_and_says_which_hole(tmp_path) -> None:
    _with_rows(tmp_path)
    (tmp_path / "environment.json").write_text("{broken")

    marker = artifacts.backfill_environment(
        tmp_path, verified_by="akshith", note="from the launch record", environment=_env()
    )

    assert marker["backfilled_over"] == "malformed"


@pytest.mark.parametrize("bad,why", [
    ({**_env(), "python": ""}, "missing python"),
    ({**_env(), "numerical_core": {"numpy": "1.26.4"}}, "scikit-learn is missing"),
    ({**_env(), "numerical_core": {**_env()["numerical_core"], "scipy": ""}}, "scipy is missing"),
    ({"python": "3.11.15"}, "missing numerical_core"),
])
def test_backfill_environment_requires_a_complete_record(tmp_path, bad, why) -> None:
    """An incomplete record answers nothing about comparability, which is the only reason it exists."""
    _with_rows(tmp_path)

    with pytest.raises(ValueError, match="COMPLETE attributed record"):
        artifacts.backfill_environment(tmp_path, verified_by="a", note="b", environment=bad)
    assert why  # documents the specific hole each case pokes


def test_completion_validation_checks_the_environment_schema_not_just_its_hash(tmp_path) -> None:
    """The bytes being unchanged says nothing about whether the record answers its question."""
    _finished(tmp_path)
    ok, _ = artifacts.validate_run_complete(tmp_path)
    assert ok

    # rewrite it as a well-formed but USELESS record, and re-mark so the hash still matches
    IOU.write_json(tmp_path / "environment.json", {"schema": 1, "numerical_core": {"numpy": "1.26.4"}})
    marker = artifacts.read_run_complete(tmp_path)
    marker["artifacts"]["environment.json"]["sha256"] = artifacts.sha256_file(tmp_path / "environment.json")
    IOU.write_json(tmp_path / "run_complete.json", marker)

    ok, problems = artifacts.validate_run_complete(tmp_path)

    assert not ok
    assert any("missing python" in p for p in problems)
    assert any("scikit-learn is missing" in p for p in problems)


# --- real-path regressions: a REFUSED resume must not touch anything ---------
#
# The guards live in artifacts, but the ORDER lives in the callers: check_run_manifest ->
# write_environment -> invalidate_run_complete. Get that order wrong and a refused resume deletes
# the completion marker of the finished run it just declined to touch. Only a test through the
# real entry points can see that, so these drive main._run_tabular_pair and
# runstate._run_segmentation_pair rather than the artifacts helpers.


def _seed_finished_dir(results_dir, env: dict) -> bytes:
    """A directory that looks like a previously-FINISHED pair. Returns the marker's bytes."""
    results_dir.mkdir(parents=True, exist_ok=True)
    rows = [_row()]
    IOU.append_jsonl(results_dir / "probe_results.jsonl", rows)
    for name in ("probe_results.csv", "summary.csv", "deltas.csv"):
        (results_dir / name).write_text('{"a": 1}\n')
    IOU.write_json(results_dir / "environment.json", env)
    IOU.write_json(results_dir / "run_manifest.json", _STUB_MANIFEST)
    artifacts.write_run_complete(
        results_dir, run_manifest_sha256="sig", expected_keys={_key()}, rows=rows,
    )
    return (results_dir / "run_complete.json").read_bytes()


def _tabular_bench():
    from dataio.get_input import Benchmark, ModalitySeries, NativeSeries

    n = 6

    def series(width, bands):
        return ModalitySeries(
            [np.ones((1, width), dtype=np.float32) for _ in range(n)],
            [np.array([0], dtype=np.int64) for _ in range(n)],
            [np.array([15.0], dtype=np.float32) for _ in range(n)],
            [np.array([2020], dtype=np.int64) for _ in range(n)],
            bands,
        )

    return Benchmark(
        name="fake", label_kind="binary",
        native=NativeSeries(series(2, ["B2", "B3"]), series(1, ["VV"]), series(1, ["temp"])),
        labels=np.array([0, 1, 0, 1, 0, 1], dtype=np.int64),
        groups=np.array(["a", "a", "b", "b", "c", "c"], dtype=object),
        latlon=np.zeros((n, 2), dtype=np.float32),
        years=np.full(n, 2020, dtype=np.int64),
    )


def _run_tabular(monkeypatch, tmp_path, *, overwrite=False):
    import main

    bench = _tabular_bench()

    class FakeBenchMod:
        BENCHMARK = "fake"
        LABEL_KIND = "binary"
        SPLIT_REGIMES = ["random_id"]

        @staticmethod
        def make_targets(loaded):
            return loaded.labels, loaded.groups

    monkeypatch.setattr(main.EV, "load_benchmark", lambda _n: FakeBenchMod)
    monkeypatch.setattr(main.cacheutils, "OUTPUT_DIR", tmp_path)
    monkeypatch.setattr(main.cacheutils, "cached_bench", lambda *a, **k: bench)
    monkeypatch.setattr(main.runstate, "build_run_manifest", lambda *a, **k: _STUB_MANIFEST)
    monkeypatch.setattr(main.compat, "input_modalities", lambda _m: set())
    monkeypatch.setattr(main.cacheutils, "load_cached_embeddings",
                        lambda *a, **k: np.zeros((6, 2), dtype=np.float32))
    # PHASE B: main consumes splits from data/splits/ instead of constructing them; stub the
    # consumption to yield no splits so this exercises the environment/overwrite/marker contract on
    # an empty run.
    monkeypatch.setattr(main.split_artifacts, "load_tabular_splits", lambda *a, **k: [])
    main._run_tabular_pair(
        "fake", "raw", [0], 0, ["random_id"], ["probing"], ["logistic"],
        {"source": [1.0], "target": [0]}, False, overwrite, False, {},  # s2_only=False
    )


@pytest.mark.parametrize("env_fault,expected", [
    ("incompatible", artifacts.EnvironmentMismatchError),
    ("absent", artifacts.EnvironmentProvenanceError),
    ("malformed", artifacts.EnvironmentProvenanceError),
])
def test_tabular_refused_resume_preserves_run_complete_byte_for_byte(
    monkeypatch, tmp_path, env_fault, expected
) -> None:
    results_dir = tmp_path / "results" / "raw" / "fake"
    before = _seed_finished_dir(results_dir, _env(sklearn="1.9.0"))
    if env_fault == "absent":
        (results_dir / "environment.json").unlink()
    elif env_fault == "malformed":
        (results_dir / "environment.json").write_text("{truncated")
    monkeypatch.setattr(artifacts, "capture_environment", lambda repo=None: _env(sklearn="1.7.2"))

    with pytest.raises(expected):
        _run_tabular(monkeypatch, tmp_path)

    assert (results_dir / "run_complete.json").read_bytes() == before, (
        "a refused resume destroyed the completion marker of a finished run"
    )
    assert artifacts.has_result_rows(results_dir), "a refused resume destroyed the rows"


def test_tabular_overwrite_removes_rows_before_recording_the_environment(monkeypatch, tmp_path) -> None:
    """The overwrite contract: provenance is replaced only once the rows it described are gone."""
    results_dir = tmp_path / "results" / "raw" / "fake"
    _seed_finished_dir(results_dir, _env(sklearn="1.7.2"))
    seen = {}
    real = artifacts.write_environment

    def spy(rd, repo=None, *, overwrite_mode=False):
        seen["rows_present_at_write"] = artifacts.has_result_rows(rd)
        return real(rd, repo, overwrite_mode=overwrite_mode)

    monkeypatch.setattr(artifacts, "capture_environment", lambda repo=None: _env(sklearn="1.9.0"))
    monkeypatch.setattr(artifacts, "write_environment", spy)
    monkeypatch.setattr(__import__("main").artifacts, "write_environment", spy)

    with pytest.raises(artifacts.IncompleteRunError):   # the stub yields no splits
        _run_tabular(monkeypatch, tmp_path, overwrite=True)

    assert seen["rows_present_at_write"] is False, (
        "the caller must delete probe_results.jsonl BEFORE the environment record is replaced"
    )
    env = json.loads((results_dir / "environment.json").read_text())
    assert env["numerical_core"]["scikit-learn"] == "1.9.0"


def test_tabular_split_refusal_preserves_finished_dir_even_under_overwrite(monkeypatch, tmp_path) -> None:
    """PHASE B req 1: splits are validated BEFORE any mutation, so a split refusal leaves the finished
    dir (rows / environment / run_manifest / run_complete) byte-for-byte unchanged even
    with overwrite_mode=True."""
    import main
    from evals import split_artifacts as SA

    results_dir = tmp_path / "results" / "raw" / "fake"
    _seed_finished_dir(results_dir, _env(sklearn="1.9.0"))
    snap = {n: (results_dir / n).read_bytes() for n in
            ("probe_results.jsonl", "environment.json", "run_manifest.json", "run_complete.json")}

    class FakeBenchMod:
        BENCHMARK = "fake"
        LABEL_KIND = "binary"
        SPLIT_REGIMES = ["random_id"]

        @staticmethod
        def make_targets(loaded):
            return loaded.labels, loaded.groups

    monkeypatch.setattr(main.EV, "load_benchmark", lambda _n: FakeBenchMod)
    monkeypatch.setattr(main.cacheutils, "OUTPUT_DIR", tmp_path)
    monkeypatch.setattr(main.cacheutils, "cached_bench", lambda *a, **k: _tabular_bench())
    monkeypatch.setattr(main.compat, "input_modalities", lambda _m: set())
    monkeypatch.setattr(main.cacheutils, "load_cached_embeddings", lambda *a, **k: np.zeros((6, 2), dtype=np.float32))
    monkeypatch.setattr(main.cacheutils, "SCRATCH", tmp_path / "empty_splits")  # no data/splits -> refuse

    with pytest.raises(SA.SplitArtifactError, match="no split log"):
        main._run_tabular_pair(
            "fake", "raw", [0], 0, ["random_id"], ["probing"], ["logistic"],
            {"source": [1.0], "target": [0]}, False, True, False, {},  # overwrite_mode=True
        )
    for n, b in snap.items():
        assert (results_dir / n).read_bytes() == b, f"{n} was mutated by a REFUSED split load under OVERWRITE"


def test_dense_split_refusal_preserves_finished_dir_even_under_overwrite(monkeypatch, tmp_path) -> None:
    """PHASE B req 1, dense path."""
    from evals import split_artifacts as SA
    from utils import runstate as RS

    results_dir = tmp_path / "results" / "raw" / "pastis"
    _seed_finished_dir(results_dir, _env(sklearn="1.9.0"))
    snap = {n: (results_dir / n).read_bytes() for n in
            ("probe_results.jsonl", "environment.json", "run_manifest.json", "run_complete.json")}

    monkeypatch.setattr(RS.cacheutils, "OUTPUT_DIR", tmp_path)
    monkeypatch.setattr(RS.cacheutils, "cached_bench", lambda *a, **k: SimpleNamespace(
        name="pastis", data_quality=None, patch_classes=None, n_samples=4, patches=[]))
    monkeypatch.setattr(RS.cacheutils, "require_dense_cache", lambda *a, **k: tmp_path / "emb")
    monkeypatch.setattr(RS.cacheutils, "SCRATCH", tmp_path / "empty_splits")  # no data/splits -> refuse

    with pytest.raises(SA.SplitArtifactError, match="no split log"):
        RS._run_segmentation_pair(
            "pastis", "raw", [0], 10, ["random_id"], ["probing"], ["logistic"],
            {"source": [1.0], "target": [0]}, False, True, True, False, {},  # overwrite_mode=True
        )
    for n, b in snap.items():
        assert (results_dir / n).read_bytes() == b, f"{n} was mutated by a REFUSED dense split load under OVERWRITE"


# --- PHASE B req 1: split validation is FAIL-FAST (before the encoder ever runs) ----------
#
# With RUN_STAGES = ["gen_embeddings", "probing"], the encoder is scheduled to run. But splits are
# loaded + validated immediately after the benchmark, BEFORE extract_and_cache /
# extract_dense_and_cache, cache require, digest, or any results-dir mutation. So a missing/invalid
# split refuses the pair WITHOUT ever invoking the (multi-GPU-hour) encoder or touching the tree.


def test_tabular_missing_splits_refuse_before_the_encoder_runs(monkeypatch, tmp_path) -> None:
    import main
    from evals import split_artifacts as SA

    class FakeBenchMod:
        BENCHMARK = "fake"
        LABEL_KIND = "binary"
        SPLIT_REGIMES = ["random_id"]

        @staticmethod
        def make_targets(loaded):
            return loaded.labels, loaded.groups

    def _never(*_a, **_k):
        raise AssertionError("the encoder ran despite missing splits")

    monkeypatch.setattr(main.EV, "load_benchmark", lambda _n: FakeBenchMod)
    monkeypatch.setattr(main.cacheutils, "OUTPUT_DIR", tmp_path / "out")
    monkeypatch.setattr(main.cacheutils, "cached_bench", lambda *a, **k: _tabular_bench())
    monkeypatch.setattr(main.cacheutils, "SCRATCH", tmp_path / "empty")  # no data/splits -> refuse
    monkeypatch.setattr(main.cacheutils, "extract_and_cache", _never)
    monkeypatch.setattr(main.cacheutils, "load_cached_embeddings", _never)

    with pytest.raises(SA.SplitArtifactError, match="no split log"):
        main._run_tabular_pair(
            "fake", "raw", [0], 0, ["random_id"], ["gen_embeddings", "probing"], ["logistic"],
            {"source": [1.0], "target": [0]}, False, False, False, {},
        )
    # never called (both stubs raise AssertionError, which is not SplitArtifactError) AND no tree
    assert not (tmp_path / "out").exists(), "a refused split load created/mutated the results tree"


def test_dense_missing_splits_refuse_before_the_encoder_runs(monkeypatch, tmp_path) -> None:
    from evals import split_artifacts as SA
    from utils import runstate as RS

    def _never(*_a, **_k):
        raise AssertionError("the dense encoder ran despite missing splits")

    monkeypatch.setattr(RS.cacheutils, "OUTPUT_DIR", tmp_path / "out")
    monkeypatch.setattr(RS.cacheutils, "cached_bench", lambda *a, **k: SimpleNamespace(
        name="pastis", data_quality=None, patch_classes=None, n_samples=4, patches=[]))
    monkeypatch.setattr(RS.cacheutils, "SCRATCH", tmp_path / "empty")  # no data/splits -> refuse
    monkeypatch.setattr(RS.cacheutils, "extract_dense_and_cache", _never)
    monkeypatch.setattr(RS.cacheutils, "require_dense_cache", _never)

    with pytest.raises(SA.SplitArtifactError, match="no split log"):
        RS._run_segmentation_pair(
            "pastis", "raw", [0], 10, ["random_id"], ["gen_embeddings", "probing"], ["logistic"],
            {"source": [1.0], "target": [0]}, False, False, True, False, {},
        )
    assert not (tmp_path / "out").exists(), "a refused dense split load created/mutated the results tree"


def _dense_stubs(monkeypatch, tmp_path):
    from evals import split_artifacts as SA
    from utils import runstate as RS

    # PHASE B: runstate consumes dense splits from data/splits/ (function-local import, so patch the
    # module object). An empty load exercises the environment gate on a pair that evaluates nothing.
    monkeypatch.setattr(SA, "load_dense_splits", lambda *a, **k: {0: []})
    monkeypatch.setattr(RS.cacheutils, "OUTPUT_DIR", tmp_path)
    monkeypatch.setattr(RS, "build_run_manifest", lambda *a, **k: _STUB_MANIFEST)
    monkeypatch.setattr(RS.cacheutils, "cached_bench", lambda *a, **k: SimpleNamespace(
        name="pastis", data_quality=None, patch_classes=None, n_samples=4))
    # The dense cache is content-addressed off the benchmark; stub the lookup so the test reaches
    # the environment gate without needing real dense tiles on disk.
    monkeypatch.setattr(RS.cacheutils, "require_dense_cache", lambda *a, **k: tmp_path / "emb")
    monkeypatch.setattr(RS.cacheutils, "dense_embedding_cache_dir", lambda *a, **k: tmp_path / "emb")
    return RS


def _run_dense(RS, *, overwrite=False):
    RS._run_segmentation_pair(
        "pastis", "raw", [0], 10, ["random_id"], ["probing"], ["logistic"],
        {"source": [1.0], "target": [0]}, False, overwrite, False, False, {},  # s2_only=False
    )


@pytest.mark.parametrize("env_fault,expected", [
    ("incompatible", artifacts.EnvironmentMismatchError),
    ("absent", artifacts.EnvironmentProvenanceError),
    ("malformed", artifacts.EnvironmentProvenanceError),
])
def test_dense_refused_resume_preserves_run_complete_byte_for_byte(
    monkeypatch, tmp_path, env_fault, expected
) -> None:
    results_dir = tmp_path / "results" / "raw" / "pastis"
    before = _seed_finished_dir(results_dir, _env(sklearn="1.9.0"))
    rows_before = (results_dir / "probe_results.jsonl").read_bytes()
    if env_fault == "absent":
        (results_dir / "environment.json").unlink()
    elif env_fault == "malformed":
        (results_dir / "environment.json").write_text("{truncated")
    RS = _dense_stubs(monkeypatch, tmp_path)
    monkeypatch.setattr(artifacts, "capture_environment", lambda repo=None: _env(sklearn="1.7.2"))

    with pytest.raises(expected):
        _run_dense(RS)

    assert (results_dir / "run_complete.json").read_bytes() == before, (
        "a refused dense resume destroyed the completion marker of a finished run"
    )
    assert (results_dir / "probe_results.jsonl").read_bytes() == rows_before


def test_dense_overwrite_removes_rows_before_recording_the_environment(monkeypatch, tmp_path) -> None:
    """Same contract as the tabular path: provenance is replaced only once its rows are gone."""
    results_dir = tmp_path / "results" / "raw" / "pastis"
    _seed_finished_dir(results_dir, _env(sklearn="1.7.2"))
    RS = _dense_stubs(monkeypatch, tmp_path)
    seen = {}
    real = artifacts.write_environment

    def spy(rd, repo=None, *, overwrite_mode=False):
        seen["rows_present_at_write"] = artifacts.has_result_rows(rd)
        return real(rd, repo, overwrite_mode=overwrite_mode)

    monkeypatch.setattr(artifacts, "capture_environment", lambda repo=None: _env(sklearn="1.9.0"))
    monkeypatch.setattr(RS.artifacts, "write_environment", spy)

    with pytest.raises(Exception):  # noqa: B017 - the stubbed pair cannot complete; order is the point
        _run_dense(RS, overwrite=True)

    assert seen["rows_present_at_write"] is False, (
        "the dense caller must delete probe_results.jsonl BEFORE the environment record is replaced"
    )
    assert json.loads((results_dir / "environment.json").read_text())["numerical_core"]["scikit-learn"] == "1.9.0"


# --- the HEALTHY dense path: the caller must actually be able to call the cell ------------
#
# Everything above stops at a refusal gate, and test_erm_characterization calls
# _run_segmentation_cell DIRECTLY. So nothing exercised the single line where the production
# caller builds the cell's positional arguments -- and that line drifted out of sync with the
# signature (a stale method_name ahead of family). The whole suite stayed green while the dense
# pair was guaranteed to die with TypeError on its first real job. Only a run that SCHEDULES one
# can see it, so this drives _run_segmentation_pair to completion over a real on-disk cache with
# the real random_id regime, stubbing only what needs a network or a multi-GB corpus.


def _dense_cache_on_disk(root, *, folds=(1, 2, 3, 4, 5), patches=12, pixels=30, dim=5, classes=3):
    """A minimal real dense cache: fold_<f>/<patch>_<r>_<c>{,.labels}.npy, linearly separable.

    Folds are PASTIS's real TRAIN/VAL/TEST_FOLDS -- the benchmark module here is the production
    one, not a stub, because run_probes_segmentation internally re-resolves load_benchmark("pastis")
    and delegates to it. Stubbing the module out would only hide that.
    """
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


def _dense_mock_bench(pids):
    """A minimal PASTIS-shaped bench exposing exactly the v2 dense surface the random_id regime and
    the runtime touch: patch ids, per-patch class sets, per-patch Sentinel tile, and patch objects."""
    patches = [SimpleNamespace(patch_id=p, fold=1, tile=f"T3{p % 4}") for p in pids]
    return SimpleNamespace(
        name="pastis",
        patches=patches,
        patch_ids=lambda folds=None, _p=pids: list(_p),
        patch_class_sets=lambda ids=None, _p=pids: {int(q): {0, 1, 2} for q in (ids if ids is not None else _p)},
        patch_tiles={p.patch_id: p.tile for p in patches},  # @property dict on the real bench
        patch_latlon={},
    )


def _publish_dense_random_splits(splits_root, emb_dir, *, seed=0):
    """Generate + publish a schema-v2 random_id DenseSourceTargetSplit matching a real dense cache.

    This is the dense integration fixture: the runtime CONSUMES patch splits from data/splits/, so a
    real dense-cell test must first publish canonical v2 artifacts (source-only patch partitions) for
    the cache's patches, exactly as tools/generate_splits.generate_dense would.
    """
    import evals.benchmarks.pastis as pastis_mod
    from evals import split_artifacts as SA
    from evals.regimes import random_id
    from utils import cacheutils as C

    all_folds = set(pastis_mod.TRAIN_FOLDS) | set(pastis_mod.VAL_FOLDS) | set(pastis_mod.TEST_FOLDS)
    pids = [int(p) for p in C.dense_fold_patches(emb_dir, all_folds)]
    bench = _dense_mock_bench(pids)
    dense_split = next(iter(random_id.iter_dense_source_target_splits(bench, pastis_mod, seed)))
    domain_of = {int(k): str(v) for k, v in random_id.patch_domains(bench, pastis_mod).items()}
    rows, summary = SA.build_dense_leaf(
        "pastis", "random_id", seed, dense_split=dense_split,
        audit_events=[], all_patch_ids=pids, domain_of=domain_of,
        class_sets={int(p): {0, 1, 2} for p in pids}, patch_latlon={},
    )
    splitfix.freeze(splits_root, [(rows, summary)])
    return pids


def _run_dense_healthy(monkeypatch, tmp_path):
    from utils import runstate as RS

    emb_dir = _dense_cache_on_disk(tmp_path / "emb")
    pids = _publish_dense_random_splits(tmp_path / "splits", emb_dir, seed=0)
    # PHASE B: the runtime reads splits from the single canonical location cacheutils.SCRATCH/"splits"
    # (no RB_SPLITS_DIR); redirect SCRATCH so the pair consumes the temp artifacts.
    monkeypatch.setattr(RS.cacheutils, "SCRATCH", tmp_path)

    monkeypatch.setattr(RS.cacheutils, "OUTPUT_DIR", tmp_path / "out")
    # PHASE B: runtime builds patch_fold from bench.patches and patch_tile from bench.patch_tiles();
    # the synthetic cache reuses patch ids across folds, and a source-only random_id split spans all
    # folds, so a single fold (its cache-layout dir) is consistent for every patch.
    patches = [SimpleNamespace(patch_id=p, fold=1, tile=f"T3{p % 4}") for p in pids]
    monkeypatch.setattr(RS.cacheutils, "cached_bench", lambda *a, **k: SimpleNamespace(
        name="pastis", data_quality=None, patch_classes=None, n_samples=4,
        patches=patches, patch_ids=lambda folds=None, _p=pids: list(_p),
        patch_tiles={p.patch_id: p.tile for p in patches}))  # @property dict on the real bench
    monkeypatch.setattr(RS.cacheutils, "require_dense_cache", lambda *a, **k: emb_dir)
    monkeypatch.setattr(RS, "build_run_manifest", lambda *a, **k: _STUB_MANIFEST)
    monkeypatch.setattr(artifacts, "capture_environment", lambda repo=None: _env(sklearn="1.9.0"))

    RS._run_segmentation_pair(
        "pastis", "raw", [0], 10_000, ["random_id"], ["probing"], ["logistic"],
        {"source": [1.0], "target": [0]}, False, False, True, False, {},  # s2_only=False
    )
    return tmp_path / "out" / "results" / "raw" / "pastis"


def test_dense_pair_schedules_and_executes_a_real_cell(monkeypatch, tmp_path) -> None:
    """The caller reaches the cell, the cell runs, and the pair marks itself complete."""
    results_dir = _run_dense_healthy(monkeypatch, tmp_path)

    rows = IOU.read_jsonl(results_dir / "probe_results.jsonl")
    assert rows, "the healthy dense pair scheduled no work -- no cell was executed"

    # write_run_complete raises IncompleteRunError rather than publishing a partial marker, so
    # reaching here already means the pair finished; validate it the way a reader would.
    ok, problems = artifacts.validate_run_complete(
        results_dir, expected_signature=runstate.run_manifest_digest(_STUB_MANIFEST)
    )
    assert ok, problems


def test_dense_pair_passes_the_cell_its_arguments_in_the_right_order(monkeypatch, tmp_path) -> None:
    """The exact drift that TypeError hid: method_name occupying `family`.

    Removing the stale argument is only half the fix -- the remaining ten had to stay aligned.
    If `family` still received method_name the rows would come back stamped probe_family="erm",
    which no gate downstream would reject: "erm" is a legal string in that column.
    """
    results_dir = _run_dense_healthy(monkeypatch, tmp_path)
    rows = IOU.read_jsonl(results_dir / "probe_results.jsonl")

    assert {r["probe_family"] for r in rows} == {"logistic"}, "family got the wrong argument"
    assert {r["method"] for r in rows} == {"erm"}
    assert {r["model"] for r in rows} == {"raw"}
    assert {r["benchmark"] for r in rows} == {"pastis"}
    assert {r["split_regime"] for r in rows} == {"random_id"}
    # source-only: random_id has no target region to sweep (HAS_TARGET=False)
    assert {r["budget_type"] for r in rows} == {"source"}
    assert {r["evaluation_split"] for r in rows} == {"validation", "test"}
    # max_dense_pixels reached the cell as a pixel budget, not as some other positional value
    assert all(r["n_test"] > 0 for r in rows)


def test_dense_pair_feeds_the_probe_the_cached_embedding_width(monkeypatch, tmp_path) -> None:
    """No coordinate columns: the probe sees the 5-dim cache, not 5+2."""
    from utils import runstate as RS

    seen = {}
    real = RS.cacheutils.load_dense_samples

    def spy(*a, **k):
        out = real(*a, **k)
        seen.setdefault("width", out[0].shape[1])
        return out

    monkeypatch.setattr(RS.cacheutils, "load_dense_samples", spy)
    _run_dense_healthy(monkeypatch, tmp_path)

    assert seen["width"] == 5, "the dense loader augmented the frozen embedding"


# --- the HEALTHY OFFICIAL dense path: zero-shot on the fold-5 target_test patches -----------------


def _dense_cache_official_on_disk(root, *, patches_per_fold=3, pixels=30, dim=5, classes=3):
    """A real dense cache with DISTINCT patch ids per fold (folds 1-5), for the official fold split."""
    rng = np.random.default_rng(0)
    fold_patches = {}
    for fold in range(1, 6):
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


def _run_dense_official(monkeypatch, tmp_path):
    from evals import split_artifacts as SA
    from evals.benchmarks import pastis as pastis_mod
    from evals.regimes import official
    from utils import runstate as RS

    emb_dir, fp = _dense_cache_official_on_disk(tmp_path / "emb")
    patches = [SimpleNamespace(patch_id=p, fold=f, tile=f"T3{p % 4}") for f, pids in fp.items() for p in pids]
    all_pids = [p.patch_id for p in patches]
    bench = SimpleNamespace(
        name="pastis", data_quality=None, patch_classes=None, n_samples=4, patches=patches,
        patch_ids=lambda folds=None, _p=all_pids: list(_p),
        patch_class_sets=lambda ids=None, _p=all_pids: {int(q): {0, 1, 2} for q in (ids if ids is not None else _p)},
        patch_tiles={p.patch_id: p.tile for p in patches},
        patch_latlon={p.patch_id: (0.0, 0.0) for p in patches},
    )
    # publish a canonical official dense leaf (folds 1-3 source_train, 4 source_val, 5 target_test)
    dsplit = next(iter(official.iter_dense_source_target_splits(bench, pastis_mod, 0)))
    domain_of = {int(k): str(v) for k, v in official.patch_domains(bench, pastis_mod).items()}
    rows, summary = SA.build_dense_leaf(
        "pastis", "official", 0, dense_split=dsplit, audit_events=[],
        all_patch_ids=all_pids, domain_of=domain_of,
        class_sets={int(p): {0, 1, 2} for p in all_pids}, patch_latlon=dict(bench.patch_latlon),
    )
    splitfix.freeze(tmp_path / "splits", [(rows, summary)])

    monkeypatch.setattr(RS.cacheutils, "SCRATCH", tmp_path)
    monkeypatch.setattr(RS.cacheutils, "OUTPUT_DIR", tmp_path / "out")
    monkeypatch.setattr(RS.cacheutils, "cached_bench", lambda *a, **k: bench)
    monkeypatch.setattr(RS.cacheutils, "require_dense_cache", lambda *a, **k: emb_dir)
    monkeypatch.setattr(RS, "build_run_manifest", lambda *a, **k: _STUB_MANIFEST)
    monkeypatch.setattr(artifacts, "capture_environment", lambda repo=None: _env(sklearn="1.9.0"))

    RS._run_segmentation_pair(
        "pastis", "raw", [0], 10_000, ["official"], ["probing"], ["logistic"],
        {"source": [1.0], "target": [0]}, False, False, True, False, {},  # s2_only=False
    )
    return tmp_path / "out" / "results" / "raw" / "pastis"


def test_dense_pair_consumes_official_zero_shot_end_to_end(monkeypatch, tmp_path) -> None:
    """official dense (has_target=True, supports_target_labels=False): the pair schedules a real cell,
    evaluates ZERO-SHOT on the fold-5 target_test patches (source budgets only, NO target sweep), and
    completes."""
    results_dir = _run_dense_official(monkeypatch, tmp_path)
    rows = IOU.read_jsonl(results_dir / "probe_results.jsonl")

    assert rows, "the official dense pair scheduled no work"
    assert {r["split_regime"] for r in rows} == {"official"}
    assert {r["holdout"] for r in rows} == {"fold_5"}
    assert {r["budget_type"] for r in rows} == {"source"}   # zero-shot: no target sweep
    assert {r["evaluation_split"] for r in rows} == {"validation", "test"}
    ok, problems = artifacts.validate_run_complete(
        results_dir, expected_signature=runstate.run_manifest_digest(_STUB_MANIFEST)
    )
    assert ok, problems


def _run_dense_geographic(monkeypatch, tmp_path, *, write_predictions=False, probe_cap=None,
                          controlled_cap=None):
    from evals import split_artifacts as SA
    from evals import split_spec
    from evals.benchmarks import pastis as pastis_mod
    from evals.regimes import geographic_ood as geo
    from utils import runstate as RS

    tiles = list(split_spec.PASTIS.geographic_targets)  # 4 Sentinel tiles rotate as LODO targets
    centers = {t: (45.0 + 3 * i, -1.0 + 3 * i) for i, t in enumerate(tiles)}  # tiles spatially separate
    rng = np.random.default_rng(0)
    # 64 patches per tile: when a tile is the LODO target its 80% pool (51 patches) clears the label-
    # access feasibility floor max(LABEL_ACCESS_COUNTS)=50, and the 3-tile source (~152 train patches)
    # clears it too, so the full PATCH-level route suite fires for every tile.
    patches, pid = [], 100
    for tile in tiles:
        la, lo = centers[tile]
        for k in range(64):
            fold = (pid % 5) + 1
            d = tmp_path / "emb" / f"fold_{fold}"
            d.mkdir(parents=True, exist_ok=True)
            y = np.tile(np.arange(3), 5)[:15].astype(np.int64)
            x = (rng.normal(size=(15, 5)) + y[:, None] * 1.5).astype(np.float32)
            np.save(d / f"{pid}_0_0.npy", x)
            np.save(d / f"{pid}_0_0.labels.npy", y)
            patches.append(SimpleNamespace(patch_id=pid, fold=fold, tile=tile, latlon=(la + k * 0.005, lo + k * 0.005)))
            pid += 1
    emb_dir = tmp_path / "emb"
    all_pids = [p.patch_id for p in patches]
    bench = SimpleNamespace(
        name="pastis", data_quality=None, patch_classes=None, n_samples=4, patches=patches,
        patch_ids=lambda folds=None, _p=all_pids: list(_p),
        patch_class_sets=lambda ids=None, _p=all_pids: {int(q): {0, 1, 2} for q in (ids if ids is not None else _p)},
        patch_tiles={p.patch_id: p.tile for p in patches},
        patch_latlon={p.patch_id: p.latlon for p in patches},
    )
    # publish ALL tile-LODO leaves via the real generator, so the pair consumes canonical artifacts
    from evals.regimes import random_id as rid

    domain_of = {int(k): str(v) for k, v in geo.patch_domains(bench, pastis_mod).items()}
    labels, built, la_specs = [], [], []
    for dsplit in geo.iter_dense_source_target_splits(bench, pastis_mod, 0):
        rows, summary = SA.build_dense_leaf(
            "pastis", "geographic_ood", 0, dense_split=dsplit, audit_events=[],
            all_patch_ids=all_pids, domain_of=domain_of,
            class_sets={int(p): {0, 1, 2} for p in all_pids}, patch_latlon=dict(bench.patch_latlon), purge_km=2.0,
        )
        built.append((rows, summary))
        labels.append(dsplit.label)
        if dsplit.supports_target_labels:  # geographic_ood headline target: also carries label_access.csv
            la_specs.append((
                str(dsplit.label),
                [str(int(p)) for p in sorted(dsplit.source_train_patches)],
                [str(int(p)) for p in sorted(dsplit.target_label_pool_patches)],
                [str(int(p)) for p in sorted(dsplit.target_test_patches)],
            ))
    # random_id (random_patch) source-only leaf -> the Stage-5 source_ID_reference anchor.
    rid_domain = {int(k): str(v) for k, v in rid.patch_domains(bench, pastis_mod).items()}
    rsplit = next(iter(rid.iter_dense_source_target_splits(bench, pastis_mod, 0)))
    rid_rows, rid_summary = SA.build_dense_leaf(
        "pastis", "random_id", 0, dense_split=rsplit, audit_events=[], all_patch_ids=all_pids,
        domain_of=rid_domain, class_sets={int(p): {0, 1, 2} for p in all_pids},
        patch_latlon=dict(bench.patch_latlon), purge_km=0.0,
    )
    built.append((rid_rows, rid_summary))
    splitfix.freeze(tmp_path / "splits", built)
    for label, src_ids, pool_ids, test_ids in la_specs:  # frozen patch-id orders, sibling of assignments.csv
        la_rows = SA.build_label_access_rows(seed=0, source_ids=src_ids, target_pool_ids=pool_ids, target_test_ids=test_ids)
        splitfix.attach_label_access(tmp_path / "splits", "pastis", 0, label, la_rows)

    monkeypatch.setattr(RS.cacheutils, "SCRATCH", tmp_path)
    monkeypatch.setattr(RS.cacheutils, "OUTPUT_DIR", tmp_path / "out")
    monkeypatch.setattr(RS.cacheutils, "cached_bench", lambda *a, **k: bench)
    monkeypatch.setattr(RS.cacheutils, "require_dense_cache", lambda *a, **k: emb_dir)
    monkeypatch.setattr(RS, "build_run_manifest", lambda *a, **k: _STUB_MANIFEST)
    monkeypatch.setattr(artifacts, "capture_environment", lambda repo=None: _env(sklearn="1.9.0"))
    if probe_cap is not None:
        monkeypatch.setattr(RS.perf, "PROBE_CAP", probe_cap)   # exercised through the real dense dispatch
    if controlled_cap is not None:                             # the label-access TOTAL-budget cap
        monkeypatch.setattr(RS, "CONTROLLED_TOTAL_BUDGET_CAP", controlled_cap)

    RS._run_segmentation_pair(
        "pastis", "raw", [0], 10_000, ["random_id", "geographic_ood"], ["probing"], ["logistic"],
        {"source": [1.0], "target": [0]}, False, False, True, write_predictions, {},
    )
    return tmp_path / "out" / "results" / "raw" / "pastis", tiles


def test_dense_pair_consumes_geographic_tile_lodo_few_shot_end_to_end(monkeypatch, tmp_path) -> None:
    """geographic_ood dense (has_target=True, supports_target_labels=True): tile-LODO, the pair runs the
    PATCH-level label-access route suite (replacing the legacy target sweep) alongside the zero-shot
    source sweep, scoring every route on the frozen target_test patches, and completes -- which means the
    patch-level semantic validation inside write_run_complete passed."""
    from evals import split_artifacts as SA

    results_dir, tiles = _run_dense_geographic(monkeypatch, tmp_path)
    rows = IOU.read_jsonl(results_dir / "probe_results.jsonl")

    # a real label-access run also runs random_id (the source_ID_reference anchor); scope the geographic
    # assertions to its rows.
    assert {r["split_regime"] for r in rows} == {"geographic_ood", "random_id"}
    geo_rows = [r for r in rows if r["split_regime"] == "geographic_ood"]
    assert {r["holdout"] for r in geo_rows} == {str(t) for t in tiles}     # one fold per Sentinel tile
    # the label-access suite REPLACES the legacy target sweep -- source sweep + label_access, no "target".
    assert {r["budget_type"] for r in geo_rows} == {"source", "label_access"}
    assert not [r for r in rows if r["budget_type"] == "target"]

    la = [r for r in geo_rows if r["budget_type"] == "label_access"]
    assert la and {r["label_budget_unit"] for r in la} == {"patches"}   # dense unit is PATCHES
    for tile in tiles:
        tla = [r for r in la if r["holdout"] == str(tile)]
        assert {(r["label_access_route"], r["label_budget"], r["evaluation_split"]) for r in tla} == set(
            SA.label_access_expected_rows()
        )  # exactly the canonical route x budget set, all on the frozen target_test
        tt = [r for r in tla if r["evaluation_split"] == SA.EVAL_TARGET_TEST]
        assert len(tt) == len(SA.label_access_expected_rows())
        # label access emits NO complete_target row -- the deployment estimand is the E1 source row.
        assert not [r for r in tla if r["evaluation_split"] == SA.EVAL_COMPLETE_TARGET]
        # patch-level accounting: every route scored on the SAME frozen target_test patch count; the
        # E1 deployment scope covers the whole region (pool + test), so its n_test is strictly larger.
        assert len({r["n_test"] for r in tt}) == 1
        deploy = [r for r in geo_rows if r["holdout"] == str(tile) and r["budget_type"] == "source"
                  and r["evaluation_split"] == SA.EVAL_COMPLETE_TARGET]
        assert len(deploy) == 1 and deploy[0]["n_test"] > tt[0]["n_test"]

    ok, problems = artifacts.validate_run_complete(
        results_dir, expected_signature=runstate.run_manifest_digest(_STUB_MANIFEST)
    )
    assert ok, problems


def test_dense_geographic_dispatch_honors_the_controlled_budget_cap(monkeypatch, tmp_path) -> None:
    """The PRODUCTION dense dispatch threads CONTROLLED_TOTAL_BUDGET_CAP into the label-access suite.

    Under the fixed-budget contract the cap is applied to the BUDGET (``B_dC = min(B_d, C)``) and never
    after route construction -- a post-hoc cap would shrink a mixed route's realized total and silently
    break the fixed-budget claim. So every allocation point still totals exactly ``B_dC``, and the pair
    still completes (the patch-level semantic validation passes at the capped budget)."""
    from evals import split_artifacts as SA

    results_dir, _tiles = _run_dense_geographic(monkeypatch, tmp_path, controlled_cap=30)
    rows = IOU.read_jsonl(results_dir / "probe_results.jsonl")
    la = [r for r in rows if r["budget_type"] == "label_access"]
    assert la
    assert {r["controlled_budget_cap"] for r in la} == {30}
    assert {r["allocation_total_budget"] for r in la} == {30}      # B_dC = min(B_d, C) = C
    assert all(r["benchmark_budget"] >= 30 for r in la)            # the frozen B_d is the uncapped one
    # the fixed-budget allocation curve is drawn at the CAPPED budget: total pinned, composition moves.
    alloc = [r for r in la if r["label_access_route"] == SA.ROUTE_FIXED_BUDGET_ALLOCATION]
    assert {r["label_budget"] for r in alloc} == set(SA.ALLOCATION_PERCENTS)
    for r in alloc:
        k = SA.allocation_target_count(int(r["label_budget"]), 30)
        assert (r["n_total_labels"], r["n_target_labels"], r["n_source_labels"]) == (30, k, 30 - k)
    ok, problems = artifacts.validate_run_complete(
        results_dir, expected_signature=runstate.run_manifest_digest(_STUB_MANIFEST)
    )
    assert ok, problems


def test_dense_geographic_streams_predictions_with_identity(monkeypatch, tmp_path) -> None:
    """With WRITE_PREDICTIONS on, the dense label-access suite streams per-pixel predictions to
    predictions.jsonl through the real pair, each record carrying stable patch/pixel identity, route,
    evaluation split, budget, seed, and patch-level supervision counts."""
    from evals import split_artifacts as SA

    results_dir, tiles = _run_dense_geographic(monkeypatch, tmp_path, write_predictions=True)
    preds_path = results_dir / "predictions.jsonl"
    assert preds_path.exists()
    preds = IOU.read_jsonl(preds_path)
    assert preds
    required = {"patch_id", "tile_row", "tile_col", "pixel_index", "sample_id", "label_access_route",
                "evaluation_split", "label_budget", "seed", "budget_type", "n_source_labels",
                "n_target_labels", "n_total_labels", "label_budget_unit"}
    sample = preds[0]
    assert required <= set(sample)
    assert sample["sample_id"] == f'{sample["patch_id"]}:{sample["tile_row"]}:{sample["tile_col"]}:{sample["pixel_index"]}'
    assert {p["label_budget_unit"] for p in preds} == {"patches"}
    assert {p["holdout"] for p in preds} == {str(t) for t in tiles}
    # every canonical route appears in the streamed predictions, all on the frozen target_test
    assert {p["label_access_route"] for p in preds} == set(SA.LABEL_ACCESS_ROUTES)
    assert {p["evaluation_split"] for p in preds} == {SA.EVAL_TARGET_TEST}


def test_dense_prediction_resume_is_transactional(monkeypatch, tmp_path) -> None:
    """Crash-window: a family appended predictions but its result rows never published. On resume the
    pair keeps ONLY the predictions of fully-done families, so the orphan family's records are dropped
    and re-running cannot duplicate them."""
    from utils import runstate as RS

    results_dir, _tiles = _run_dense_geographic(monkeypatch, tmp_path, write_predictions=True)
    preds_path = results_dir / "predictions.jsonl"
    original = IOU.read_jsonl(preds_path)
    assert original
    # Simulate the crash window: predictions for a family whose result rows never landed (holdout GHOST,
    # so its _fam_key is not among the done families reconstructed from probe_results.jsonl).
    orphan = {**original[0], "holdout": "GHOST", "patch_id": -1, "tile_row": 0, "tile_col": 0,
              "pixel_index": 0, "sample_id": "-1:0:0:0"}
    IOU.append_jsonl(preds_path, [orphan])
    assert any(p.get("holdout") == "GHOST" for p in IOU.read_jsonl(preds_path))

    # Resume the SAME pair (monkeypatches still active). All real families are done -> no jobs run, but
    # the orphan's predictions are pruned, and the real predictions are neither dropped nor duplicated.
    RS._run_segmentation_pair(
        "pastis", "raw", [0], 10_000, ["random_id", "geographic_ood"], ["probing"], ["logistic"],
        {"source": [1.0], "target": [0]}, False, False, True, True, {},
    )
    after = IOU.read_jsonl(preds_path)
    assert not any(p.get("holdout") == "GHOST" for p in after)
    assert len(after) == len(original)   # real predictions intact, no duplication


# --- the HEALTHY tabular path: consume published splits end-to-end -----------------------
#
# The dense healthy tests above prove the dense caller reaches its cell. This proves the tabular
# caller does too -- over REAL published canonical split artifacts, a real _run_tabular_pair probing
# cell, real logistic probes, and the full completion contract. build_run_manifest is NOT stubbed, so
# the run manifest actually carries the consumed split set it bound itself to (req 3 checks that).


def _ch_like_bench(n_per=12):
    """A synthetic cropharvest-shaped bench the real ch.make_targets accepts, with enough samples for
    a stratified random_id split. Only the attributes _run_tabular_pair touches are populated."""
    centers = {"kenya": (0.5, 37.0), "togo": (8.0, 1.0), "ethiopia": (9.0, 40.0)}
    rng = np.random.default_rng(0)
    groups, labels, latlon, sids = [], [], [], []
    for dom, (clat, clon) in centers.items():
        for i in range(n_per):
            groups.append(dom)
            labels.append(i % 2)  # balanced two classes per domain
            latlon.append((clat + rng.normal(0, 0.05), clon + rng.normal(0, 0.05)))
            sids.append(f"{dom}_{i}")
    return SimpleNamespace(
        name="cropharvest",
        groups=np.asarray(groups, dtype=object),
        labels=np.asarray(labels, dtype=np.int64),
        latlon=np.asarray(latlon, dtype=float),
        sample_ids=np.asarray(sids, dtype=object),
        years=None,
        data_quality=None,
        available_modalities=lambda: {"s2"},
    )


def _publish_tabular_random_splits(splits_root, bench, *, seed=0):
    """Publish a schema-v2 random_id SourceTargetSplit for `bench` -- the runtime CONSUMES these."""
    from evals import split_artifacts as SA
    from evals.benchmarks import cropharvest as ch
    from evals.regimes import random_id

    y, _g = ch.make_targets(bench)
    domains = random_id.sample_domains(bench, ch)
    labels_seen, built = [], []
    for split in random_id.iter_source_target_splits(bench, ch, seed):
        rows, summary = SA.build_tabular_leaf(
            "cropharvest", "random_id", seed, split=split, domains=domains,
            labels=y, sample_ids=bench.sample_ids, audit_events=[],
        )
        built.append((rows, summary))
        labels_seen.append(str(split.label))
    splitfix.freeze(splits_root, built)
    return labels_seen


def _run_tabular_healthy(monkeypatch, tmp_path):
    import main
    from evals.benchmarks import cropharvest as ch

    bench = _ch_like_bench()
    labels = _publish_tabular_random_splits(tmp_path / "splits", bench, seed=0)

    # embeddings linearly separable by label so the logistic probe fits and emits rows
    y, _g = ch.make_targets(bench)
    emb = np.zeros((len(bench.sample_ids), 4), dtype=np.float32)
    emb[np.asarray(y) == 1] = 1.0

    monkeypatch.setattr(main.EV, "load_benchmark", lambda _n: ch)  # the REAL cropharvest module
    monkeypatch.setattr(main.cacheutils, "SCRATCH", tmp_path)              # splits under tmp_path/"splits"
    monkeypatch.setattr(main.cacheutils, "OUTPUT_DIR", tmp_path / "out")
    monkeypatch.setattr(main.cacheutils, "cached_bench", lambda *a, **k: bench)
    monkeypatch.setattr(main.cacheutils, "load_cached_embeddings", lambda *a, **k: emb)
    monkeypatch.setattr(main.compat, "input_modalities", lambda _m: {"s2"})
    monkeypatch.setattr(artifacts, "capture_environment", lambda repo=None: _env(sklearn="1.9.0"))
    # build_run_manifest is intentionally NOT stubbed: the real builder runs end to end here.

    main._run_tabular_pair(
        "cropharvest", "raw", [0], 0, ["random_id"], ["probing"], ["logistic"],
        {"source": [1.0], "target": [0]}, False, False, False, {},  # s2_only / overwrite / strict = False
    )
    return tmp_path / "out" / "results" / "raw" / "cropharvest", labels


def test_tabular_pair_consumes_published_splits_end_to_end(monkeypatch, tmp_path) -> None:
    from evals import split_artifacts as SA

    results_dir, labels = _run_tabular_healthy(monkeypatch, tmp_path)

    rows = IOU.read_jsonl(results_dir / "probe_results.jsonl")
    assert rows, "the healthy tabular pair scheduled no work -- no cell was executed"
    assert {r["seed"] for r in rows} == {0}
    assert {r["split_regime"] for r in rows} == {"random_id"}
    assert {r["holdout"] for r in rows} == set(labels)
    assert {r["model"] for r in rows} == {"raw"}
    assert {r["benchmark"] for r in rows} == {"cropharvest"}
    assert {r["probe_family"] for r in rows} == {"logistic"}

    # the frozen assignments.csv leaf the runtime consumed exists under the canonical splits root
    assert SA.assignments_path(tmp_path / "splits", "cropharvest", "random_id", 0, labels[0]).is_file()

    # completion validation succeeds against the run's own manifest signature
    man = json.loads((results_dir / "run_manifest.json").read_text())
    ok, problems = artifacts.validate_run_complete(
        results_dir, expected_signature=runstate.run_manifest_digest(man)
    )
    assert ok, problems


# --- the HEALTHY OFFICIAL path: consume a published official split, ZERO-SHOT on target_test ------
#
# official is has_target=True / supports_target_labels=False: fit source_train, calibrate source_val,
# evaluate ZERO-SHOT on target_test -- source budgets only, NO target-budget sweep. This proves the
# runtime routes that capability end to end over a real published official leaf.


def _ch_official_bench(n_source=40, n_target=12):
    """A cropharvest-shaped bench with real `togo`/`togo-eval` provenance ids for the official split."""
    rng = np.random.default_rng(0)
    sids, labels, groups, latlon = [], [], [], []
    i = 0
    for prov, n in (("togo", n_source), ("togo-eval", n_target)):
        for k in range(n):
            sids.append(f"{i}_{prov}.h5")
            labels.append(k % 2)
            groups.append("togo")
            latlon.append((8.0 + rng.normal(0, 0.05), 1.0 + rng.normal(0, 0.05)))
            i += 1
    return SimpleNamespace(
        name="cropharvest", groups=np.asarray(groups, dtype=object),
        labels=np.asarray(labels, dtype=np.int64), latlon=np.asarray(latlon, dtype=float),
        sample_ids=np.asarray(sids, dtype=object), years=None, data_quality=None,
        available_modalities=lambda: {"s2"},
    )


def _publish_official_tabular_split(splits_root, bench, *, seed=0):
    from evals import split_artifacts as SA
    from evals.benchmarks import cropharvest as ch
    from evals.regimes import official

    y, _g = ch.make_targets(bench)
    domains = official.sample_domains(bench, ch)
    split = next(iter(official.iter_source_target_splits(bench, ch, seed)))
    rows, summary = SA.build_tabular_leaf(
        "cropharvest", "official", seed, split=split, domains=domains, labels=y,
        sample_ids=bench.sample_ids, audit_events=[],
    )
    splitfix.freeze(splits_root, [(rows, summary)])
    return [split.label]


def test_tabular_pair_consumes_official_zero_shot_end_to_end(monkeypatch, tmp_path) -> None:
    import main
    from evals import split_artifacts as SA
    from evals.benchmarks import cropharvest as ch

    bench = _ch_official_bench()
    labels = _publish_official_tabular_split(tmp_path / "splits", bench, seed=0)

    y, _g = ch.make_targets(bench)
    emb = np.zeros((len(bench.sample_ids), 4), dtype=np.float32)
    emb[np.asarray(y) == 1] = 1.0

    monkeypatch.setattr(main.EV, "load_benchmark", lambda _n: ch)
    monkeypatch.setattr(main.cacheutils, "SCRATCH", tmp_path)
    monkeypatch.setattr(main.cacheutils, "OUTPUT_DIR", tmp_path / "out")
    monkeypatch.setattr(main.cacheutils, "cached_bench", lambda *a, **k: bench)
    monkeypatch.setattr(main.cacheutils, "load_cached_embeddings", lambda *a, **k: emb)
    monkeypatch.setattr(main.compat, "input_modalities", lambda _m: {"s2"})
    monkeypatch.setattr(artifacts, "capture_environment", lambda repo=None: _env(sklearn="1.9.0"))

    main._run_tabular_pair(
        "cropharvest", "raw", [0], 0, ["official"], ["probing"], ["logistic"],
        {"source": [1.0], "target": [0]}, False, False, False, {},
    )
    results_dir = tmp_path / "out" / "results" / "raw" / "cropharvest"
    rows = IOU.read_jsonl(results_dir / "probe_results.jsonl")

    assert rows, "the official pair scheduled no work"
    assert {r["split_regime"] for r in rows} == {"official"}
    assert {r["holdout"] for r in rows} == set(labels) == {"togo"}
    # ZERO-SHOT: official emits source budgets ONLY (no target-label sweep) evaluated on target_test.
    assert {r["budget_type"] for r in rows} == {"source"}
    assert {r["evaluation_split"] for r in rows} == {"test"}
    # the eval set is the togo-eval target_test (fixed release evaluation)
    assert {r["n_test"] for r in rows} == {12}

    assert SA.assignments_path(tmp_path / "splits", "cropharvest", "official", 0, labels[0]).is_file()


# --- the HEALTHY GEOGRAPHIC path: source zero-shot + the label-access route suite ----------------
#
# geographic_ood is has_target=True / supports_target_labels=True and is the label-access HEADLINE
# regime: the source budgets evaluate zero-shot on target_test, AND the frozen label_access.csv drives
# the 13 label-access routes (source_only / source_plus_target(k) / target_only_full /
# source_plus_target_full / matched_source / matched_target / fixed_total_mixed(k)), each scored on the
# SAME frozen target_test, plus the single source_only complete-target diagnostic. This suite REPLACES
# the old target-budget sweep. The test proves the runtime routes that capability end to end and that
# target_test is invariant across every route.


def _ch_geo_bench(per=40):
    """A cropharvest-shaped bench: kenya is the LODO target; togo/ethiopia/lem-brazil are the source."""
    centers = {"kenya": (0.5, 37.0), "togo": (8.0, 1.0), "ethiopia": (9.0, 40.0), "lem-brazil": (-12.0, -55.0)}
    rng = np.random.default_rng(0)
    groups, labels, latlon, sids = [], [], [], []
    for dom, (la, lo) in centers.items():
        for i in range(per):
            groups.append(dom)
            labels.append(i % 2)
            latlon.append((la + rng.normal(0, 0.05), lo + rng.normal(0, 0.05)))
            sids.append(f"{dom}_{i}")
    return SimpleNamespace(
        name="cropharvest", groups=np.asarray(groups, dtype=object), labels=np.asarray(labels, dtype=np.int64),
        latlon=np.asarray(latlon, dtype=float), sample_ids=np.asarray(sids, dtype=object),
        years=None, data_quality=None, available_modalities=lambda: {"s2"},
    )


def _publish_geographic_kenya_split(splits_root, bench, *, seed=0):
    from evals import split_artifacts as SA
    from evals.benchmarks import cropharvest as ch
    from evals.regimes import geographic_ood as geo
    from evals.regimes import random_id as rid

    y, _g = ch.make_targets(bench)
    domains = geo.sample_domains(bench, ch)
    split = next(s for s in geo.iter_source_target_splits(bench, ch, seed) if s.label == "kenya")
    geo_rows, geo_summary = SA.build_tabular_leaf(
        "cropharvest", "geographic_ood", seed, split=split, domains=domains, labels=y,
        sample_ids=bench.sample_ids, audit_events=[], purge_km=50.0,
    )
    # A real label-access run also publishes random_id -- its full-source in-distribution result is the
    # Stage-5 source_ID_reference anchor. Publish it alongside so the contrasts resolve.
    rsplit = next(iter(rid.iter_source_target_splits(bench, ch, seed)))
    rid_rows, rid_summary = SA.build_tabular_leaf(
        "cropharvest", "random_id", seed, split=rsplit, domains=rid.sample_domains(bench, ch), labels=y,
        sample_ids=bench.sample_ids, audit_events=[], purge_km=0.0,
    )
    splitfix.freeze(splits_root, [(geo_rows, geo_summary), (rid_rows, rid_summary)])
    # The geographic_ood headline target also carries a frozen label_access.csv sibling (built from the
    # SAME source_train / target_label_pool / target_test the runtime loads), exactly as the generator does.
    sids = [str(s) for s in np.asarray(bench.sample_ids).tolist()]
    la_rows = SA.build_label_access_rows(
        seed=seed,
        source_ids=[sids[int(i)] for i in split.source_train],
        target_pool_ids=[sids[int(i)] for i in split.target_label_pool],
        target_test_ids=[sids[int(i)] for i in split.target_test],
    )
    splitfix.attach_label_access(splits_root, "cropharvest", seed, str(split.label), la_rows)
    return split


def test_tabular_pair_consumes_geographic_few_shot_end_to_end(monkeypatch, tmp_path) -> None:
    import main
    from evals.benchmarks import cropharvest as ch

    # per=70: kenya's 80% target_label_pool (56) and the 3-region source_train (168) both clear the
    # label-access feasibility floor (>= max(LABEL_ACCESS_COUNTS)=50), so the full route suite fires.
    bench = _ch_geo_bench(per=70)
    split = _publish_geographic_kenya_split(tmp_path / "splits", bench, seed=0)

    y, _g = ch.make_targets(bench)
    emb = np.zeros((len(bench.sample_ids), 4), dtype=np.float32)
    emb[np.asarray(y) == 1] = 1.0

    monkeypatch.setattr(main.EV, "load_benchmark", lambda _n: ch)
    monkeypatch.setattr(main.cacheutils, "SCRATCH", tmp_path)
    monkeypatch.setattr(main.cacheutils, "OUTPUT_DIR", tmp_path / "out")
    monkeypatch.setattr(main.cacheutils, "cached_bench", lambda *a, **k: bench)
    monkeypatch.setattr(main.cacheutils, "load_cached_embeddings", lambda *a, **k: emb)
    monkeypatch.setattr(main.compat, "input_modalities", lambda _m: {"s2"})
    monkeypatch.setattr(artifacts, "capture_environment", lambda repo=None: _env(sklearn="1.9.0"))

    # A real label-access run also runs random_id (the source_ID_reference anchor for Stage-5 contrasts).
    main._run_tabular_pair(
        "cropharvest", "raw", [0], 0, ["random_id", "geographic_ood"], ["probing"], ["logistic"],
        {"source": [1.0], "target": [0, 5]}, False, False, False, {},
    )
    results_dir = tmp_path / "out" / "results" / "raw" / "cropharvest"
    rows = IOU.read_jsonl(results_dir / "probe_results.jsonl")

    from evals import contrasts
    from evals import split_artifacts as SA

    assert {r["split_regime"] for r in rows} == {"geographic_ood", "random_id"}
    geo_rows = [r for r in rows if r["split_regime"] == "geographic_ood"]
    assert {r["holdout"] for r in geo_rows} == {"kenya"}
    # geographic_ood headline runs the canonical LABEL-ACCESS suite (which REPLACES the old target-budget
    # sweep -- no legacy budget_type="target" rows) alongside the unchanged source-budget sweep.
    assert {r["budget_type"] for r in geo_rows} == {"source", "label_access"}
    assert not [r for r in rows if r["budget_type"] == "target"]

    la = [r for r in geo_rows if r["budget_type"] == "label_access"]
    tt = [r for r in la if r["evaluation_split"] == SA.EVAL_TARGET_TEST]
    # the whole suite is scored on target_test; label access emits NO complete_target row.
    assert len(tt) == len(la) == len(SA.label_access_expected_rows())
    assert not [r for r in la if r["evaluation_split"] == SA.EVAL_COMPLETE_TARGET]
    assert {(r["label_access_route"], r["label_budget"], r["evaluation_split"]) for r in la} == set(
        SA.label_access_expected_rows()
    )
    # target-test INVARIANCE: every route is scored on the SAME frozen target_test.
    assert {r["n_test"] for r in tt} == {len(split.target_test)}
    # the deployment estimand rides on the ordinary full-source (E1) fit, scored on the WHOLE region.
    deploy = [r for r in geo_rows if r["budget_type"] == "source"
              and r["evaluation_split"] == SA.EVAL_COMPLETE_TARGET]
    assert len(deploy) == 1
    assert deploy[0]["n_test"] == len(split.target_label_pool) + len(split.target_test)
    assert deploy[0]["label_budget"] == 1.0 and deploy[0]["label_access_route"] == ""

    # label accounting per route: the allocation curve pins the TOTAL at the realized budget B and only
    # moves its composition; the additive routes hold the COMPLETE source pool and add k on top.
    S = len(split.source_train)
    by = {(r["label_access_route"], r["label_budget"]): r for r in tt}
    B = by[(SA.ROUTE_FIXED_BUDGET_ALLOCATION, 0)]["allocation_total_budget"]
    for f in SA.ALLOCATION_PERCENTS:
        k = SA.allocation_target_count(f, B)
        row = by[(SA.ROUTE_FIXED_BUDGET_ALLOCATION, f)]
        assert row["n_total_labels"] == B
        assert row["n_target_labels"] == k and row["n_source_labels"] == B - k
    for k in SA.LABEL_ACCESS_COUNTS:
        assert by[(SA.ROUTE_SOURCE_PLUS_TARGET, k)]["n_total_labels"] == S + k
        assert by[(SA.ROUTE_SOURCE_PLUS_TARGET, k)]["n_source_labels"] == S
        assert by[(SA.ROUTE_SOURCE_PLUS_TARGET, k)]["n_target_labels"] == k
    # target_only_full trains on the pool alone (no source); source_plus_target_full uses S + whole pool.
    assert by[(SA.ROUTE_TARGET_ONLY_FULL, 0)]["n_source_labels"] == 0
    assert by[(SA.ROUTE_TARGET_ONLY_FULL, 0)]["n_target_labels"] == len(split.target_label_pool)
    assert by[(SA.ROUTE_SOURCE_PLUS_TARGET_FULL, 0)]["n_total_labels"] == S + len(split.target_label_pool)

    # the source budgets evaluate zero-shot on the same fixed target_test ("test" scope), AND report
    # the untouched within-source reference on the manifest source_test partition ("source_test" scope).
    source_test_eval = [r for r in geo_rows if r["budget_type"] == "source" and r["evaluation_split"] == "test"]
    assert source_test_eval and {r["n_test"] for r in source_test_eval} == {len(split.target_test)}
    within_source = [r for r in geo_rows if r["budget_type"] == "source" and r["evaluation_split"] == "source_test"]
    assert within_source and {r["n_test"] for r in within_source} == {len(split.source_test)}
    # non-label-access rows carry the empty route identity; every geographic row carries the headline role.
    assert all(r["label_access_route"] == "" for r in geo_rows if r["budget_type"] == "source")
    assert {r["target_role"] for r in geo_rows} == {"headline"}

    # Stage 5: the paired-contrast artifacts were written and the run COMPLETED (run_complete.json only
    # exists if write_run_complete's recompute-and-compare passed), with the anchor resolved from the
    # random_id full-source result and the artifact hashes recorded in the marker.
    assert (results_dir / contrasts.CONTRAST_FILE).exists()
    assert (results_dir / contrasts.CONTRAST_SUMMARY_FILE).exists()
    marker = artifacts.read_run_complete(results_dir)
    assert marker is not None
    assert contrasts.CONTRAST_FILE in marker["artifacts"] and contrasts.CONTRAST_SUMMARY_FILE in marker["artifacts"]
    paired, _summary = contrasts.compute_contrasts(rows)
    assert {r["contrast"] for r in paired} == {c[0] for c in SA.LABEL_ACCESS_CONTRASTS}
    assert {r["target"] for r in paired} == {"kenya"}


# --- dirty-tree identity is content-sensitive --------------------------------


def _repo(root, files: dict[str, str], *, untracked: dict[str, str] | None = None):
    """A throwaway git repo with one commit, then the given tracked edits + untracked files.

    Committer identity and dates are fixed so two repos built from identical content produce the
    identical commit -- otherwise `tree_identity` would differ for a trivial reason and the tests
    below would prove nothing.
    """
    import subprocess

    root.mkdir(parents=True, exist_ok=True)
    env = {
        **os.environ,
        # Isolate from the developer's real git config. Without this the test inherits whatever
        # the machine sets -- e.g. commit.gpgsign=true, which fails non-interactively with
        # "gpg: signing failed: No pinentry" -- so the suite would pass or fail depending on
        # whose laptop it ran on.
        "GIT_CONFIG_GLOBAL": os.devnull,
        "GIT_CONFIG_SYSTEM": os.devnull,
        "GIT_AUTHOR_NAME": "t", "GIT_AUTHOR_EMAIL": "t@t",
        "GIT_COMMITTER_NAME": "t", "GIT_COMMITTER_EMAIL": "t@t",
        # Fixed dates so identical content yields an identical commit; otherwise tree_identity
        # would differ for a trivial reason and these tests would prove nothing.
        "GIT_AUTHOR_DATE": "2026-01-01T00:00:00 +0000",
        "GIT_COMMITTER_DATE": "2026-01-01T00:00:00 +0000",
    }

    def run(*args):
        subprocess.run(args, cwd=str(root), capture_output=True, check=True, env=env)

    run("git", "init", "-q", "-b", "main")
    (root / "base.py").write_text("x = 0\n")
    run("git", "add", "-A")
    run("git", "commit", "-qm", "base")
    for name, body in files.items():
        (root / name).write_text(body)
    for name, body in (untracked or {}).items():
        (root / name).write_text(body)
    return root


def test_identity_differs_for_different_edits_to_the_same_filenames(tmp_path) -> None:
    """A dirty-FILE-LIST cannot tell two different edits apart; the content hash must."""
    a = artifacts.git_identity(_repo(tmp_path / "a", {"base.py": "x = 1\n"}))
    b = artifacts.git_identity(_repo(tmp_path / "b", {"base.py": "x = 2\n"}))

    assert a["dirty_files"] == b["dirty_files"] == ["base.py"]  # identical lists...
    assert a["commit"] == b["commit"]                           # ...and identical commits
    assert a["tracked_diff_sha256"] != b["tracked_diff_sha256"]
    assert a["tree_identity"] != b["tree_identity"], "different code must not share an identity"


def test_identity_differs_for_different_untracked_source_content(tmp_path) -> None:
    a = artifacts.git_identity(_repo(tmp_path / "a", {}, untracked={"new.py": "def f(): return 1\n"}))
    b = artifacts.git_identity(_repo(tmp_path / "b", {}, untracked={"new.py": "def f(): return 2\n"}))

    assert a["untracked_source_sha256"] != b["untracked_source_sha256"]
    assert a["tree_identity"] != b["tree_identity"]


def test_identity_is_stable_for_identical_content(tmp_path) -> None:
    a = artifacts.git_identity(_repo(tmp_path / "a", {"base.py": "x = 1\n"}, untracked={"u.py": "y\n"}))
    b = artifacts.git_identity(_repo(tmp_path / "b", {"base.py": "x = 1\n"}, untracked={"u.py": "y\n"}))

    assert a["tree_identity"] == b["tree_identity"], "identical code must share an identity"


def test_identity_retains_the_human_readable_dirty_list(tmp_path) -> None:
    ident = artifacts.git_identity(_repo(tmp_path / "a", {"base.py": "x = 9\n"}, untracked={"u.py": "y\n"}))

    assert ident["dirty"] is True
    assert "base.py" in ident["dirty_files"]
    assert ident["n_untracked_source_files"] == 1
    assert ident["commit"] and ident["tree_identity"]


def test_identity_is_null_but_honest_without_a_git_repo(tmp_path) -> None:
    """Three of the machines that produce results are rsync copies with no .git."""
    ident = artifacts.git_identity(tmp_path)

    assert ident["commit"] is None and ident["tree_identity"] is None
    assert "not recoverable" in ident["note"]


# --- completeness -----------------------------------------------------------


def _key(seed=0, regime="geographic_ood", holdout="kenya", method="erm", family="logistic",
         bt="source", lb=1.0, es="test", route=""):
    return (seed, regime, holdout, method, family, bt, lb, es, route)


def _row(**over):
    k = _key(**over)
    return dict(zip(artifacts.CELL_KEY_FIELDS, k, strict=True))


def test_completeness_accepts_an_exact_match() -> None:
    expected = {_key(lb=b) for b in (0.05, 1.0)}
    rows = [_row(lb=b) for b in (0.05, 1.0)]

    comp = artifacts.completeness(expected, rows)

    assert comp["ok"] and comp["expected"] == 2 and comp["actual_rows"] == 2


def test_completeness_detects_a_missing_cell() -> None:
    expected = {_key(lb=b) for b in (0.05, 0.25, 1.0)}
    rows = [_row(lb=b) for b in (0.05, 1.0)]

    comp = artifacts.completeness(expected, rows)

    assert not comp["ok"]
    assert comp["missing"] == [list(_key(lb=0.25))]
    assert comp["unexpected"] == [] and comp["duplicate"] == []


def test_completeness_detects_an_unexpected_cell() -> None:
    comp = artifacts.completeness({_key(lb=1.0)}, [_row(lb=1.0), _row(lb=0.05)])

    assert not comp["ok"] and comp["unexpected"] == [list(_key(lb=0.05))]


def test_completeness_detects_a_duplicate_cell() -> None:
    comp = artifacts.completeness({_key()}, [_row(), _row()])

    assert not comp["ok"]
    assert comp["duplicate"] == [list(_key())]
    assert comp["actual_rows"] == 2 and comp["actual_cells"] == 1


# --- run_complete.json ------------------------------------------------------


def _finished(tmp_path, *, keys=None, rows=None, signature="sigABC", **kw):
    keys = keys if keys is not None else {_key(lb=b) for b in (0.05, 1.0)}
    rows = rows if rows is not None else [_row(lb=b) for b in (0.05, 1.0)]
    IOU.append_jsonl(tmp_path / "probe_results.jsonl", rows)
    for name in ("probe_results.csv", "summary.csv", "deltas.csv"):
        (tmp_path / name).write_text('{"a": 1}\n')
    IOU.write_json(tmp_path / "environment.json", _env())  # a COMPLETE record; the schema is validated
    return artifacts.write_run_complete(
        tmp_path, run_manifest_sha256=signature, expected_keys=keys, rows=rows, **kw
    )


def test_a_finished_run_validates(tmp_path) -> None:
    _finished(tmp_path)

    ok, problems = artifacts.validate_run_complete(tmp_path, expected_signature="sigABC")

    assert ok, problems


def test_marker_refuses_when_a_required_artifact_is_absent(tmp_path) -> None:
    rows = [_row()]
    IOU.append_jsonl(tmp_path / "probe_results.jsonl", rows)
    # summary.csv / deltas.csv / environment.json / split_manifest.json never written

    with pytest.raises(artifacts.IncompleteRunError, match="required artifact\\(s\\) absent"):
        artifacts.write_run_complete(tmp_path, run_manifest_sha256="s", expected_keys={_key()}, rows=rows)

    assert artifacts.read_run_complete(tmp_path) is None


def test_marker_refuses_when_a_planned_cell_is_missing(tmp_path) -> None:
    """A crashed or skipped cell must not leave a table that reads as finished."""
    with pytest.raises(artifacts.IncompleteRunError, match="never produced a row"):
        _finished(tmp_path, keys={_key(lb=b) for b in (0.05, 0.25, 1.0)},
                  rows=[_row(lb=b) for b in (0.05, 1.0)])

    assert artifacts.read_run_complete(tmp_path) is None


def test_marker_refuses_on_unexpected_and_duplicate_cells(tmp_path) -> None:
    with pytest.raises(artifacts.IncompleteRunError, match="never planned"):
        _finished(tmp_path, keys={_key(lb=1.0)}, rows=[_row(lb=1.0), _row(lb=0.05)])
    with pytest.raises(artifacts.IncompleteRunError, match="more than once"):
        _finished(tmp_path, keys={_key()}, rows=[_row(), _row()])


def test_marker_refuses_when_the_pair_dropped_a_regime(tmp_path) -> None:
    """Even though every planned cell landed: the plan itself was short a regime."""
    with pytest.raises(artifacts.IncompleteRunError, match="declared regime\\(s\\) did not run"):
        _finished(tmp_path, regime_problems=[("cropharvest", "random_id", "domain assignment failed")])

    assert artifacts.read_run_complete(tmp_path) is None


def test_marker_refuses_when_a_probe_cell_was_skipped(tmp_path) -> None:
    with pytest.raises(artifacts.IncompleteRunError, match="skipped after a degenerate fit"):
        _finished(tmp_path, cell_failures=[
            {"method": "erm", "holdout": "kenya", "label_budget": 0.05, "reason": "ValueError: empty"}
        ])


def test_marker_hashes_the_required_artifacts(tmp_path) -> None:
    marker = _finished(tmp_path)

    for name in artifacts.REQUIRED_ARTIFACTS:
        assert marker["artifacts"][name]["sha256"], f"{name} not hashed"
    assert "environment.json" in marker["artifacts"]
    assert "split_ref.json" not in artifacts.REQUIRED_ARTIFACTS  # retired: no per-pair split_ref


def test_validation_catches_a_stale_derived_csv(tmp_path) -> None:
    _finished(tmp_path)
    (tmp_path / "summary.csv").write_text("tampered\n")

    ok, problems = artifacts.validate_run_complete(tmp_path)

    assert not ok and any("summary.csv: sha256 changed" in p for p in problems)


def test_validation_catches_a_tampered_environment_record(tmp_path) -> None:
    _finished(tmp_path)
    (tmp_path / "environment.json").write_text('{"numerical_core": {"scikit-learn": "0.1"}}')

    ok, problems = artifacts.validate_run_complete(tmp_path)

    assert not ok and any("environment.json: sha256 changed" in p for p in problems)


def test_validation_treats_a_null_hash_as_an_error(tmp_path) -> None:
    """A null hash is a hole, not a pass -- it means the artifact was absent at completion."""
    _finished(tmp_path)
    marker = artifacts.read_run_complete(tmp_path)
    marker["artifacts"]["deltas.csv"]["sha256"] = None
    IOU.write_json(tmp_path / "run_complete.json", marker)

    ok, problems = artifacts.validate_run_complete(tmp_path)

    assert not ok and any("null sha256" in p for p in problems)


def test_validation_parses_jsonl_rather_than_counting_lines(tmp_path) -> None:
    """A corrupt row is exactly the condition worth catching, and a line count cannot see it."""
    _finished(tmp_path)
    p = tmp_path / "probe_results.jsonl"
    p.write_text(p.read_text().replace('"label_budget": 1.0', '"label_budget": BROKEN'))

    ok, problems = artifacts.validate_run_complete(tmp_path, check_hashes=False)

    assert not ok and any("corrupt" in p for p in problems)


def test_validation_catches_duplicate_cells_on_disk(tmp_path) -> None:
    _finished(tmp_path)
    IOU.append_jsonl(tmp_path / "probe_results.jsonl", [_row(lb=1.0)])

    ok, problems = artifacts.validate_run_complete(tmp_path, check_hashes=False)

    assert not ok
    assert any("duplicate cell key" in p for p in problems)


def test_validation_catches_rows_appended_after_completion(tmp_path) -> None:
    _finished(tmp_path)
    IOU.append_jsonl(tmp_path / "probe_results.jsonl", [_row(lb=0.25)])

    ok, problems = artifacts.validate_run_complete(tmp_path, check_hashes=False)

    assert not ok and any("parses to 3 rows, marker recorded 2" in p for p in problems)


def test_a_started_but_unfinished_run_does_not_validate(tmp_path) -> None:
    IOU.write_json(tmp_path / "run_manifest.json", {"stub": "started"})
    IOU.append_jsonl(tmp_path / "probe_results.jsonl", [_row()])

    ok, problems = artifacts.validate_run_complete(tmp_path)

    assert not ok and "started, not known finished" in problems[0]


def test_marker_is_written_atomically_and_dropped_on_resume(tmp_path) -> None:
    _finished(tmp_path)

    assert [p.name for p in tmp_path.iterdir() if ".tmp" in p.name] == []
    assert artifacts.invalidate_run_complete(tmp_path) is True
    assert artifacts.read_run_complete(tmp_path) is None
    assert artifacts.invalidate_run_complete(tmp_path) is False


def test_a_corrupt_marker_is_treated_as_absent(tmp_path) -> None:
    _finished(tmp_path)
    (tmp_path / "run_complete.json").write_text("{not json")

    ok, _ = artifacts.validate_run_complete(tmp_path)

    assert not ok


# --- historical backfill ----------------------------------------------------


def _historical(tmp_path, n=5):
    IOU.append_jsonl(tmp_path / "probe_results.jsonl", [_row(lb=float(i)) for i in range(n)])
    for name in ("probe_results.csv", "summary.csv", "deltas.csv"):
        (tmp_path / name).write_text("a\n1\n")
    IOU.write_json(tmp_path / "environment.json", _env())
    # A genuinely historical dir carries the OLD run_signature.txt; the (kept) run-complete backfill
    # reads it to preserve that signature. New runs never need this path.
    (tmp_path / "run_signature.txt").write_text("historicalsig")


def test_historical_run_fails_validation_then_passes_after_backfill(tmp_path) -> None:
    """Every canonical results directory predates the marker; a validator that just required it
    would reject all 80 at once, so backfill is the escape hatch."""
    _historical(tmp_path, n=5)
    ok, _ = artifacts.validate_run_complete(tmp_path)
    assert not ok

    marker = artifacts.backfill_run_complete(
        tmp_path, verified_by="akshith", note="counted from the run config",
        expected_cells=5,
    )

    ok, problems = artifacts.validate_run_complete(tmp_path)
    assert ok, problems
    assert marker["signature"] == "historicalsig"
    assert marker["backfilled"] is True and marker["verified_by"] == "akshith"


def test_backfill_requires_an_independently_derived_expected_cells(tmp_path) -> None:
    """Defaulting it to the observed rows would make expected == actual a tautology and certify a
    truncated directory as complete -- the exact failure this mechanism exists to catch."""
    _historical(tmp_path, n=5)

    with pytest.raises(TypeError):
        artifacts.backfill_run_complete(tmp_path, verified_by="a", note="b")  # no expected_cells
    for bad in (0, -1, True):
        with pytest.raises(ValueError, match="positive, independently-derived"):
            artifacts.backfill_run_complete(tmp_path, verified_by="a", note="b", expected_cells=bad)


def test_backfill_rejects_a_row_count_disagreement(tmp_path) -> None:
    """The directory is not the run it is claimed to be."""
    _historical(tmp_path, n=5)

    with pytest.raises(ValueError, match="5 rows on disk but 9 cells were asserted"):
        artifacts.backfill_run_complete(tmp_path, verified_by="a", note="b", expected_cells=9)


def test_backfill_rejects_corrupt_jsonl(tmp_path) -> None:
    _historical(tmp_path, n=3)
    p = tmp_path / "probe_results.jsonl"
    p.write_text(p.read_text() + '{"seed": BROKEN}\n')

    with pytest.raises(ValueError, match="malformed JSONL row"):
        artifacts.backfill_run_complete(tmp_path, verified_by="a", note="b", expected_cells=4)


def test_backfill_rejects_duplicate_cell_keys(tmp_path) -> None:
    _historical(tmp_path, n=3)
    IOU.append_jsonl(tmp_path / "probe_results.jsonl", [_row(lb=0.0)])  # dupe of the first

    with pytest.raises(ValueError, match="duplicate cell key"):
        artifacts.backfill_run_complete(tmp_path, verified_by="a", note="b", expected_cells=4)


def test_backfill_rejects_missing_final_artifacts(tmp_path) -> None:
    IOU.append_jsonl(tmp_path / "probe_results.jsonl", [_row()])

    with pytest.raises(FileNotFoundError, match="absent artifact"):
        artifacts.backfill_run_complete(tmp_path, verified_by="a", note="b", expected_cells=1)


def test_backfill_refuses_to_be_anonymous(tmp_path) -> None:
    """So it cannot be swept over a directory tree in a loop."""
    _historical(tmp_path, n=1)

    with pytest.raises(ValueError, match="requires verified_by and note"):
        artifacts.backfill_run_complete(tmp_path, verified_by="", note="x", expected_cells=1)
    with pytest.raises(ValueError, match="requires verified_by and note"):
        artifacts.backfill_run_complete(tmp_path, verified_by="akshith", note="", expected_cells=1)


# --- dense per-regime completeness ------------------------------------------
# The v1 base.segmentation_fold_configs "declared regime yielded nothing is recorded, not silent"
# accounting is removed. Its schema-v2 equivalent -- a requested regime that yields zero leaves is a
# hard consumption-time refusal -- is pinned by
# test_split_parity.test_phase_b_refuses_zero_yield_requested_regime (tabular) and applies to dense
# too, since the central-log discovery in split_artifacts gates load_tabular_splits and load_dense_splits.


def test_exit_code_is_nonzero_when_a_declared_regime_did_not_run() -> None:
    """The hole that made the random_id/official TypeError silent: the banner already printed;
    nothing acted on it, so the shard reported success with 2 of 4 regimes evaluated."""
    import main as MAIN

    assert MAIN.shard_exit_code([], []) == 0
    assert MAIN.shard_exit_code([], [("cropharvest", "random_id", "domain assignment failed")]) == 1
    assert MAIN.shard_exit_code([("raw", "pastis", "boom")], []) == 1
    assert MAIN.shard_exit_code([("raw", "pastis", "boom")], [("x", "y", "z")]) == 1


def test_skipped_probe_cells_are_recorded_in_the_accumulator() -> None:
    """A skipped cell writes no row; without this the only trace is one line in a long log."""
    from utils import perfutils as PERF

    PERF.clear_cell_failures()
    PERF._record_cell_failure(
        {"seed": 0, "split_regime": "geographic_ood", "holdout": "kenya", "method": "erm",
         "probe_family": "logistic"},
        0.05, "source", ValueError("empty train array"),
    )

    assert len(PERF.CELL_FAILURES) == 1
    rec = PERF.CELL_FAILURES[0]
    assert rec["holdout"] == "kenya" and rec["label_budget"] == 0.05
    assert "empty train array" in rec["reason"]
    PERF.clear_cell_failures()


@pytest.mark.parametrize("name", [
    "run.sbatch", "job.slurm", "launch.sh", "conf.toml", "conf.yaml", "conf.json",
    "setup.cfg", "tox.ini", "deps.lock", "requirements.txt", "requirements-dev.txt",
    "Dockerfile", "Makefile",
])
def test_untracked_launcher_and_config_content_changes_the_identity(tmp_path, name) -> None:
    """A run submitted by a different sbatch script, or resolved from a different lock, is a
    different run even when every .py is byte-identical."""
    a = artifacts.git_identity(_repo(tmp_path / "a", {}, untracked={name: "VALUE=1\n"}))
    b = artifacts.git_identity(_repo(tmp_path / "b", {}, untracked={name: "VALUE=2\n"}))

    assert a["n_untracked_source_files"] == 1, f"{name} was not counted as source"
    assert a["untracked_source_sha256"] != b["untracked_source_sha256"]
    assert a["tree_identity"] != b["tree_identity"], f"{name} content did not reach the identity"


@pytest.mark.parametrize("name", [
    "results.jsonl", "summary.csv", "emb.npy", "fig.png", "tiles.h5", "bench.pkl", "notes.txt",
])
def test_untracked_data_and_artifacts_do_not_perturb_the_identity(tmp_path, name) -> None:
    """Datasets and generated artifacts are not code. A results file dropped in the tree must not
    change the identity of the code that produced it -- and `notes.txt` shows why arbitrary .txt
    is excluded while requirements*.txt is not."""
    a = artifacts.git_identity(_repo(tmp_path / "a", {}, untracked={name: "1\n"}))
    b = artifacts.git_identity(_repo(tmp_path / "b", {}, untracked={name: "2\n"}))

    assert a["n_untracked_source_files"] == 0, f"{name} was wrongly treated as source"
    assert a["tree_identity"] == b["tree_identity"]


def test_gitignored_files_never_reach_the_identity(tmp_path) -> None:
    """`--exclude-standard` is what keeps viz/data's 79 MB snapshot out of every run's identity."""
    root = _repo(tmp_path / "a", {}, untracked={".gitignore": "secret.py\n", "secret.py": "x = 1\n"})
    a = artifacts.git_identity(root)
    (root / "secret.py").write_text("x = 999\n")
    b = artifacts.git_identity(root)

    assert a["tree_identity"] == b["tree_identity"], "an ignored file perturbed the identity"


def test_identity_covers_a_realistic_mixed_dirty_tree(tmp_path) -> None:
    """Tracked edit + untracked module + untracked launcher, all at once."""
    def build(marker: str):
        return artifacts.git_identity(_repo(
            tmp_path / marker,
            {"base.py": f"x = {marker!r}\n"},
            untracked={"mod.py": "def f(): ...\n", "run.sbatch": f"#SBATCH --time={marker}\n"},
        ))

    a, b = build("a"), build("b")

    assert a["n_untracked_source_files"] == 2
    assert a["dirty_files"] == b["dirty_files"]          # same filenames...
    assert a["tree_identity"] != b["tree_identity"]      # ...different code
