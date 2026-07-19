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
from evals.regimes import base as RB
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
    for name in ("probe_results.csv", "summary.csv", "deltas.csv", "split_ref.json"):
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
    # consumption to yield no splits (the old iter_splits stub's intent) so this exercises the
    # environment/overwrite/marker contract on an empty run.
    monkeypatch.setattr(main.split_artifacts, "load_tabular_splits", lambda *a, **k: ([], []))
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
    dir (rows / split_ref / environment / run_manifest / run_complete) byte-for-byte unchanged even
    with overwrite_mode=True."""
    import main
    from evals import split_artifacts as SA

    results_dir = tmp_path / "results" / "raw" / "fake"
    _seed_finished_dir(results_dir, _env(sklearn="1.9.0"))
    snap = {n: (results_dir / n).read_bytes() for n in
            ("probe_results.jsonl", "split_ref.json", "environment.json", "run_manifest.json", "run_complete.json")}

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

    with pytest.raises(SA.SplitArtifactError, match="no canonical splits"):
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
            ("probe_results.jsonl", "split_ref.json", "environment.json", "run_manifest.json", "run_complete.json")}

    monkeypatch.setattr(RS.cacheutils, "OUTPUT_DIR", tmp_path)
    monkeypatch.setattr(RS.cacheutils, "cached_bench", lambda *a, **k: SimpleNamespace(
        name="pastis", data_quality=None, patch_classes=None, n_samples=4, patches=[]))
    monkeypatch.setattr(RS.cacheutils, "require_dense_cache", lambda *a, **k: tmp_path / "emb")
    monkeypatch.setattr(RS.cacheutils, "SCRATCH", tmp_path / "empty_splits")  # no data/splits -> refuse

    with pytest.raises(SA.SplitArtifactError, match="no canonical splits"):
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

    with pytest.raises(SA.SplitArtifactError, match="no canonical splits"):
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

    with pytest.raises(SA.SplitArtifactError, match="no canonical splits"):
        RS._run_segmentation_pair(
            "pastis", "raw", [0], 10, ["random_id"], ["gen_embeddings", "probing"], ["logistic"],
            {"source": [1.0], "target": [0]}, False, False, True, False, {},
        )
    assert not (tmp_path / "out").exists(), "a refused dense split load created/mutated the results tree"


def _dense_stubs(monkeypatch, tmp_path):
    from evals import split_artifacts as SA
    from utils import runstate as RS

    # PHASE B: runstate consumes dense splits from data/splits/ (function-local import, so patch the
    # module object). Empty configs reproduce the old segmentation_fold_configs stub's intent.
    monkeypatch.setattr(SA, "load_dense_splits", lambda *a, **k: ({0: []}, []))
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


def _publish_dense_random_splits(splits_root, emb_dir, *, seed=0):
    """Generate + publish random_id dense split artifacts matching a real dense cache (Phase B).

    This is the dense integration fixture: the runtime now CONSUMES patch splits from data/splits/,
    so a real dense-cell test must first publish canonical artifacts for the cache's patches.
    """
    import evals.benchmarks.pastis as pastis_mod
    from evals import split_artifacts as SA
    from evals.regimes import base as RB
    from utils import cacheutils as C

    all_folds = set(pastis_mod.TRAIN_FOLDS) | set(pastis_mod.VAL_FOLDS) | set(pastis_mod.TEST_FOLDS)
    pids = [int(p) for p in C.dense_fold_patches(emb_dir, all_folds)]
    bench = SimpleNamespace(name="pastis", patch_ids=lambda folds=None, _p=pids: list(_p))
    dense_cache = dict(
        all_patch_ids=pids,
        fold_of={p: sorted(all_folds)[0] for p in pids},  # random_id has no exclusions; stats unchecked here
        class_sets={p: {0, 1, 2} for p in pids},
        patch_latlon={},
    )
    cfgs = list(RB.segmentation_fold_configs(pastis_mod, ["random_id"], seed=seed, emb_dir=emb_dir, bench=bench))
    for _regime, cfg in cfgs:
        params = {"train_folds": sorted(cfg.train_folds), "val_folds": sorted(cfg.val_folds),
                  "test_folds": sorted(cfg.test_folds), "assembly_seed": 0}
        spec, eligible = SA.build_dense_leaf("pastis", "random_id", seed, cfg=cfg, bench=bench,
                                             params=params, audit_events=[], **dense_cache)
        SA.publish_leaf(splits_root, spec, eligible)
    SA.write_generation(splits_root, "pastis", "random_id", seed, requested=["random_patch"],
                        yielded=[c.label for _, c in cfgs], dropped=[], audit_events=[], regime_problems=[])
    return pids


def _run_dense_healthy(monkeypatch, tmp_path):
    from utils import runstate as RS

    emb_dir = _dense_cache_on_disk(tmp_path / "emb")
    pids = _publish_dense_random_splits(tmp_path / "splits", emb_dir, seed=0)
    # PHASE B: the runtime reads splits from the single canonical location cacheutils.SCRATCH/"splits"
    # (no RB_SPLITS_DIR); redirect SCRATCH so the pair consumes the temp artifacts.
    monkeypatch.setattr(RS.cacheutils, "SCRATCH", tmp_path)

    monkeypatch.setattr(RS.cacheutils, "OUTPUT_DIR", tmp_path / "out")
    # PHASE B: runtime builds patch_fold from bench.patches; the synthetic cache reuses patch ids
    # across folds, and random_id spans all folds, so any single fold is consistent for every patch.
    patches = [SimpleNamespace(patch_id=p, fold=1) for p in pids]
    monkeypatch.setattr(RS.cacheutils, "cached_bench", lambda *a, **k: SimpleNamespace(
        name="pastis", data_quality=None, patch_classes=None, n_samples=4,
        patches=patches, patch_ids=lambda folds=None, _p=pids: list(_p)))
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
    """Publish random_id tabular split artifacts for `bench` -- the runtime CONSUMES these."""
    from evals import split_artifacts as SA
    from evals.benchmarks import cropharvest as ch
    from evals.regimes import base as RB

    y, _g = ch.make_targets(bench)
    labels_seen = []
    for (label, train, val, test, domains, has_target, group_kind, source_val, source_test) in \
            RB.iter_splits("random_id", bench, y, None, seed):
        spec, eligible = SA.build_tabular_leaf(
            "cropharvest", "random_id", seed, label=label,
            train=train, val=val, test=test, source_val=source_val, source_test=source_test,
            domains=domains, labels=y, sample_ids=bench.sample_ids, has_target=has_target,
            group_kind=group_kind, params={"assembly_seed": 0}, audit_events=[],
        )
        SA.publish_leaf(splits_root, spec, eligible)
        labels_seen.append(str(label))
    SA.write_generation(splits_root, "cropharvest", "random_id", seed, requested=["random_id"],
                        yielded=labels_seen, dropped=[], audit_events=[], regime_problems=[])
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
    # build_run_manifest is intentionally NOT stubbed: only the real builder records consumed_splits.

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

    # split_ref.json records the consumed relative leaf paths (canonical committed location)
    expected = [f"cropharvest/random_id/0/{SA.holdout_dirname(labels[0])}"]
    ref = json.loads((results_dir / "split_ref.json").read_text())
    assert ref["consumed_leaves"] == expected
    assert ref["splits_location"] == "data/splits"

    # run_manifest.json binds the SAME consumed split set (resume identity)
    man = json.loads((results_dir / "run_manifest.json").read_text())
    assert man["consumed_splits"]["leaves"] == expected

    # completion validation succeeds against the run's own manifest signature
    ok, problems = artifacts.validate_run_complete(
        results_dir, expected_signature=runstate.run_manifest_digest(man)
    )
    assert ok, problems


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
         bt="source", lb=1.0, es="test"):
    return (seed, regime, holdout, method, family, bt, lb, es)


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
    for name in ("probe_results.csv", "summary.csv", "deltas.csv", "split_ref.json"):
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


def test_marker_hashes_environment_and_split_ref(tmp_path) -> None:
    marker = _finished(tmp_path)

    for name in artifacts.REQUIRED_ARTIFACTS:
        assert marker["artifacts"][name]["sha256"], f"{name} not hashed"
    assert "environment.json" in marker["artifacts"]
    assert "split_ref.json" in marker["artifacts"]


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
    for name in ("probe_results.csv", "summary.csv", "deltas.csv", "split_ref.json"):
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


class _EmptyDenseRegime:
    NAME = "spatial_cluster_ood"
    GROUP_KIND = "spatial_cluster"
    HAS_TARGET = True

    @staticmethod
    def iter_dense_splits(bench_mod, *, emb_dir, seed, bench=None):
        return iter(())  # the real regime prints and returns on any ValueError


def _dense_mod():
    return SimpleNamespace(BENCHMARK="pastis", TRAIN_FOLDS=[1], VAL_FOLDS=[2], TEST_FOLDS=[3])


@pytest.fixture(autouse=True)
def _clean_problems():
    RB.clear_regime_problems()
    yield
    RB.clear_regime_problems()


def test_dense_regime_yielding_nothing_is_recorded_not_silent(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(RB, "load_regime", lambda name: _EmptyDenseRegime)

    configs = list(RB.segmentation_fold_configs(
        _dense_mod(), ["spatial_cluster_ood"], seed=0, emb_dir=tmp_path, strict_mode=False
    ))

    assert configs == []
    assert len(RB.REGIME_PROBLEMS) == 1
    benchmark, regime, reason = RB.REGIME_PROBLEMS[0]
    assert (benchmark, regime) == ("pastis", "spatial_cluster_ood")
    assert "0 dense fold configs" in reason


def test_dense_regime_yielding_nothing_raises_under_strict_mode(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(RB, "load_regime", lambda name: _EmptyDenseRegime)

    with pytest.raises(RuntimeError, match="declared regime did not run"):
        list(RB.segmentation_fold_configs(
            _dense_mod(), ["spatial_cluster_ood"], seed=0, emb_dir=tmp_path, strict_mode=True
        ))


def test_dense_regime_that_yields_configs_reports_no_problem(monkeypatch, tmp_path) -> None:
    class _Good(_EmptyDenseRegime):
        @staticmethod
        def iter_dense_splits(bench_mod, *, emb_dir, seed, bench=None):
            yield RB.DenseSplit("fold_3", {1}, {2}, {3})

    monkeypatch.setattr(RB, "load_regime", lambda name: _Good)

    configs = list(RB.segmentation_fold_configs(
        _dense_mod(), ["spatial_cluster_ood"], seed=0, emb_dir=tmp_path, strict_mode=True
    ))

    assert len(configs) == 1
    assert RB.REGIME_PROBLEMS == []


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
