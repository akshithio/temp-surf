from __future__ import annotations

import numpy as np
import pytest

import main
from dataio.get_input import Benchmark, ModalitySeries, NativeSeries
from utils import runstate


def test_run_stage_set_accepts_embedding_and_probe_stages() -> None:
    assert runstate.validate_run_stages(["gen_embeddings", "probing"]) == {"gen_embeddings", "probing"}
    assert runstate.validate_run_stages(["probing"]) == {"probing"}


def test_run_stage_set_rejects_unknown_stage() -> None:
    with pytest.raises(ValueError, match="Unknown RUN_STAGES"):
        runstate.validate_run_stages(["probing", "not-a-stage"])


def test_run_stage_set_rejects_empty_config() -> None:
    with pytest.raises(ValueError, match="RUN_STAGES must include"):
        runstate.validate_run_stages([])


def test_main_dispatches_config_to_run_pair(monkeypatch) -> None:
    calls = []
    monkeypatch.setattr(main, "LAUNCH_GPU_SHARDS", False)
    monkeypatch.setattr(main.compat, "eligible_models", lambda bm: ["raw"])
    monkeypatch.setattr(main.gputils, "take_shard", lambda pairs: pairs[:1])
    monkeypatch.setattr(main.gputils, "shard_indices", lambda: (0, 1))
    monkeypatch.setattr(main.gputils, "device", lambda: "cpu")
    monkeypatch.setattr(main, "run_pair", lambda **kwargs: calls.append(kwargs))
    monkeypatch.setattr(main.regime_base, "report_regime_problems", lambda: None)

    assert main.main() == 0
    assert len(calls) == 1
    call = calls[0]
    assert call["benchmark_name"] == main.BENCHMARKS[0]
    assert call["model_name"] == "raw"
    assert call["seeds"] == main.SEEDS
    assert call["max_samples"] == main.MAX_SAMPLES
    assert call["max_dense_pixels"] == main.MAX_DENSE_PIXELS
    assert call["split_regimes"] == main.SPLIT_REGIMES
    assert call["run_stages"] == main.RUN_STAGES
    assert call["active_probes"] == main.ACTIVE_PROBES
    assert call["budget_regimes"] == main.BUDGET_REGIMES
    assert call["overwrite_mode"] == main.OVERWRITE_MODE
    assert call["strict_mode"] == main.STRICT_MODE
    assert call["enc_kwargs"] == {"device": "cpu"}


def test_main_source_budgets_include_explicit_full_source_anchor() -> None:
    assert main.BUDGET_REGIMES["source"][-1] == 1.0


def test_s2_only_uses_original_coordinates_for_split_regimes(monkeypatch, tmp_path) -> None:
    n = 6

    def series(width: int, bands: list[str]) -> ModalitySeries:
        return ModalitySeries(
            [np.ones((1, width), dtype=np.float32) for _ in range(n)],
            [np.array([0], dtype=np.int64) for _ in range(n)],
            [np.array([15.0], dtype=np.float32) for _ in range(n)],
            [np.array([2020], dtype=np.int64) for _ in range(n)],
            bands,
        )

    latlon = np.array([[10.0, 20.0], [11.0, 21.0], [12.0, 22.0], [13.0, 23.0], [14.0, 24.0], [15.0, 25.0]], dtype=np.float32)
    bench = Benchmark(
        name="fake",
        label_kind="binary",
        native=NativeSeries(series(2, ["B2", "B3"]), series(1, ["VV"]), series(1, ["temp"])),
        labels=np.array([0, 1, 0, 1, 0, 1], dtype=np.int64),
        groups=np.array(["a", "a", "b", "b", "c", "c"], dtype=object),
        latlon=latlon,
        years=np.full(n, 2020, dtype=np.int64),
    )

    class FakeBenchMod:
        BENCHMARK = "fake"
        LABEL_KIND = "binary"
        SPLIT_REGIMES = ["spatial_cluster_ood"]

        @staticmethod
        def make_targets(loaded):
            return loaded.labels, loaded.groups

    seen = {}
    monkeypatch.setenv("RB_S2_ONLY", "1")
    monkeypatch.setattr(main.EV, "load_benchmark", lambda _name: FakeBenchMod)
    monkeypatch.setattr(main.cacheutils, "OUTPUT_DIR", tmp_path)
    monkeypatch.setattr(main.cacheutils, "bench_tag", lambda *_args, **_kwargs: "tag")
    monkeypatch.setattr(main.cacheutils, "cached_bench", lambda *_args, **_kwargs: bench)
    monkeypatch.setattr(main.runstate, "run_signature", lambda *_args, **_kwargs: "sig")
    monkeypatch.setattr(main.compat, "input_modalities", lambda _model: set())

    def load_embeddings(loaded, *_args, **_kwargs):
        seen["embedding_latlon"] = np.asarray(loaded.latlon).copy()
        seen["embedding_modalities"] = loaded.available_modalities()
        return np.zeros((n, 2), dtype=np.float32)

    def iter_splits(_regime, loaded, *_args, **_kwargs):
        seen["regime_latlon"] = np.asarray(loaded.latlon).copy()
        seen["regime_modalities"] = loaded.available_modalities()
        yield from ()

    monkeypatch.setattr(main.cacheutils, "load_cached_embeddings", load_embeddings)
    monkeypatch.setattr(main.regime_base, "iter_splits", iter_splits)

    main._run_tabular_pair(
        "fake",
        "raw",
        [0],
        None,
        0,
        ["spatial_cluster_ood"],
        ["probing"],
        ["logistic"],
        {"source": [1.0], "target": [0]},
        True,
        False,
        {},
    )

    assert np.all(seen["embedding_latlon"] == 0.0)
    assert "latlon" not in seen["embedding_modalities"]
    np.testing.assert_array_equal(seen["regime_latlon"], latlon)
    assert "latlon" in seen["regime_modalities"]


def test_compatibility_table_excludes_only_blocked_pairs() -> None:
    from evals import compat

    assert compat.eligible_models("cropharvest") == ["presto", "olmoearth", "galileo", "raw"]
    assert compat.eligible_models("eurocropsml") == ["presto", "olmoearth", "galileo", "tessera", "raw"]
    assert compat.eligible_models("breizhcrops") == ["olmoearth", "galileo", "presto", "tessera", "raw"]
    assert compat.eligible_models("pastis") == ["tessera", "olmoearth", "galileo", "agrifm", "presto", "raw"]

    assert not compat.is_eligible("cropharvest", "tessera")
    assert not compat.is_eligible("cropharvest", "agrifm")
    assert not compat.is_eligible("eurocropsml", "agrifm")
    assert not compat.is_eligible("breizhcrops", "agrifm")
    assert compat.is_eligible("pastis", "agrifm")
