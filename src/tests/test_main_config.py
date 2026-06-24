from __future__ import annotations

import pytest

import main
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
    assert call["enc_kwargs"] == {"device": "cpu"}


def test_main_source_budgets_include_explicit_full_source_anchor() -> None:
    assert main.BUDGET_REGIMES["source"][-1] == 1.0


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
