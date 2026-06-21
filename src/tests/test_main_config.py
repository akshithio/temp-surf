from __future__ import annotations

import pytest

import main


def test_run_stage_set_accepts_embedding_and_probe_stages() -> None:
    assert main._run_stage_set(["gen_embeddings", "probing"]) == {"gen_embeddings", "probing"}
    assert main._run_stage_set(["probing"]) == {"probing"}


def test_run_stage_set_rejects_unknown_stage() -> None:
    with pytest.raises(ValueError, match="Unknown RUN_STAGES"):
        main._run_stage_set(["probing", "not-a-stage"])


def test_run_stage_set_rejects_empty_config() -> None:
    with pytest.raises(ValueError, match="RUN_STAGES must include"):
        main._run_stage_set([])
