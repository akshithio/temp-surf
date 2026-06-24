"""Run-stage validation, result signatures, and resume-state helpers."""

from __future__ import annotations

import os

from utils import cacheutils
from utils import ioutils as IOU

VALID_RUN_STAGES = {"gen_embeddings", "probing"}


def validate_run_stages(run_stages: list[str]) -> set[str]:
    """Validate the configured run stages."""
    stages = set(run_stages)
    unknown = stages - VALID_RUN_STAGES
    if unknown:
        raise ValueError(f"Unknown RUN_STAGES entries: {sorted(unknown)}. Valid entries: {sorted(VALID_RUN_STAGES)}")
    if not stages:
        raise ValueError("RUN_STAGES must include at least one stage.")
    return stages


def run_signature(
    model_name: str,
    tag: str,
    split_regimes,
    seeds,
    enc_kwargs,
    *,
    active_probes,
    budget_regimes,
    max_samples,
    max_dense_pixels,
) -> str:
    """Fingerprint the result-defining experiment inputs and source files."""
    src = cacheutils.REPO / "src"
    code = cacheutils._hash_files(
        src / "evals" / "probes.py",
        src / "evals" / "evals.py",
        src / "evals" / "regimes" / "base.py",
        src / "utils" / "ioutils.py",
        src / "utils" / "cacheutils.py",
        src / "utils" / "runstate.py",
        *cacheutils._model_source_files(model_name),
        *sorted((src / "evals" / "regimes").glob("*.py")),
    )
    enc = {k: v for k, v in sorted(enc_kwargs.items()) if k != "device"}
    parts = [
        f"tag={tag}",
        f"ckpt={cacheutils._checkpoint_fingerprint(model_name, enc_kwargs.get('weights_path'))}",
        f"probes={active_probes}",
        f"budgets={budget_regimes}",
        f"seeds={list(seeds)}",
        f"regimes={sorted(split_regimes)}",
        f"max_samples={max_samples}",
        f"max_dense_pixels={max_dense_pixels}",
        f"enc={enc}",
        f"code={code}",
    ]
    return cacheutils._hash_str("|".join(map(str, parts)))


def check_run_signature(results_dir, signature: str, *, overwrite_mode: bool) -> None:
    """Refuse to resume a result directory from a different experiment."""
    if overwrite_mode:
        return
    sig_path = results_dir / "run_signature.txt"
    rows_path = results_dir / "probe_results.jsonl"
    if sig_path.exists():
        existing = sig_path.read_text().strip()
        if existing != signature:
            raise RuntimeError(
                f"Refusing to resume {results_dir}: signature {existing[:10]!r} != {signature[:10]!r} "
                "(different experiment config, or a corrupt/partial signature). Set OVERWRITE_MODE=True "
                "or remove the directory."
            )
    elif rows_path.exists() and rows_path.stat().st_size > 0:
        raise RuntimeError(
            f"Refusing to resume {results_dir}: it has results but NO run_signature.txt (a pre-guard "
            "or foreign run). Verify they match this config and write the signature, or use "
            "OVERWRITE_MODE=True."
        )


def publish_run_signature(results_dir, signature: str) -> None:
    """Write the run signature atomically."""
    results_dir.mkdir(parents=True, exist_ok=True)
    sig_path = results_dir / "run_signature.txt"
    tmp = sig_path.with_name(sig_path.name + ".tmp")
    tmp.write_text(signature)
    os.replace(tmp, sig_path)


def budget_row_key(row):
    """Budget-level identity of a result or prediction row."""
    return (
        row.get("seed"),
        row.get("split_regime"),
        row.get("holdout"),
        row.get("method"),
        row.get("probe_family"),
        row.get("budget_type"),
        row.get("label_budget"),
    )


def prune_partial_budgets(rows, rows_path, preds_path, rerun_keys):
    """Remove existing rows/predictions for budgets that are about to be regenerated."""
    if not rerun_keys:
        return rows
    kept = [r for r in rows if budget_row_key(r) not in rerun_keys]
    if len(kept) != len(rows):
        rows_path.unlink(missing_ok=True)
        IOU.append_jsonl(rows_path, kept)
    preds = IOU.read_jsonl(preds_path)
    kept_preds = [p for p in preds if budget_row_key(p) not in rerun_keys]
    if len(kept_preds) != len(preds):
        preds_path.unlink(missing_ok=True)
        IOU.append_jsonl(preds_path, kept_preds)
    return kept
