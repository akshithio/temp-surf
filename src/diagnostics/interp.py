"""Feature-importance diagnostics for frozen encoder runs.

This module is deliberately config-free. Callers pass the benchmark subset,
encoder/probe objects, metric function, and metadata explicitly, then write the
returned rows to the single flat output table.
"""

from __future__ import annotations

import csv
from collections.abc import Callable, Iterable
from dataclasses import dataclass, fields, is_dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[2]
DEFAULT_IMPORTANCE_PATH = REPO / "data" / "output" / "feature_importance.csv"


@dataclass(frozen=True)
class BandSpec:
    modality: str
    band: str
    index: int
    band_group: str


@dataclass
class AttributionBatch:
    embeddings: torch.Tensor
    inputs: torch.Tensor | dict[str, torch.Tensor]
    specs: list[BandSpec] | dict[str, list[BandSpec]]


OUTPUT_COLUMNS = [
    "encoder",
    "task",
    "benchmark",
    "seed",
    "split_regime",
    "holdout",
    "condition",
    "train_regime",
    "importance_method",
    "perturbation",
    "modality",
    "band",
    "band_group",
    "baseline_score",
    "perturbed_score",
    "delta_score",
    "baseline_prob_mean",
    "perturbed_prob_mean",
    "delta_prob_mean",
    "importance",
    "normalized_importance",
    "rank",
    "n_samples",
    "repeat",
]


def _band_group(modality: str, band: str) -> str:
    b = band.upper()
    if modality == "s1":
        return "sar"
    if modality == "climate":
        if b in {"ELEVATION", "SLOPE", "DEM"}:
            return "topography"
        return "climate"
    if b in {"B2", "B3", "B4"}:
        return "visible"
    if b in {"B5", "B6", "B7", "B8A"}:
        return "red_edge"
    if b == "B8":
        return "nir"
    if b in {"B11", "B12"}:
        return "swir"
    if b == "NDVI":
        return "index"
    return "other"


def specs_for(channels: list[tuple[str, str]]) -> list[BandSpec]:
    """BandSpec list for an encoder's input-tensor channels, given (modality, band) in order.

    Encoders can feed bands in their own order/layout, so each ``forward_for_attribution`` builds its own specs that align with its
    tensor's channel axis -- not ``band_specs(bench)`` (which is in benchmark order).
    """
    return [BandSpec(m, b, i, _band_group(m, b)) for i, (m, b) in enumerate(channels)]


def band_specs(bench: Any) -> list[BandSpec]:
    """Return raw-input band metadata for every modality in a benchmark."""
    specs: list[BandSpec] = []
    for modality, bands_attr in [
        ("s2", "s2_bands"),
        ("s1", "s1_bands"),
        ("climate", "climate_bands"),
    ]:
        for idx, band in enumerate(getattr(bench, bands_attr, [])):
            specs.append(BandSpec(modality, str(band), idx, _band_group(modality, str(band))))
    return specs


def subset_benchmark(bench: Any, indices: Iterable[int]) -> Any:
    """Return a row-subset benchmark while preserving metadata fields."""
    idx = np.asarray(list(indices), dtype=np.int64)
    sample_fields = {
        "s2",
        "s1",
        "climate",
        "s2_mask",
        "s1_mask",
        "climate_mask",
        "doy",
        "labels",
        "groups",
        "latlon",
    }
    updates = {name: np.asarray(getattr(bench, name))[idx].copy() for name in sample_fields if hasattr(bench, name)}
    if is_dataclass(bench):
        valid = {f.name for f in fields(bench)}
        return replace(bench, **{k: v for k, v in updates.items() if k in valid})

    data = dict(getattr(bench, "__dict__", {}))
    data.update(updates)
    return type(bench)(**data)


def perturb_band(bench: Any, spec: BandSpec, mode: str = "permute", seed: int = 0) -> Any:
    """Perturb a single raw input band and return a benchmark copy."""
    if mode not in {"permute", "zero"}:
        raise ValueError(f"Unknown perturbation mode {mode!r}; expected 'permute' or 'zero'.")
    out = subset_benchmark(bench, np.arange(getattr(bench, spec.modality).shape[0]))
    arr = getattr(out, spec.modality).copy()
    if spec.index < 0 or spec.index >= arr.shape[2]:
        raise IndexError(f"{spec.modality}.{spec.band} index {spec.index} is outside shape {arr.shape}.")
    if mode == "zero":
        arr[:, :, spec.index] = 0.0
    else:
        order = np.random.default_rng(seed).permutation(arr.shape[0])
        if arr.shape[0] > 1 and np.array_equal(order, np.arange(arr.shape[0])):
            order = np.roll(order, 1)
        arr[:, :, spec.index] = arr[order, :, spec.index]
    setattr(out, spec.modality, arr)
    return out


def _score_value(score: Any) -> float:
    if isinstance(score, dict):
        if "score" in score:
            return float(score["score"])
        if len(score) != 1:
            raise ValueError("score_fn returned a dict with multiple metrics; return a float or {'score': value}.")
        return float(next(iter(score.values())))
    return float(score)


def _prob_mean(probe: Any, x: np.ndarray) -> float:
    if not hasattr(probe, "predict_proba"):
        return float("nan")
    prob = probe.predict_proba(x)
    if prob.ndim != 2 or prob.shape[1] == 0:
        return float("nan")
    col = 1 if prob.shape[1] > 1 else 0
    return float(np.mean(prob[:, col]))


def _normalize_and_rank(rows: list[dict[str, Any]]) -> None:
    groups: dict[tuple[Any, ...], list[dict[str, Any]]] = {}
    key_names = ["importance_method", "perturbation", "repeat"]
    for row in rows:
        key = tuple(row.get(k) for k in key_names)
        groups.setdefault(key, []).append(row)
    for group_rows in groups.values():
        denom = float(sum(max(0.0, float(r["importance"])) for r in group_rows))
        order = sorted(range(len(group_rows)), key=lambda i: float(group_rows[i]["importance"]), reverse=True)
        for rank, pos in enumerate(order, start=1):
            row = group_rows[pos]
            row["rank"] = rank
            row["normalized_importance"] = float(row["importance"] / denom) if denom > 0 else 0.0


def permutation_importance(
    encoder: Any,
    probe: Any,
    bench: Any,
    y: np.ndarray,
    score_fn: Callable[[Any, np.ndarray, np.ndarray], float | dict[str, float]],
    specs: list[BandSpec] | None = None,
    metadata: dict[str, Any] | None = None,
    *,
    mode: str = "permute",
    repeats: int = 1,
    seed: int = 0,
    baseline_embeddings: np.ndarray | None = None,
) -> list[dict[str, Any]]:
    """Compute raw-band permutation or zero-ablation importance rows."""
    specs = list(specs or band_specs(bench))
    meta = dict(metadata or {})
    y = np.asarray(y)
    baseline = baseline_embeddings if baseline_embeddings is not None else encoder.encode(bench)
    baseline_score = _score_value(score_fn(probe, baseline, y))
    baseline_prob = _prob_mean(probe, baseline)

    rows: list[dict[str, Any]] = []
    for repeat in range(repeats):
        for spec in specs:
            perturbed = perturb_band(bench, spec, mode=mode, seed=seed + repeat * 1009 + spec.index)
            x_pert = encoder.encode(perturbed)
            pert_score = _score_value(score_fn(probe, x_pert, y))
            pert_prob = _prob_mean(probe, x_pert)
            delta_score = baseline_score - pert_score
            rows.append(
                {
                    **meta,
                    "importance_method": "permutation",
                    "perturbation": mode,
                    "modality": spec.modality,
                    "band": spec.band,
                    "band_group": spec.band_group,
                    "baseline_score": baseline_score,
                    "perturbed_score": pert_score,
                    "delta_score": delta_score,
                    "baseline_prob_mean": baseline_prob,
                    "perturbed_prob_mean": pert_prob,
                    "delta_prob_mean": baseline_prob - pert_prob if np.isfinite(baseline_prob) and np.isfinite(pert_prob) else float("nan"),
                    "importance": max(0.0, float(delta_score)),
                    "normalized_importance": 0.0,
                    "rank": 0,
                    "n_samples": int(len(y)),
                    "repeat": repeat,
                }
            )
    _normalize_and_rank(rows)
    return rows


def _probe_linear_params(probe: Any, embeddings: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    obj = probe
    scaler = None
    if hasattr(probe, "named_steps"):
        scaler = probe.named_steps.get("standardscaler")
        obj = probe.named_steps.get("logisticregression", probe)
    if not hasattr(obj, "coef_"):
        raise TypeError("gradient_input_importance needs a linear probe with coef_.")
    coef = torch.as_tensor(obj.coef_, dtype=embeddings.dtype, device=embeddings.device)
    intercept = torch.as_tensor(getattr(obj, "intercept_", np.zeros(coef.shape[0])), dtype=embeddings.dtype, device=embeddings.device)
    x = embeddings
    if scaler is not None:
        mean = torch.as_tensor(scaler.mean_, dtype=x.dtype, device=x.device)
        scale = torch.as_tensor(scaler.scale_, dtype=x.dtype, device=x.device)
        x = (x - mean) / torch.clamp(scale, min=1e-12)
    return x @ coef.T + intercept, coef


def _iter_input_tensors(batch: AttributionBatch) -> list[tuple[str, torch.Tensor, list[BandSpec]]]:
    if isinstance(batch.inputs, dict):
        spec_map = batch.specs if isinstance(batch.specs, dict) else {}
        return [(k, v, list(spec_map[k])) for k, v in batch.inputs.items() if k in spec_map]
    if isinstance(batch.specs, dict):
        raise TypeError("Tensor inputs require a list of BandSpec, not a dict.")
    return [("input", batch.inputs, list(batch.specs))]


def _channel_saliency(tensor: torch.Tensor) -> np.ndarray:
    if tensor.grad is None:
        return np.zeros(tensor.shape[2], dtype=np.float32)
    sal = torch.abs(tensor.grad * tensor)
    reduce_dims = tuple(i for i in range(sal.ndim) if i != 2)
    return sal.mean(dim=reduce_dims).detach().cpu().numpy().astype(np.float32)


def gradient_input_importance(
    attribution_fn: Callable[[Any], AttributionBatch | tuple[torch.Tensor, torch.Tensor | dict[str, torch.Tensor], Any]],
    probe: Any,
    bench: Any,
    metadata: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    """Compute gradient × input saliency rows for a frozen encoder + fitted linear probe."""
    meta = dict(metadata or {})
    raw = attribution_fn(bench)
    batch = raw if isinstance(raw, AttributionBatch) else AttributionBatch(*raw)
    for _name, tensor, _specs in _iter_input_tensors(batch):
        tensor.retain_grad()
    logits, _ = _probe_linear_params(probe, batch.embeddings)
    if logits.shape[1] == 1:
        objective = logits[:, 0].mean()
    else:
        objective = logits.gather(1, logits.detach().argmax(dim=1, keepdim=True)).mean()
    objective.backward()

    rows: list[dict[str, Any]] = []
    for _name, tensor, specs in _iter_input_tensors(batch):
        vals = _channel_saliency(tensor)
        for spec, val in zip(specs, vals, strict=False):
            rows.append(
                {
                    **meta,
                    "importance_method": "grad_input",
                    "perturbation": "",
                    "modality": spec.modality,
                    "band": spec.band,
                    "band_group": spec.band_group,
                    "baseline_score": float("nan"),
                    "perturbed_score": float("nan"),
                    "delta_score": float("nan"),
                    "baseline_prob_mean": float("nan"),
                    "perturbed_prob_mean": float("nan"),
                    "delta_prob_mean": float("nan"),
                    "importance": float(val),
                    "normalized_importance": 0.0,
                    "rank": 0,
                    "n_samples": int(getattr(bench, "n_samples", tensor.shape[0])),
                    "repeat": 0,
                }
            )
    _normalize_and_rank(rows)
    return rows


def write_importance(
    rows: list[dict[str, Any]],
    output_path: str | Path | None = None,
    append: bool = True,
) -> Path:
    """Write rows to one flat CSV and return the path."""
    path = Path(output_path) if output_path is not None else DEFAULT_IMPORTANCE_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    mode = "a" if append and path.exists() else "w"
    columns = list(OUTPUT_COLUMNS)
    for row in rows:
        for key in row:
            if key not in columns:
                columns.append(key)
    with path.open(mode, newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        if mode == "w":
            writer.writeheader()
        writer.writerows(rows)
    return path
