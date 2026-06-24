"""Model/benchmark eligibility and model input footprints."""

from __future__ import annotations

BENCHMARKS: tuple[str, ...] = ("cropharvest", "eurocropsml", "breizhcrops", "pastis")
MODEL_ORDER: tuple[str, ...] = ("presto", "tessera", "agrifm", "olmoearth", "galileo", "raw")

BLOCKED_MODELS: dict[str, set[str]] = {
    "cropharvest": {"tessera", "agrifm"},
    "eurocropsml": {"agrifm"},
    "breizhcrops": {"agrifm"},
}

RUN_RANK: dict[tuple[str, str], int] = {
    ("cropharvest", "presto"): 1,
    ("cropharvest", "olmoearth"): 1,
    ("cropharvest", "galileo"): 1,
    ("eurocropsml", "presto"): 4,
    ("eurocropsml", "olmoearth"): 4,
    ("eurocropsml", "galileo"): 4,
    ("eurocropsml", "tessera"): 6,
    ("breizhcrops", "presto"): 3,
    ("breizhcrops", "tessera"): 6,
    ("breizhcrops", "olmoearth"): 1,
    ("breizhcrops", "galileo"): 1,
    ("pastis", "presto"): 8,
    ("pastis", "tessera"): 1,
    ("pastis", "agrifm"): 4,
    ("pastis", "olmoearth"): 1,
    ("pastis", "galileo"): 1,
}

MODEL_INPUT_MODALITIES: dict[str, tuple[str, ...]] = {
    "raw": ("s2", "s1", "climate"),
    "presto": ("s2", "s1", "climate", "latlon", "time"),
    "galileo": ("s2", "s1", "time"),
    "olmoearth": ("s2", "time"),
    "tessera": ("s2", "s1", "time"),
    "agrifm": ("s2",),
}


def rank(benchmark: str, model: str) -> int | None:
    """Return the table rank for an eligible non-raw pair."""
    return RUN_RANK.get((benchmark, model))


def is_eligible(benchmark: str, model: str) -> bool:
    """True when the model should run on the benchmark."""
    return benchmark in BENCHMARKS and model in MODEL_ORDER and model not in BLOCKED_MODELS.get(benchmark, set())


def eligible_models(benchmark: str) -> list[str]:
    """Models that should run on ``benchmark``, sorted by the compatibility table rank."""
    if benchmark not in BENCHMARKS:
        raise KeyError(f"Benchmark {benchmark!r} not in compatibility table. Known: {sorted(BENCHMARKS)}")
    order = {model: idx for idx, model in enumerate(MODEL_ORDER)}
    models = [model for model in MODEL_ORDER if is_eligible(benchmark, model)]
    return sorted(models, key=lambda model: (RUN_RANK.get((benchmark, model), 999), order[model]))


def input_modalities(model: str) -> tuple[str, ...]:
    """Modalities consumed by ``model``; empty tuple if unregistered."""
    return MODEL_INPUT_MODALITIES.get(model, ())
