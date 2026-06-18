"""Model × benchmark compatibility matrix — drives automatic model selection.

This is the executable form of the design doc's `(4) A Compatibility Matrix`
(and the table in the README). Each ``(benchmark, model)`` cell carries two
independent grades:

  * **precedent**  — how strongly the literature sanctions the baseline.
  * **adaptation** — how far our wrapper had to bend the model's native input.

The two collapse to a single runnability ``RANK`` (1 = best). A pair is eligible
to run iff its adaptation is below ``RUN_CUTOFF`` (i.e. ``CLEAN`` or ``MINOR``):
a 🔴 ``SEVERE`` adaptation produces a number that misrepresents the model, so it
is excluded from the cross-model comparison rather than run.

``main.py`` consumes :func:`eligible_models` so that you specify only the
*benchmarks* — the model set for each benchmark is read off this matrix.
"""

from __future__ import annotations

from enum import IntEnum


class Precedent(IntEnum):
    """How strongly the literature sanctions a (model, benchmark) baseline."""

    NONE = 0  # 🆕 no published precedent — runs, but the number is exploratory
    THIRD_PARTY = 1  # ⚠️ only a third-party re-implementation ran it (not the authors)
    EQUIVALENT = 2  # 🔗 own paper ran a directly-equivalent benchmark, not this one
    PAPER = 3  # 📄 own paper ran this exact benchmark


class Adaptation(IntEnum):
    """How far our wrapper bends the model's native input form."""

    CLEAN = 0  # native input — no adaptation
    MINOR = 1  # 🚧 defensible reframing or tolerable missing input
    SEVERE = 2  # 🔴 off-distribution input, or a data form the model has no pathway for


# Curated runnability rank for every (precedent, adaptation) pair (1 = best). This
# is NOT a closed-form of the two axes: the order is the one fixed in the design
# doc's "Ranking & run cutoff" table (e.g. 📄🚧 deliberately outranks a clean
# ⚠️/🔗/🆕). Edit here to match the doc; nothing else depends on the exact integers.
RANK: dict[tuple[Precedent, Adaptation], int] = {
    (Precedent.PAPER, Adaptation.CLEAN): 1,
    (Precedent.PAPER, Adaptation.MINOR): 2,
    (Precedent.THIRD_PARTY, Adaptation.CLEAN): 3,
    (Precedent.EQUIVALENT, Adaptation.CLEAN): 4,
    (Precedent.NONE, Adaptation.CLEAN): 5,
    (Precedent.EQUIVALENT, Adaptation.MINOR): 6,
    (Precedent.THIRD_PARTY, Adaptation.MINOR): 7,
    (Precedent.NONE, Adaptation.MINOR): 8,
    (Precedent.PAPER, Adaptation.SEVERE): 9,
    (Precedent.EQUIVALENT, Adaptation.SEVERE): 10,
    (Precedent.THIRD_PARTY, Adaptation.SEVERE): 11,
    (Precedent.NONE, Adaptation.SEVERE): 12,
}

# A pair runs iff its adaptation is strictly below this (CLEAN or MINOR; never SEVERE).
RUN_CUTOFF: Adaptation = Adaptation.SEVERE

_P, _E, _T, _N = Precedent.PAPER, Precedent.EQUIVALENT, Precedent.THIRD_PARTY, Precedent.NONE
_CLEAN, _MINOR, _SEVERE = Adaptation.CLEAN, Adaptation.MINOR, Adaptation.SEVERE

# benchmark name -> model name -> (precedent, adaptation). Mirrors the README matrix.
# Benchmark keys are the BENCHMARKS module names used in main.py; model keys are the
# cacheutils.ENCODERS keys.
MATRIX: dict[str, dict[str, tuple[Precedent, Adaptation]]] = {
    "cropharvest": {
        "presto": (_P, _CLEAN),
        "olmoearth": (_P, _CLEAN),
        "galileo": (_P, _CLEAN),
        "tessera": (_E, _SEVERE),
        "agrifm": (_N, _SEVERE),
    },
    "eurocropsml": {
        "presto": (_E, _CLEAN),
        "olmoearth": (_E, _CLEAN),
        "galileo": (_E, _CLEAN),
        "tessera": (_E, _MINOR),
        "agrifm": (_N, _SEVERE),
    },
    "breizhcrops": {
        "presto": (_T, _CLEAN),
        "olmoearth": (_P, _CLEAN),
        "galileo": (_P, _CLEAN),
        "tessera": (_E, _MINOR),
        "agrifm": (_N, _SEVERE),
    },
    "pastis_r": {
        "presto": (_N, _MINOR),
        "olmoearth": (_P, _CLEAN),
        "galileo": (_P, _CLEAN),
        "tessera": (_P, _CLEAN),
        "agrifm": (_E, _CLEAN),
    },
}


def grade(benchmark: str, model: str) -> tuple[Precedent, Adaptation] | None:
    """Return the (precedent, adaptation) cell for a pair, or None if ungraded."""
    return MATRIX.get(benchmark, {}).get(model)


def rank(benchmark: str, model: str) -> int | None:
    """Runnability rank (1 = best) for a pair, or None if ungraded."""
    cell = grade(benchmark, model)
    return RANK[cell] if cell is not None else None


def is_eligible(benchmark: str, model: str) -> bool:
    """True iff the pair is graded and its adaptation is below the run cutoff."""
    cell = grade(benchmark, model)
    return cell is not None and cell[1] < RUN_CUTOFF


def eligible_models(benchmark: str) -> list[str]:
    """Models that should run on ``benchmark``, best-rank first.

    Raises ``KeyError`` if the benchmark is absent from the matrix, so a typo or a
    new, ungraded benchmark fails loudly rather than silently running nothing.
    """
    if benchmark not in MATRIX:
        raise KeyError(f"Benchmark {benchmark!r} not in compatibility matrix. Known: {sorted(MATRIX)}")
    runnable = [model for model in MATRIX[benchmark] if is_eligible(benchmark, model)]
    return sorted(runnable, key=lambda model: RANK[MATRIX[benchmark][model]])
