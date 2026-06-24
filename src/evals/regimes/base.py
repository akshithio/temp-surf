"""Shared types for split regimes.

A *regime* owns two things:

    (1) the domain basis: how each sample is assigned a domain label
    (2) the split strategy — how those domains become train/val/test
"""

from __future__ import annotations

import importlib
from dataclasses import dataclass, field

import numpy as np


def _empty() -> np.ndarray:
    return np.empty(0, dtype=np.int64)


@dataclass(frozen=True)
class Split:
    """One train/val/test partition produced by a regime.

    ``label`` identifies the held-out domain (or fold). ``val`` may be empty when a
    regime trains on the full non-target pool and leaves threshold calibration to
    the probe's own internal split. ``domain`` is the raw domain value held out (e.g. the
    region ``"Estonia"`` behind label ``"Estonia"``); the runner uses it to detect a
    leave-one-domain-out regime that silently dropped a domain. Defaults to ``label``.
    """

    label: str
    train: np.ndarray
    test: np.ndarray
    val: np.ndarray = field(default_factory=_empty)
    domain: str | None = None


def geography_domains(bench) -> np.ndarray:
    """Default domain assignment: the benchmark's native region/source groups."""
    return np.asarray(bench.groups, dtype=object)


REGIME_PROBLEMS: list[tuple[str, str, str]] = []


def load_regime(regime_name: str):
    """Import a split-regime module."""
    return importlib.import_module(f"evals.regimes.{regime_name}")


def regime_problem(benchmark: str, regime: str, reason: str, *, overwrite_mode: bool) -> None:
    """Surface a declared regime that did not run."""
    REGIME_PROBLEMS.append((benchmark, regime, reason))
    if overwrite_mode:
        raise RuntimeError(f"declared regime did not run -- {benchmark}/{regime}: {reason}")
    bar = "!" * 78
    print(
        f"\n{bar}\n!! REGIME DECLARED BUT DID NOT RUN -- {benchmark}/{regime}\n!! {reason}"
        f"\n!! (OVERWRITE_MODE is False for this run; it would be a hard failure with OVERWRITE_MODE=True)\n{bar}\n",
        flush=True,
    )


def report_regime_problems() -> None:
    """Print a consolidated list of regimes that were declared but did not run."""
    if not REGIME_PROBLEMS:
        return
    bar = "=" * 78
    print(f"\n{bar}\nREGIMES DECLARED BUT NOT RUN ({len(REGIME_PROBLEMS)}):", flush=True)
    for benchmark, regime, reason in REGIME_PROBLEMS:
        print(f"  - {benchmark}/{regime}: {reason}", flush=True)
    print(f"{bar}\n", flush=True)


def iter_splits(split_regime, bench, y, holdouts, seed, *, overwrite_mode: bool, val_group=None):
    """Yield split metadata and regime-assigned domain labels."""
    regime = load_regime(split_regime)
    bench_name = getattr(bench, "name", "?")
    try:
        domains = np.asarray(regime.assign_domains(bench), dtype=object)
    except Exception as exc:
        regime_problem(
            bench_name,
            split_regime,
            f"domain assignment failed ({type(exc).__name__}: {exc})",
            overwrite_mode=overwrite_mode,
        )
        return
    if len(domains) != len(y):
        raise ValueError(
            f"{split_regime}.assign_domains returned {len(domains)} domains for {len(y)} labels"
        )
    n_unknown = int(np.isin(domains.astype(str), ("unknown", "nan")).sum())
    if n_unknown:
        print(
            f"   [{bench_name}/{split_regime}] {n_unknown}/{len(domains)} samples have no domain "
            f"(unknown/nan coords) and are excluded from this regime's holdouts",
            flush=True,
        )
    n_splits = 0
    yielded_labels: set[str] = set()
    yielded_domains: set[str] = set()
    for split in regime.iter_splits(y, domains, seed=seed, holdouts=holdouts, val_group=val_group):
        n_splits += 1
        yielded_labels.add(str(split.label))
        yielded_domains.add(str(getattr(split, "domain", None) or split.label))
        yield split.label, split.train, split.val, split.test, domains, regime.HAS_TARGET, regime.GROUP_KIND
    if n_splits == 0:
        labels = sorted({str(d) for d in domains})
        shown = labels[:8] + (["..."] if len(labels) > 8 else [])
        regime_problem(
            bench_name,
            split_regime,
            f"produced 0 splits (domain labels seen: {shown})",
            overwrite_mode=overwrite_mode,
        )
    elif getattr(regime, "USES_CURATED_HOLDOUTS", False):
        missing = [str(h) for h in (holdouts or []) if str(h) not in yielded_labels]
        if missing:
            regime_problem(
                bench_name,
                split_regime,
                f"curated holdout(s) dropped (no valid split): {missing}",
                overwrite_mode=overwrite_mode,
            )
    elif getattr(regime, "LEAVE_ONE_DOMAIN_OUT", False):
        attempted = {str(d) for d in domains if str(d) not in ("unknown", "nan")}
        missing = sorted(attempted - yielded_domains)
        if missing:
            regime_problem(
                bench_name,
                split_regime,
                f"domain(s) dropped (no valid split): {missing}",
                overwrite_mode=overwrite_mode,
            )


def segmentation_fold_configs(bench_mod, regimes, *, overwrite_mode: bool):
    """Yield dense fold configs for segmentation regimes."""
    for regime_name in regimes:
        fold_iter = getattr(load_regime(regime_name), "iter_fold_splits", None)
        if fold_iter is None:
            regime_problem(
                getattr(bench_mod, "BENCHMARK", "?"),
                regime_name,
                "no dense (segmentation) realization -- regime exposes no iter_fold_splits",
                overwrite_mode=overwrite_mode,
            )
            continue
        for label, train_folds, val_folds, test_folds in fold_iter(bench_mod):
            yield (regime_name, label, train_folds, val_folds, test_folds)
