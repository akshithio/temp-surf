from __future__ import annotations

import csv
from types import SimpleNamespace

import numpy as np
import torch

from dataio.get_input import Benchmark
from diagnostics.interp import (
    AttributionBatch,
    BandSpec,
    band_specs,
    gradient_input_importance,
    permutation_importance,
    perturb_band,
    subset_benchmark,
    write_importance,
)
from evals.benchmarks.cropharvest import CH_CLIMATE_BANDS, CH_S1_BANDS, CH_S2_BANDS


def _bench() -> Benchmark:
    n, t = 4, 3
    s2 = np.zeros((n, t, len(CH_S2_BANDS)), dtype=np.float32)
    s2[:2, :, 0] = 0.0
    s2[2:, :, 0] = 10.0
    s2[:, :, 1] = np.arange(n, dtype=np.float32)[:, None]
    return Benchmark(
        name="tiny",
        label_kind="binary",
        s2=s2,
        s1=np.ones((n, t, len(CH_S1_BANDS)), dtype=np.float32),
        climate=np.full((n, t, len(CH_CLIMATE_BANDS)), 2.0, dtype=np.float32),
        s2_mask=np.ones((n, t), dtype=np.float32),
        s1_mask=np.ones((n, t), dtype=np.float32),
        climate_mask=np.ones((n, t), dtype=np.float32),
        doy=np.tile(np.arange(1, t + 1, dtype=np.float32), (n, 1)),
        labels=np.array([0, 0, 1, 1], dtype=np.int64),
        groups=np.array(["a", "b", "c", "d"], dtype=object),
        latlon=np.arange(n * 2, dtype=np.float32).reshape(n, 2),
        s2_bands=CH_S2_BANDS,
        s1_bands=CH_S1_BANDS,
        climate_bands=CH_CLIMATE_BANDS,
    )


class FakeModel:
    def encode(self, bench):
        return np.stack([bench.s2[:, :, 0].mean(axis=1), bench.s2[:, :, 1].mean(axis=1)], axis=1).astype(np.float32)


class FakeProbe:
    coef_ = np.array([[1.0, 0.0]], dtype=np.float32)
    intercept_ = np.array([-5.0], dtype=np.float32)

    def predict(self, x):
        return (x[:, 0] >= 5.0).astype(np.int64)

    def predict_proba(self, x):
        p = 1.0 / (1.0 + np.exp(-(x[:, 0] - 5.0)))
        return np.stack([1.0 - p, p], axis=1)


def _accuracy(probe, x, y):
    return float(np.mean(probe.predict(x) == y))


def test_subset_benchmark_preserves_sample_alignment() -> None:
    bench = _bench()

    out = subset_benchmark(bench, [3, 1])

    assert out.s2.shape == (2, bench.timesteps, len(CH_S2_BANDS))
    assert out.s1.shape == (2, bench.timesteps, len(CH_S1_BANDS))
    assert out.climate.shape == (2, bench.timesteps, len(CH_CLIMATE_BANDS))
    np.testing.assert_array_equal(out.labels, bench.labels[[3, 1]])
    np.testing.assert_array_equal(out.groups, bench.groups[[3, 1]])
    np.testing.assert_array_equal(out.latlon, bench.latlon[[3, 1]])
    assert out.s2_bands == bench.s2_bands
    assert out.s1_bands == bench.s1_bands
    assert out.climate_bands == bench.climate_bands


def test_perturb_band_permute_changes_only_selected_band() -> None:
    bench = _bench()
    spec = BandSpec("s2", CH_S2_BANDS[0], 0, "visible")

    out = perturb_band(bench, spec, mode="permute", seed=0)

    assert not np.array_equal(out.s2[:, :, 0], bench.s2[:, :, 0])
    np.testing.assert_array_equal(out.s2[:, :, 1:], bench.s2[:, :, 1:])
    np.testing.assert_array_equal(out.s1, bench.s1)
    np.testing.assert_array_equal(out.climate, bench.climate)


def test_perturb_band_zero_zeros_only_selected_band() -> None:
    bench = _bench()
    spec = BandSpec("s2", CH_S2_BANDS[1], 1, "visible")

    out = perturb_band(bench, spec, mode="zero", seed=0)

    assert np.all(out.s2[:, :, 1] == 0.0)
    np.testing.assert_array_equal(out.s2[:, :, 0], bench.s2[:, :, 0])
    np.testing.assert_array_equal(out.s2[:, :, 2:], bench.s2[:, :, 2:])


def test_permutation_importance_ranks_known_useful_band_first() -> None:
    bench = _bench()
    specs = [s for s in band_specs(bench) if s.modality == "s2" and s.index in {0, 1}]

    rows = permutation_importance(
        FakeModel(),
        FakeProbe(),
        bench,
        bench.labels,
        _accuracy,
        specs=specs,
        metadata={"model": "fake", "benchmark": "unit"},
        seed=2,
    )

    top = min(rows, key=lambda r: r["rank"])
    assert top["band"] == CH_S2_BANDS[0]
    assert top["importance"] > 0
    assert sum(r["normalized_importance"] for r in rows) == 1.0


def test_gradient_input_importance_returns_finite_band_values() -> None:
    bench = SimpleNamespace(n_samples=3)
    specs = [
        BandSpec("s2", "B2", 0, "visible"),
        BandSpec("s2", "B3", 1, "visible"),
    ]

    def callback(_bench):
        x = torch.tensor(
            [
                [[1.0, 3.0], [2.0, 4.0]],
                [[2.0, 1.0], [3.0, 1.0]],
                [[4.0, 2.0], [5.0, 2.0]],
            ],
            requires_grad=True,
        )
        emb = torch.stack([x[:, :, 0].mean(dim=1), x[:, :, 1].mean(dim=1)], dim=1)
        return AttributionBatch(emb, x, specs)

    rows = gradient_input_importance(callback, FakeProbe(), bench, metadata={"model": "fake"})

    assert [r["band"] for r in rows] == ["B2", "B3"]
    assert all(np.isfinite(r["importance"]) and r["importance"] >= 0 for r in rows)
    assert rows[0]["importance"] > rows[1]["importance"]


def test_write_importance_writes_flat_csv(tmp_path) -> None:
    path = tmp_path / "feature_importance.csv"
    rows = [{"model": "fake", "benchmark": "unit", "importance_method": "permutation", "importance": 1.5}]

    out = write_importance(rows, output_path=path, append=False)

    assert out == path
    with path.open(newline="", encoding="utf-8") as f:
        read = list(csv.DictReader(f))
    assert read[0]["model"] == "fake"
    assert read[0]["importance"] == "1.5"
