from __future__ import annotations

import csv
from types import SimpleNamespace

import numpy as np
import torch

from dataio.get_input import (
    Benchmark,
    ModalitySeries,
    NativeSeries,
    _synthetic_month_doy,
)
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
    """Native-contract fixture: S2 band 0 separates the two label groups (0 vs 10), band 1 carries
    the sample index. Each sample has 3 acquisitions (months 0-2)."""
    n, t = 4, 3
    months = np.arange(t, dtype=np.int64)
    doy = _synthetic_month_doy(12)[months]
    s2_series = []
    for i in range(n):
        s = np.zeros((t, len(CH_S2_BANDS)), dtype=np.float32)
        s[:, 0] = 10.0 if i >= 2 else 0.0
        s[:, 1] = float(i)
        s2_series.append(s)
    s1_series = [np.ones((t, len(CH_S1_BANDS)), dtype=np.float32) for _ in range(n)]
    clim_series = [np.full((t, len(CH_CLIMATE_BANDS)), 2.0, dtype=np.float32) for _ in range(n)]

    def _mod(series, bands):
        return ModalitySeries(series, [months] * n, [doy] * n, [np.full(t, 2020, np.int64)] * n, bands)

    native = NativeSeries(_mod(s2_series, CH_S2_BANDS), _mod(s1_series, CH_S1_BANDS), _mod(clim_series, CH_CLIMATE_BANDS))
    return Benchmark(
        name="tiny",
        label_kind="binary",
        native=native,
        labels=np.array([0, 0, 1, 1], dtype=np.int64),
        groups=np.array(["a", "b", "c", "d"], dtype=object),
        latlon=np.arange(n * 2, dtype=np.float32).reshape(n, 2),
        years=np.full(n, 2020, dtype=np.int64),
    )


def _band_cols(bench, modality: str) -> np.ndarray:
    """Stack a modality's per-sample series into (n, t, c) (fixtures use equal-length samples)."""
    return np.stack([np.asarray(v) for v in getattr(bench.native, modality).values])


class FakeModel:
    def encode(self, bench):
        vals, _doy, _months, _bands = bench.native_series("s2")
        b0 = np.array([v[:, 0].mean() for v in vals], dtype=np.float32)
        b1 = np.array([v[:, 1].mean() for v in vals], dtype=np.float32)
        return np.stack([b0, b1], axis=1)


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

    assert out.n_samples == 2
    assert out.monthly("s2")[0].shape == (2, 12, len(CH_S2_BANDS))
    assert len(out.native.s1.values) == 2
    assert len(out.native.climate.values) == 2
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

    base, new = _band_cols(bench, "s2"), _band_cols(out, "s2")
    assert not np.array_equal(new[:, :, 0], base[:, :, 0])
    np.testing.assert_array_equal(new[:, :, 1:], base[:, :, 1:])
    np.testing.assert_array_equal(_band_cols(out, "s1"), _band_cols(bench, "s1"))
    np.testing.assert_array_equal(_band_cols(out, "climate"), _band_cols(bench, "climate"))


def test_perturb_band_zero_zeros_only_selected_band() -> None:
    bench = _bench()
    spec = BandSpec("s2", CH_S2_BANDS[1], 1, "visible")

    out = perturb_band(bench, spec, mode="zero", seed=0)

    base, new = _band_cols(bench, "s2"), _band_cols(out, "s2")
    assert np.all(new[:, :, 1] == 0.0)
    np.testing.assert_array_equal(new[:, :, 0], base[:, :, 0])
    np.testing.assert_array_equal(new[:, :, 2:], base[:, :, 2:])


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
