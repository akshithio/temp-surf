"""EuroCropsML frozen 40-class population mask: the 7 globally-rare 6-digit HCAT classes (22
parcels) are dropped at load, before any regime. Synthetic preprocess dir; no real data.
"""

from __future__ import annotations

import numpy as np

import evals.benchmarks.eurocropsml as euro
from evals import split_spec as S


def _write_parcel(pre, country, code, n=3):
    """One EuroCropsML npz whose stem encodes country prefix + HCAT code."""
    data = np.ones((n, 13), dtype=np.float32)
    dates = np.array(["2020-05-01", "2020-06-01", "2020-07-01"], dtype="datetime64[D]")[:n]
    center = np.array([25.0, 59.0], dtype=np.float64)  # [lon, lat]
    np.savez(pre / f"{country}_x_{code}.npz", data=data, dates=dates, center=center)


def _code6_of(sample_id: str) -> str:
    return str(sample_id)[:-4].split("_")[-1][:6]


def test_rare_six_digit_classes_are_dropped_at_load(tmp_path):
    pre = tmp_path / "eurocropsml" / "preprocess"
    pre.mkdir(parents=True)
    # kept classes
    _write_parcel(pre, "EE", "3301010000")
    _write_parcel(pre, "LV", "3302000000")
    _write_parcel(pre, "PT", "3301020000")
    # removed classes (their 6-digit prefix is in the frozen mask)
    _write_parcel(pre, "EE", "3301120000")  # 330112
    _write_parcel(pre, "PT", "3301260000")  # 330126
    _write_parcel(pre, "LV", "3304030000")  # 330403

    bench = euro.load_benchmark(root=tmp_path, shuffle=False)

    kept6 = {_code6_of(s) for s in bench.sample_ids}
    assert kept6 == {"330101", "330200", "330102"}
    assert not (kept6 & set(S.EUROCROPS_REMOVED_CLASSES)), "a masked class survived"
    assert len(bench.sample_ids) == 3, "removed parcels were not dropped"
    # make_targets still works on the masked population
    y, groups = euro.make_targets(bench)
    assert len(y) == 3 and set(groups) == {"Estonia", "Latvia", "Portugal"}


def test_max_samples_returns_that_many_eligible_parcels(tmp_path):
    """The mask is applied BEFORE _select_files/max_samples: a max_samples subset returns that many
    eligible parcels, never silently short by masked ones it happened to draw."""
    pre = tmp_path / "eurocropsml" / "preprocess"
    pre.mkdir(parents=True)
    for i in range(5):  # 5 eligible classes 330100..330104
        _write_parcel(pre, "EE", f"33010{i}0000")
    _write_parcel(pre, "EE", "3301120000")  # 330112 masked
    _write_parcel(pre, "PT", "3301260000")  # 330126 masked
    _write_parcel(pre, "LV", "3304030000")  # 330403 masked

    bench = euro.load_benchmark(root=tmp_path, shuffle=True, seed=0, max_samples=4)

    assert len(bench.sample_ids) == 4, "max_samples did not return 4 ELIGIBLE parcels"
    kept6 = {_code6_of(s) for s in bench.sample_ids}
    assert not (kept6 & set(S.EUROCROPS_REMOVED_CLASSES)), "a masked parcel was returned"


def test_mask_is_the_seven_spec_classes():
    assert set(S.EUROCROPS_REMOVED_CLASSES) == {
        "330112", "330117", "330125", "330126", "330130", "330310", "330403"
    }
    assert sum(S.EUROCROPS_REMOVED_CLASSES.values()) == 22
