from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from evals.benchmarks import cropharvest, eurocropsml, pastis_r


def test_bin_crop_class_targets() -> None:
    bench = SimpleNamespace(labels=np.array([0.0, 1.0]), groups=np.array(["a", "b"], dtype=object))

    y, groups = cropharvest.make_targets(bench)

    assert cropharvest.BENCHMARK == "cropharvest"
    assert cropharvest.LABEL_KIND == "binary"
    assert y.dtype == np.int64
    np.testing.assert_array_equal(y, np.array([0, 1]))
    np.testing.assert_array_equal(groups, bench.groups)


def test_crop_class() -> None:
    label_names = ["3301010500", "3301019999", "3302000000"]
    bench = SimpleNamespace(labels=np.array([0, 1, 2, 0]), groups=np.array(["LV", "LV", "EE", "PT"], dtype=object), label_names=label_names)

    y, groups = eurocropsml.make_targets(bench)

    assert eurocropsml.BENCHMARK == "eurocropsml"
    assert eurocropsml.LABEL_KIND == "multiclass"
    np.testing.assert_array_equal(y, np.array([0, 0, 1, 0]))
    np.testing.assert_array_equal(groups, bench.groups)


def test_pastis_crop_seg_protocol() -> None:
    bench = SimpleNamespace(groups=np.array([1, 2, 3, 4, 5]))

    y, groups = pastis_r.make_targets(bench)

    assert pastis_r.BENCHMARK == "pastis_r"
    assert pastis_r.LABEL_KIND == "segmentation"
    assert pastis_r.TRAIN_FOLDS == {1, 2, 3}
    assert pastis_r.VAL_FOLDS == {4}
    assert pastis_r.TEST_FOLDS == {5}
    assert y.size == 0
    np.testing.assert_array_equal(groups, bench.groups)
