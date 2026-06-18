from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from evals.tasks import bin_crop_class, crop_class, pastis_crop_seg


def test_bin_crop_class_targets() -> None:
    bench = SimpleNamespace(labels=np.array([0.0, 1.0]), groups=np.array(["a", "b"], dtype=object))

    y, groups = bin_crop_class.make_targets(bench)

    assert bin_crop_class.BENCHMARK == "cropharvest"
    assert bin_crop_class.TASK_KIND == "binary"
    assert y.dtype == np.int64
    np.testing.assert_array_equal(y, np.array([0, 1]))
    np.testing.assert_array_equal(groups, bench.groups)


def test_crop_class() -> None:
    label_names = ["3301010500", "3301019999", "3302000000"]
    bench = SimpleNamespace(labels=np.array([0, 1, 2, 0]), groups=np.array(["LV", "LV", "EE", "PT"], dtype=object), label_names=label_names)

    y, groups = crop_class.make_targets(bench)

    assert crop_class.BENCHMARK == "eurocropsml"
    assert crop_class.TASK_KIND == "multiclass"
    np.testing.assert_array_equal(y, np.array([0, 0, 1, 0]))
    np.testing.assert_array_equal(groups, bench.groups)


def test_pastis_crop_seg_protocol() -> None:
    bench = SimpleNamespace(groups=np.array([1, 2, 3, 4, 5]))

    y, groups = pastis_crop_seg.make_targets(bench)

    assert pastis_crop_seg.BENCHMARK == "pastis"
    assert pastis_crop_seg.TASK_KIND == "segmentation"
    assert pastis_crop_seg.TRAIN_FOLDS == {1, 2, 3}
    assert pastis_crop_seg.VAL_FOLDS == {4}
    assert pastis_crop_seg.TEST_FOLDS == {5}
    assert y.size == 0
    np.testing.assert_array_equal(groups, bench.groups)
