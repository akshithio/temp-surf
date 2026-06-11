from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from evals.tasks import bin_crop_class, crop_class, pheno_reg, yield_reg


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


def test_pheno_reg_targets() -> None:
    bench = SimpleNamespace(labels=np.array([101, 155]), groups=np.array(["upper", "lower"], dtype=object))

    y, groups = pheno_reg.make_targets(bench)

    assert pheno_reg.BENCHMARK == "sickle"
    assert pheno_reg.TASK_KIND == "regression"
    assert y.dtype == np.float32
    np.testing.assert_array_equal(groups, bench.groups)


def test_yield_reg_targets() -> None:
    bench = SimpleNamespace(labels=np.array([5.2, 7.4]), groups=np.array(["Brazil", "Germany"], dtype=object))

    y, groups = yield_reg.make_targets(bench)

    assert yield_reg.BENCHMARK == "yieldsat"
    assert yield_reg.TASK_KIND == "regression"
    assert yield_reg.HOLDOUTS == ["Argentina", "Brazil", "Germany", "Uruguay"]
    assert y.dtype == np.float32
    np.testing.assert_allclose(y, np.array([5.2, 7.4], dtype=np.float32))
    np.testing.assert_array_equal(groups, bench.groups)
