from __future__ import annotations

import numpy as np

from dataio.get_input import Benchmark
from models.agrifm import AgriFMModel
from models.olmoearth import OlmoEarthModel
from models.presto import PrestoModel
from models.tessera import TESSERA_S2_BANDS, TesseraModel
from utils import cacheutils


class FakePrestoModel:
    pass


def test_presto_is_registered_without_loading_external_weights() -> None:
    mod = cacheutils.build_model("presto", load_model=lambda weights_path: FakePrestoModel())

    assert isinstance(mod, PrestoModel)


def test_model_pool_wrappers_are_registered_without_loading_external_weights() -> None:
    assert AgriFMModel.name == "agrifm"
    assert TesseraModel.name == "tessera"
    assert cacheutils.MODELS["agrifm"] == ("models.agrifm", "AgriFMModel")
    assert cacheutils.MODELS["tessera"] == ("models.tessera", "TesseraModel")


def _benchmark() -> Benchmark:
    n, t = 2, 12
    return Benchmark(
        name="test",
        label_kind="binary",
        s2=np.arange(n * t * 11, dtype=np.float32).reshape(n, t, 11) + 1000,
        s1=np.full((n, t, 2), -12.0, dtype=np.float32),
        climate=np.zeros((n, t, 0), dtype=np.float32),
        s2_mask=np.array([[1] * t, [1] * 6 + [0] * 6], dtype=np.float32),
        s1_mask=np.array([[1] * t, [0] * t], dtype=np.float32),
        climate_mask=np.zeros((n, t), dtype=np.float32),
        doy=np.tile(np.arange(15, 360, 30, dtype=np.float32)[:t], (n, 1)),
        labels=np.array([0, 1]),
        groups=np.array(["a", "b"]),
        latlon=np.zeros((n, 2), dtype=np.float32),
        s2_bands=["B2", "B3", "B4", "B5", "B6", "B7", "B8", "B8A", "B11", "B12", "NDVI"],
        s1_bands=["VV", "VH"],
        climate_bands=[],
        years=np.array([2020, 2021]),
    )


def test_tessera_uses_v11_band_order_and_buckets_only_valid_observations() -> None:
    model = TesseraModel()
    groups = model._prepare_streams(_benchmark())

    assert TESSERA_S2_BANDS == ["B4", "B2", "B3", "B8", "B8A", "B5", "B6", "B7", "B11", "B12"]
    assert set(groups) == {(16, 16), (8, 8)}
    assert sum(len(indices) for indices, _s2, _s1 in groups.values()) == 2


def test_olmoearth_builds_one_timestamp_and_mask_per_observation(monkeypatch) -> None:
    class MaskValue:
        class _Value:
            def __init__(self, value):
                self.value = value

        ONLINE_MODEL = _Value(0)
        MISSING = _Value(3)

    class Modality:
        SENTINEL2_L2A = object()

    class Strategy:
        COMPUTED = object()

    class Normalizer:
        def __init__(self, _strategy):
            pass

        def normalize(self, _modality, values):
            return values

    import sys
    import types

    monkeypatch.setitem(sys.modules, "olmoearth_pretrain.datatypes", types.SimpleNamespace(MaskValue=MaskValue))
    monkeypatch.setitem(sys.modules, "olmoearth_pretrain.data.constants", types.SimpleNamespace(Modality=Modality))
    monkeypatch.setitem(
        sys.modules,
        "olmoearth_pretrain.data.normalize",
        types.SimpleNamespace(Normalizer=Normalizer, Strategy=Strategy),
    )
    images, masks, timestamps = OlmoEarthModel()._bench_to_olmoearth(_benchmark())

    assert images.shape == (2, 1, 1, 12, 12)
    assert masks.shape == (2, 1, 1, 12, 1)
    assert timestamps.shape == (2, 12, 3)
    assert np.all(masks[1, :, :, 6:, :] == 3)
    assert np.array_equal(timestamps[:, :, 2], np.array([[2020] * 12, [2021] * 12]))
