from __future__ import annotations

import json
import sys
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from dataio.get_input import Benchmark, ModalitySeries, NativeSeries, _synthetic_month_doy
from models.agrifm import AgriFMModel, _split_s2_checkpoint
from models.olmoearth import OLMOEARTH_BATCH_SIZE, OlmoEarthModel
from models.presto import PrestoModel
from models.tessera import TESSERA_S2_BANDS, TesseraModel
from utils import cacheutils
from utils.models.galileoutil import GalileoNativeModel


class FakePrestoModel:
    pass


def test_presto_is_registered_without_loading_external_weights() -> None:
    mod = cacheutils.build_model("presto", load_model=lambda weights_path: FakePrestoModel())

    assert isinstance(mod, PrestoModel)


def test_presto_default_loader_returns_pinned_encoder_api(tmp_path, monkeypatch) -> None:
    import models.presto as presto

    encoder = object()

    class FakeFullModel:
        def __init__(self):
            self.encoder = encoder

        def load_state_dict(self, _state):
            pass

        def eval(self):
            return self

    class FakePresto:
        construct = FakeFullModel

    weights = tmp_path / "presto.pth"
    weights.touch()
    monkeypatch.setattr(presto, "Presto", FakePresto)
    monkeypatch.setattr(torch, "load", lambda *_args, **_kwargs: {})

    assert presto._default_load_model(str(weights)) is encoder


def test_model_pool_wrappers_are_registered_without_loading_external_weights() -> None:
    assert AgriFMModel.name == "agrifm"
    assert TesseraModel.name == "tessera"
    assert cacheutils.MODELS["agrifm"] == ("models.agrifm", "AgriFMModel")
    assert cacheutils.MODELS["tessera"] == ("models.tessera", "TesseraModel")


def test_agrifm_splits_released_encoder_checkpoint_namespace() -> None:
    patch, backbone = _split_s2_checkpoint(
        {
            "encoder.S2_patch_emd.weights": torch.tensor([1.0]),
            "encoder.S2_patch_emd.bias": torch.tensor([2.0]),
            "encoder.backbone.layers.0.weight": torch.tensor([3.0]),
            "encoder.HLSL30_patch_emd.weights": torch.tensor([4.0]),
        }
    )

    assert set(patch) == {"weights", "bias"}
    assert set(backbone) == {"layers.0.weight"}


def _benchmark() -> Benchmark:
    """Native-contract fixture: sample 0 has 12 monthly S2+S1 acquisitions; sample 1 has 6 S2
    acquisitions (months 0-5) and no S1 -- so the monthly view masks sample 1's months 6-11 and
    TESSERA buckets the two samples to (16, 16) and (8, 8)."""
    s2_bands = ["B2", "B3", "B4", "B5", "B6", "B7", "B8", "B8A", "B11", "B12", "NDVI"]
    doy12 = _synthetic_month_doy(12)
    s2v = [np.arange(12 * 11, dtype=np.float32).reshape(12, 11) + 1000,
           np.arange(6 * 11, dtype=np.float32).reshape(6, 11) + 2000]
    s1v = [np.full((12, 2), -12.0, dtype=np.float32), np.zeros((0, 2), dtype=np.float32)]
    native = NativeSeries(
        s2=ModalitySeries(
            s2v, [np.arange(12), np.arange(6)], [doy12, doy12[:6]],
            [np.full(12, 2020), np.full(6, 2021)], s2_bands,
        ),
        s1=ModalitySeries(
            s1v, [np.arange(12), np.zeros(0, np.int64)], [doy12, np.zeros(0, np.float32)],
            [np.full(12, 2020), np.zeros(0, np.int64)], ["VV", "VH"],
        ),
        climate=ModalitySeries.absent(2),
    )
    return Benchmark(
        name="test",
        label_kind="binary",
        native=native,
        labels=np.array([0, 1]),
        groups=np.array(["a", "b"]),
        latlon=np.zeros((2, 2), dtype=np.float32),
        years=np.array([2020, 2021]),
    )


def test_tessera_uses_v11_band_order_and_buckets_only_valid_observations() -> None:
    model = TesseraModel()
    groups = model._prepare_streams(_benchmark())

    assert TESSERA_S2_BANDS == ["B4", "B2", "B3", "B8", "B8A", "B5", "B6", "B7", "B11", "B12"]
    assert set(groups) == {(16, 16), (8, 8)}
    assert sum(len(indices) for indices, _s2, _s1 in groups.values()) == 2


def test_tessera_loads_released_encoder_namespace(tmp_path, monkeypatch) -> None:
    import models.tessera as tessera

    loaded = {}

    class FakeModel:
        def load_state_dict(self, state, strict=False):
            loaded.update(state)
            return [], []

        def to(self, _device):
            return self

        def eval(self):
            return self

        def parameters(self):
            return []

    weights = tmp_path / "tessera_v1_1_mpc_encoder.pt"
    weights.touch()
    monkeypatch.setattr(tessera, "TesseraV11Model", FakeModel)
    monkeypatch.setattr(
        tessera.torch,
        "load",
        lambda *_args, **_kwargs: {
            "model_state": {"s2_backbone.transformer_encoder.layers.0.weight": np.array([1.0])}
        },
    )

    TesseraModel(weights_path=weights, device="cpu")._ensure_loaded()

    assert "s2_backbone.transformer_model.layers.0.weight" in loaded
    assert "s2_backbone.transformer_encoder.layers.0.weight" not in loaded


def test_galileo_loader_accepts_pinned_encoder_config(tmp_path, monkeypatch) -> None:
    folder = tmp_path / "galileo"
    folder.mkdir()
    config = {
        "model": {
            "encoder": {
                "embedding_size": 768,
                "depth": 12,
                "num_heads": 12,
                "mlp_ratio": 4,
                "max_sequence_length": 24,
                "freeze_projections": False,
                "drop_path": 0.1,
                "max_patch_size": 8,
            }
        }
    }
    (folder / "config.json").write_text(json.dumps(config))
    (folder / "model.pt").touch()

    class FakeGalileo(GalileoNativeModel):
        def __init__(self, **kwargs):
            torch.nn.Module.__init__(self)
            self.kwargs = kwargs

        def load_state_dict(self, state_dict, strict=True):
            self.loaded_state = state_dict

    monkeypatch.setattr(torch, "load", lambda *_args, **_kwargs: {})
    loaded = FakeGalileo.load_from_folder(folder, torch.device("cpu"))

    assert loaded.kwargs == config["model"]["encoder"]


def test_olmoearth_builds_one_timestamp_and_mask_per_observation(monkeypatch) -> None:
    class MaskValue:
        class _Value:
            def __init__(self, value):
                self.value = value

        ONLINE_ENCODER = _Value(0)
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


def test_olmoearth_default_loader_returns_pinned_encoder_api(tmp_path, monkeypatch) -> None:
    import sys
    import types

    import models.olmoearth as olmoearth

    class FakeEncoder:
        def eval(self):
            return self

    encoder = FakeEncoder()
    model_dir = tmp_path / "olmoearth"
    model_dir.mkdir()
    (model_dir / "config.json").touch()
    (model_dir / "weights.pth").touch()
    monkeypatch.setitem(
        sys.modules,
        "olmoearth_pretrain.model_loader",
        types.SimpleNamespace(load_model_from_path=lambda _path: types.SimpleNamespace(encoder=encoder)),
    )

    assert olmoearth._default_load_model(model_dir) is encoder


def test_olmoearth_uses_fixed_high_throughput_batch_contract() -> None:
    assert OLMOEARTH_BATCH_SIZE == 2048
    assert OlmoEarthModel().batch_size == OLMOEARTH_BATCH_SIZE


def test_olmoearth_encode_uses_batch_invariant_general_path(monkeypatch) -> None:
    calls = []

    class Sample:
        def __init__(self, sentinel2_l2a, sentinel2_l2a_mask, timestamps):
            self.sentinel2_l2a = sentinel2_l2a
            self.sentinel2_l2a_mask = sentinel2_l2a_mask
            self.timestamps = timestamps

    class Encoder:
        def __call__(self, sample, *, fast_pass, patch_size):
            calls.append((fast_pass, patch_size))
            pooled = torch.zeros((sample.sentinel2_l2a.shape[0], 768))
            return {"tokens_and_masks": SimpleNamespace(sentinel2_l2a=pooled)}

    monkeypatch.setitem(
        sys.modules,
        "olmoearth_pretrain.datatypes",
        SimpleNamespace(MaskedOlmoEarthSample=Sample),
    )
    monkeypatch.setitem(
        sys.modules,
        "olmoearth_pretrain.nn.pooling",
        SimpleNamespace(
            PoolingType=SimpleNamespace(MEAN="mean"),
            pool_unmasked_tokens=lambda tokens, _pooling: tokens.sentinel2_l2a,
        ),
    )
    model = OlmoEarthModel(batch_size=2, _model=Encoder())
    monkeypatch.setattr(
        model,
        "_bench_to_olmoearth",
        lambda _bench: (
            np.zeros((5, 1, 1, 12, 12), dtype=np.float32),
            np.zeros((5, 1, 1, 12, 1), dtype=np.float32),
            np.zeros((5, 12, 3), dtype=np.int64),
        ),
    )

    assert model.encode(object()).shape == (5, 768)
    assert calls == [(False, 1), (False, 1), (False, 1)]


def test_pretrained_classification_smoke_runs_under_pytest() -> None:
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required for pretrained model smoke coverage")
    if not (cacheutils.INPUT_ROOT / "cropharvest").exists():
        pytest.skip("CropHarvest input is not staged")
    from tests import smoke_models

    smoke_models.main()


def test_pretrained_dense_smoke_runs_under_pytest() -> None:
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required for dense pretrained model smoke coverage")
    if not (cacheutils.INPUT_ROOT / "pastis").exists():
        pytest.skip("PASTIS input is not staged")
    from tests import smoke_pastis

    smoke_pastis.main()
