"""CropHarvest metadata: Mali merge into the canonical region, and provenance retained separately.

The canonical geographic region (17) is what geographic/random/spatial regimes leave out; the
original provenance (22 source datasets) is retained for the official Togo split, which needs the
un-merged Togo-source vs Togo-eval distinction the canonical region collapses. No real data.
"""

from __future__ import annotations

import json
from types import SimpleNamespace

import h5py
import numpy as np
import pytest

import evals.benchmarks.cropharvest as ch
from evals import split_spec as S


@pytest.mark.parametrize("dataset,region", [
    ("mali", "mali"),
    ("mali-non-crop", "mali"),          # THE new merge
    ("Mali_2019", "mali"),
    ("togo", "togo"),
    ("togo-eval", "togo"),              # eval collapses into canonical togo
    ("lem_brazil", "lem-brazil"),
    ("kenya-non-crop", "kenya"),
    ("rwanda", "rwanda"),
    ("ethiopia", "ethiopia"),           # unmerged regions pass through
    ("croplands", "croplands"),
    ("geowiki-landcover-2017", "geowiki-landcover-2017"),
])
def test_canonical_region_merges(dataset, region):
    assert ch.canonical_region(dataset) == region
    assert ch._ch_geo_group(dataset) == region  # alias agrees


def test_provenance_is_recovered_from_stable_id_and_differs_from_canonical():
    # mali-non-crop provenance is retained even though its canonical region is 'mali'
    assert ch.provenance_dataset("1619_central-asia.h5") == "central-asia"
    assert ch.provenance_dataset("42_mali-non-crop.h5") == "mali-non-crop"
    assert ch.canonical_region("mali-non-crop") == "mali"
    # togo source vs togo-eval: distinct provenance, same canonical region
    assert ch.provenance_dataset("7_togo.h5") == "togo"
    assert ch.provenance_dataset("9_togo-eval.h5") == "togo-eval"
    assert ch.canonical_region("togo") == ch.canonical_region("togo-eval") == "togo"


def test_provenance_groups_array():
    bench = SimpleNamespace(sample_ids=np.asarray(
        ["1_mali.h5", "2_mali-non-crop.h5", "3_togo.h5", "4_togo-eval.h5"], dtype=object))
    prov = ch.provenance_groups(bench)
    assert list(prov) == ["mali", "mali-non-crop", "togo", "togo-eval"]
    # canonical collapses the pairs the official split must keep apart
    canon = [ch.canonical_region(p) for p in prov]
    assert canon == ["mali", "mali", "togo", "togo"]


def test_all_canonical_regions_are_in_the_spec_set():
    # every region the merge can emit for the known collections is one of the 17 canonical regions
    known_provenance = [
        "central-asia", "croplands", "ethiopia", "geowiki-landcover-2017", "ile-de-france",
        "kenya", "kenya-non-crop", "lem-brazil", "mali", "mali-non-crop", "martinique-france",
        "reunion-france", "rwanda", "sudan", "tanzania", "togo", "togo-eval", "uganda",
        "usa-kern", "zimbabwe",
    ]
    emitted = {ch.canonical_region(p) for p in known_provenance}
    assert emitted <= set(S.CROPHARVEST_CANONICAL_REGIONS)
    assert "mali" in emitted and "mali-non-crop" not in emitted  # merged, not a separate region


def test_malformed_file_is_in_the_frozen_exclusion_set():
    assert "1619_central-asia.h5" in S.CROPHARVEST_EXCLUDED_FILES


def test_region_mapping_has_one_source_of_truth():
    # the benchmark module delegates to split_spec; it holds no merge rules of its own
    assert ch.canonical_region is S.cropharvest_canonical_region
    assert ch._ch_geo_group is ch.canonical_region
    assert ("mali", "mali") in S.CROPHARVEST_REGION_MERGES  # rules live in split_spec


# --- loader integration: the frozen exclusion is data_quality, not a STRICT_MODE failure ---
def _write_ch(base, samples, *, extra_files=()):
    arrays = base / "cropharvest" / "features" / "arrays"
    arrays.mkdir(parents=True, exist_ok=True)
    feats = []
    for idx, dataset, is_crop, lat, lon in samples:
        with h5py.File(arrays / f"{idx}_{dataset}.h5", "w") as f:
            f.create_dataset("array", data=np.ones((12, 18), dtype=np.float32))
        feats.append({"properties": {"index": idx, "dataset": dataset, "is_crop": is_crop,
                                     "lat": lat, "lon": lon, "export_end_date": "2021-02-01"}})
    for name in extra_files:  # present-but-excluded files (content never read)
        with h5py.File(arrays / name, "w") as f:
            f.create_dataset("array", data=np.ones((1, 1), dtype=np.float32))
    (base / "cropharvest" / "labels.geojson").write_text(json.dumps({"features": feats}))


def test_frozen_exclusion_is_data_quality_not_a_strict_mode_failure(tmp_path, monkeypatch):
    _write_ch(tmp_path,
              samples=[(0, "kenya", 1, 0.5, 37.0), (1, "togo", 0, 8.0, 1.0)],
              extra_files=["1619_central-asia.h5"])  # the frozen malformed file, present on disk
    monkeypatch.setenv("STRICT_MODE", "1")  # even strict: a FROZEN exclusion must not raise

    bench = ch.load_benchmark(root=tmp_path, shuffle=False)

    assert len(bench.sample_ids) == 2, "the excluded file leaked into the population"
    assert set(bench.groups.tolist()) == {"kenya", "togo"}
    fx = bench.data_quality.get("frozen_exclusions")
    assert fx and fx[0]["file"] == "1619_central-asia.h5" and "malformed" in fx[0]["reason"]
    assert "skipped_inputs" not in bench.data_quality  # not an unexpected skip
