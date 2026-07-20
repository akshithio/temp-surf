"""Realized-geography audit for the frozen CropHarvest ``geographic_ood`` leaves.

CropHarvest domains are PROVENANCE labels, not polygons. Two collections -- ``croplands`` and
``geowiki-landcover-2017`` -- are globally distributed, so on every fold they contribute source points
that can sit inside the held-out target's territory. The only spatial filter the split applies is a
nearest-target-sample distance purge, which bounds but does not establish territorial exclusion: a
source point 51 km from every LABELLED target sample is retained no matter where it actually lies.

This driver reads the already-frozen ``assignments.csv`` leaves together with the authoritative
coordinates in ``labels.geojson`` and reports, per (seed, target), what the realized source pool
actually contains:

  * minimum retained-source-to-target great-circle distance (the purge's realized floor);
  * purged counts broken down by original provenance collection;
  * retained source points falling INSIDE the target's realized footprint, where the footprint is the
    convex hull of the target's own labelled coordinates -- a strict LOWER BOUND on territory, so any
    hit is definitive containment rather than a near miss;
  * the same count against the hull buffered by the purge radius;
  * class balance (is_crop) of target pool/test and of the in-footprint source points;
  * declared target role (headline rotation vs supplementary stress).

Read-only with respect to ``data/splits/``: nothing here writes, moves, or regenerates a split.

No command-line arguments: edit the CONFIG block below and run it.

    python tools/audit_cropharvest_geography.py
"""

from __future__ import annotations

import csv
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

_SRC = Path(__file__).resolve().parents[1] / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from evals import split_spec  # noqa: E402
from evals.benchmarks import cropharvest as CH  # noqa: E402
from evals.regimes import geographic_ood as GEO  # noqa: E402
from utils import cacheutils  # noqa: E402

# === Configuration ===========================================================
SPLITS_ROOT = cacheutils.SCRATCH / "splits"                       # data/splits/
LABELS_GEOJSON = cacheutils.SCRATCH / "input" / "benchmarks" / "cropharvest" / "labels.geojson"
REPORT_PATH = cacheutils.SCRATCH / "logs" / "cropharvest_geography_audit.json"
SEEDS = [0, 1, 2]
WRITE_REPORT = True     # False = print the table only, write no JSON
# =============================================================================

EARTH_RADIUS_KM = 6371.0088
SOURCE_PARTITIONS = ("source_train", "source_val", "source_test")
TARGET_PARTITIONS = ("target_label_pool", "target_test")
STATUS_ASSIGNED, STATUS_PURGED = "assigned", "purged"


def _load_coords() -> dict[str, tuple[float, float, int]]:
    """``{stable_id: (lat, lon, is_crop)}`` straight from the authoritative labels file."""
    geo = json.loads(Path(LABELS_GEOJSON).read_text())
    out: dict[str, tuple[float, float, int]] = {}
    for feature in geo["features"]:
        p = feature["properties"]
        lat = float(p["lat"]) if p.get("lat") is not None else float("nan")
        lon = float(p["lon"]) if p.get("lon") is not None else float("nan")
        out[f"{int(p['index'])}_{p['dataset']}.h5"] = (lat, lon, int(p["is_crop"]))
    return out


def _read_leaf(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def _haversine_min(src: np.ndarray, tgt: np.ndarray) -> float:
    """Minimum great-circle distance (km) from any source point to any target point."""
    if len(src) == 0 or len(tgt) == 0:
        return float("nan")
    from sklearn.neighbors import BallTree

    tree = BallTree(np.deg2rad(tgt), metric="haversine")
    return float(tree.query(np.deg2rad(src), k=1, return_distance=True)[0].min() * EARTH_RADIUS_KM)


def _inside_footprint(points: np.ndarray, target_points: np.ndarray, buffer_km: float) -> np.ndarray:
    """Boolean mask of ``points`` inside the target's footprint, buffered by ``buffer_km``.

    Delegates to the PRODUCTION implementation (``geographic_ood.target_footprint``) so the audit can
    never report a different territory than the split generator excludes. Passing ``buffer_km=0``
    measures the bare convex hull -- a strict lower bound on territory, so a hit there is definitive
    containment rather than a near miss.
    """
    import shapely

    if len(points) == 0 or len(target_points) == 0:
        return np.zeros(len(points), dtype=bool)
    # A zero buffer is not a valid production footprint; use a millimetre so the hull itself is tested.
    buffer_m = max(float(buffer_km) * 1000.0, 1e-3)
    footprint, _hull, transformer, _crs = GEO.target_footprint(
        target_points, buffer_m, where="audit_cropharvest_geography"
    )
    px, py = transformer.transform(points[:, 1].tolist(), points[:, 0].tolist())
    return np.asarray(shapely.intersects_xy(footprint, np.asarray(px), np.asarray(py)), dtype=bool)


def audit_leaf(rows: list[dict[str, str]], coords: dict[str, tuple[float, float, int]],
               purge_km: float) -> dict:
    """One (seed, target) leaf: realized purge floor, provenance of purges, territorial containment."""
    src_ids = [r["stable_id"] for r in rows
               if r["partition"] in SOURCE_PARTITIONS and r["status"] == STATUS_ASSIGNED]
    tgt_ids = [r["stable_id"] for r in rows
               if r["partition"] in TARGET_PARTITIONS and r["status"] == STATUS_ASSIGNED]
    purged_ids = [r["stable_id"] for r in rows if r["status"] == STATUS_PURGED]

    def latlon(ids):
        pts = np.asarray([coords[i][:2] for i in ids if i in coords], dtype=np.float64)
        return pts[np.isfinite(pts).all(axis=1)] if len(pts) else np.empty((0, 2))

    src_pts, tgt_pts = latlon(src_ids), latlon(tgt_ids)
    src_keep = [i for i in src_ids if i in coords and np.isfinite(coords[i][:2]).all()]

    inside = _inside_footprint(src_pts, tgt_pts, 0.0)              # bare hull: definitive containment
    inside_buf = _inside_footprint(src_pts, tgt_pts, purge_km)     # what the split actually excludes

    prov = [CH.provenance_dataset(i) for i in src_keep]
    inside_prov = Counter(p for p, hit in zip(prov, inside, strict=True) if hit)
    global_inside = sum(v for k, v in inside_prov.items()
                        if split_spec.cropharvest_canonical_region(k) in split_spec.CROPHARVEST_GLOBAL_COLLECTIONS)

    return {
        "n_source": len(src_ids),
        "n_target_pool": sum(1 for r in rows if r["partition"] == "target_label_pool"),
        "n_target_test": sum(1 for r in rows if r["partition"] == "target_test"),
        "n_purged": len(purged_ids),
        "min_source_target_km": _haversine_min(src_pts, tgt_pts),
        "purged_by_provenance": dict(sorted(Counter(CH.provenance_dataset(i) for i in purged_ids).items())),
        "source_inside_footprint": int(inside.sum()),
        "source_inside_footprint_by_provenance": dict(sorted(inside_prov.items())),
        "source_inside_global_collections": int(global_inside),
        "source_inside_buffered_footprint": int(inside_buf.sum()),
        "target_class_counts": dict(sorted(Counter(
            coords[i][2] for i in tgt_ids if i in coords).items())),
        "inside_class_counts": dict(sorted(Counter(
            coords[i][2] for i, hit in zip(src_keep, inside, strict=True) if hit).items())),
    }


def main() -> int:
    root = Path(SPLITS_ROOT) / "cropharvest" / "geographic_ood"
    if not root.is_dir():
        print(f"no frozen geographic_ood leaves under {root}", flush=True)
        return 1
    coords = _load_coords()
    spec = split_spec.CROPHARVEST
    purge_km = float(spec.purge_km)
    print(f"loaded {len(coords)} coordinates; purge radius {purge_km} km\n", flush=True)

    report: dict[str, dict] = defaultdict(dict)
    header = (f"{'seed':>4} {'target':<20} {'role':<13} {'source':>8} {'purged':>7} "
              f"{'minkm':>8} {'INSIDE':>7} {'global':>7} {'buf':>7}")
    print(header)
    print("-" * len(header))
    for seed in SEEDS:
        seed_dir = root / str(seed)
        if not seed_dir.is_dir():
            continue
        for target_dir in sorted(p for p in seed_dir.iterdir() if p.is_dir()):
            leaf = target_dir / "assignments.csv"
            if not leaf.is_file():
                continue
            target = target_dir.name
            role = "headline" if target in spec.geographic_targets else "supplementary"
            res = audit_leaf(_read_leaf(leaf), coords, purge_km)
            res["target_role"] = role
            report[str(seed)][target] = res
            print(f"{seed:>4} {target:<20} {role:<13} {res['n_source']:>8} {res['n_purged']:>7} "
                  f"{res['min_source_target_km']:>8.1f} {res['source_inside_footprint']:>7} "
                  f"{res['source_inside_global_collections']:>7} "
                  f"{res['source_inside_buffered_footprint']:>7}", flush=True)

    total_inside = sum(v["source_inside_footprint"] for s in report.values() for v in s.values())
    total_global = sum(v["source_inside_global_collections"] for s in report.values() for v in s.values())
    print(f"\ntotal retained source points inside a target footprint: {total_inside} "
          f"(from globally distributed collections: {total_global})", flush=True)

    if WRITE_REPORT:
        Path(REPORT_PATH).parent.mkdir(parents=True, exist_ok=True)
        Path(REPORT_PATH).write_text(json.dumps(dict(report), indent=2, sort_keys=True) + "\n")
        print(f"wrote {REPORT_PATH}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
