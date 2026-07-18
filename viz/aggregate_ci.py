"""Step-1 uncertainty aggregation: surface the region-bootstrap CIs the harness
already writes (deltas.csv: delta_ci_lo/hi, ood_std) into a paper-ready table.
Source = collated baseline (ERM) bundles. No rerun. CIs are across held-out
regions (n_holdouts), not seeds -- seeds come in Step 2."""
import csv
import glob
import os
import statistics as st

PRIMARY = {"cropharvest": "calibrated_f1", "eurocropsml": "macro_f1",
           "breizhcrops": "macro_f1", "pastis": "miou"}
ROOT = "data/output/collated/baseline"

def load_deltas(bench, model):
    f = f"{ROOT}/{bench}/{model}/deltas.csv"
    if not os.path.exists(f):
        return []
    return [r for r in csv.DictReader(open(f))
            if r.get("method") == "erm" and r.get("metric") == PRIMARY[bench]]

def fnum(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return float("nan")

rows_out = []
for bench in ["cropharvest", "eurocropsml", "breizhcrops", "pastis"]:
    for mdir in sorted(glob.glob(f"{ROOT}/{bench}/*")):
        model = os.path.basename(mdir)
        d = load_deltas(bench, model)
        by_regime = {r["ood_regime"]: r for r in d}
        geo = by_regime.get("geographic_ood")
        if not geo:
            continue
        rows_out.append({
            "benchmark": bench, "model": model,
            "id": fnum(geo["id"]),
            "ood_geo": fnum(geo["ood"]), "ood_geo_std": fnum(geo["ood_std"]),
            "delta_geo": fnum(geo["delta"]),
            "delta_ci_lo": fnum(geo["delta_ci_lo"]), "delta_ci_hi": fnum(geo["delta_ci_hi"]),
            "off_ood": fnum(by_regime["official"]["ood"]) if "official" in by_regime else float("nan"),
            "off_std": fnum(by_regime["official"]["ood_std"]) if "official" in by_regime else float("nan"),
            "off_delta": fnum(by_regime["official"]["delta"]) if "official" in by_regime else float("nan"),
            "off_ci_lo": fnum(by_regime["official"]["delta_ci_lo"]) if "official" in by_regime else float("nan"),
            "off_ci_hi": fnum(by_regime["official"]["delta_ci_hi"]) if "official" in by_regime else float("nan"),
            "spa_ood": fnum(by_regime["spatial_cluster_ood"]["ood"]) if "spatial_cluster_ood" in by_regime else float("nan"),
            "spa_std": fnum(by_regime["spatial_cluster_ood"]["ood_std"]) if "spatial_cluster_ood" in by_regime else float("nan"),
            "worst": fnum(geo["ood_worst_region"]),
        })

# --- write source table ---
out = "viz/tables/10_decomposition_ci.md"
with open(out, "w") as f:
    f.write("# Region-bootstrap CIs on baseline (ERM) claims\n\n")
    f.write("Source: collated `baseline/*/*/deltas.csv` (harness-computed). CI = 95% "
            "bootstrap over held-out regions; `ood_std` = std across held-out regions. "
            "Single-seed; seed replication is Step 2.\n\n")
    f.write("| benchmark | model | ID | OOD(geo)±std | Δ(geo) [95% CI] | Δ(official) [95% CI] | worst region |\n")
    f.write("|:--|:--|--:|--:|:--|:--|--:|\n")
    for r in rows_out:
        f.write(f"| {r['benchmark']} | {r['model']} | {r['id']:.3f} | "
                f"{r['ood_geo']:.3f}±{r['ood_geo_std']:.3f} | "
                f"{r['delta_geo']:.3f} [{r['delta_ci_lo']:.3f}, {r['delta_ci_hi']:.3f}] | "
                f"{r['off_delta']:.3f} [{r['off_ci_lo']:.3f}, {r['off_ci_hi']:.3f}] | "
                f"{r['worst']:.3f} |\n")

# --- split-flip disjointness check (CropHarvest: official vs geographic drop CIs) ---
print("=== SPLIT-FLIP: official-drop CI vs geographic-drop CI (disjoint => real) ===")
for r in rows_out:
    if r["benchmark"] != "cropharvest":
        continue
    disjoint = r["off_ci_lo"] > r["delta_ci_hi"] or r["delta_ci_lo"] > r["off_ci_hi"]
    print(f"  {r['model']:<10} geo Δ [{r['delta_ci_lo']:.3f},{r['delta_ci_hi']:.3f}]  "
          f"official Δ [{r['off_ci_lo']:.3f},{r['off_ci_hi']:.3f}]  "
          f"=> {'DISJOINT' if disjoint else 'overlap'}")

# --- regime bars + error bars for the split figure (CropHarvest) ---
print("\n=== SPLIT FIGURE error bars (CropHarvest: regime score +/- ood_std) ===")
for r in rows_out:
    if r["benchmark"] != "cropharvest":
        continue
    print(f"  {r['model']:<10} random_id={r['id']:.3f}(-)  official={r['off_ood']:.3f}+/-{r['off_std']:.3f}  "
          f"geo={r['ood_geo']:.3f}+/-{r['ood_geo_std']:.3f}  spatial={r['spa_ood']:.3f}+/-{r['spa_std']:.3f}")

# --- worst-region figure error bars (per benchmark: mean-OOD +/- ood_std, mean over models) ---
print("\n=== WORST-REGION figure: per-benchmark mean-OOD std (mean of per-model ood_std) ===")
for bench in ["cropharvest", "eurocropsml", "breizhcrops", "pastis"]:
    br = [r for r in rows_out if r["benchmark"] == bench]
    if not br:
        continue
    mean_ood = st.mean(r["ood_geo"] for r in br)
    mean_std = st.mean(r["ood_geo_std"] for r in br)
    mean_worst = st.mean(r["worst"] for r in br)
    print(f"  {bench:<12} mean_ood={mean_ood:.3f} avg_region_std={mean_std:.3f} worst={mean_worst:.3f}")
print(f"\nwrote {out}")
