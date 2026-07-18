"""viz: figures for the experiments run Mon 2026-07-06 -> Mon 2026-07-13.

Six experiment families (all reading cached probe_results.jsonl pulled from the
gilbreth consolidation into viz/data/<box>/<experiment>/results/<model>/<bench>/):
  1. output-erm-full-20260711        main 3-seed ERM baseline (+calibration, budget sweep)
  2. output-probecap-20260711        probe-capacity ablation (kNN/MLP, UNCAPPED)
  3. output-probecap-capped-20260712 capped iteration of the above
  4. output-capsens-c*-20260712      cap-value sensitivity (breizh, olmoearth)
  5. output-s2only-*-20260712        common-S2 modality control
  6. output-fixedbudget-*-2026071{3,4}  fixed-budget probe family (+euro 25/50/100k)

Run:  python3 viz/make_figures.py
"""
import collections
import glob
import json
import os

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.join(os.path.dirname(__file__), "data")
FIG = os.path.join(os.path.dirname(__file__), "figures")
os.makedirs(FIG, exist_ok=True)

PRIMARY = {"cropharvest": "calibrated_f1", "eurocropsml": "macro_f1",
           "breizhcrops": "macro_f1", "pastis": "miou"}
MODELS = ["raw", "presto", "olmoearth", "galileo", "tessera", "agrifm"]
COL = {"raw": "#b5482a", "presto": "#4e79a7", "olmoearth": "#77a06a",
       "galileo": "#5b6b8c", "tessera": "#e08a3c", "agrifm": "#8c8c8c"}
FAMS = ["logistic", "mlp", "knn"]
FAMLBL = {"logistic": "logistic", "mlp": "MLP", "knn": "$k$NN"}


def rows(exp_glob):
    """Yield result rows from experiments whose path matches exp_glob."""
    for f in glob.glob(f"{ROOT}/{exp_glob}/results/*/*/probe_results.jsonl"):
        p = f.split(os.sep)
        model, bench = p[-3], p[-2]
        with open(f) as fh:
            for line in fh:
                try:
                    r = json.loads(line)
                except Exception:
                    continue
                r["_model"], r["_bench"] = model, bench
                yield r


def metric(r):
    """Primary metric for the row's benchmark (handles __s2only suffix)."""
    b = r["_bench"].split("__")[0]
    return r.get(PRIMARY.get(b, "macro_f1"))


def ood_cell(r, regime="geographic_ood", budget=0):
    return (r.get("split_regime") == regime and r.get("evaluation_split") == "full"
            and r.get("label_budget") == budget)


def deploy_cell(r):
    """The 'deployment metric' cell for this row's regime.

    NOTE the asymmetry in the harness's schema (verified from the data):
      random_id  -> the ID anchor: evaluation_split='test', label_budget=1.0
                    (label_budget here is a SOURCE FRACTION, not a target count;
                     random_id has NO 'full'/0 row).
      OOD regimes-> zero-shot on the full held-out region: 'full' / 0.
    (Oracle = <regime>/'held_out'/-1; few-shot = 'held_out'/{5,10,25,50}.)
    """
    if r.get("split_regime") == "random_id":
        return r.get("evaluation_split") == "test" and r.get("label_budget") == 1.0
    return r.get("evaluation_split") == "full" and r.get("label_budget") == 0


def mean(v):
    v = [x for x in v if x is not None]
    return float(np.mean(v)) if v else None


def savefig(fig, name):
    p = os.path.join(FIG, name)
    fig.savefig(p, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {name}")


# ---------------------------------------------------------------- 01 fixed-budget probe family
def fig01():
    panels = [("breizhcrops", "*/output-fixedbudget-breizh100k-*", "BreizhCrops (macro-F1)"),
              ("eurocropsml", "*/output-fixedbudget-euro100k-*", "EuroCropsML (macro-F1)"),
              ("cropharvest", "*/output-fixedbudget-crop100k-*", "CropHarvest (cal-F1)")]
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.4))
    for ax, (_bench, g, title) in zip(axes, panels, strict=True):
        acc = collections.defaultdict(lambda: collections.defaultdict(list))
        for r in rows(g):
            if ood_cell(r) and r.get("probe_family") in FAMS:
                acc[r["_model"]][r["probe_family"]].append(metric(r))
        for m in MODELS:
            if m not in acc:
                continue
            ys = [mean(acc[m].get(f, [])) for f in FAMS]
            if all(y is None for y in ys):
                continue
            ax.plot(range(3), ys, marker="o", ms=4, lw=2.2 if m == "raw" else 1.4,
                    color=COL[m], label=m, zorder=3 if m == "raw" else 2)
        ax.set_xticks(range(3))
        ax.set_xticklabels([FAMLBL[f] for f in FAMS])
        ax.set_title(title, fontsize=10)
        ax.grid(alpha=.25)
    axes[0].set_ylabel("zero-shot OOD")
    axes[-1].legend(fontsize=7, frameon=False, loc="best")
    fig.suptitle("Fixed-budget probe family @100k — geographic OOD (exp 6, Jul 13)", fontsize=11)
    savefig(fig, "01_fixedbudget_probe_family.png")


# ---------------------------------------------------------------- 02 shared sample-budget sensitivity
def fig02():
    budgets = [
        ("25k", [
            "cranberry/output-fixedbudget-euro25k-cranberry-20260713",
            "avocado/output-fixedbudget-euro25k-tesraw-avocado-20260714",
        ]),
        ("50k", ["avocado/output-fixedbudget-euro50k-avocado-20260713"]),
        ("100k", ["gilbreth-native/output-fixedbudget-euro100k-gil-20260713"]),
    ]

    # The 25k experiment finished across two machines. Cranberry contains two
    # partial Tessera rows that were subsequently rerun in the complete avocado
    # shard; deduplicate by evaluation-cell identity so those rows are not
    # counted twice.
    budget_rows = {}
    for budget, patterns in budgets:
        unique = {}
        for pattern in patterns:
            for r in rows(pattern):
                key = (
                    r["_model"], r["_bench"], r.get("probe_family"),
                    r.get("split_regime"), r.get("evaluation_split"),
                    r.get("label_budget"), r.get("holdout"), r.get("seed"),
                )
                unique[key] = r
        budget_rows[budget] = list(unique.values())

    fig, axes = plt.subplots(1, 3, figsize=(12, 3.4), sharey=True)
    for ax, fam in zip(axes, FAMS, strict=True):
        family_values = []
        for m in MODELS:
            ys = []
            for budget, _ in budgets:
                v = [metric(r) for r in budget_rows[budget]
                     if ood_cell(r) and r.get("probe_family") == fam and r["_model"] == m]
                ys.append(mean(v))
            if all(y is None for y in ys):
                continue
            family_values.append(ys)
            ax.plot(range(3), ys, marker="o", ms=4, lw=2.2 if m == "raw" else 1.4,
                    color=COL[m], label={
                        "raw": "Raw", "presto": "Presto", "olmoearth": "OlmoEarth",
                        "galileo": "Galileo", "tessera": "TESSERA", "agrifm": "AgriFM",
                    }[m])
        max_swing = max(max(v) - min(v) for v in family_values)
        ax.set_xticks(range(3))
        ax.set_xticklabels([b for b, _ in budgets])
        ax.set_xlabel("shared sample budget")
        ax.set_title(f"{FAMLBL[fam]}  ·  max swing = {max_swing:.3f}", fontsize=9.5)
        ax.grid(alpha=.25)
    axes[0].set_ylabel("EuroCropsML OOD macro-F1")
    handles, labels = axes[-1].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", bbox_to_anchor=(.5, -.01),
               ncol=len(labels), fontsize=8, frameon=False)
    fig.suptitle("Shared sample-budget sensitivity — EuroCropsML geographic OOD", fontsize=11)
    fig.subplots_adjust(bottom=.23, top=.82)
    savefig(fig, "02_reference_budget_sensitivity.png")


# ---------------------------------------------------------------- 03 regime comparison (split effect)
def fig03():
    regimes = ["random_id", "spatial_cluster_ood", "geographic_ood", "official"]
    benches = ["cropharvest", "eurocropsml", "breizhcrops", "pastis"]
    acc = collections.defaultdict(lambda: collections.defaultdict(list))
    for r in rows("*/output-erm-full-20260711*"):
        if deploy_cell(r) and r.get("probe_family", "logistic") == "logistic":
            acc[r["_bench"]][r.get("split_regime")].append(metric(r))
    fig, ax = plt.subplots(figsize=(9, 3.6))
    w, x = 0.2, np.arange(len(benches))
    for i, reg in enumerate(regimes):
        ys = [mean(acc[b].get(reg, [])) or np.nan for b in benches]
        ax.bar(x + (i - 1.5) * w, ys, w, label=reg)
    ax.set_xticks(x)
    ax.set_xticklabels(benches)
    ax.set_ylabel("deployment metric")
    ax.grid(alpha=.25, axis="y")
    ax.legend(fontsize=8, frameon=False)
    ax.set_title("Split-regime comparison — the same task moves by >30 points (exp 1, Jul 11)", fontsize=11)
    savefig(fig, "03_regime_comparison.png")


# ---------------------------------------------------------------- 04 worst region
def fig04():
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.4))
    for ax, bench in zip(axes, ["eurocropsml", "breizhcrops"], strict=True):
        per = collections.defaultdict(list)
        for r in rows("*/output-erm-full-20260711*"):
            if r["_bench"] == bench and ood_cell(r) and r.get("probe_family", "logistic") == "logistic":
                per[r.get("holdout")].append(metric(r))
        items = sorted(((h, mean(v)) for h, v in per.items() if mean(v) is not None),
                       key=lambda t: t[1])
        if not items:
            continue
        ax.bar(range(len(items)), [v for _, v in items], color="#4e79a7")
        ax.axhline(np.mean([v for _, v in items]), ls="--", color="#b5482a",
                   label=f"mean {np.mean([v for _,v in items]):.3f}")
        ax.set_xticks(range(len(items)))
        ax.set_xticklabels([str(h) for h, _ in items], rotation=35, ha="right", fontsize=7)
        ax.set_title(f"{bench} — per held-out region", fontsize=10)
        ax.legend(fontsize=8, frameon=False)
        ax.grid(alpha=.25, axis="y")
    axes[0].set_ylabel("zero-shot OOD")
    fig.suptitle("Worst-region collapse — means hide near-zero regions (exp 1)", fontsize=11)
    savefig(fig, "04_worst_region.png")


# ---------------------------------------------------------------- 05 calibration ID vs OOD
def fig05():
    mets = ["ece", "nll", "brier"]
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.4))
    for ax, mt in zip(axes, mets, strict=True):
        idv, oodv, labels = [], [], []
        for m in MODELS:
            i = [r.get(mt) for r in rows("*/output-erm-full-20260711*")
                 if r["_bench"] == "cropharvest" and r.get("split_regime") == "random_id"
                 and deploy_cell(r) and r["_model"] == m]
            o = [r.get(mt) for r in rows("*/output-erm-full-20260711*")
                 if r["_bench"] == "cropharvest" and ood_cell(r) and r["_model"] == m]
            if mean(i) is None or mean(o) is None:
                continue
            labels.append(m)
            idv.append(mean(i))
            oodv.append(mean(o))
        x = np.arange(len(labels))
        ax.bar(x - .2, idv, .4, label="ID", color="#77a06a")
        ax.bar(x + .2, oodv, .4, label="OOD", color="#b5482a")
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=20, fontsize=8)
        ax.set_title(mt.upper(), fontsize=10)
        ax.grid(alpha=.25, axis="y")
    axes[0].legend(fontsize=8, frameon=False)
    fig.suptitle("Calibration degrades OOD — CropHarvest (exp 1)", fontsize=11)
    savefig(fig, "05_calibration_id_vs_ood.png")


# ---------------------------------------------------------------- 06 common-S2 control
def fig06():
    pairs = []
    for m in MODELS:
        nat = [metric(r) for r in rows("*/output-erm-full-20260711*")
               if r["_bench"] == "cropharvest" and ood_cell(r) and r["_model"] == m]
        s2 = [metric(r) for r in rows("*/output-s2only-cropharvest-*")
              if r["_bench"].startswith("cropharvest") and ood_cell(r) and r["_model"] == m]
        if mean(nat) is None or mean(s2) is None:
            continue
        pairs.append((m, mean(nat), mean(s2)))
    if not pairs:
        print("  (06 skipped: no overlap)")
        return
    fig, ax = plt.subplots(figsize=(7, 3.4))
    x = np.arange(len(pairs))
    ax.bar(x - .2, [p[1] for p in pairs], .4, label="native", color="#4e79a7")
    ax.bar(x + .2, [p[2] for p in pairs], .4, label="common-S2", color="#e08a3c")
    for i, p in enumerate(pairs):
        ax.text(i, max(p[1], p[2]) + .006, f"{p[2]-p[1]:+.3f}", ha="center", fontsize=7)
    ax.set_xticks(x)
    ax.set_xticklabels([p[0] for p in pairs])
    ax.set_ylabel("CropHarvest OOD cal-F1")
    ax.legend(fontsize=8, frameon=False)
    ax.grid(alpha=.25, axis="y")
    ax.set_title("Common-S2 modality control — equalizing input (exp 5, Jul 12)", fontsize=11)
    savefig(fig, "06_s2only_control.png")


# ---------------------------------------------------------------- 07 sample-budget scaling
def fig07():
    caps = [("50k", "*/output-capsens-c50000-*"), ("100k", "*/output-capsens-c100000-*"),
            ("200k", "*/output-capsens-c200000-*")]
    fig, ax = plt.subplots(figsize=(6.5, 3.4))
    any_line = False
    for fam in ("logistic", "mlp"):
        ys = []
        for _, g in caps:
            v = [metric(r) for r in rows(g) if ood_cell(r) and r.get("probe_family") == fam]
            ys.append(mean(v))
        if all(y is None for y in ys):
            continue
        delta = ys[-1] - ys[0]
        ax.plot(range(3), ys, marker="o", ms=6, lw=2,
                label=f"{FAMLBL[fam]}  ($\\Delta_{{50k\\to200k}}={delta:+.3f}$)")
        ax.annotate(f"{ys[-1]:.3f}", (2, ys[-1]), xytext=(7, 0),
                    textcoords="offset points", va="center", fontsize=8.5)
        any_line = True
    if not any_line:
        print("  (07 skipped: no capsens OOD rows)")
        plt.close(fig)
        return
    ax.set_xticks(range(3))
    ax.set_xticklabels([c for c, _ in caps])
    ax.set_xlabel("shared sample budget")
    ax.set_ylabel("BreizhCrops OOD macro-F1")
    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, fontsize=8.5, frameon=False, loc="lower center",
               bbox_to_anchor=(.5, -.01), ncol=2)
    ax.grid(alpha=.25)
    ax.set_title("Sample-budget scaling — BreizhCrops / OlmoEarth\n"
                 "logistic and MLP, geographic OOD", fontsize=11)
    fig.subplots_adjust(bottom=.23)
    savefig(fig, "07_capsens_breizh.png")


# ---------------------------------------------------------------- 08 probecap across regimes
def fig08():
    regimes = ["random_id", "spatial_cluster_ood", "geographic_ood", "official"]
    acc = collections.defaultdict(lambda: collections.defaultdict(list))
    for r in rows("*/output-probecap*"):
        if r["_bench"] == "cropharvest" and deploy_cell(r) and r.get("probe_family") in FAMS:
            acc[r["probe_family"]][r.get("split_regime")].append(metric(r))
    fig, ax = plt.subplots(figsize=(7.5, 3.4))
    w, x = 0.25, np.arange(len(regimes))
    plotted = False
    for i, fam in enumerate(FAMS):
        ys = [mean(acc[fam].get(rg, [])) or np.nan for rg in regimes]
        if all(np.isnan(y) for y in ys):
            continue
        ax.bar(x + (i - 1) * w, ys, w, label=FAMLBL[fam])
        plotted = True
    if not plotted:
        print("  (08 skipped)")
        plt.close(fig)
        return
    ax.set_xticks(x)
    ax.set_xticklabels(regimes, rotation=15, fontsize=8)
    ax.set_ylabel("CropHarvest cal-F1")
    ax.legend(fontsize=8, frameon=False)
    ax.grid(alpha=.25, axis="y")
    ax.set_title("Probe-capacity across ALL regimes — only copy of these cells (exp 2/3)", fontsize=11)
    savefig(fig, "08_probecap_multiregime.png")


if __name__ == "__main__":
    print("building viz figures...")
    for fn in (fig01, fig02, fig03, fig04, fig05, fig06, fig07, fig08):
        try:
            fn()
        except Exception as e:
            print(f"  !! {fn.__name__} failed: {type(e).__name__}: {e}")
    print("done ->", FIG)
