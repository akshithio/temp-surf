# Post-Hoc Robustness for Agricultural Earth Observation Foundation Models

**Student:** Akshith Garapati · **Faculty Advisor:** Prof. Dharmendra Saraswat ·
**Lab Mentor:** Rajiv Ranjan
**Target submission:** NeurIPS 2026 CCAI Workshop (Aug 22, 2026) / ICLR 2027 (Oct 2026)

Given an existing Earth-observation foundation model, how do you make it reliably
deployable in a *new* agricultural region — with few or zero target-region labels —
**without retraining any encoder weights**?

This repository treats EO encoders as **frozen** and asks how much deployment
performance can be recovered with post-hoc, probe-level adaptation. The encoder is
run once per condition and its embeddings are cached; every adaptation method and
probe operates on those cached matrices, so no encoder weights are ever touched.

> **Central contribution.** Establish a deployment-realistic, **classification-first**
> evaluation framework for **geographic robustness** in frozen agricultural EO
> encoders, and demonstrate whether post-hoc adaptation recovers a measurable
> fraction of the transfer gap between in-distribution and geographically
> out-of-distribution conditions — without modifying encoder weights.

---

## Scope & priorities (read this first)

The project is deliberately constrained. Of everything the pipeline *can* run, the
**first-priority core is narrow**, and the rest is explicitly secondary — to be
attempted only after the core is stable across the three core models.

- **Robustness axes: 1 of 3 is core.** **Geographic** transfer (strict holdout) is
  the promised contribution. **Sensor** dropout and **temporal** sparsity are
  secondary extensions, added only after the geographic protocol is stable.
- **Tasks: 2 of 4 are core.** **Classification** — `bin-crop-class` (CropHarvest)
  and `crop-class` (EuroCropsML) — is the promised task family. The regression
  tasks `pheno-reg` (SICKLE) and `yield-reg` (YieldSAT) are optional extensions.
- **Encoders: 3 are core.** **Presto, AgriFM, TESSERA.** OlmoEarth is retained only
  as an optional general-purpose comparator once the three core models work.

**We expand beyond this core only after establishing success on the core — i.e. a
rigorous geographic-holdout *classification* evaluation across Presto, AgriFM, and
TESSERA, with calibrated metrics, label-budget curves, and at least one post-hoc
adaptation method beating a no-adaptation baseline.**

| Scope | Item | Status | Why |
|---|---|---|---|
| **Core robustness axis** | Geographic transfer (strict holdout) | **Required** | Most directly aligned with deployment; supported by current preliminary results. |
| **Core task family** | Classification: `bin-crop-class` (CropHarvest), `crop-class` (EuroCropsML) | **Required** | Aligns with the current pipeline, metrics, and post-hoc methods. |
| **Core encoders** | Presto, AgriFM, TESSERA | **Required** | Agriculturally specialized with public code/weights. |
| Secondary axis | Sensor / modality dropout | Optional (if time permits) | Deployment-relevant, but must not block the core paper. |
| Secondary axis | Temporal-support sparsity | Optional (if time permits) | Important but more complex to validate cleanly. |
| Optional task | Regression: `pheno-reg` (SICKLE), `yield-reg` (YieldSAT) | Optional (if time permits) | Different metrics/adaptation logic; must not compete with the classification-first claim. |
| Optional encoder | OlmoEarth | Optional comparator | General-purpose multimodal EO baseline; less directly agricultural. |
| Out of scope (v1) | Phenological shift, cross-year shift, map-level coherence, adversarial robustness, physical consistency | Future work | Important but too broad for the current timeline. |

---

## Approach

Every experiment is the same pipeline:

```text
task spec ─▶ get_input ─▶ corrupt ─▶ encode (frozen) ─▶ cache
                                                          │
                                       ┌──────────────────┘
                                       ▼
              strict geographic holdout split        ◀── core axis
                                       │
              {ERM, GRIT, DFR, TENT, …} feature transform
                                       │
              calibrated probe ─▶ label-budget sweep ─▶ tables
```

### Evaluation protocol

The protocol is the primary contribution; it defines the deployment-realistic
conditions under which every encoder and method is measured. **The minimum promised
protocol is geographic holdout under the `clean` input condition** — sensor and
temporal stress are optional add-ons.

- **Strict geographic holdout (core).** One source region is held out entirely from
  probe training and used as the test set (`evals.STRICT_HOLDOUTS`: togo, ethiopia,
  lem-brazil, rwanda, togo-eval for CropHarvest; Estonia for the EuroCropsML
  transnational split). This enforces target-region exclusion from probe
  training/fine-tuning. Strict *SSL* exclusion (the encoder also never pretrained on
  the region) is an upstream property of the public weights and is treated as an
  **auditable variable, not assumed** — the preliminary grouped ≈ strict finding is a
  diagnostic about *downstream* exposure, not proof of SSL exclusion.
- **Label-budget curves (`evals.BUDGETS`).** Probe trained at increasing target-label
  budgets `n ∈ {0, 5, 10, 25, 50}` (and source-fraction budgets as a secondary
  diagnostic). This is where methods diverge: zero-label / pair-based methods work at
  `n = 0`, last-layer methods need a moderate budget.
- **Calibrated metrics.** `calibrated_f1` (F1 at a source-validation-optimized
  threshold) is the primary metric; default-0.5 F1 misrepresents transfer under
  shift. Also reported: AUROC, ECE, balanced accuracy (binary); macro/weighted F1,
  balanced accuracy, macro AUROC (multiclass). Regression metrics (RMSE/MAE/R²/
  correlation) apply only to the optional regression tasks.
- **Stress conditions (secondary).** Sensor/temporal corruptions are applied at the
  input by `dataio.corrupt` and run **only after the clean geographic baseline is
  stable**. The geographic axis is the holdout split itself (always on), so
  `ACTIVE_AXES = []` runs the core scope (clean only); adding `"sensorial"` /
  `"temporal"` turns on the optional stress conditions.

| Condition | Scope | Meaning |
|---|---|---|
| `clean` | **Core** | No corruption — the required geographic-holdout baseline. |
| `sensor_off_s2` | Optional | Sentinel-2 bands zeroed (optical blackout: clouds/monsoon). |
| `sensor_off_s1` | Optional | Sentinel-1 bands zeroed (SAR gap / preprocessing failure). |
| `sensor_off_climate` | Optional | Climate bands zeroed. |
| `temporal_drop_{30,50,70}` | Optional | Retain 70/50/30% of timesteps (reduced observation support). |
| `s2_off_tdrop50`, `s1_off_tdrop50` | Optional | Compound stress (sensor-off + temporal drop). |

### Frozen encoders (`src/models/`)

| Priority | Encoder | Params | Input | Source |
|---|---|---:|---|---|
| **Core** | **Presto** | ~0.8M | pixel time series, S1+S2+ERA5 → `(N, 128)` | open weights |
| **Core** | **AgriFM** | ~88M (measured) | multi-source temporal patches (S2 branch) → `(N, 1024)` | open code + CC0 weights |
| **Core** | **TESSERA** | ~58M (nominal) | pixel time series, S1+S2 → `(N, 128)` | open CC0 model |
| Optional | OlmoEarth Base | ~90M | spatial patches, multi-modal (run per-pixel H=W=1) | open weights |

- **AgriFM** runs the published S2 Video-Swin encoder. The official stack wants
  compiled MMCV (no Python-3.13 wheel), but the wrapper imports the repo's real
  `SwinTransformer3D` with mmseg's registries stubbed out — no mmcv needed (see
  Usage). Condition-sensitive and stress-testable.
- **TESSERA** runs the published S1+S2 transformer on each sample's time series, so
  it **is condition-sensitive and stress-testable** (not the old precomputed-lookup
  product). Needs the checkpoint (`TESSERA_WEIGHTS`). Caveat: S1 is fed in dB while
  TESSERA trained on its own S1 scale — read S1-driven numbers accordingly
  (`src/models/tessera.py`).

Parameter counts are nominal integration signals; verify from the loaded checkpoint
before reporting as exact.

### Adaptation methods (`src/methods/`)

Fitted **feature transforms** on the frozen embedding space — drop-in for
`evals.run_probes(transform=...)`; `None` is the ERM baseline. Each method is a flat
file exposing `variants(task_kind)`.

| Method | File | Status |
|---|---|---|
| **GRIT** | `src/methods/grit.py` | Implemented |
| **DFR** | `src/methods/dfr.py` | Implemented |
| **TENT** | `src/methods/tent.py` | Implemented |

The final method set (candidates: DFR, T3A, prompt tuning, conformal prediction) is
chosen **after** the geographic baseline failures are characterized, targeting the
observed failure modes rather than added decoratively.

- **GRIT** projects embeddings orthogonal to a spurious subspace estimated from
  counterfactual pair differences — *geographic* (same-label, different-region pairs)
  or *missingness* (clean-vs-corrupted views). `rank` is the key knob; sweep it.

### Tasks (`src/evals/tasks/`)

| Priority | Task | Benchmark | Kind | Label source |
|---|---|---|---|---|
| **Core** | `bin-crop-class` | CropHarvest | binary | real `is_crop` (**primary development task**) |
| **Core** | `crop-class` | EuroCropsML | multiclass | real crop-type (transnational transfer; preferred replication) |
| Optional | `pheno-reg` | SICKLE | regression | observed harvest day-of-season (external field annotation) |
| Optional | `yield-reg` | YieldSAT | regression | field-mean combine-harvester crop yield |

The two classification tasks are the promised core. The two regression tasks are
optional extensions that test whether the framework generalizes beyond
classification — added only once the classification protocol, baselines, and at least
one adaptation method are already stable.

## Repository layout

```text
.
├── src/
│   ├── main.py              # orchestrator: task → encode/cache → {methods} → tables
│   ├── dataio/get_input.py  # Benchmark loader (CropHarvest, EuroCropsML, SICKLE, YieldSAT) + corrupt()
│   ├── models/              # frozen encoder wrappers: presto, olmoearth, tessera, agrifm
│   ├── evals/
│   │   ├── evals.py         # splits, calibrated/multiclass/regression probes, budget sweep
│   │   └── tasks/           # per-task label + metric specs
│   ├── methods/             # method files: grit.py, dfr.py, tent.py, …
│   └── utils/
│       ├── cacheutils.py    # content-keyed bench/embedding cache; scratch + EMB_DTYPE
│       ├── gputils.py       # split (encoder, task) work across GPUs (fan-out launcher)
│       ├── perf.py          # MACs / timing instrumentation
│       └── ioutils.py       # CSV/JSONL writing + result aggregation
├── data/                    # (git-ignored; cache/output may be wiped, input is not)
│   ├── input/               # staged raw datasets (see below)
│   ├── cache/               # bench pickles + encoder caches (re-derived on miss)
│   └── output/
│       ├── embeddings/<bench>/<encoder>/n<N>/<condition>.npy
│       └── results/<encoder>/<task>/{probe_results.jsonl,*.csv,summary.csv}
└── notebooks/               # eda/ (per-dataset) and models/ (per-encoder)
```

### Expected data layout

```text
data/input/cropharvest/
    labels.geojson
    features/arrays/<index>_<dataset>.h5      # one (T, C) array per sample
data/input/eurocropsml/
    preprocess/*.npz                          # one (T, 13) array per parcel
    split/latvia_portugal_vs_estonia/...      # official transnational split
data/input/sickle/
    sickle_dataset_tabular.csv                # phenology / yield annotations
    images/{S2,S1}/npy/<uid>/*.npz            # per-acquisition band chips
    masks/10m/<uid>.tif                       # plot / phenology / yield rasters
data/input/yieldsat/
    preprocessed-24-ts/<Country>/merge_s2-soil-dem-weather-coords.nc
```

### SICKLE data provenance (optional `pheno-reg` task)

The original SICKLE release (`sickle_dataset.zip`, ~6.5 GB, access-gated by a
Google Form) is **not** staged whole. Only the subset the frozen-encoder pipeline
consumes is extracted into `data/input/sickle/` (~2.6 GB). What is filtered out:

**At extraction (never staged):**

- `images/L8/**` — all **Landsat-8** imagery; the encoders here ingest Sentinel-1/2 only.
- `images/{S2,S1}/tif/**` — the GeoTIFF mirror of the `npy` chips (redundant).
- `masks/3m/**` and `masks/30m/**` — only the **10 m** masks are kept (Sentinel/Presto scale).
- `weights.zip` — SICKLE's own pretrained weights (we use frozen *external* encoders).
- `sickle_toy_dataset.zip` — the small toy release.

**At load time (`load_sickle`):**

- Only `PADDY_BIN == 1` rows with a valid (`> 0`) target day are used → **907** paddy
  samples for the harvesting target (sowing similar; transplanting smaller).
- **S2:** only `B2,B3,B4,B5,B6,B7,B8,B8A,B11,B12` are read (NDVI computed); `B1`, `B9`,
  and the `AOT/SCL/MSK_CLDPRB/WVP` QA layers are dropped.
- **S1:** only `VV,VH` are read (the `angle` band is dropped).
- Each plot's series is the **mean over its plot-mask pixels** (`== PLOT_ID`, resized
  per band), nearest-resampled onto a fixed grid (12 steps for Presto).
- The mask GeoTIFFs are **not georeferenced**, so `lat/lon` is a fixed Cauvery-Delta
  centroid — location-keyed encoders are degenerate on SICKLE. No climate modality.

### YieldSAT data provenance (optional `yield-reg` task)

Access-gated; two release formats exist (a flexible per-field `Raw/` release and an
ML-ready NetCDF `Preprocessed/` release). Only the **ML-ready NetCDF release** is
staged:

```text
data/input/yieldsat/preprocessed-24-ts/{Argentina,Brazil,Germany,Uruguay}/merge_s2-soil-dem-weather-coords.nc
```

The raw flexible release is intentionally not copied in (redundant for this
frozen-embedding benchmark; the loader only needs the ML-ready
`sample(index,time_step,band)`, `target(index)`, `times`, `field_shared_name`,
`crop`, `country`, `year` variables).

At load time (`load_yieldsat`): rows are grouped by `field_shared_name`; `target` is
averaged over field pixels to a field-level yield (t/ha); S2 bands
`B02,B03,B04,B05,B06,B07,B08,B8A,B11,B12` are averaged and NDVI computed; auxiliary
`temp_mean,total_prec,dem` are exposed as the `temperature,precipitation,elevation`
climate modality; there is no Sentinel-1 (zeros + zero mask); strict-holdout groups
are countries. If staged under `temp/YieldSAT/`, extract with:

```bash
mkdir -p data/input/yieldsat/preprocessed-24-ts
for country in Argentina Brazil Germany Uruguay; do
  mkdir -p "data/input/yieldsat/preprocessed-24-ts/${country}"
  unzip -n "temp/YieldSAT/Preprocessed/${country}/${country}.zip" \
    -d "data/input/yieldsat/preprocessed-24-ts/${country}"
done
```

The four preprocessed zips expand to about 136 GiB — check free space first.

## Usage

Edit the config block at the top of `src/main.py`, then run:

```bash
python src/main.py
```

### Recommended core-scope configuration

To run the first-priority core (classification, geographic axis, three core models):

```python
ACTIVE_ENCODERS = ["presto", "agrifm", "tessera"]   # the 3 core models
TASKS           = ["bin-crop-class", "crop-class"]  # 2/4 tasks: classification only
ACTIVE_AXES     = []                                # no stress axes → clean only
ACTIVE_METHODS  = []                                # [] = ERM baseline; then add one method
SEEDS           = [0, 1, 2]
```

The **geographic** axis is the core robustness contribution, but it is the
always-on strict-holdout *split* itself (target-budget 0 = zero-shot transfer), not
a stress condition — so an empty `ACTIVE_AXES` already runs it under the `clean`
input. `ACTIVE_AXES` only toggles the **optional** stress axes. Add `"sensorial"` /
`"temporal"`, OlmoEarth, or the regression tasks (`pheno-reg`, `yield-reg`) only
after the core is stable.

### Configuration reference

```python
ACTIVE_ENCODERS = [...]   # presto, agrifm, tessera (core); olmoearth (optional)
TASKS           = [...]   # bin-crop-class, crop-class (core); pheno-reg, yield-reg (optional)
ACTIVE_AXES     = [...]   # optional stress axes: sensorial, temporal ([] = clean only; geographic holdout is always on)
ACTIVE_METHODS  = [...]   # grit, dfr, tent, … ([] = ERM baseline only)
ACTIVE_HOLDOUTS = None    # None → each task's defaults; ["togo"]; or {"bin-crop-class": ["togo"]}
ACTIVE_CONDITIONS = None  # None → all (within ACTIVE_AXES); or an explicit subset
MAX_SAMPLES = None        # benchmark samples (None = all)
SEEDS = [0, 1, 2]         # each seed re-runs the eval loop; summary.csv reports mean ± std
OVERWRITE_MODE = "skip"   # "skip" = resume only missing cells; "override" = wipe + rerun
ENCODER_KWARGS = {}       # per-encoder kwargs (device auto-set to the visible GPU)
```

**AgriFM setup.** The wrapper expects the official repo at `../AgriFM` and the
checkpoint at `../AgriFM/AgriFM.pth` (override with `AGRIFM_REPO` / `AGRIFM_WEIGHTS`):

```bash
git clone https://github.com/flyakon/AgriFM ../AgriFM
curl -L https://glass.hku.hk/casual/AgriFM/AgriFM.pth -o ../AgriFM/AgriFM.pth   # 363 MB, CC0
```

The official environment recommends Python 3.9 + compiled MMCV, **but this wrapper
does not need mmcv**: AgriFM's model math is pure torch, and mmcv only enters via
mmseg's registry decorators, so we stub those out and import the repo's real
`SwinTransformer3D` directly (deps: `torch`, `timm`, `einops`, `mmengine`). Runs on
Python 3.13. See `src/models/agrifm.py`.

**Other encoder dependencies** (not on PyPI in the usual way):

```bash
pip install git+https://github.com/nasaharvest/presto.git   # Presto
pip install olmoearth-pretrain                               # OlmoEarth (optional comparator)
pip install torch thop                                       # TESSERA (runs the published model)
```

TESSERA's weights are not redistributed: download the published checkpoint (e.g.
`best_model_fsdp_20250427_084307.pt`) and set `TESSERA_WEIGHTS` (or drop it at
`<cache>/tessera/`).

**Holdouts by task (`ACTIVE_HOLDOUTS`):**

| Task | Available holdouts |
|---|---|
| `bin-crop-class` | `togo`, `ethiopia`, `lem-brazil`, `rwanda`, `togo-eval` |
| `crop-class` | `Estonia` |
| `pheno-reg` | `Coastal Cauvery`, `Lower Cauvery`, `Upper Cauvery`, `Middle Vennar`, `Coastal Vennar` |
| `yield-reg` | `Argentina`, `Brazil`, `Germany`, `Uruguay` |

### Parallelism and resumption

Everything is **cache-backed and resumable**, so a crash (or `Ctrl-C`) only loses the
in-flight probe cell:

- Assembled benchmarks are pickle-cached under `data/cache/benchmark/` — re-runs load
  one file instead of re-reading tens of thousands of small inputs.
- Encoder embeddings are cached per `(benchmark, encoder, N, condition)` with atomic
  writes; only missing conditions trigger a forward pass.
- Probe results are appended to `results/<encoder>/<task>/probe_results.jsonl` as each
  `(seed, holdout, condition, method, budget_type)` cell finishes; on restart,
  finished cells are skipped (`OVERWRITE_MODE="skip"`) and tables regenerated.

To **split work across GPUs** (digital-ag has 2× RTX 3090), run the fan-out launcher;
it starts one process per GPU, each pinned via `CUDA_VISIBLE_DEVICES` and handed a
disjoint round-robin slice of the `(encoder, task)` pairs:

```bash
cd src && python utils/gputils.py         # one process per detected GPU
cd src && python utils/gputils.py 2       # force 2 shards
```

### Big-disk scratch (when `$HOME` is small/full)

Staged input always lives in `data/input/`. Everything regenerable (benchmark
pickles, embeddings, results, shard logs) is written under `$ROBUSTNESS_SCRATCH` when
set — point it at a roomy/fast disk so a near-full `$HOME` can't crash a run:

```bash
export ROBUSTNESS_SCRATCH=/var/tmp/robustness   # e.g. a big NVMe partition
```

Unset, it defaults to `data/`. `./sync.sh setup` creates the remote scratch dirs and
`./sync.sh pull` reads results back. The embedding cache is stored as `float16`
(`EMB_DTYPE` in `src/utils/cacheutils.py`) and loaded back as `float32`, halving its
footprint with no effect on the probes.

### When to delete the cache

**Short answer: you don't.** Cache keys are content-aware, so they self-invalidate —
a stale entry is simply never read. What gets folded into each key:

| cache | key includes | self-invalidates when you change… |
|---|---|---|
| `cache/benchmark/<bench>__<params>__code-<h>__data-<h>.pkl` | bench params + **hash of `get_input.py`** + **fingerprint of `data/input/<bench>/`** | the loader / `corrupt()` code, or the staged input data |
| `output/embeddings/<bench>/<encoder>/n<N>_b<benchhash>_e<enchash>/<cond>.npy` | the bench identity **+ hash of `models/<encoder>.py`** | the benchmark *or* the encoder's code |

So: edit a loader/`corrupt()` → that bench pickle and all embeddings from it rebuild;
edit an encoder → only that encoder's embeddings rebuild; re-stage a dataset → input
fingerprint changes → rebuild (a cheap top-level fingerprint, so a surgical in-place
edit of one deep file may not bump it — `touch` the dataset dir if so); edit
probe/method/task code → nothing in cache changes. Deleting cache is never
destructive — `data/input/` is the only source of truth. Old hashes accumulate;
`rm -rf data/cache data/output/embeddings` to reclaim space.

## Status

Infrastructure is largely complete; the empirical matrix is not.

- **Built:** the full pipeline; all four encoder wrappers (Presto, AgriFM, TESSERA
  core + OlmoEarth optional); GRIT, DFR, TENT; CropHarvest, EuroCropsML, SICKLE,
  YieldSAT ingestion; content-keyed caching; GPU fan-out; resumable runs.
- **Preliminary results (CropHarvest, strict geographic holdout):** calibrated `τ*`
  materially changes encoder rankings vs. default `τ = 0.5`; grouped ≈ strict
  (0.508 vs. 0.510 F1) at the *downstream* level; LEM-Brazil is a consistent failure
  region (a candidate geographic spurious subspace). An optional sensor-stress
  diagnostic (S2-off F1 0.597 → 0.102, recovering to 0.54–0.58 when the probe is
  retrained on the degraded condition) suggests much of the collapse is probe-level
  shift, not loss of embedding signal.
- **Next (core first):** full geographic-holdout *classification* baselines across
  Presto, AgriFM, TESSERA (the cross-encoder failure map, contributions C1/C2); then
  one post-hoc method (DFR or balanced last-layer retraining) measured against the
  no-adaptation baseline. Sensor/temporal axes, regression tasks, OlmoEarth, and
  further methods (T3A, conformal, prompt tuning) follow only once the core is stable.
```
