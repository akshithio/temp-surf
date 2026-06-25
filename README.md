# Post-Hoc Robustness for Agricultural Earth Observation Foundation Models

Given an existing Earth-observation foundation model, how do you make it reliably
deployable in a new agricultural region, with few or zero target-region labels,
without retraining model weights? This repository treats EO models as frozen:
the model is run once, embeddings are cached, and every adaptation
method or probe operates on those cached matrices.

> **Central contribution.** Establish a deployment-realistic evaluation framework
> for geographic robustness in frozen agricultural EO models, then measure
> whether post-hoc adaptation recovers a meaningful fraction of the transfer gap
> without modifying model weights.

---

## Current Scope

- **Robustness axis:** deployment-style domain transfer through geographic holdouts.
- **Benchmarks:** binary crop/non-crop (CropHarvest), multiclass crop-type
  (EuroCropsML, BreizhCrops), and semantic segmentation (PASTIS-R).
- **Models:** Presto, OlmoEarth v1.1-Base, Galileo v1 Base, AgriFM, and TESSERA v1.1.
- **Adaptation:** ERM baseline plus optional post-hoc feature transforms.

Success for this pass means clean, reproducible domain-holdout baselines with
cached frozen embeddings and complete probe outputs. Additional models,
benchmarks, and domain bases should be added only after their input contracts are
explicitly designed and tested.

---

## Evaluation Protocol

Every experiment cell is defined by the cross product of model, benchmark, seed,
split regime (which determines the number of actual train/test splits), method,
and label budget. Each split produces one probe fit per budget level.

**Models**

| Model               | Embedding dim | Input                                   |
| ------------------- | ------------: | --------------------------------------- |
| Presto              |           128 | S1+S2+ERA5/SRTM pixel time series       |
| OlmoEarth v1.1-Base |           768 | S2 L2A spatial chips (H,W,T,12)         |
| Galileo v1 (Base)   |           768 | S2 spatial chips (H,W,T,10) + NDVI      |
| AgriFM              |          1024 | S2 time series adapted to the S2 branch |
| TESSERA v1.1        |           128 | S1+S2 time series                       |


**Benchmarks**

| Benchmark | Name | Kind | Holdouts | Metric family | Label source |
|---|---|---|---|---|---|
| `cropharvest` | CropHarvest | binary crop/non-crop | kenya, togo, ethiopia, lem-brazil, rwanda | F1, AUROC, calibrated F1, ECE, Brier, NLL, balanced accuracy | real `is_crop` |
| `eurocropsml` | EuroCropsML | multiclass crop type | Estonia | macro/weighted F1, balanced accuracy, accuracy, macro AUC | real crop type |
| `breizhcrops` | BreizhCrops | multiclass crop type | frh04 | macro/weighted F1, balanced accuracy, accuracy, macro AUC | real crop type |
| `pastis` | PASTIS-R | semantic segmentation | official folds 1-3/4/5 | mIoU, pixel accuracy, macro/weighted F1 | per-pixel crop type |

PASTIS-R is streamed lazily as four 64x64 tiles per source patch. S2 and
ascending-orbit S1 observations are monthly aggregated. Class 19 (void) is
removed, while background class 0 remains part of the evaluation. OlmoEarth,
Galileo, and AgriFM use their native spatial feature grids; Presto and TESSERA
encode bounded pixel batches. Dense tile caches are resumable.

**Model × benchmark compatibility**

Not every model runs meaningfully on every benchmark. The compatibility table in
[`src/evals/compat.py`](src/evals/compat.py) is the single source of truth: you
specify only `BENCHMARKS`, and the runner reads off which models run on each.

| Model | CropHarvest | EuroCropsML | BreizhCrops | PASTIS-R |
|---|:--:|:--:|:--:|:--:|
| Presto | ✅ (1) | ✅ (4) | ✅ (3) | ✅ (8) |
| TESSERA | ❌ (10) | ✅ (6) | ✅ (6) | ✅ (1) |
| AgriFM | ❌ (12) | ❌ (12) | ❌ (12) | ✅ (4) |
| OlmoEarth | ✅ (1) | ✅ (4) | ✅ (1) | ✅ (1) |
| Galileo | ✅ (1) | ✅ (4) | ✅ (1) | ✅ (1) |
| raw baseline | ✅ (n/a) | ✅ (n/a) | ✅ (n/a) | ✅ (n/a) |

Every benchmark clears at least three eligible learned models, plus the raw baseline.

**Split regimes**

Each regime owns both its domain assignment and splitting logic in its own file
under [`src/evals/regimes/`](src/evals/regimes/) (named exactly after the
regime). A regime first assigns each sample to a domain basis, then decides how
those domains become train/test splits:

| Regime | Domain basis | Splits per benchmark | Description |
|---|---|---:|---|
| `random_id` | geography | 1 | Random stratified 80/10/10 split. Train and test share regions/domains (in-distribution upper bound). |
| `official` | geography | benchmark-defined | Benchmark-recommended split. CropHarvest: Kenya/Brazil/Togo targets. EuroCropsML: Estonia. BreizhCrops: frh04 test with frh03 validation. PASTIS-R: folds 1-3/4/5. |
| `geographic_ood` | geography | strict stress splits | Stricter-than-official geographic stress. CropHarvest uses lat/lon spatial blocks plus a distance purge. EuroCropsML, BreizhCrops, and PASTIS-R use leave-one-geographic-unit-out over all countries/regions/folds with disjoint OOD validation. |
| `spatial_cluster_ood` | spatial clusters | 1 | Coordinate-only spatial stress split. K-means clusters lat/lon, test uses one spatial extreme, val uses the opposite extreme, train uses the remaining clusters after distance purging. Runs on all four benchmarks. |

Each regime yields a train/val/test split; the binary probe calibrates its
decision threshold on that held-out `val`, falling back to an internal split only
when a regime supplies no val. These regimes drive the classification benchmarks.

**PASTIS-R (segmentation)** uses dense split configs from the same regime modules:
`random_id` is an 80/10/10 split over whole patches, `official` uses the
published fold assignment (1-3 train / 4 val / 5 test), and `geographic_ood`
iterates leave-one-fold-out across all five spatial folds. `spatial_cluster_ood`
clusters PASTIS-R patch centroids and evaluates whole held-out patches.

**Budget types**

| Type | Levels | Meaning |
|---|---|---|
| Target budgets | `[0, 5, 10, 25, 50, -1]` | `0` = zero-shot OOD; `5..50` = absolute count of target labels added (nested); `-1` = target-ID oracle (trains on the 80% target pool). |
| Source budgets | `[0.05, 0.10, 0.25, 1.00]` | Fraction of source training data used. |

Each budget level fits a calibrated probe and scores the metrics on the test set.
For `geographic_ood` and `spatial_cluster_ood`, the source-domain training pool
also withholds source-side diagnostics when there is enough data:
`source_validation` and `source_test`. These rows are scored by the same fitted
probe as the OOD rows; they are diagnostic ID-within-source checks, not delta
anchors.

**Target evaluation scope (`evaluation_split`).** The target sweep draws ONE fixed 80/20
target split (a nested ordering of the 80% pool, a fixed 20% test). Every target-budget row
is tagged with an `evaluation_split`:

- `held_out` — scored on the fixed 20% test. Used by all budgets so the few-shot curve and the
  inherent-difficulty decomposition (zero-shot vs the `-1` oracle) compare like-for-like on the
  same samples.
- `full` — emitted only for budget `0`: zero-shot scored on the **whole** target domain. This is
  the **primary deployment OOD** estimand: `compute_deltas` reads the `full` rows for `ood` /
  `delta` / `ood_worst_region` / the secondary `ood_<axis>`, and the `held_out` rows for
  `ood_matched` / `target_id` / `adjusted_delta`.

Downstream tables (`summary.csv`, the resume completion key) key on `evaluation_split`, so the
two scopes are never averaged together.

**Metric roles**

Every result directory writes `metric_roles.json`, separating deployment metrics
from diagnostic metrics. Deployment metrics are the headline values that match
the expected use case; diagnostic metrics explain why the deployment metric moved.

| Label family | Deployment metrics | Diagnostic metrics |
|---|---|---|
| binary | `calibrated_f1`, `calibrated_balanced_accuracy`, `worst_group_calibrated_f1`, `worst_group_calibrated_balanced_accuracy` | default-threshold F1/balanced accuracy, AUROC, oracle target-optimal F1, threshold, ECE, Brier, NLL |
| multiclass | `macro_f1`, `balanced_accuracy`, `worst_group_macro_f1`, `worst_group_balanced_accuracy` | weighted F1, accuracy, macro AUC |
| segmentation | `miou`, `mean_per_tile_miou`, `worst_tile_miou` | pixel accuracy, macro/weighted F1, tile/class-count diagnostics |

Classification probe rows also include first-class worst-group fields
computed from the evaluated test domains (`worst_group`, `worst_group_metric`,
`worst_group_score`, and metric-specific `worst_group_*` columns). On `random_id`
rows this is the subpopulation shift view: train/test contain all domains, then
the row reports the worst test domain rather than only the average.

---

## Current Limitations

- **EuroCropsML is loaded as native irregular S2 series.** Model adapters choose their own
  temporal view: monthly-cadence models request calendar-month composites, while native-series
  models consume the original acquisition sequence.
- **EuroCropsML preprocessed inputs are S2-only.** Missing S1 and climate channels are
  masked for Presto on that benchmark.

---

## Post-Hoc Methods

Out of scope for now: the pipeline runs the plain **ERM** probe on the frozen embeddings
(no post-hoc adaptation). The generic `transform` hook in `evals` is retained (always
`None`), so a fitted-feature-transform method axis (e.g. GRIT/DFR/TENT) can be reintroduced
once the baseline failures are characterized — targeting observed failure modes rather than
adding decorative comparisons.

---

## Repository Layout

```text
.
├── src/
│   ├── main.py              # orchestrator: benchmark -> encode/cache -> probe (ERM) -> tables
│   ├── dataio/get_input.py  # Benchmark loader + shared degradation protocol
│   ├── models/              # active frozen model wrappers
│   ├── evals/
│   │   ├── evals.py         # budget sweeps and shared protocol constants
│   │   ├── probes/          # probe implementations and scoring
│   │   ├── compat.py        # model × benchmark matrix -> which models run per benchmark
│   │   ├── regimes/         # one file per regime; each owns domain assignment + splitting
│   │   └── benchmarks/      # per-benchmark label + metric specs
│   └── utils/
│       ├── cacheutils.py    # content-keyed benchmark and embedding cache
│       ├── gputils.py       # split work across GPUs
│       ├── ioutils.py       # CSV/JSONL writing + result aggregation
│       └── perfutils.py     # timing and static diagnostics
├── data/                    # git-ignored; input is source of truth
│   ├── input/               # staged benchmarks + model artifacts
│   ├── cache/
│   │   ├── benchmark/       # assembled benchmark pickles
│   │   └── embeddings/      # cached frozen-embedding .npy per (bench, model, signature)
│   └── output/
│       └── results/         # probe results, predictions, summaries per <model>/<benchmark>/
└── notebooks/
```

### Expected Data Layout

```text
data/input/benchmarks/cropharvest/
    labels.geojson
    features/arrays/<index>_<dataset>.h5
data/input/benchmarks/eurocropsml/
    preprocess/*.npz
    split/latvia_portugal_vs_estonia/...
data/input/benchmarks/pastis/
    metadata.geojson
    DATA_S2/S2_<patch>.npy
    DATA_S1A/S1A_<patch>.npy
    ANNOTATIONS/TARGET_<patch>.npy
data/input/models/presto/model-f317d103.pth
data/input/models/olmoearth-v1_1-base/
data/input/models/agrifm/AgriFM.pth
data/input/models/agrifm/source/
data/input/models/tessera/tessera_v1_1_mpc_encoder.pt
```

---

## Usage

### Tooling

The project uses a conda + uv split:

- **Conda** owns the interpreter and base command-line tools. `environment.yml`
  creates the `robustness` environment with Python 3.11, `uv`, and `ruff`.
- **uv** owns Python package installation from `pyproject.toml` and `uv.lock`.
  Use `uv pip install ...` inside the activated conda environment rather than
  installing project dependencies directly with pip.
- **`pyproject.toml`** is the single source of truth for package metadata,
  runtime dependencies, optional dev/notebook extras, and ruff configuration.
- **`uv.lock`** is checked in so both local and remote installs resolve the same
  Python dependency graph.
- **ruff** is the lint/import-format tool. Its config lives under `[tool.ruff]`
  in `pyproject.toml`.
- **pytest** is the test runner. Tests live under `src/tests/`.

Presto is installed separately with `--no-deps` because its published package
pulls an old benchmark dependency stack that conflicts with the Python 3.11
environment. The import-time dependencies needed by this repository are listed
directly in `pyproject.toml`.

OlmoEarth v1.1 requires PyTorch 2.7.1. The project pins PyTorch 2.7.1 and
torchvision 0.22.1 for every model rather than maintaining a second model-specific
environment. `uv.lock` pins the current upstream `olmoearth_pretrain` source because
the PyPI 0.1.0 release predates v1.1's linear patch embed. The model's `config.json` and `weights.pth` are downloaded from
`allenai/OlmoEarth-v1_1-Base` on first use into `data/input/models/olmoearth-v1_1-base/`.

To build the environment, use the same conda environment name:

```bash
conda env create -f environment.yml
conda activate robustness
uv pip install -e ".[dev,notebooks]"
uv pip install --no-deps "git+https://github.com/nasaharvest/presto.git@11e207a668a34336ced1d8e492a1bd5849b96c4a"
```

If the environment already exists, refresh the conda-managed base tools and then
sync the project dependencies through uv:

```bash
conda env update -n robustness -f environment.yml --prune
conda activate robustness
uv pip install -e ".[dev,notebooks]"
uv pip install --no-deps "git+https://github.com/nasaharvest/presto.git@11e207a668a34336ced1d8e492a1bd5849b96c4a"
```

Run checks from the activated environment:

```bash
ruff check src
python -m pytest src/tests
```



Edit the config block at the top of `src/main.py`, then run:

```bash
python src/main.py
```

Recommended core-scope configuration (you list only the benchmarks; the
compatibility matrix decides which models run on each):

```python
BENCHMARKS = ["cropharvest", "eurocropsml", "breizhcrops", "pastis"]
RUN_STAGES = ["gen_embeddings", "probing"]
SPLIT_REGIMES = ["random_id", "official", "geographic_ood", "spatial_cluster_ood"]
ACTIVE_PROBES = ["logistic"]
BUDGET_REGIMES = {
    "source": [0.05, 0.10, 0.25, 1.0],
    "target": [0, 5, 10, 25, 50, EV.TARGET_ID_UPPER_BOUND],
}
SEEDS = [0]
```

Configuration reference:

```python
BENCHMARKS = ["cropharvest", "eurocropsml", "breizhcrops", "pastis"]
RUN_STAGES = ["gen_embeddings", "probing"]
SPLIT_REGIMES = ["random_id", "official", "geographic_ood", "spatial_cluster_ood"]
ACTIVE_PROBES = ["logistic"]  # add "mlp" later if linear-probe gaps are ambiguous
BUDGET_REGIMES = {
    "source": [0.05, 0.10, 0.25, 1.0],
    "target": [0, 5, 10, 25, 50, EV.TARGET_ID_UPPER_BOUND],
}
MAX_SAMPLES = None            # None = all samples
MAX_DENSE_PIXELS = 50_000     # sampled PASTIS pixels per fold partition
SEEDS = [0]                   # expand to [0, 1, 2] for additional bulk runs
OVERWRITE_MODE = False        # False resumes; True overwrites cached outputs
STRICT_MODE = True            # True raises on recoverable data/regime issues
LAUNCH_GPU_SHARDS = False     # True launches one main.py worker per local GPU
```

Set `RUN_STAGES = ["gen_embeddings"]` to build or refresh embedding caches without
running probes. Set `RUN_STAGES = ["probing"]` to read existing embedding caches
and run probes only; this fails loudly if the matching cache is missing.

There is no model list to configure: `src/evals/compat.py` is the single source
of truth for which models are eligible per benchmark.

The default Presto checkpoint path is:

```text
data/input/models/presto/model-f317d103.pth
```

If it is missing, the wrapper downloads `torchgeo/presto`'s `model-f317d103.pth` into
that path.

TESSERA uses the encoder-only v1.1 MPC checkpoint:

```text
data/input/models/tessera/tessera_v1_1_mpc_encoder.pt
```

Download it from the [official TESSERA v1.1 release](https://drive.google.com/file/d/1t-gfTxi3Hg_uJXpJ9etROCRgKt2myfJ2/view).
The wrapper uses the checkpoint's 192-dimensional model and retains the released
128-dimensional downstream prefix. Its S2 band order and MPC normalization constants
match the official v1.1 inference code.

### Holdouts By Benchmark

| Benchmark | Available holdouts |
|---|---|
| `cropharvest` | `kenya`, `togo`, `ethiopia`, `lem-brazil`, `rwanda` |
| `eurocropsml` | `Estonia` |
| `breizhcrops` | `frh04` |

### Parallelism And Resumption

Everything is cache-backed and resumable:
- Assembled benchmarks are pickle-cached under `data/cache/benchmark/`.
- Model embeddings are cached per `(benchmark, model, benchmark signature)`
  with atomic writes.
- Probe results append to `probe_results.jsonl`; per-sample predictions append to
  `predictions.jsonl` for every probe cell.
- Restarting with `OVERWRITE_MODE=False` skips finished cells and regenerates derived
  tables from the JSONL logs.

To split work across GPUs:

Set `LAUNCH_GPU_SHARDS = True` in `src/main.py`, then run `cd src && python main.py`.



### Cache Invalidation

Deleting generated cache is safe since cache keys are content-aware and `data/input/` is the only source of truth. 

| Cache | Key includes | Rebuilds when you change |
|---|---|---|
| `data/cache/benchmark/<bench>__<tag>.pkl` | benchmark params, `get_input.py` hash, input-data fingerprint | loader code or staged benchmark inputs |
| `data/cache/embeddings/<bench>/<model>/<signature>/baseline.npy` | benchmark identity and model source hash | benchmark inputs or model code |
