# Model × Benchmark Compatibility

Scope: the 5-model pool (**Presto, TESSERA, AgriFM, OlmoEarth, Galileo**) against the narrowed
benchmark set for the top-3 agricultural tasks:

| Benchmark | Task (deployer question) | Form |
|---|---|---|
| **CropHarvest** | cropland (binary) + crop-type classification | pixel/point time series |
| **EuroCropsML** | crop-type classification | parcel time series (S2-only) |
| **BreizhCrops** | crop-type classification | field/pixel time series (S2 L1C) |
| **PASTIS-R** | crop-type **mapping** (semantic/panoptic seg) | 128×128 patch time series, **S2 + S1** |

We use **PASTIS-R** (not PASTIS or PASTIS-HD): it is a superset of PASTIS (adds Sentinel-1 to
the same patches), and PASTIS-HD's only extra is SPOT 6/7 VHR optical (1.5 m), which **none of
the 5 models can consume** — so PASTIS-R is exactly the union of modalities the pool uses.

There are **two matrices**:

1. **Input-contract compatibility** — purely mechanical: does the model's encoder accept the
   benchmark's data form/modalities, as implemented in `src/models/*`?
2. **Literature-grounded precedent** — does a published paper actually run this model (or a
   directly-equivalent benchmark) so we can copy its implementation for the baseline?

---

## Matrix 2 — literature-grounded precedent

| Model | CropHarvest | EuroCropsML | BreizhCrops | PASTIS-R |
|---|:--:|:--:|:--:|:--:|
| **Presto** | ✅ | 🚧 | ⚠️ | ❌ |
| **TESSERA** | 🚧 | ❌ | ❌ | ✅ |
| **AgriFM** | ❌ | ❌ | ❌ | ✅ |
| **OlmoEarth** | 🚧 | 🚧 | 🚧 | ✅ |
| **Galileo** | 🚧 | 🚧 | 🚧 | ✅ |

**Legend:** ✅ paper-validated (the model is run on this — or a directly-equivalent — benchmark in a
published paper) · 🚧 **caveated in Matrix 1, but the model's _own_ paper shows it runs → copy that
paper's implementation for the baseline** · ⚠️ precedent exists **only from a third-party paper that
re-ran this model as a baseline** (not the model's own paper) — usable, but weaker than 🚧 · ❌ no
published precedent for this model on this benchmark form.

The reason for most of the 🚧 and/or ⚠️ is **pixel vs spatial**. The 3 classification benchmarks are pixel/parcel
time series → **Presto** is the clean fit; the 3 spatial models either can't (**AgriFM** needs real
chips) or run **degenerate 1×1** (OlmoEarth/Galileo wrappers force H=W=1). **PASTIS-R flips it**: it
ships spatial S2+S1 patches → native home for **AgriFM / OlmoEarth / Galileo**, and the most faithful
source for **TESSERA** (dense S1+S2 on disk).

> The ⚠️ vs 🚧 distinction is deliberate: e.g. **Presto's own paper does _not_ evaluate BreizhCrops**;
> only **Galileo re-ran Presto on BreizhCrops** as a baseline (Galileo Table 6 = 63.0). That's a real
> precedent we can copy, but it's the Galileo authors' setup, not Presto's — hence ⚠️, not 🚧.

### Coverage per eval (target ≥ 3 compatible models)

Counting ✅ + 🚧 + ⚠️ (i.e. native, own-paper-precedented, or third-party-precedented):

| Eval | ✅ | 🚧 | ⚠️ | total | ≥3? |
|---|---|---|---|:--:|:--:|
| **CropHarvest** | Presto | TESSERA, OlmoEarth, Galileo | — | 4 | ✓ |
| **EuroCropsML** | — | Presto, OlmoEarth, Galileo | — | 3 | ✓ |
| **BreizhCrops** | — | OlmoEarth, Galileo | Presto | 3 | ✓ (3rd via ⚠️) |
| **PASTIS-R** | TESSERA, AgriFM, OlmoEarth, Galileo | — | — | 4 | ✓ |

Every chosen eval reaches **≥3** compatible models with the current 5-model pool — **no additions
required**. Two are at the floor and lean on weaker evidence:
- **BreizhCrops** = 3 only because Presto counts via ⚠️ (Galileo's baseline re-run). For an
  own-paper-only bar, add a pixel-time-series FM whose own paper runs BreizhCrops (e.g. **AnySat**,
  arXiv:2412.14123 — a generalist that does crop-type classification; confirm BreizhCrops specifically).
- **EuroCropsML** = 3, all 🚧 via *equivalent* parcel/patch crop-type benchmarks (no model's paper uses
  EuroCropsML by name). Adding **AnySat** or a EuroCrops-native model would strengthen it.

### Per-cell evidence

**Presto** ([arXiv:2304.14065](https://arxiv.org/abs/2304.14065)) — evaluates on CropHarvest
(Togo/Kenya/Brazil, binary pixel), **S2-Agri** (parcel crop-type), EuroSAT, TreeSatAI, fuel-moisture,
algae. Not BreizhCrops/EuroCropsML/PASTIS in its own paper, but **Galileo re-runs Presto on
BreizhCrops** (Galileo Table 6 = 63.0).
- CropHarvest ✅ (own paper). · EuroCropsML 🚧 (own paper ran S2-Agri = parcel crop-type S2, the same
  task/modality as EuroCropsML). · BreizhCrops ⚠️ (**not** in Presto's own paper; only Galileo re-ran
  Presto on it as a baseline, Table 6 = 63.0 — third-party precedent). ·
  PASTIS-R ❌ (pixel-only model; never run on PASTIS; would need a per-pixel hack + the spatial pathway).

**TESSERA** ([arXiv:2506.20380](https://arxiv.org/abs/2506.20380)) — evaluates on **PASTIS-R**
(parcel segmentation, France), **Austrian Crop** (pixel crop-type + crop segmentation), TreeSatAI-TS,
Biomassters (AGB), Borneo canopy height. Pixel-wise **S1+S2** model.
- PASTIS-R ✅ (**TESSERA's own paper runs PASTIS-R**). · CropHarvest 🚧 (Austrian-Crop precedent for
  pixel crop-type with S1+S2; caveat = CropHarvest's staged arrays are the wrong radiometric product —
  L1C TOA + S1 dB vs TESSERA's L2A + S1 RTC). · EuroCropsML ❌ / BreizhCrops ❌ (both are **S2-only**;
  TESSERA's crop-type precedent uses S1+S2, so there's no precedent for an S1-starved run).

**AgriFM** ([arXiv:2505.21357](https://arxiv.org/abs/2505.21357)) — evaluates on **EuroCrops** crop
**mapping**/field-boundary/early-season (Auvergne-Rhône-Alpes, France), HLS30+MODIS rice mapping,
WorldCereal winter-wheat — i.e. a **spatial crop-mapping / segmentation** model.
- PASTIS-R ✅ (native spatial S2; crop-semantic-segmentation is exactly AgriFM's task — its paper used
  EuroCrops-ARA/rice/wheat mapping rather than PASTIS specifically, but PASTIS is the standard benchmark
  for it). · CropHarvest / EuroCropsML / BreizhCrops ❌ (all point/parcel-classification; AgriFM has no
  point pathway. Note its EuroCrops use is crop *mapping*/segmentation, not the EuroCropsML parcel-
  classification benchmark — relevant only if we later add a crop-mapping EuroCrops task).

**OlmoEarth** ([arXiv:2511.13655](https://arxiv.org/abs/2511.13655); v1.1
[arXiv:2605.20804](https://arxiv.org/abs/2605.20804)) — evaluates on **BreizhCrops**, **CropHarvest**
(Togo, PRC), **m-SA-crop-type**, **PASTIS**.
- PASTIS-R ✅ (own paper runs PASTIS). · CropHarvest 🚧 (own paper runs CropHarvest; Matrix-1 caveat =
  our wrapper feeds 1×1 chips). · BreizhCrops 🚧 (own paper runs BreizhCrops; 1×1 caveat). ·
  EuroCropsML 🚧 (own paper runs m-SA-crop-type + CropHarvest = same crop-type-classification family;
  EuroCropsML itself not in the paper).

**Galileo** ([arXiv:2502.09356](https://arxiv.org/abs/2502.09356)) — evaluates on **CropHarvest**
(Table 6), **BreizhCrops** (Table 6), **PASTIS** (Table 17), plus GeoBench, m-SA-crop-type, MADOS,
Sen1Floods11.
- PASTIS-R ✅ (own paper, Table 17). · CropHarvest 🚧 (own paper Table 6; 1×1 caveat in our wrapper). ·
  BreizhCrops 🚧 (own paper Table 6; 1×1 caveat). · EuroCropsML 🚧 (own paper runs m-SA-crop-type +
  CropHarvest crop-type; EuroCropsML itself not in the paper).

---

## How to read the two together

- A **✅ in both** matrices is a clean baseline we can stand up immediately by copying the source
  paper's probe setup: **Presto×CropHarvest**, and (once the spatial pathway exists)
  **TESSERA/AgriFM/OlmoEarth/Galileo × PASTIS-R**.
- A **🚧** means the mechanical fit is awkward (1×1 spatial, S2-only, wrong product) **but the model's
  own paper demonstrates the run** — so we replicate that paper's preprocessing/probe rather than
  inventing one. This covers the spatial FMs on the classification benchmarks.
- A **⚠️** means the only precedent is a *third-party* paper that re-ran the model as a baseline (e.g.
  Galileo's Table 6 running Presto on BreizhCrops). Still copyable, but it's not the model authors'
  setup — verify it before trusting the baseline.
- A **❌** means no published precedent for that model on that benchmark *form* — don't force it.

## Structural note (applies to every PASTIS-R cell)

PASTIS-R is scored with the **standard frozen-encoder segmentation protocol the source papers use**:
run the frozen encoder over each patch → dense per-pixel (or per-token, upsampled) feature map →
**linear probe per pixel** → **mIoU** (Galileo Table 17, "segmentation via linear probing"; OlmoEarth
evaluates PASTIS the same way). The repo is per-sample classification today (`(N,T,C)` pixel
`Benchmark` + logistic probe), so supporting PASTIS-R adds a spatial `Benchmark` + dense per-pixel
probe + mIoU metrics — **this pathway is being built**, and it is the accepted literature approach, not
a bespoke one.

Why AgriFM/TESSERA are ❌ on the three classification benchmarks: those provide point/parcel samples
with no spatial extent (AgriFM needs image patches), and for TESSERA no faithful dense S1+S2 series
(CropHarvest's arrays are the wrong radiometric product; EuroCropsML/BreizhCrops have no S1). PASTIS-R
provides spatial S2+S1 patches on disk, so both consume it directly.

> EuroCropsML caveat: **no model's paper benchmarks EuroCropsML by name** — the 🚧's there rest on
> directly-equivalent parcel/patch crop-type benchmarks (Presto→S2-Agri; OlmoEarth/Galileo→m-SA-crop-type
> + CropHarvest). Copy those probe/normalization setups.
