# MAV Biodiversity — Species Distribution Modeling

Ducks Unlimited Incorporated and Ducks Unlimited Canada are building species distribution models (SDMs) to estimate probability of occurrence across the Mississippi Alluvial Valley (MAV). The workflow follows methods similar to [Paterson et al. (2024)](https://www.sciencedirect.com/science/article/pii/S0006320724003161), adapted for MAV counties and a multi-taxa species list.

**Data sources**


| Taxa                          | Occurrence data                                                 |
| ----------------------------- | --------------------------------------------------------------- |
| Birds                         | [eBird](https://ebird.org) Basic Dataset (EBD) custom downloads |
| Mammals, reptiles, amphibians | [GBIF](https://www.gbif.org)                                    |


**Processing stack:** Google Earth Engine (GEE) for covariate extraction and local Python (Polars, DuckDB, elapid) for data prep and MaxEnt modeling.

---

## Quick start

1. [Set up the conda environment](#conda-environment)
2. Configure `EBIRD_API_KEY` (see below)
3. Run tests to verify the install:

```bash
cd dataprep
python -m unittest test_species_manifest test_gap_species -v
python -m pytest test_convert_ebird_downloads.py test_partition_ebird.py -v
```

---

## End-to-end workflow

```
┌─────────────────────────────────────────────────────────────────────────────┐
│ 1. Define AOI + species list (Excel manifest)                               │
└─────────────────────────────────────────────────────────────────────────────┘
         │
         ├─ Birds ──► EBD Custom Download TSVs ──► convert_ebird_downloads.py
         │              (/mnt/f/biodiversity/ebird)                  (/mnt/f/biodiversity/ebirdpolars)
         │
         └─ Herps/Mammals ──► pullgbif.ipynb ──► /mnt/f/biodiversity/gbif/*.csv
         │
         ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│ 2. GEE covariate extraction (notebooks)                                     │
│    geeBackgroundToCSV → background points + covariates                      │
│    geeDataFromPoints  → per-species param CSVs                              │
│    GEEcreateRaster / GEEcreateEnvRaster → inference rasters                 │
└─────────────────────────────────────────────────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│ 3. MaxEnt modeling                                                          │
│    groupBGpoints.ipynb → background_avian.csv / background_herp.csv / …     │
│    maxent_model.py + runBatch_maxent.ipynb → tuned models per species       │
└─────────────────────────────────────────────────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│ 4. Seasonal inference                                                       │
│    run_batch_inference.py / runBatch_inference.ipynb → probability rasters  │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Recommended order of operations

1. **Define scope** — AOI geometry and species list in `dataprep/mavBiodiversityToolSpeciesList.xlsx`.
2. **Acquire occurrences**
  - **Non-avian:** run `dataprep/pullgbif.ipynb` → CSVs under `/mnt/f/biodiversity/gbif`.
  - **Avian:** request [EBD Custom Downloads](https://support.ebird.org/en/support/solutions/articles/48000838205-download-ebird-data) per species for MAV states (AR/LA/MS), save as `ebd_<pkg>/ebd_<pkg>.txt` under `/mnt/f/biodiversity/ebird`.
3. **Convert eBird TSVs to hive Parquet** (run in a terminal, not a notebook cell):
  ```bash
   conda activate biodiversity
   cd dataprep
   python -u convert_ebird_downloads.py 2>&1 | tee /mnt/f/biodiversity/logs/ebird_downloads_to_parquet.log
  ```
   Output: `/mnt/f/biodiversity/ebirdpolars/scientific_name=<species>/*.parquet`
4. **Check pipeline gaps** (optional but recommended):
  ```bash
   export EBIRD_API_KEY='your-key'
   python -u download_ebird_api_mav.py --mode manifest   # species codes + download instructions
   python -u download_ebird_api_mav.py --mode gaps       # missing EBD / parquet
   python -u download_ebird_api_mav.py --mode pipeline   # through param CSV stage
  ```
5. **Extract covariates in GEE** — run `geeBackgroundToCSV.ipynb`, then `geeDataFromPoints.ipynb` (see [GEE notebooks](#gee-notebooks)).
6. **Build group background CSVs** — `groupBGpoints.ipynb` writes `background_avian.csv`, `background_herp.csv`, `background_mammal.csv`.
7. **Train MaxEnt models** — `runBatch_maxent.ipynb` (uses `maxent_model.py`).
8. **Create inference rasters** — `GEEcreateRaster.ipynb` + `GEEcreateEnvRaster.ipynb`, then `run_batch_inference.py` (or `runBatch_inference.ipynb`).

---

## Repository layout

```
biodiversity/
├── README.md                    # this file
├── environment.yml              # conda environment definition
├── methods.txt                  # brief workflow notes (superseded by README)
├── maxent_model.py              # MaxEnt training pipeline (elapid)
├── maxent_model.ipynb           # single-species model example
├── maxent_inference.ipynb       # single-species seasonal inference (interactive)
├── run_batch_inference.py       # batch inference CLI (joblib, no papermill)
├── runBatch_maxent.ipynb        # parallel batch training (Excel manifest)
├── runBatch_inference.ipynb     # parallel batch inference (same as CLI)
├── test_run_inference.py        # unit tests for inference runner
└── dataprep/
    ├── species_manifest.py      # Excel species list → spgroup mapping
    ├── paths.py                 # default data read/write paths
    ├── gap_species.py           # pipeline gap detection + test helpers
    ├── ebird_polars_io.py       # shared Polars I/O for eBird → parquet
    ├── convert_ebird_downloads.py
    ├── download_ebird_api_mav.py
    ├── partition_ebird.py       # optional: partition national EBD file
    ├── mavBiodiversityToolSpeciesList.xlsx
    ├── mav_counties_4326.parquet / .geojson   # MAV AOI
    ├── pullgbif.ipynb
    ├── geeDataFromPoints.ipynb
    ├── geeBackgroundToCSV.ipynb
    ├── groupBGpoints.ipynb
    ├── GEEcreateRaster.ipynb
    ├── GEEcreateEnvRaster.ipynb
    └── test_*.py                # unit tests
```

---

## Conda environment

All scripts and notebooks use the `**biodiversity**` conda environment defined in `[environment.yml](environment.yml)`.

### Prerequisites

Install [Miniforge](https://github.com/conda-forge/miniforge) or [Miniconda](https://docs.conda.org/en/latest/miniconda.html) with `conda` available in your shell.

### Create the environment

From the repository root:

```bash
conda env create -f environment.yml
conda activate biodiversity
```

This installs Python 3.13 and packages from **conda-forge**, including Polars, pandas, geopandas, DuckDB, elapid, scikit-learn, rasterio, GDAL, openpyxl, papermill, pytest, JupyterLab, and the Google Earth Engine Python API.

### Update an existing environment

After pulling changes to `environment.yml`:

```bash
conda env update -f environment.yml --prune
```

### Jupyter kernel (optional)

To run notebooks in JupyterLab with this environment:

```bash
conda activate biodiversity
python -m ipykernel install --user --name biodiversity --display-name "biodiversity"
```

Then select the **biodiversity** kernel in JupyterLab.

### eBird API key

Copy `dataprep/.env.example` to `dataprep/.env` (gitignored) or export in your shell:

```bash
export EBIRD_API_KEY='your-ebird-api-token'
```

Required for `download_ebird_api_mav.py` when resolving eBird taxonomy codes. Get a token from [eBird API key management](https://ebird.org/api/keygen).

Set `EBIRD_INSECURE_SSL=0` if you need strict SSL certificate verification (default is `1`).

### Google Earth Engine

GEE notebooks authenticate against project `biodiversity-478015`. Run once per machine:

```python
import ee
ee.Authenticate()
ee.Initialize(project='biodiversity-478015')
```

---

## Data paths (defaults)

Default paths below are used throughout notebooks and scripts. Override with environment variables where noted (see table).


| Path                                              | Contents                                                           |
| ------------------------------------------------- | ------------------------------------------------------------------ |
| `dataprep/mav_counties_4326.parquet`              | MAV county AOI (EPSG:4326)                                         |
| `dataprep/mavBiodiversityToolSpeciesList.xlsx`    | Master species list (`Common Name`, `Scientific name`, `Group`)    |
| `/mnt/f/biodiversity/ebird/`                      | Raw EBD custom-download folders (`ebd_<pkg>/ebd_<pkg>.txt`)        |
| `/mnt/f/biodiversity/ebirdpolars/`                | Hive Parquet (`scientific_name=<key>/*.parquet`)                   |
| `/mnt/f/biodiversity/gbif/`                       | GBIF occurrence CSVs (`<species>_<year>.csv`)                      |
| `/mnt/f/biodiversity/param_csvs/`                 | GEE-exported covariate CSVs (`<species>_subset0.csv`, …)           |
| `/mnt/f/biodiversity/param_csvs/background_*.csv` | Group background points (avian / herp / mammal)                    |
| `/mnt/f/biodiversity/ppp_paramsoutput/<species>/` | Model artifacts (`.pkl`, plots, config JSON, prediction TIFs)      |
| `/mnt/f/biodiversity/ppp_paramsoutput/<species>/inference_stack_<season>.tif` | Cached covariate stack per species/season (built once, reused) |
| `/mnt/f/biodiversity/rasters/inference/`          | Inference covariate GeoTIFFs (`all_months_*.tif`, `spring.tif`, …) |
| `/mnt/f/biodiversity/modelprep/`                  | Legacy papermill output notebooks (optional; batch inference no longer uses papermill) |
| `/mnt/f/biodiversity/logs/`                       | Long-running job logs (convert, partition)                         |


All defaults are defined in `dataprep/paths.py`. Set `BIODIVERSITY_DATA_ROOT` to use a different root directory.

**Environment variables**


| Variable                           | Used by                                            | Default                                  |
| ---------------------------------- | -------------------------------------------------- | ---------------------------------------- |
| `BIODIVERSITY_DATA_ROOT`           | `dataprep/paths.py`                                | `/mnt/f/biodiversity`                    |
| `EBIRD_API_KEY`                    | `download_ebird_api_mav.py`                        | — (required for taxonomy lookup)         |
| `EBIRD_INSECURE_SSL`               | eBird API                                          | `1` (set `0` to verify SSL certificates) |
| `EBIRD_INPUT` / `EBIRD_INPUT_ROOT` | `partition_ebird.py`, `convert_ebird_downloads.py` | `/mnt/f/biodiversity/ebird`              |
| `EBIRD_OUTPUT`                     | partition / convert scripts                        | `/mnt/f/biodiversity/ebirdpolars`        |


---

## Species manifest

`dataprep/species_manifest.py` reads the Excel workbook and normalizes names for the pipeline:

```python
from species_manifest import load_species_manifest, get_species_entry

manifest = load_species_manifest()  # default: mavBiodiversityToolSpeciesList.xlsx
entry = get_species_entry(manifest, "Protonotaria citrea")
# → scientific_name, excel_group, spgroup
```

**Group mapping**


| Excel `Group`        | `spgroup` (background / model) |
| -------------------- | ------------------------------ |
| birds                | avian                          |
| amphibians, reptiles | herp                           |
| mammals              | mammal                         |


Batch notebooks (`runBatch_maxent.ipynb`, `runBatch_inference.ipynb`) load the manifest instead of hard-coded species lists.

---

## Occurrence data prep

### GBIF (mammals, reptiles, amphibians)

`dataprep/pullgbif.ipynb` queries GBIF per species and year, clips to the MAV AOI, and writes CSVs to `/mnt/f/biodiversity/gbif`. `geeDataFromPoints.ipynb` prefers GBIF CSV when present, otherwise falls back to eBird Parquet.

### eBird (birds)

The eBird API 2.0 **cannot** bulk-download multi-year history (historic endpoint is one calendar day per request). For MAV birds:

1. Use **EBD Custom Download** (web form) per species / region / date range.
2. Place extracts under `/mnt/f/biodiversity/ebird`.
3. Convert locally with `convert_ebird_downloads.py` or `download_ebird_api_mav.py --mode convert`.

`download_ebird_api_mav.py` modes:


| Mode           | Purpose                                                     |
| -------------- | ----------------------------------------------------------- |
| `manifest`     | Write JSON of species codes + EBD download instructions     |
| `gaps`         | List species missing EBD folders or parquet partitions      |
| `convert`      | Convert local TSVs → hive parquet (default)                 |
| `pipeline`     | Full status through param CSV stage                         |
| `seed-test`    | Write synthetic test parquet for gap species (testing only) |
| `api-historic` | Opt-in daily API loop (not recommended)                     |


### Optional: national EBD partition

For a single large national EBD file (~440 GB), `partition_ebird.py` buckets by scientific-name prefix and writes hive parquet incrementally. Resumable via `.complete` markers. Run from a terminal, not inside a Jupyter cell.

See `dataprep/ebirdtoparquet.ipynb` for dry-run checks and progress monitoring.

---

## GEE notebooks

All GEE notebooks use the MAV AOI from `mav_counties_4326.parquet` (often convex-hull buffered by 10 km for focal statistics).

### Covariates extracted at occurrence and background points

For each point, `geeDataFromPoints.ipynb` and `geeBackgroundToCSV.ipynb` compute:

**Land cover** — [Google Dynamic World v1](https://developers.google.com/earth-engine/datasets/catalog/GOOGLE_DYNAMICWORLD_V1) class percentages within **500 m** and **10 km** buffers:


| Class ID | Label              |
| -------- | ------------------ |
| 0        | water              |
| 1        | trees              |
| 2        | grass              |
| 3        | flooded_vegetation |
| 4        | crops              |
| 5        | shrub_and_scrub    |
| 6        | built              |
| 7        | bare               |
| 8        | snow_and_ice       |


**Climate** — [Daymet v4](https://developers.google.com/earth-engine/datasets/catalog/NASA_ORNL_DAYMET_V4) seasonal means within 10 km: day length, precipitation, max/min air temperature.

**Seasons**


| Season | Months        |
| ------ | ------------- |
| Winter | Dec, Jan, Feb |
| Spring | Mar, Apr, May |
| Summer | Jun, Jul, Aug |
| Fall   | Sep, Oct, Nov |


### `geeDataFromPoints.ipynb`

- **Species queue:** computed at runtime from the Excel manifest + disk state via `gap_species.species_for_gee_export()` (preview: `python -u dataprep/gap_species.py --gee-queue`). No hardcoded species list.
- Loads occurrences from `/mnt/f/biodiversity/gbif/<species>*.csv` (herps/mammals) or DuckDB query against `/mnt/f/biodiversity/ebirdpolars` (birds).
- eBird filter: 2017–2024, AOI intersection, stationary or effort < 0.1 km.
- Exports `{species}_subset0.csv`, `{species}_subset1.csv` to `/mnt/f/biodiversity/param_csvs`.
- Skips species whose param CSVs already exist.

**Typical GEE workflow:**

```bash
python -u dataprep/check_occurrences.py --acquire   # ensure GBIF CSVs + eBird parquet
python -u dataprep/gap_species.py --gee-queue       # preview species needing GEE export
# then run geeDataFromPoints.ipynb
```

### `geeBackgroundToCSV.ipynb`

Generates ~20k random background points within the AOI and exports the same covariate set. Output feeds `groupBGpoints.ipynb` / `group_bg_points.py`.

### `groupBGpoints.ipynb` / `group_bg_points.py`

Builds group-specific background CSVs for MaxEnt from the **Excel manifest** (via `species_manifest.py`):

- **`background_avian.csv`** — birds with param CSVs on disk
- **`background_herp.csv`** — amphibians + reptiles
- **`background_mammal.csv`** — mammals

Uses presence points from `{species}_subset*.csv` for a KDE bias surface, then selects nearest points from `background_pts.csv`.

```bash
cd dataprep
python -u group_bg_points.py          # no plots
python -u group_bg_points.py --plot   # KDE diagnostic maps
```

### `GEEcreateRaster.ipynb` / `GEEcreateEnvRaster.ipynb`

Build spatial covariate stacks for **inference** (current conditions):

- **Land cover:** mode of Dynamic World classes for 2025.
- **Climate:** Daymet means over 2017–2024 (Daymet 2025 not yet available).

Rasters land under `/mnt/f/biodiversity/rasters/inference/`.

---

## MaxEnt modeling

Core logic lives in `maxent_model.py` (extracted from `maxent_model.ipynb`). Training uses the [elapid](https://earthlab.github.io/elapid/) MaxEnt implementation.

### Group-specific predictors

`prepare_predictors()` applies taxon-specific temporal covariates:


| Excel group                   | `month` / `season` in model      |
| ----------------------------- | -------------------------------- |
| birds                         | Kept as **categorical** features |
| amphibians, reptiles, mammals | **Dropped**                      |


Model config is saved per species as `model_config_<species>.json` (includes `excel_group`, predictor list, categorical features).

### Single species

```python
from maxent_model import run_maxent_species

run_maxent_species(
    sp="protonotaria_citrea",
    spgroup="avian",
    parambasedir="/mnt/f/biodiversity/param_csvs",
    baseoutputdir="/mnt/f/biodiversity",
    aoi_filename="mav_counties_4326.parquet",
    excel_group="birds",
)
```

Outputs under `/mnt/f/biodiversity/ppp_paramsoutput/<species>/`:

- `elapid_maxent_model_tuned_<species>.pkl`
- `model_config_<species>.json`, `predictors_<species>.txt`
- Regularization curves, ROC/PR curves, feature importance, diagnostic plots

### Batch training

`runBatch_maxent.ipynb` loads the Excel manifest, parallelizes across species with joblib, and reads group backgrounds (`background_avian.csv`, `background_herp.csv`, `background_mammal.csv`). Set `TEST_MODE = True` to run only gap-test species.

---

## Inference

Core logic lives in `maxent_model.py`: `build_inference_covariate_stack()` mosaics Dynamic World monthly rasters with the seasonal Daymet raster into a cached flat GeoTIFF, then `predict_raster_with_elapid()` applies the trained model.

**Outputs** under `/mnt/f/biodiversity/ppp_paramsoutput/<species>/`:

- `inference_stack_<season>.tif` — cached covariate stack (reused unless `--force-stack`)
- `predictions_prob_<species>_<season>.tif`
- `predictions_binary_<species>_<season>.tif`

**Seasons:** `spring`, `summer`, `fall`, `winter`.

### Single species (interactive)

`maxent_inference.ipynb` is for one-off runs and debugging. For scripting, call:

```python
from maxent_model import run_inference_species

run_inference_species(
    sp="dryophytes_cinereus",
    data_root="/mnt/f/biodiversity",
    seasons=["spring"],
)
```

### Batch inference (recommended)

Use `run_batch_inference.py` at the repo root. It calls `run_inference_species()` directly — **no papermill** — and parallelizes **one species per worker** (four seasons run serially inside each worker). A full 57-species rerun with `--jobs 8` typically finishes in **~30–60 minutes** vs several hours with the old papermill path.

```bash
conda activate biodiversity
cd /path/to/biodiversity

# Full rerun (rebuild cached stacks after all_months raster refresh)
python run_batch_inference.py --mode all --jobs 8 --force-stack

# Only missing prediction TIFs (default)
python run_batch_inference.py --mode missing_only --jobs 8

# Single species / subset of seasons
python run_batch_inference.py --species dryophytes_cinereus --seasons spring,summer
```

| Flag | Default | Purpose |
| ---- | ------- | ------- |
| `--data-root` | `/mnt/f/biodiversity` | Pipeline root |
| `--mode` | `missing_only` | `all` reruns every queued job; `missing_only` skips complete species/seasons |
| `--jobs` | `8` | Parallel species workers (increase only if CPU/disk are not saturated) |
| `--force-stack` | off | Rebuild `inference_stack_*.tif` caches |
| `--verbose` | off | Print band mapping and predict diagnostics (debugging) |
| `--species` | manifest | Comma-separated species override |
| `--seasons` | all 4 | Comma-separated season list |

`runBatch_inference.ipynb` uses the same functions and joblib pattern; set `run_mode`, `jobs`, and `force_stack` in the notebook. After a run, `audit_inference_outputs()` reports complete vs modelable species (57 modelable in the current manifest; 7 MaxEnt-failed species are excluded).

**Bottleneck:** first-time stack build (~20 s for small hulls, ~1–2 min for large bird hulls). Cached stacks make reruns much faster.

---

## Pipeline gap tracking

`dataprep/gap_species.py` classifies each species by pipeline stage:


| Stage                   | Meaning                           |
| ----------------------- | --------------------------------- |
| `need_ebd_download`     | No EBD folder for species code    |
| `need_convert`          | EBD present, no parquet partition |
| `need_gee_export`       | Parquet present, no param CSVs    |
| `ready_for_maxent`      | Param CSVs exist                  |
| `gbif_only`             | Non-eBird taxa (GBIF path)        |
| `not_in_ebird_taxonomy` | Species code not resolved         |


`GAP_TEST_SPECIES` lists species used for end-to-end pipeline testing when real EBD data is missing. `seed_test_parquet()` copies a small sample from an existing partition for dry runs.

---

## CLI scripts reference

Run with the `biodiversity` env active.

### Batch inference (repo root)

```bash
python run_batch_inference.py --mode all --jobs 8 --force-stack
python run_batch_inference.py --mode missing_only --jobs 8
python run_batch_inference.py --species protonotaria_citrea --seasons spring,summer
```

### Data prep (`dataprep/`)

```bash
cd dataprep

# Convert all EBD downloads under /mnt/f/biodiversity/ebird
python -u convert_ebird_downloads.py
python -u convert_ebird_downloads.py --dry-run
python -u convert_ebird_downloads.py --only ebd_US_comyel_relApr-2026

# MAV species orchestration
export EBIRD_API_KEY='...'
python -u download_ebird_api_mav.py --mode manifest
python -u download_ebird_api_mav.py --mode convert
python -u download_ebird_api_mav.py --mode pipeline

# Optional national file partition
EBIRD_INPUT=/path/to/ebd_US_relSep-2025.txt EBIRD_OUTPUT=/mnt/f/biodiversity/ebirdpolars \
  python -u partition_ebird.py
```

**Important:** Long-running convert/partition jobs should run in a **terminal**, not inside a Jupyter cell (memory and process limits).

---

## Testing

```bash
conda activate biodiversity

# Inference runner (repo root)
python -m pytest test_run_inference.py -v

cd dataprep

# stdlib unittest (species manifest + prepare_predictors + gap helpers)
python -m unittest test_species_manifest test_gap_species -v

# pytest (eBird I/O scripts)
python -m pytest test_convert_ebird_downloads.py test_partition_ebird.py -v
```

---

## References

- Paterson, J. et al. (2024). [Species distribution modeling methods paper](https://www.sciencedirect.com/science/article/pii/S0006320724003161)
- [eBird EBD Custom Download help](https://support.ebird.org/en/support/solutions/articles/48000838205-download-ebird-data)
- [eBird API 2.0 docs](https://documenter.getpostman.com/view/664302/S1ENwy59)
- [Google Dynamic World](https://developers.google.com/earth-engine/datasets/catalog/GOOGLE_DYNAMICWORLD_V1)
- [NASA Daymet v4](https://developers.google.com/earth-engine/datasets/catalog/NASA_ORNL_DAYMET_V4)
- [elapid MaxEnt library](https://earthlab.github.io/elapid/)

---

## License

GPL-3.0 — see [LICENSE](LICENSE).