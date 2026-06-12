# MAV Biodiversity — Species Distribution Modeling

Ducks Unlimited Incorporated and Ducks Unlimited Canada are building species distribution models (SDMs) to estimate probability of occurrence across the Mississippi Alluvial Valley (MAV). The workflow follows methods similar to [Paterson et al. (2024)](https://www.sciencedirect.com/science/article/pii/S0006320724003161), adapted for MAV counties and a multi-taxa species list.

**Data sources**

| Taxa | Occurrence data |
|------|-----------------|
| Birds | [eBird](https://ebird.org) Basic Dataset (EBD) custom downloads |
| Mammals, reptiles, amphibians | [GBIF](https://www.gbif.org) |

**Processing stack:** Google Earth Engine (GEE) for covariate extraction, local Python (Polars, DuckDB, elapid) for data prep and MaxEnt modeling, WSL2 for heavy I/O jobs.

---

## Quick start

```bash
conda activate rapids-25.10
cd dataprep
```

If `python3` still resolves to system Python after activation, prepend the env:

```bash
export PATH="$CONDA_PREFIX/bin:$PATH"
```

Copy `dataprep/.env.example` to `dataprep/.env` (or export in your shell) and set `EBIRD_API_KEY` when resolving eBird taxonomy codes.

Run unit tests:

```bash
cd dataprep
python3 -m unittest test_species_manifest test_gap_species -v
python3 -m pytest test_convert_ebird_downloads.py test_partition_ebird.py -v
```

---

## End-to-end workflow

```
┌─────────────────────────────────────────────────────────────────────────────┐
│ 1. Define AOI + species list (Excel manifest)                               │
└─────────────────────────────────────────────────────────────────────────────┘
         │
         ├─ Birds ──► EBD Custom Download TSVs ──► convert_ebird_downloads.py
         │              (/mnt/f/ebird)                  (/mnt/c/ebirdpolars)
         │
         └─ Herps/Mammals ──► pullgbif.ipynb ──► /mnt/f/gbif/*.csv
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
│    maxent_inference.ipynb + runBatch_inference.ipynb → probability rasters  │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Recommended order of operations

1. **Define scope** — AOI geometry and species list in `dataprep/mavBiodiversityToolSpeciesList.xlsx`.
2. **Acquire occurrences**
   - **Non-avian:** run `dataprep/pullgbif.ipynb` → CSVs under `/mnt/f/gbif`.
   - **Avian:** request [EBD Custom Downloads](https://support.ebird.org/en/support/solutions/articles/48000838205-download-ebird-data) per species for MAV states (AR/LA/MS), save as `ebd_<pkg>/ebd_<pkg>.txt` under `/mnt/f/ebird`.
3. **Convert eBird TSVs to hive Parquet** (WSL terminal, not a notebook cell):

   ```bash
   conda activate rapids-25.10
   cd dataprep
   python -u convert_ebird_downloads.py 2>&1 | tee /mnt/f/ebird_downloads_to_parquet.log
   ```

   Output: `/mnt/c/ebirdpolars/scientific_name=<species>/*.parquet`

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
8. **Create inference rasters** — `GEEcreateRaster.ipynb` + `GEEcreateEnvRaster.ipynb`, then `runBatch_inference.ipynb`.

---

## Repository layout

```
biodiversity/
├── README.md                    # this file
├── methods.txt                  # brief workflow notes (superseded by README)
├── maxent_model.py              # MaxEnt training pipeline (elapid)
├── maxent_model.ipynb           # single-species model example
├── maxent_inference.ipynb       # single-species seasonal inference
├── runBatch_maxent.ipynb        # parallel batch training (Excel manifest)
├── runBatch_inference.ipynb     # parallel batch inference (papermill)
└── dataprep/
    ├── species_manifest.py      # Excel species list → spgroup mapping
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

## Environment

Use the **`rapids-25.10`** conda environment for all Python scripts and notebooks in this repo.

Key dependencies: Polars, pandas, geopandas, DuckDB, elapid, scikit-learn, rasterio, openpyxl (Excel I/O), Google Earth Engine Python API (for GEE notebooks).

GEE notebooks authenticate against project `biodiversity-478015`:

```python
ee.Authenticate()
ee.Initialize(project='biodiversity-478015')
```

---

## Data paths (defaults)

Paths assume WSL mounts (`/mnt/c`, `/mnt/f`). Override with environment variables where noted.

| Path | Contents |
|------|----------|
| `dataprep/mav_counties_4326.parquet` | MAV county AOI (EPSG:4326) |
| `dataprep/mavBiodiversityToolSpeciesList.xlsx` | Master species list (`Common Name`, `Scientific name`, `Group`) |
| `/mnt/f/ebird/` | Raw EBD custom-download folders (`ebd_<pkg>/ebd_<pkg>.txt`) |
| `/mnt/c/ebirdpolars/` | Hive Parquet (`scientific_name=<key>/*.parquet`) |
| `/mnt/f/gbif/` | GBIF occurrence CSVs (`<species>_<year>.csv`) |
| `/mnt/f/readyparams/param_csvs/` | GEE-exported covariate CSVs (`<species>_subset0.csv`, …) |
| `/mnt/f/readyparams/param_csvs/background_*.csv` | Group background points (avian / herp / mammal) |
| `/mnt/f/readyparams/ppp_paramsoutput/<species>/` | Model artifacts (`.pkl`, plots, config JSON) |
| `/mnt/f/readyparams/rasters/inference/` | Inference covariate GeoTIFFs (`all_months_*.tif`, `spring.tif`, …) |
| `/mnt/f/readyparams/modelprep/` | Papermill output notebooks from batch runs |

**Environment variables**

| Variable | Used by | Default |
|----------|---------|---------|
| `EBIRD_API_KEY` | `download_ebird_api_mav.py` | — (required for taxonomy lookup) |
| `EBIRD_INSECURE_SSL` | eBird API | `1` (skip SSL verify on WSL) |
| `EBIRD_INPUT` / `EBIRD_INPUT_ROOT` | `partition_ebird.py`, `convert_ebird_downloads.py` | per-script |
| `EBIRD_OUTPUT` | partition / convert scripts | `/mnt/c/ebirdpolars` or `/mnt/f/ebirdpolars` |

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

| Excel `Group` | `spgroup` (background / model) |
|---------------|--------------------------------|
| birds | avian |
| amphibians, reptiles | herp |
| mammals | mammal |

Batch notebooks (`runBatch_maxent.ipynb`, `runBatch_inference.ipynb`) load the manifest instead of hard-coded species lists.

---

## Occurrence data prep

### GBIF (mammals, reptiles, amphibians)

`dataprep/pullgbif.ipynb` queries GBIF per species and year, clips to the MAV AOI, and writes CSVs to `/mnt/f/gbif`. `geeDataFromPoints.ipynb` prefers GBIF CSV when present, otherwise falls back to eBird Parquet.

### eBird (birds)

The eBird API 2.0 **cannot** bulk-download multi-year history (historic endpoint is one calendar day per request). For MAV birds:

1. Use **EBD Custom Download** (web form) per species / region / date range.
2. Place extracts under `/mnt/f/ebird`.
3. Convert locally with `convert_ebird_downloads.py` or `download_ebird_api_mav.py --mode convert`.

`download_ebird_api_mav.py` modes:

| Mode | Purpose |
|------|---------|
| `manifest` | Write JSON of species codes + EBD download instructions |
| `gaps` | List species missing EBD folders or parquet partitions |
| `convert` | Convert local TSVs → hive parquet (default) |
| `pipeline` | Full status through param CSV stage |
| `seed-test` | Write synthetic test parquet for gap species (testing only) |
| `api-historic` | Opt-in daily API loop (not recommended) |

### Optional: national EBD partition

For a single large national EBD file (~440 GB), `partition_ebird.py` buckets by scientific-name prefix and writes hive parquet incrementally. Resumable via `.complete` markers. Run from a terminal, not inside a Jupyter cell.

See `dataprep/ebirdtoparquet.ipynb` for dry-run checks and progress monitoring.

---

## GEE notebooks

All GEE notebooks use the MAV AOI from `mav_counties_4326.parquet` (often convex-hull buffered by 10 km for focal statistics).

### Covariates extracted at occurrence and background points

For each point, `geeDataFromPoints.ipynb` and `geeBackgroundToCSV.ipynb` compute:

**Land cover** — [Google Dynamic World v1](https://developers.google.com/earth-engine/datasets/catalog/GOOGLE_DYNAMICWORLD_V1) class percentages within **500 m** and **10 km** buffers:

| Class ID | Label |
|----------|-------|
| 0 | water |
| 1 | trees |
| 2 | grass |
| 3 | flooded_vegetation |
| 4 | crops |
| 5 | shrub_and_scrub |
| 6 | built |
| 7 | bare |
| 8 | snow_and_ice |

**Climate** — [Daymet v4](https://developers.google.com/earth-engine/datasets/catalog/NASA_ORNL_DAYMET_V4) seasonal means within 10 km: day length, precipitation, max/min air temperature.

**Seasons**

| Season | Months |
|--------|--------|
| Winter | Dec, Jan, Feb |
| Spring | Mar, Apr, May |
| Summer | Jun, Jul, Aug |
| Fall | Sep, Oct, Nov |

### `geeDataFromPoints.ipynb`

- Reads species from a list (or manifest-driven workflow).
- Loads occurrences from `/mnt/f/gbif/<species>*.csv` or DuckDB query against `/mnt/c/ebirdpolars`.
- eBird filter: 2017–2024, AOI intersection, stationary or effort &lt; 0.1 km.
- Exports `{species}_subset0.csv`, `{species}_subset1.csv` to `/mnt/f/readyparams/param_csvs`.
- Skips species whose param CSVs already exist.

### `geeBackgroundToCSV.ipynb`

Generates ~20k random background points within the AOI and exports the same covariate set. Output feeds `groupBGpoints.ipynb`.

### `GEEcreateRaster.ipynb` / `GEEcreateEnvRaster.ipynb`

Build spatial covariate stacks for **inference** (current conditions):

- **Land cover:** mode of Dynamic World classes for 2025.
- **Climate:** Daymet means over 2017–2024 (Daymet 2025 not yet available).

Rasters land under `/mnt/f/readyparams/rasters/inference/`.

---

## MaxEnt modeling

Core logic lives in `maxent_model.py` (extracted from `maxent_model.ipynb`). Training uses the [elapid](https://earthlab.github.io/elapid/) MaxEnt implementation.

### Group-specific predictors

`prepare_predictors()` applies taxon-specific temporal covariates:

| Excel group | `month` / `season` in model |
|-------------|----------------------------|
| birds | Kept as **categorical** features |
| amphibians, reptiles, mammals | **Dropped** |

Model config is saved per species as `model_config_<species>.json` (includes `excel_group`, predictor list, categorical features).

### Single species

```python
from maxent_model import run_maxent_species

run_maxent_species(
    sp="protonotaria_citrea",
    spgroup="avian",
    parambasedir="/mnt/f/readyparams/param_csvs",
    baseoutputdir="/mnt/f/readyparams",
    aoi_filename="mav_counties_4326.parquet",
    excel_group="birds",
)
```

Outputs under `/mnt/f/readyparams/ppp_paramsoutput/<species>/`:

- `elapid_maxent_model_tuned_<species>.pkl`
- `model_config_<species>.json`, `predictors_<species>.txt`
- Regularization curves, ROC/PR curves, feature importance, diagnostic plots

### Batch training

`runBatch_maxent.ipynb` loads the Excel manifest, parallelizes across species with joblib, and reads group backgrounds (`background_avian.csv`, `background_herp.csv`, `background_mammal.csv`). Set `TEST_MODE = True` to run only gap-test species.

---

## Inference

`maxent_inference.ipynb` applies a trained model to inference rasters for one species and season (`tp`: `spring`, `summer`, `fall`, `winter`). It mosaics Dynamic World monthly rasters with the seasonal Daymet raster, applies the model, and writes:

- `predictions_prob_<species>_<season>.tif`
- `predictions_binary_<species>_<season>.tif`

`runBatch_inference.ipynb` loops all manifest species × four seasons via papermill.

---

## Pipeline gap tracking

`dataprep/gap_species.py` classifies each species by pipeline stage:

| Stage | Meaning |
|-------|---------|
| `need_ebd_download` | No EBD folder for species code |
| `need_convert` | EBD present, no parquet partition |
| `need_gee_export` | Parquet present, no param CSVs |
| `ready_for_maxent` | Param CSVs exist |
| `gbif_only` | Non-eBird taxa (GBIF path) |
| `not_in_ebird_taxonomy` | Species code not resolved |

`GAP_TEST_SPECIES` lists species used for end-to-end pipeline testing when real EBD data is missing. `seed_test_parquet()` copies a small sample from an existing partition for dry runs.

---

## CLI scripts reference

Run from `dataprep/` with `rapids-25.10` active.

```bash
# Convert all EBD downloads under /mnt/f/ebird
python -u convert_ebird_downloads.py
python -u convert_ebird_downloads.py --dry-run
python -u convert_ebird_downloads.py --only ebd_US_comyel_relApr-2026

# MAV species orchestration
export EBIRD_API_KEY='...'
python -u download_ebird_api_mav.py --mode manifest
python -u download_ebird_api_mav.py --mode convert
python -u download_ebird_api_mav.py --mode pipeline

# Optional national file partition
EBIRD_INPUT=/path/to/ebd_US_relSep-2025.txt EBIRD_OUTPUT=/mnt/f/ebirdpolars \
  python -u partition_ebird.py
```

**Important:** Long-running convert/partition jobs must run in a **WSL terminal**, not inside a Jupyter cell (memory and process limits).

---

## Testing

```bash
conda activate rapids-25.10
cd dataprep

# stdlib unittest (species manifest + prepare_predictors + gap helpers)
python3 -m unittest test_species_manifest test_gap_species -v

# pytest (eBird I/O scripts)
python3 -m pytest test_convert_ebird_downloads.py test_partition_ebird.py -v
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
