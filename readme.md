# UNified Model

**Predicting Education Outcomes for Displaced Children in Data-Scarce Contexts**

**Team Craic – OECD & UNHCR Datathon 2025**  
Matt Murtagh-White, William Paja, Wooyong Jung

---

## Overview

This repository implements the **UNified model**, a machine learning pipeline for predicting educational outcomes among displaced children in data-scarce environments. The model is trained on microdata where available and leverages geospatial features (amenity access, proximity to conflict, distance to capital) to generate out-of-sample predictions.

The pipeline is demonstrated on:

- **Iraq**, using the 2021 Multi-Cluster Needs Assessment (MCNA)
- **Uganda**, using the 2018 Joint Multi-Sector Needs Assessment (J-MSNA) and Uganda National Panel Survey (UNPS)

[Our storymap report is available here](https://storymaps.arcgis.com/stories/f34f331fc8864c2788cc651c061a4cc0)

---

## How to Replicate the Iraq Pipeline

### 1. **Get the Required Datasets**

_The primary microdata cannot be redistributed via GitHub._ You will need to download it manually from official sources:

- **MCNA 2021 (Iraq)**  
  [UNHCR Microdata Library – MCNA 2021 Iraq](https://microdata.unhcr.org/index.php/catalog/913)

- **OpenStreetMap (OSM)** amenity data  
  Download or extract filtered `.csv` from [OpenStreetMap](https://www.openstreetmap.org) via Overpass or HOT Export tools. You'll need amenity names and coordinates for Iraq. **Available in repo.**

- **ACLED Conflict Data (Iraq, 2021)**  
  [ACLED Iraq Dataset – 2021](https://acleddata.com/data/)

- **Camp Coordinates**  
  Manually curated using UNHCR Iraq portal, ReliefWeb, and Google Maps, **available in repo.**

- **District Boundaries (IRQ ADM2)**  
  Available from [Humanitarian Data Exchange (HDX)](https://data.humdata.org) or UN OCHA shapefiles. File used: `irq_admbnda_adm2_cso_20190603.shp`.

---

### 2. **Prepare Data**

```bash
cd src/iraq
python prepare_data_iraq.py
```

This script:

- Loads MCNA microdata
- Assigns camp/non-camp household flags
- Merges in camp coordinates
- Computes spatial features (amenity counts, distances, conflict counts, etc.)
- Writes a flat feature file

---

### 3. **Train Iraq Models**

There are three model types:

#### a) Full Model (with microdata)

```bash
python iraq_full_model.py
```

- Trains separate LightGBM models for camp and non-camp households
- Incorporates economic, demographic, and spatial features
- Outputs:
  - SHAP summary CSVs
  - Beeswarm plots
  - ROC and PR curves
  - `.pkl` model files

#### b) Geospatial-only Model (Unified – Iraq)

```bash
python iraq_unified_geospatial_only.py
```

- Trains a **unified** model using only spatial features
- Outputs predictions and diagnostics for use in data-scarce regions

#### c) Iraq Grid Inference

```bash
python grid_walk_iraq.py
```

- Constructs a 10km synthetic grid over Iraq’s districts
- Computes features for each grid cell
- Applies trained geospatial model
- Writes predictions as GeoPackage and CSV

---

## How to Replicate the Uganda Pipeline

### 1. **Get Required Datasets**

Again, most microdata must be downloaded manually:

- **J-MSNA 2018 (Uganda)**  
  [UNHCR Microdata Library – Uganda 2018](https://microdata.unhcr.org/index.php/catalog/229)

- **Uganda National Panel Survey (UNPS 2018–2019)**  
  [World Bank Microdata Library – UNPS](https://microdata.worldbank.org/index.php/catalog/3820)

- **OpenStreetMap Amenities (Uganda)**  
  As with Iraq, filtered via Overpass or HOT Export. **Available in Repo**.

- **ACLED Conflict Data (Uganda, 2018)**  
  [ACLED Uganda Dataset – 2018](https://acleddata.com/data/)

- **Refugee settlement coordinates**  
  Manually matched to `settlement` column in J-MSNA using UNHCR/ReliefWeb/Google Maps. **Available in Repo**.

- **Uganda District Shapefile (ADM2)**  
  From [HDX](https://data.humdata.org/) or UBOS/NIRA shapefiles.

---

### 2. **Prepare Data**

```bash
cd src/uganda
python prepare_data_uganda.py
```

This will:

- Parse J-MSNA microdata
- Detect households with children
- Flag formal education attendance
- Merge camp coordinates
- Compute spatial features for each household

---

### 3. **Train Unified Model (Uganda)**

```bash
python unified_uganda.py
```

- Trains a single LightGBM model on:
  - J-MSNA microdata
  - Uganda National Panel Survey (UNPS) 2018–2019
- Uses spatial features only (no economic or child labor indicators)
- Outputs:
  - SHAP summary and beeswarm plots
  - Full SHAP matrix
  - ROC/PR curves
  - Trained model `.pkl` file

---

### 4. **Generate Grid Predictions**

```bash
python grid_walk_uganda.py
```

- Creates a 30km fishnet grid across Uganda’s districts
- Computes spatial features at each point
- Applies unified model to generate predictions
- Saves to GeoPackage and CSV

---

## Key Outputs

All models write outputs to:

```
output/
├── geospatial_only_model_iraq_output/
├── iraq_all_data_geospatial_output/
├── iraq_full_model_output/
└── uga_full_model_output/
```

Trained models are saved to:

```
weights/
├── geospatial_only_model_iraq/
├── iraq_all_geospatial_data/
├── iraq_model_full/
└── uga_model_full/
```

---

## References

- [UNHCR Microdata Library](https://microdata.unhcr.org/)
- [ACLED Conflict Data](https://acleddata.com/data/)
- [OpenStreetMap](https://www.openstreetmap.org)
- [World Bank Microdata (UNPS)](https://microdata.worldbank.org/index.php/catalog/3820)
- [Humanitarian Data Exchange (HDX)](https://data.humdata.org/)

---

## Citation

> OECD & UNHCR Datathon 2025  
> _The UNified Model: Predicting Education Outcomes for Displaced Children in Data-Scarce Contexts_  
> Authors: Matt Murtagh-White, William Paja, Wooyong Jung

For further details, consult the full report at `https://storymaps.arcgis.com/stories/f34f331fc8864c2788cc651c061a4cc0`.
