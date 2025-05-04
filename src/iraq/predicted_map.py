#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
predicted_map.py

Load point-level enrollment probabilities, aggregate them
to Iraq’s ADM2 districts (mean per district), write out
a district-level shapefile, and plot a choropleth map of
average predicted enrollment.
"""
from pathlib import Path
import geopandas as gpd
import matplotlib.pyplot as plt
import pandas as pd

Path("output_iraq").mkdir(exist_ok=True)
Path("output_iraq/UNified_predictions").mkdir(exist_ok=True)
# ────────────── CONFIG ──────────────────
GRID_GPKG      = Path("prepared_data_iraq/predictions_unified.gpkg")
DISTRICTS_SHP  = Path("data/iraq/iraq_district_shapefile/"
                       "irq_admbnda_adm2_cso_20190603.shp")
OUTPUT_SHP     = Path("output_iraq/UNified_predictions/districts_avg_pred_enroll.shp")
OUTPUT_FIG     = Path("output_iraq/UNified_predictions/district_enrollment_heatmap.png")

# ────────────────── LOAD DATA ───────────────────────────
# 1) load your grid of point predictions
grid = gpd.read_file(GRID_GPKG, layer="grid")

# 2) load ADM2 district polygons
districts = gpd.read_file(DISTRICTS_SHP)

# ─────────────── ensure same CRS ────────────────────────
# project both to a common CRS (here EPSG:3857 for plotting)
grid      = grid.to_crs(epsg=3857)
districts = districts.to_crs(epsg=3857)

# ───────────── spatial join & aggregation ───────────────
# 3) join each point to the district polygon it falls in
joined = gpd.sjoin(
    grid[["pred_enroll", "geometry"]],
    districts[["geometry"]],
    how="inner",
    predicate="within"
)

# 4) compute mean prediction per district
mean_pred = (
    joined
    .groupby("index_right")["pred_enroll"]
    .mean()
    .rename("avg_pred_enroll")
)

# 5) attach back to the districts GeoDataFrame
districts = districts.reset_index()  # ensures there's a column "index"
districts["avg_pred_enroll"] = districts["index"].map(mean_pred)

# ───────── after step 5 ────────────────────────
# create a plain DataFrame with just the district name and mean prediction
table = districts[['ADM2_EN', 'avg_pred_enroll']].copy()

# (Optional) sort by descending probability
table = table.sort_values('avg_pred_enroll', ascending=False)

# display to console
print(table.to_string(index=False))

# save to CSV for downstream use
table.to_csv("districts_avg_pred_enroll.csv", index=False)
print("Wrote CSV: districts_avg_pred_enroll.csv")

# ─── Optional: create categorical levels ───────
# e.g. split into tertiles: Low, Medium, High
table['enroll_level'] = pd.qcut(
    table['avg_pred_enroll'],
    q=3,
    labels=['Low', 'Medium', 'High']
)

# show the first few rows with categories
print("\nWith enrollment‐level categories:")
print(table.head().to_string(index=False))


# ─────────────── write out shapefile ────────────────────
districts[[
    "index",
    "ADM2_EN",
    "avg_pred_enroll",
    "geometry"
]].to_file(OUTPUT_SHP)
print(f"Wrote district-level shapefile: {OUTPUT_SHP}")

# ───────────────────── PLOT ────────────────────────────
fig, ax = plt.subplots(1, 1, figsize=(10, 8))

districts.plot(
    column       = "avg_pred_enroll",
    cmap         = "viridis",
    linewidth    = 0.5,
    edgecolor    = "gray",
    legend       = True,
    legend_kwds  = {
        "label": "Mean predicted enrollment probability",
        "shrink": 0.6
    },
    ax           = ax,
    missing_kwds = {
        "color": "lightgrey",
        "label": "No data"
    }
)

ax.set_title(
    "Average Predicted School-Attendance Probability by District",
    fontsize=14
)
ax.set_axis_off()

plt.tight_layout()
plt.savefig(OUTPUT_FIG, dpi=300)
plt.show()
