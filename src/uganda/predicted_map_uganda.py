#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
predicted_map_uganda.py
──────────────────────────────────
Aggregate point-level enrollment probabilities to Uganda’s ADM-2
districts and draw a choropleth of mean predicted enrollment.

Outputs
-------
• districts_avg_pred_enroll_uga.geojson
• district_enrollment_heatmap_uga.png
"""
from pathlib import Path
import geopandas as gpd
import matplotlib.pyplot as plt

# ───────────── CONFIG ─────────────
GRID_GPKG       = Path("output/predictions_camp_uga.gpkg")
DISTRICTS_SHP   = Path("data/uganda/Uganda_Districts-2020---136-wgs84/Uganda_Districts-2020---136-wgs84.shp")
OUTPUT_GEOJSON  = Path("output/districts_avg_pred_enroll_uga.geojson")
OUTPUT_FIG       = Path("output/district_enrollment_heatmap_uga.png")

# ───────────── LOAD DATA ─────────────
print("Loading point predictions …")
grid = gpd.read_file(GRID_GPKG, layer="grid")

print("Loading ADM-2 boundaries …")
districts = gpd.read_file(DISTRICTS_SHP)

# ───────────── PREP CRS ─────────────
grid      = grid.to_crs(epsg=3857)
districts = districts.to_crs(epsg=3857)

print(districts.columns)

# ───────────── JOIN & AGGREGATE ─────
print("Spatial join (points → districts) …")
joined = gpd.sjoin(
    grid[["pred_enroll", "geometry"]],
    districts[["geometry"]],
    how="inner",
    predicate="within"
)

print("Computing mean per district …")
mean_pred = (
    joined
    .groupby("index_right")["pred_enroll"]
    .mean()
    .rename("avg_pred_enroll")
)

# reset index to get a numeric 'index' column
districts = districts.reset_index(drop=False)

# map the computed means back into the districts GeoDataFrame
districts["avg_pred_enroll"] = districts["index"].map(mean_pred)



# ───────────── NAME HANDLING ─────────
# Identify which column holds the district name
name_candidates = ("ADM2_EN", "ADM2_NAME", "NAME_2", "dname2019")
for col in name_candidates:
    if col in districts.columns:
        districts = districts.rename(columns={col: "district_name"})
        break
else:
    raise KeyError(
        f"None of the expected name columns {name_candidates} "
        "were found in the districts shapefile."
    )

# ───────────── SAVE GEOJSON ───────
out_cols = ["index", "district_name", "avg_pred_enroll", "geometry"]
districts[out_cols].to_file(OUTPUT_GEOJSON, driver="GeoJSON")
print(f"→ wrote GeoJSON  {OUTPUT_GEOJSON}")

# ───────────── PLOT MAP ─────────────
fig, ax = plt.subplots(figsize=(10, 8))
districts.plot(
    column       = "avg_pred_enroll",
    cmap         = "viridis",
    linewidth    = 0.4,
    edgecolor    = "gray",
    legend       = True,
    legend_kwds  = dict(
        label  = "Mean predicted enrollment probability",
        shrink = 0.6
    ),
    ax           = ax,
    missing_kwds = dict(
        color  = "lightgrey",
        label  = "No data"
    )
)

ax.set_title(
    "Average Predicted School-Attendance Probability by District (Uganda)",
    fontsize=13
)
ax.set_axis_off()

plt.tight_layout()
plt.savefig(OUTPUT_FIG, dpi=300)
print(f"→ wrote PNG       {OUTPUT_FIG}")
plt.show()
