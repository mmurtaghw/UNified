#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
grid_walk_Iraq.py

Generate a regular grid of points over Iraq’s districts, compute spatial features
(amenity counts/distances, conflict-event counts, distance to Baghdad) in parallel
for each point, apply the non-camp LightGBM enrollment model, and save predictions.

Outputs
-------
– predictions.gpkg (layer “grid”) with columns:
    x, y, pred_enroll, plus all spatial feature columns
"""
from pathlib import Path
from datetime import datetime
import numpy as np
import pandas as pd
import geopandas as gpd
import joblib
from shapely.geometry import Point
from joblib import Parallel, delayed
import unicodedata
import re

# ────────────── CONFIG ──────────────────
DISTRICTS_SHP = Path("data/iraq/iraq_district_shapefile/irq_admbnda_adm2_cso_20190603.shp")
MODEL_PKL     = Path("weights/iraq_all_geospatial_data/lgbm_enroll_unified.pkl")
OUTPUT_GPKG   = Path("prepared_data_iraq/predictions_unified.gpkg")
AMENITY_CSV      = Path("data/iraq/iraq_amenities_with_districts.csv")
CONFLICT_CSV     = Path("data/iraq/2021_conflict_data_iraq.csv")

GRID_SPACING  = 5_000   # metres between grid points
BUFFER_RAD    = 5_000    # metres for amenity & conflict counts

N_JOBS = 4 # check your processor for this

CRS_WGS84     = "EPSG:4326"
CRS_M         = 3857

# Baghdad centre (lon, lat)
BAGHDAD_LL    = (44.36611, 33.31528)

def load_amenities_gdf():
    df = pd.read_csv(AMENITY_CSV, dtype=str)
    df["lat"] = pd.to_numeric(df["lat"], errors="coerce")
    df["lon"] = pd.to_numeric(df["lon"], errors="coerce")
    df.dropna(subset=["lat","lon"], inplace=True)
    return gpd.GeoDataFrame(df, geometry=[Point(xy) for xy in zip(df.lon,df.lat)], crs=CRS_WGS84)

def load_conflict_gdf():
    df = pd.read_csv(CONFLICT_CSV, usecols=["event_type","latitude","longitude"], dtype=str)
    df["latitude"]  = pd.to_numeric(df["latitude"], errors="coerce")
    df["longitude"] = pd.to_numeric(df["longitude"],errors="coerce")
    df.dropna(subset=["latitude","longitude"], inplace=True)
    return gpd.GeoDataFrame(df, geometry=[Point(xy) for xy in zip(df.longitude,df.latitude)], crs=CRS_WGS84)

def make_fishnet(bounds, spacing):
    """
    Given (minx,miny,maxx,maxy) in target CRS, return a list of Points
    spaced by `spacing` metres.
    """
    minx, miny, maxx, maxy = bounds
    xs = np.arange(minx, maxx + spacing, spacing)
    ys = np.arange(miny, maxy + spacing, spacing)
    return [Point(x, y) for x in xs for y in ys]

def safe_name(txt: str) -> str:
    txt = unicodedata.normalize("NFKD", txt).encode("ascii", "ignore").decode()
    return re.sub(r"\W+", "_", txt.lower()).strip("_")

def main():
    ts = datetime.now().strftime("%Y-%m-%d %H:%M")
    print(f"[{ts}] Loading district boundaries…")
    districts = gpd.read_file(DISTRICTS_SHP).to_crs(CRS_M)
    bounds    = districts.total_bounds  # minx, miny, maxx, maxy

    print("Building grid…")
    fishnet_pts = make_fishnet(bounds, GRID_SPACING)
    grid = gpd.GeoDataFrame(geometry=fishnet_pts, crs=CRS_M)

    # keep only those inside Iraq
    grid = (
        gpd.sjoin(grid, districts[["geometry"]],
                  how="inner", predicate="within")
           .drop(columns="index_right")
    )
    print(f" → {len(grid)} grid points inside Iraq districts")

    # 1) load spatial layers ONCE
    print("Loading amenities and building spatial index…")
    am = load_amenities_gdf().to_crs(CRS_M)
    am_sidx = am.sindex
    amen_types = sorted(am["amenity"].dropna().unique())

    print("Loading conflict events and building spatial index…")
    cf = load_conflict_gdf().to_crs(CRS_M)
    cf_sidx = cf.sindex
    conf_types = sorted(cf["event_type"].dropna().unique())

    # 2) load model & feature ordering
    print("Loading model…")
    model = joblib.load(MODEL_PKL)
    feat_names = model.booster_.feature_name()

    # 3) prepare Baghdad point
    bag_pt = (
        gpd.GeoSeries([Point(*BAGHDAD_LL)], crs=CRS_WGS84)
           .to_crs(CRS_M)[0]
    )

    def process_point(pt):
        """
        Compute features + prediction for a single Point (in CRS_M).
        Returns a dict of x,y,<features…>,pred_enroll
        """
        row = {"x": pt.x, "y": pt.y}

        # Amenity counts within 2km & 5km
        for radius in (2000, 5000):
            # spatial index → bbox candidates
            idx = list(am_sidx.intersection(pt.buffer(radius).bounds))
            nearby = am.iloc[idx]
            nearby = nearby[nearby.geometry.distance(pt) <= radius]
            vc = nearby["amenity"].value_counts()
            for a in amen_types:
                key = f"amenity_{safe_name(a)}_within_{radius//1000}km"
                row[key] = int(vc.get(a, 0))

        # Nearest distance to each amenity type
        for a in amen_types:
            col = f"dist_nearest_{safe_name(a)}_m"
            subset = am[am["amenity"] == a]
            if subset.empty:
                row[col] = np.nan
            else:
                row[col] = float(subset.geometry.distance(pt).min())

        # Conflict-event counts within 5km
        radius = 5000
        idx = list(cf_sidx.intersection(pt.buffer(radius).bounds))
        nearby = cf.iloc[idx]
        nearby = nearby[nearby.geometry.distance(pt) <= radius]
        vc = nearby["event_type"].value_counts()
        for t in conf_types:
            key = f"conflict_{safe_name(t)}_within_{radius//1000}km"
            row[key] = int(vc.get(t, 0))

        # Distance to Baghdad
        row["dist_to_baghdad_m"] = float(pt.distance(bag_pt))

        # Predict — ensure correct column order
        Xrow = pd.DataFrame([row])[feat_names]
        row["pred_enroll"] = float(model.predict_proba(Xrow)[0, 1])

        return row

    # 4) process all points in parallel
    print("Computing features & predictions in parallel …")
    results = Parallel(n_jobs=N_JOBS, verbose=5)(
        delayed(process_point)(pt) for pt in grid.geometry
    )

    # 5) assemble GeoDataFrame
    df = pd.DataFrame(results)
    grid_out = gpd.GeoDataFrame(
        df,
        geometry=[Point(x, y) for x, y in zip(df.x, df.y)],
        crs=CRS_M
    )

    # 6) write to GeoPackage
    print(f"Writing {len(grid_out)} points → {OUTPUT_GPKG}")
    grid_out.to_file(OUTPUT_GPKG, layer="grid", driver="GPKG")

    # 7) also write to CSV
    OUTPUT_CSV = OUTPUT_GPKG.with_suffix(".csv")
    print(f"Writing {len(grid_out)} points → {OUTPUT_CSV}")
    # drop the geometry column (or convert it to WKT) for CSV export
    csv_df = grid_out.drop(columns="geometry")
    csv_df.to_csv(OUTPUT_CSV, index=False)

    print("Done.")

if __name__ == "__main__":
    main()
