#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
grid_walk_uganda.py.py
──────────────────────────────
• Generate a 10-km fish-net over Uganda’s administrative boundaries
• Compute spatial features for each point
    – amenity counts (2 km / 5 km)
    – nearest distance to each amenity type
    – conflict-event counts (5 km)
    – distance to Kampala city centre
• Apply the **camp LightGBM model** (lgbm_enroll_camp.pkl)
• Save predictions + features to GeoPackage + CSV
"""
# ───────────────────────── imports ─────────────────────────
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
from shapely import wkt

# ─────────────── CONFIG ───────────────────────────────────
ADM_SHP      = Path("data/uganda/Uganda_Districts-2020---136-wgs84/Uganda_Districts-2020---136-wgs84.shp")   # ↔ Uganda ADM-2 boundary
MODEL_PKL    = Path("weights/uga_model_full/lgbm_enroll_uga_unified.pkl")               # camp model
OUTPUT_GPKG  = Path("output/predictions_camp_uga.gpkg")
AMENITY_CSV      = Path("data/uganda/uganda_filtered_amenities.csv")
CONFLICT_CSV     = Path("data/uganda/conflict_data_uganda_2018.csv")

GRID_SPACING = 5_000   # metres between grid points
CRS_WGS84    = "EPSG:4326"
CRS_M        = 3857     # metres

KAMPALA_LL   = (32.5825, 0.3476)   # lon, lat – Kampala City Square
# ----------------------------------------------------------



def load_amenities_gdf():
    df = pd.read_csv(AMENITY_CSV, dtype=str)
    # rename fclass→amenity if needed
    if "fclass" in df.columns:
        df = df.rename(columns={"fclass":"amenity"})
    # parse the WKT geometry into real Points
    df["geometry"] = df["geometry"].apply(wkt.loads)
    # now construct a GeoDataFrame directly from that geometry
    gdf = gpd.GeoDataFrame(df, geometry="geometry", crs=CRS_WGS84)
    return gdf

def load_conflict_gdf():
    df = pd.read_csv(CONFLICT_CSV, usecols=["event_type","latitude","longitude"], dtype=str)
    df["latitude"]  = pd.to_numeric(df["latitude"], errors="coerce")
    df["longitude"] = pd.to_numeric(df["longitude"],errors="coerce")
    df.dropna(subset=["latitude","longitude"], inplace=True)
    return gpd.GeoDataFrame(df, geometry=[Point(xy) for xy in zip(df.longitude,df.latitude)], crs=CRS_WGS84)

def safe_name(txt: str) -> str:
    txt = unicodedata.normalize("NFKD", txt).encode("ascii", "ignore").decode()
    return re.sub(r"\W+", "_", txt.lower()).strip("_")

def make_fishnet(bounds, spacing):
    """Return a list of Points (CRS_M) spaced by <spacing> metres."""
    minx, miny, maxx, maxy = bounds
    xs = np.arange(minx, maxx + spacing, spacing)
    ys = np.arange(miny, maxy + spacing, spacing)
    return [Point(x, y) for x in xs for y in ys]


def main() -> None:

    ts = datetime.now().strftime("%Y-%m-%d %H:%M")
    print(f"[{ts}] Loading administrative boundaries …")
    adm = gpd.read_file(ADM_SHP).to_crs(CRS_M)
    bounds = adm.total_bounds

    print("Building 10-km grid …")
    fishnet = make_fishnet(bounds, GRID_SPACING)
    grid = gpd.GeoDataFrame(geometry=fishnet, crs=CRS_M)

    # keep points inside Uganda
    grid = (
        gpd.sjoin(grid, adm[["geometry"]],
                  how="inner", predicate="within")
        .drop(columns="index_right")
    )
    print(f" → {len(grid):,} grid points inside Uganda ADM-2")

    # —— load spatial layers ONCE ————————————————
    print("Loading amenities …")
    am = load_amenities_gdf().to_crs(CRS_M)
    am_sidx = am.sindex
    amen_types = sorted(am["amenity"].dropna().unique())

    print("Loading conflict events …")
    cf = load_conflict_gdf().to_crs(CRS_M)
    cf_sidx = cf.sindex
    conf_types = sorted(cf["event_type"].dropna().unique())

    # —— load model & feature order ————————————————
    print("Loading LightGBM model …")
    model = joblib.load(MODEL_PKL)
    feat_names = model.booster_.feature_name()

    # —— ref point: Kampala ————————————————————————
    kamp_pt = (
        gpd.GeoSeries([Point(*KAMPALA_LL)], crs=CRS_WGS84)
           .to_crs(CRS_M)[0]
    )

    # —— per-point feature builder ————————————————
    def build_row(pt):
        row = {"x": pt.x, "y": pt.y}

        # amenity counts (2 km / 5 km)
        for radius in (2_000, 5_000):
            idx = list(am_sidx.intersection(pt.buffer(radius).bounds))
            nearby = am.iloc[idx]
            nearby = nearby[nearby.geometry.distance(pt) <= radius]
            vc = nearby["amenity"].value_counts()
            for a in amen_types:
                col = f"amenity_{safe_name(a)}_within_{radius//1000}km"
                row[col] = int(vc.get(a, 0))

        # nearest distance to each amenity
        for a in amen_types:
            col = f"dist_nearest_{safe_name(a)}_m"
            subset = am[am["amenity"] == a]
            row[col] = float(subset.geometry.distance(pt).min()) if not subset.empty else np.nan

        # conflict counts (5 km)
        radius = 5_000
        idx = list(cf_sidx.intersection(pt.buffer(radius).bounds))
        nearby = cf.iloc[idx]
        nearby = nearby[nearby.geometry.distance(pt) <= radius]
        vc = nearby["event_type"].value_counts()
        for t in conf_types:
            col = f"conflict_{safe_name(t)}_within_{radius//1000}km"
            row[col] = int(vc.get(t, 0))

        # distance to Kampala
        row["dist_to_kampala_m"] = float(pt.distance(kamp_pt))

        # LightGBM prediction (ensure correct column order)
        Xrow = pd.DataFrame([row], columns=feat_names)
        row["pred_enroll"] = float(model.predict_proba(Xrow)[0, 1])

        return row

    # —— parallel processing ————————————————
    print("Computing features & predictions (parallel) …")
    rows = Parallel(n_jobs=-1, verbose=5)(
        delayed(build_row)(pt) for pt in grid.geometry
    )

    # —— assemble GeoDataFrame ————————————————
    df = pd.DataFrame(rows)
    gdf_out = gpd.GeoDataFrame(
        df,
        geometry=[Point(xy) for xy in zip(df.x, df.y)],
        crs=CRS_M
    )

    # —— save outputs ————————————————
    print(f"Writing GeoPackage → {OUTPUT_GPKG}")
    gdf_out.to_file(OUTPUT_GPKG, layer="grid", driver="GPKG")

    csv_path = OUTPUT_GPKG.with_suffix(".csv")
    print(f"Writing CSV        → {csv_path}")
    gdf_out.drop(columns="geometry").to_csv(csv_path, index=False)

    print("Done.")


if __name__ == "__main__":
    main()
