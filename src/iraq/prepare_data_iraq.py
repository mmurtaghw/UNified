#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
prepare_data_iraq.py
───────────────────────────────────────────────
End-to-end pipeline that:
 1. Loads UNHCR household survey data
 2. Geocodes each household by district and, where applicable, by camp
 3. Builds spatial features (amenity counts/distances, conflict counts, distance to Baghdad)
 4. Filters to school‑age children and writes output CSV
"""
import json
import re
import unicodedata
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point

# ──────────────── Paths & CRS ──────────────────
DATA_DIR     = Path("data/iraq")
PREPARED_DIR = Path("prepared_data_iraq")
PREPARED_DIR.mkdir(exist_ok=True)

HH_SURVEY_CSV    = DATA_DIR / "UNHCR_IRQ_2022_MCNA_data_household_v2.1.csv"
MEMBER_CSV       = DATA_DIR / "UNHCR_IRQ_2022_MCNA_data_member_v2.1.csv"
GEO_COORD_CSV    = DATA_DIR / "governorate_district_coordinates.csv"
CAMPS_JSON       = DATA_DIR / "camps_with_districts.json"
AMENITY_CSV      = DATA_DIR / "iraq_amenities_with_districts.csv"
CONFLICT_CSV     = DATA_DIR / "2021_conflict_data_iraq.csv"

HH_GEOCODED_CSV  = PREPARED_DIR / "households_with_full_coords.csv"
FINAL_OUTPUT_CSV = PREPARED_DIR / "UNHCR_IRQ_2022_MCNA_school_age_children_with_spatial_features.csv"

CRS_WGS84 = "EPSG:4326"
CRS_M     = 3857   # metric projection for distance calcs
BAGHDAD_LL = (44.36611, 33.31528)  # lon, lat

# ─────────────── Utility Functions ────────────────────────
def safe_name(txt: str) -> str:
    txt = unicodedata.normalize("NFKD", txt).encode("ascii", "ignore").decode()
    return re.sub(r"\W+", "_", txt.lower()).strip("_")

_MONTH2INT = dict(jan=1,feb=2,mar=3,apr=4,may=5,jun=6,
                  jul=7,aug=8,sep=9,oct=10,nov=11,dec=12)

def parse_age_bounds(txt: str):
    txt = str(txt).strip()
    m = re.match(r"^(\d{1,2})\s*-\s*(\d{1,2})$", txt)
    if m: return int(m.group(1)), int(m.group(2))
    m = re.match(r"^(\d{1,2})\s*-\s*([A-Za-z]{3})$", txt)
    if m: return int(m.group(1)), _MONTH2INT.get(m.group(2).lower(), np.nan)
    m = re.match(r"^([A-Za-z]{3})\s*-\s*(\d{1,2})$", txt)
    if m: return _MONTH2INT.get(m.group(1).lower(), np.nan), int(m.group(2))
    m = re.match(r"^(\d{1,2})\s*\+$", txt)
    if m: return int(m.group(1)), 120
    return np.nan, np.nan

# ─────────────── Step 1: Geocode Households ────────────────────────
def geocode_households():
    df_geo  = pd.read_csv(GEO_COORD_CSV, dtype=str)
    df_data = pd.read_csv(HH_SURVEY_CSV, dtype=str)
    with open(CAMPS_JSON, 'r', encoding='utf-8') as f:
        camps_data = json.load(f)

    # normalize
    for df in (df_geo, df_data):
        for col in ("governorate_mcna","district_mcna"):
            df[col] = df[col].astype(str).str.strip().str.lower()
    if "camp_name" in df_data:
        df_data["camp_name"] = df_data["camp_name"].astype(str).str.strip().str.lower()

    # check duplicates
    dupes = (df_geo
             .groupby(["governorate_mcna","district_mcna"])
             .size().reset_index(name="count")
             .query("count>1"))
    if not dupes.empty:
        raise ValueError(f"Duplicate district coords:\n{dupes}")

    # flatten camps JSON
    camps_records = []
    for gov, info in camps_data.items():
        for camp in info.get("refugee_camps",[]):
            camps_records.append({
                "camp_name": camp["name"].strip().lower(),
                "lat_camp": camp["latitude"],
                "lon_camp": camp["longitude"]
            })
    df_camps = pd.DataFrame(camps_records)
    if df_camps.empty:
        raise ValueError("No camps found in JSON.")

    # merge district coords
    df_geo = df_geo.rename(columns={"latitude":"lat","longitude":"lon"})
    df = df_data.merge(
        df_geo[["governorate_mcna","district_mcna","lat","lon"]],
        on=["governorate_mcna","district_mcna"], how="left"
    )

    # override with camp coords
    df = df.merge(df_camps, on="camp_name", how="left")
    df["lat"] = df["lat_camp"].combine_first(df["lat"])
    df["lon"] = df["lon_camp"].combine_first(df["lon"])
    df = df.drop(columns=["lat_camp","lon_camp"])

    # warn if still missing
    missing = df[df["lat"].isna()|df["lon"].isna()]
    if not missing.empty:
        print("WARNING: some households not geocoded:")
        print(missing[["governorate_mcna","district_mcna","camp_name"]].drop_duplicates())
        print(f"Count: {len(missing)}")

    # coerce and save
    df["lat"] = pd.to_numeric(df["lat"],errors="coerce")
    df["lon"] = pd.to_numeric(df["lon"],errors="coerce")
    df.dropna(subset=["lat","lon"],inplace=True)
    df.to_csv(HH_GEOCODED_CSV, index=False)
    print(f"Geocoded households → {HH_GEOCODED_CSV}")
    return HH_GEOCODED_CSV

# ─────────────── Load GeoDataFrames ────────────────────────
def load_households_gdf():
    df = pd.read_csv(HH_GEOCODED_CSV, dtype=str)
    df["lat"] = pd.to_numeric(df["lat"], errors="coerce")
    df["lon"] = pd.to_numeric(df["lon"], errors="coerce")
    df.dropna(subset=["lat","lon"], inplace=True)
    return gpd.GeoDataFrame(df, geometry=[Point(xy) for xy in zip(df.lon,df.lat)], crs=CRS_WGS84)

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

# ─────────────── Spatial Feature Builders ────────────────────────
def amenity_counts(gdf_pts, gdf_am, radius_m):
    p = gdf_pts.to_crs(CRS_M); a = gdf_am.to_crs(CRS_M)
    sidx = a.sindex
    rows = []
    for pt in p.geometry:
        idx = list(sidx.intersection(pt.buffer(radius_m).bounds))
        nearby = a.iloc[idx]
        nearby = nearby[nearby.distance(pt)<=radius_m]
        rows.append(nearby["amenity"].value_counts())
    df = pd.DataFrame(rows).fillna(0).astype(int)
    df.columns = [f"amenity_{safe_name(c)}_within_{radius_m//1000}km" for c in df.columns]
    return df

def nearest_distances(gdf_pts, gdf_am):
    p = gdf_pts.to_crs(CRS_M); a = gdf_am.to_crs(CRS_M)
    out = {}
    for amen in sorted(a["amenity"].unique()):
        col = f"dist_nearest_{safe_name(amen)}_m"
        sub = a[a["amenity"]==amen]
        out[col] = p.geometry.apply(lambda g: sub.distance(g).min()).values
    return pd.DataFrame(out)

def conflict_counts(gdf_pts, gdf_conf, radius_m=5000):
    p = gdf_pts.to_crs(CRS_M); c = gdf_conf.to_crs(CRS_M)
    sidx = c.sindex
    types = sorted(c["event_type"].unique())
    mat = np.zeros((len(p), len(types)), dtype=int)
    for i, pt in enumerate(p.geometry):
        idx = list(sidx.intersection(pt.buffer(radius_m).bounds))
        nearby = c.iloc[idx]
        nearby = nearby[nearby.distance(pt)<=radius_m]
        vc = nearby["event_type"].value_counts()
        for j,t in enumerate(types):
            mat[i,j] = vc.get(t,0)
    cols = [f"conflict_{safe_name(t)}_within_{radius_m//1000}km" for t in types]
    return pd.DataFrame(mat, columns=cols)

def build_household_spatial_block():
    hh   = load_households_gdf()
    am   = load_amenities_gdf()
    cf   = load_conflict_gdf()
    print(f"Loaded {len(hh)} households, {len(am)} amenities, {len(cf)} conflict events")

    cnt2 = amenity_counts(hh, am, 2000).reset_index(drop=True)
    cnt5 = amenity_counts(hh, am, 5000).reset_index(drop=True)
    dist = nearest_distances(hh, am).reset_index(drop=True)
    cf5  = conflict_counts(hh, cf, 5000).reset_index(drop=True)

    # distance to Baghdad
    bag_pt = gpd.GeoSeries([Point(*BAGHDAD_LL[::-1])], crs=CRS_WGS84).to_crs(CRS_M)[0]
    hh_m   = hh.to_crs(CRS_M)
    hh["dist_to_baghdad_m"] = hh_m.geometry.distance(bag_pt)

    hh_df = hh.drop(columns="geometry").reset_index(drop=True)

    # now all five have unique 0..n-1 index
    return pd.concat([hh_df, cnt2, cnt5, dist, cf5], axis=1)

# ─────────────── Main ────────────────────────
def main():
    geocode_households()
    hh_features = build_household_spatial_block()

    members = pd.read_csv(MEMBER_CSV, dtype=str)
    low, high = zip(*members["age"].map(parse_age_bounds))
    members["age_low"], members["age_high"] = low, high

    merged = members.merge(
        hh_features, on="id", how="left", validate="many_to_one"
    )
    print(f"Merged member rows: {len(merged):,}")

    school_age = merged[
        (merged["age_low"] <= 17) & (merged["age_high"] >= 6)
    ].reset_index(drop=True)
    print(f"School-age children (6–17): {len(school_age):,}")

    school_age.to_csv(FINAL_OUTPUT_CSV, index=False)
    print(f"[{datetime.now():%Y-%m-%d %H:%M}] Written → {FINAL_OUTPUT_CSV}")

if __name__ == "__main__":
    main()
