#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
prepare_data_uganda.py
────────────────────────────────────────────────────────────────
End‑to‑end pipeline for Uganda JMSNA 2018 that:
 1. Loads raw household survey
 2. Detects child presence & formal‑education status
 3. Geocodes by settlement (camp) coordinates for households
 4. Loads & merges panel data from GSEC2, GSEC4, geo‑vars
 5. Builds spatial features for both households & panel:
    • amenity counts (2 km, 5 km)
    • nearest distances to each amenity class
    • distance to Kampala city centre
    • conflict‑event counts (5 km) by ACLED event_type
 6. Concatenates household & panel feature tables
 7. Writes final flat file

Output → output_uganda/UNHCR_UGA_2018_spatial_features.csv
"""
from pathlib import Path
from datetime import datetime
import re, unicodedata

import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point
from shapely import wkt

# ──────────────── Paths & output dirs ─────────────────────────────
DATA_DIR    = Path("data/uganda")
OUTPUT_DIR  = Path("prepared_data_uganda")
OUTPUT_DIR.mkdir(exist_ok=True)

# household survey
RAW_SURVEY_CSV = DATA_DIR / "UGA_JMSNA_2018_Anonymized.csv"
# panel raw files
GSEC2_CSV      = DATA_DIR /  "GSEC2.csv"
GSEC4_CSV      = DATA_DIR / "GSEC4.csv"
GEO_VARS_CSV   = DATA_DIR / "unps_geovars_2018_19.csv"

# camp, amenity, conflict
CAMP_CSV       = DATA_DIR / "geocoded_camps_uganda.csv"
AMENITY_CSV    = DATA_DIR / "uganda_filtered_amenities.csv"
CONFLICT_CSV   = DATA_DIR / "conflict_data_uganda_2018.csv"

OUTPUT_CSV     = OUTPUT_DIR / "UNHCR_UGA_2018_spatial_features.csv"

CRS_WGS84 = "EPSG:4326"
CRS_M     = 3857  # Web‑Mercator (metres)

# Kampala city centre
KAMPALA_LL = (32.5825, 0.3476)  # lon, lat

# ─────────────── Utility functions ─────────────────────────
def parse_range(val, which="min"):
    if pd.isnull(val):
        return None
    s = str(val).strip().lower()
    if "to" in s:
        parts = s.split("to")
        try:
            return int(parts[0].strip()) if which=="min" else int(parts[1].strip())
        except:
            return None
    try:
        return int(s)
    except:
        return None

def has_child(row):
    yes_like = {"yes","1","1 to 3","one or more","more_than_1"}
    for v in ["unaccompanied_minor","orphan","separated_minor","child_violence","uasc_monitoring_visit"]:
        if str(row.get(v,"")).strip().lower() in yes_like:
            return True
    if str(row.get("young_child_diarrhoea","")).strip().lower() in yes_like:
        return True
    if str(row.get("school_previously","")).strip().lower() in yes_like:
        return True
    for v in ["no_latrn_acces_who_female_child","no_latrn_acces_who_male_child"]:
        if str(row.get(v,"")).strip().lower() in yes_like:
            return True
    hh_min = parse_range(row.get("hh_size"),"min")
    adults_max = parse_range(row.get("num_adults"),"max")
    if hh_min is not None and adults_max is not None and hh_min > adults_max:
        return True
    return False

def determine_education_status(row):
    if row["has_child"]:
        sv = str(row.get("school_previously","")).strip().lower()
        return "not_in_formal_education" if sv and sv!="no" else "in_formal_education"
    else:
        return "in_formal_education"

def safe_name(txt):
    s = unicodedata.normalize("NFKD", str(txt)).encode("ascii","ignore").decode()
    return re.sub(r"\W+","_",s.lower()).strip("_")

# ─────────────── Load & flag households ──────────────────────
def load_and_flag_households():
    df = pd.read_csv(RAW_SURVEY_CSV, dtype=str)
    df["has_child"] = df.apply(has_child, axis=1)
    df["in_formal_education"] = df.apply(determine_education_status, axis=1)
    return df

# ─────────────── Load & merge panel raw data ─────────────────
def load_panel():
    # read
    g2 = pd.read_csv(GSEC2_CSV, dtype={"hhid":str,"PID":str})
    g4 = pd.read_csv(GSEC4_CSV, dtype={"hhid":str,"PID":str})
    geo= pd.read_csv(GEO_VARS_CSV, dtype={"hhid":str})

    # age & filter 5–17
    g2["age"] = pd.to_numeric(g2["h2q8"], errors="coerce")
    g2 = g2.dropna(subset=["age"])
    g4["in_school"] = (g4["s4q05"]==3).astype(int)

    df = pd.merge(
        g2[["hhid","PID","age"]],
        g4[["hhid","PID","in_school"]],
        on=["hhid","PID"], how="inner"
    )
    df = df[(df["age"]>=5)&(df["age"]<=17)]
    df = df.merge(
        geo[["hhid","pub_lat_mod","pub_lon_mod"]],
        on="hhid", how="left"
    )
    df = df.dropna(subset=["pub_lat_mod","pub_lon_mod"])

    # mark
    df["has_child"] = 1
    df = df.rename(columns={"in_school":"in_formal_education"})
    df["from_panel"] = 1

    # to GeoDataFrame
    df["lat"] = pd.to_numeric(df["pub_lat_mod"],errors="coerce")
    df["lon"] = pd.to_numeric(df["pub_lon_mod"],errors="coerce")
    return gpd.GeoDataFrame(df, geometry=[Point(xy) for xy in zip(df.lon,df.lat)], crs=CRS_WGS84)

# ─────────────── Geocode households ────────────────────────
def geocode_households(df):
    df = df.copy()
    df["settlement_key"] = df["settlement"].str.strip().str.lower()
    camps = pd.read_csv(CAMP_CSV, dtype=str)
    camps["settlement_key"] = camps["settlement"].str.strip().str.lower()
    df = df.merge(camps[["settlement_key","lat","lon"]], on="settlement_key", how="left", suffixes=("","_camp"))
    for axis in ("lat","lon"):
        df[axis] = pd.to_numeric(df.get(axis),errors="coerce").fillna(pd.to_numeric(df.get(f"{axis}_camp"),errors="coerce"))
    df = df.dropna(subset=["lat","lon"])
    df["from_panel"] = 0
    return gpd.GeoDataFrame(df.drop(columns=["settlement_key","lat_camp","lon_camp"],errors="ignore"),
                            geometry=[Point(xy) for xy in zip(df.lon,df.lat)],
                            crs=CRS_WGS84)

# ─────────────── Load amenities & conflict ─────────────────
def load_amenities():
    am = pd.read_csv(AMENITY_CSV, dtype=str).rename(columns={"fclass":"amenity"})
    am["geometry"] = am["geometry"].apply(wkt.loads)
    return gpd.GeoDataFrame(am, geometry="geometry", crs=CRS_WGS84)

def load_conflict():
    cf = pd.read_csv(CONFLICT_CSV, usecols=["event_type","latitude","longitude"], dtype=str)
    cf["latitude"]  = pd.to_numeric(cf["latitude"],errors="coerce")
    cf["longitude"] = pd.to_numeric(cf["longitude"],errors="coerce")
    cf = cf.dropna(subset=["latitude","longitude"])
    return gpd.GeoDataFrame(cf, geometry=[Point(xy) for xy in zip(cf.longitude,cf.latitude)], crs=CRS_WGS84)

# ───────── amenity & conflict features ─────────────────────
def amenity_counts(gdf, am, r):
    p = gdf.to_crs(CRS_M); a = am.to_crs(CRS_M); idx=a.sindex
    rows=[]
    for pt in p.geometry:
        cand=a.iloc[list(idx.intersection(pt.buffer(r).bounds))]
        near=cand[cand.distance(pt)<=r]
        rows.append(near["amenity"].value_counts())
    df=pd.DataFrame(rows).fillna(0).astype(int)
    df.columns=[f"amenity_{safe_name(c)}_within_{r//1000}km" for c in df.columns]
    return df

def nearest_distances(gdf, am):
    p=gdf.to_crs(CRS_M); a=am.to_crs(CRS_M)
    out={}
    for c in sorted(a["amenity"].unique()):
        col=f"dist_nearest_{safe_name(c)}_m"
        sub=a[a["amenity"]==c]
        out[col]=p.geometry.apply(lambda g: sub.distance(g).min() if not sub.empty else np.nan).values
    return pd.DataFrame(out)

def conflict_counts(gdf,cf,r=5000):
    p=gdf.to_crs(CRS_M); c=cf.to_crs(CRS_M); idx=c.sindex
    types=sorted(c["event_type"].unique()); mat=np.zeros((len(p),len(types)),int)
    for i,pt in enumerate(p.geometry):
        cand=c.iloc[list(idx.intersection(pt.buffer(r).bounds))]
        near=cand[cand.distance(pt)<=r]
        vc=near["event_type"].value_counts()
        for j,t in enumerate(types): mat[i,j]=vc.get(t,0)
    cols=[f"conflict_{safe_name(t)}_within_{r//1000}km" for t in types]
    return pd.DataFrame(mat,columns=cols)

# ───────── build spatial block & write ─────────────────────
def main():
    hh_raw    = load_and_flag_households()
    hh_gdf    = geocode_households(hh_raw)
    panel_gdf = load_panel()

    amenities = load_amenities()
    conflict  = load_conflict()

    def build_block(gdf, label):
        print(f"Building spatial for {label}: {len(gdf)} rows")
        cnt2 = amenity_counts(gdf, amenities, 2000).reset_index(drop=True)
        cnt5 = amenity_counts(gdf, amenities, 5000).reset_index(drop=True)
        dist = nearest_distances(gdf, amenities).reset_index(drop=True)
        kp = gpd.GeoSeries([Point(*KAMPALA_LL[::-1])],crs=CRS_WGS84).to_crs(CRS_M)[0]
        d2k = gdf.to_crs(CRS_M).geometry.distance(kp).rename("dist_to_kampala_m").reset_index(drop=True)
        cf5 = conflict_counts(gdf, conflict, 5000).reset_index(drop=True)
        base = gdf.drop(columns="geometry").reset_index(drop=True)
        return pd.concat([base, cnt2, cnt5, dist, d2k, cf5], axis=1)

    hh_feat    = build_block(hh_gdf,    "households")
    panel_feat = build_block(panel_gdf, "panel")

    unified = pd.concat([hh_feat, panel_feat], ignore_index=True, sort=False)
    unified.to_csv(OUTPUT_CSV, index=False)
    print(f"[{datetime.now():%Y-%m-%d %H:%M}] → {OUTPUT_CSV}")

if __name__=="__main__":
    main()
