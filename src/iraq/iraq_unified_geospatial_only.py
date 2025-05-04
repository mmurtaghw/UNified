#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
iraq_unified_geospatial_only.py
───────────────────────────
Predict regular formal-school attendance (target = 1) for school-age children
using ONLY spatial variables:

• OSM amenity counts / distances
• Conflict-event counts
• Distance to Baghdad

Two separate LightGBM models are trained: one for camp households,
one for non-camp. SHAP signed + |value| summaries and beeswarm plots saved.

All CSV/PNG outputs go to geospatial_only_model_iraq_output/
All model .pkl files go to geospatial_only_model_iraq/
"""
from pathlib import Path
import numpy as np
import pandas as pd
import shap, joblib, matplotlib.pyplot as plt

from lightgbm               import LGBMClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics         import roc_auc_score, average_precision_score
from sklearn.utils.class_weight import compute_sample_weight

# ─────────────── configuration ─────────────────────────────
DATA_FILE   = Path("prepared_data_iraq/UNHCR_IRQ_2022_MCNA_school_age_children_with_spatial_features.csv")
MAX_DISPLAY = 20
RANDOM_SEED = 42
Path("weights").mkdir(exist_ok=True)
Path("output_iraq").mkdir(exist_ok=True)
# ─── NEW OUTPUT DIRS ─────────────────────────────────────
OUTPUT_DIR = Path("output_iraq/geospatial_only_model_iraq_output")
MODEL_DIR  = Path("weights/geospatial_only_model_iraq")
OUTPUT_DIR.mkdir(exist_ok=True, parents=True)
MODEL_DIR.mkdir(exist_ok=True, parents=True)

# ───────────── data loader ───────────────────────────────
def load_features() -> pd.DataFrame:
    """
    Load the prepared spatial-feature CSV and retain only:
      amenity_* , dist_nearest_* , conflict_* , dist_to_baghdad_m
    plus the columns we need for training/meta.
    """
    df = pd.read_csv(DATA_FILE)

    # map target
    df["target"] = df["school_regular_attendance_formal"].map({"yes": 1, "no": 0})
    df = df[df["target"].notna()].reset_index(drop=True)

    # harmonise survey weight
    df = (
        df.rename(columns={"survey_weight_x": "survey_weight"})
          .drop(columns=[c for c in ["survey_weight_y"] if c in df], errors="ignore")
    )
    if "survey_weight" not in df:
        df["survey_weight"] = 1.0

    # keep only the wanted spatial columns
    keep_cols = [
        c for c in df.columns
        if c.startswith("amenity_")
        or c.startswith("dist_nearest_")
        or c.startswith("conflict_")
    ]
    keep_cols += ["dist_to_baghdad_m", "camp_name", "survey_weight", "target"]

    df = df[keep_cols]

    # numeric / boolean coercion
    df.replace({"yes": 1, "no": 0, "TRUE": 1, "FALSE": 0}, inplace=True)
    num_cols = [c for c in df.columns if c not in ("camp_name",)]
    for c in num_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    return df

# ──────────── modelling utils ─────────────────────────────
def train_lgbm(Xtr, ytr, Xval, yval, *, seed=RANDOM_SEED):
    """LightGBM with survey-weight × class-balance weighting + early stop."""
    sw = Xtr.pop("survey_weight").astype(float).values
    if "survey_weight" in Xval:
        Xval = Xval.drop(columns="survey_weight")
    sw *= compute_sample_weight("balanced", ytr)

    model = LGBMClassifier(
        objective="binary",
        n_estimators=5000, learning_rate=0.05,
        num_leaves=31, subsample=0.8, colsample_bytree=0.8,
        random_state=seed, early_stopping_rounds=100
    )
    model.fit(Xtr, ytr, sample_weight=sw,
              eval_set=[(Xval, yval)], eval_metric="auc")

    preds = model.predict_proba(Xval)[:, 1]
    print(f"     AUROC={roc_auc_score(yval, preds):.3f} "
          f"PR-AUC={average_precision_score(yval, preds):.3f}")
    return model

def shap_summary(model, Xval, label):
    expl = shap.TreeExplainer(model)
    sv   = expl.shap_values(Xval)
    arr  = sv[1] if isinstance(sv, list) else sv
    return pd.DataFrame(
        {f"{label}_signed": arr.mean(0),
         f"{label}_abs":    np.abs(arr).mean(0)},
        index=Xval.columns
    )

def save_beeswarm(model, Xval, tag):
    expl = shap.TreeExplainer(model)
    sv   = expl.shap_values(Xval)
    arr  = sv[1] if isinstance(sv, list) else sv
    shap.summary_plot(arr, Xval,
                      plot_type="dot", max_display=MAX_DISPLAY,
                      show=False, color_bar=True)
    plt.title(f"{tag.capitalize()} model – top {MAX_DISPLAY}", fontsize=14)
    plt.tight_layout()
    out_path = OUTPUT_DIR / f"shap_beeswarm_{tag}_top{MAX_DISPLAY}.png"
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"   → {out_path}")

# ─────────────────── main ─────────────────────────────────
def main() -> None:
    df = load_features()

    # flag camp then drop the string column
    df["in_camp"] = df["camp_name"].notna()
    df.drop(columns="camp_name", inplace=True)

    # train/val split
    X, y = df.drop(columns="target"), df["target"]
    Xtr, Xval, ytr, yval = train_test_split(
        X, y, stratify=y, test_size=0.30, random_state=RANDOM_SEED
    )

    # split into camp / non-camp sets
    camp_tr = Xtr["in_camp"]
    camp_val = Xval["in_camp"]
    Xc_tr, yc_tr   = Xtr[camp_tr].drop(columns="in_camp"),   ytr[camp_tr]
    Xc_val, yc_val = Xval[camp_val].drop(columns="in_camp"), yval[camp_val]
    Xn_tr, yn_tr   = Xtr[~camp_tr].drop(columns="in_camp"),  ytr[~camp_tr]
    Xn_val, yn_val = Xval[~camp_val].drop(columns="in_camp"),yval[~camp_val]

    # strip weight before SHAP
    for _df in (Xc_val, Xn_val):
        _df.drop(columns="survey_weight", inplace=True, errors="ignore")

    # ─── train models ────────────────────────
    print("\n=== CAMP MODEL ===")
    camp_m = train_lgbm(Xc_tr.copy(), yc_tr, Xc_val.copy(), yc_val)

    print("\n=== NON-CAMP MODEL ===")
    non_m  = train_lgbm(Xn_tr.copy(), yn_tr, Xn_val.copy(), yn_val)

    # ─── SHAP tables ────────────────────────
    shap_df = (
        pd.concat(
            [shap_summary(camp_m, Xc_val, "camp"),
             shap_summary(non_m,  Xn_val, "noncamp")],
            axis=1
        ).fillna(0)
    )
    ordered = (shap_df.filter(like="_abs")
                       .max(axis=1)
                       .sort_values(ascending=False)
                       .index)
    out_csv = OUTPUT_DIR / "shap_amenities_enrollment.csv"
    shap_df.loc[ordered].to_csv(out_csv)
    print(f"   → {out_csv}")

    # ─── beeswarm plots ─────────────────────
    save_beeswarm(camp_m, Xc_val, "camp")
    save_beeswarm(non_m,  Xn_val, "noncamp")

    # ─── save models ───────────────────────
    camp_model_path   = MODEL_DIR / "lgbm_enroll_camp.pkl"
    noncamp_model_path = MODEL_DIR / "lgbm_enroll_noncamp.pkl"
    joblib.dump(camp_m,   camp_model_path)
    joblib.dump(non_m,    noncamp_model_path)
    print(f"   → {camp_model_path}")
    print(f"   → {noncamp_model_path}\nDone.")

if __name__ == "__main__":
    main()
