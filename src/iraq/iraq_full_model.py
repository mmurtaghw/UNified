#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
iraq_full_model.py

Predict formal-education enrolment (positive = enrolled) for school-age children,
with separate LightGBM models for camp vs. non-camp households, SHAP explainability,
and per-category SHAP-value summaries.

All CSVs/PNGs → iraq_full_model_output/
All .pkl models → iraq_model/
"""
from pathlib import Path
import numpy as np
import pandas as pd
import shap
import joblib
import matplotlib.pyplot as plt

from lightgbm import LGBMClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    roc_auc_score, average_precision_score,
    roc_curve, precision_recall_curve
)
from sklearn.utils.class_weight import compute_sample_weight

# ─── CONFIG ────────────────────────────────────────────────────────────────
DATA_FILE       = Path("prepared_data_iraq/UNHCR_IRQ_2022_MCNA_school_age_children_with_spatial_features.csv")
CAMP_ONLY_DROP  = ["days_since_arrival", "days_since_return"]
TE_COLS         = ["district_mcna", "relationship", "marital_status"]
MAX_DISPLAY     = 20

Path("weights").mkdir(exist_ok=True)
Path("output_iraq").mkdir(exist_ok=True)
# ─── OUTPUT DIRS & FILENAMES ───────────────────────────────────────────────
OUTPUT_DIR = Path("output_iraq/iraq_full_model_output")
MODEL_DIR  = Path("weights/iraq_model_full")
OUTPUT_DIR.mkdir(exist_ok=True)
MODEL_DIR.mkdir(exist_ok=True)

SHAP_SUMMARY_CSV        = OUTPUT_DIR / "enrollment_shap_summary.csv"
SHAP_BY_CAT_CAMP_CSV    = OUTPUT_DIR / "enrollment_shap_by_category_camp.csv"
SHAP_BY_CAT_NONCAMP_CSV = OUTPUT_DIR / "enrollment_shap_by_category_noncamp.csv"
BEESWARM_CAMP_PNG       = OUTPUT_DIR / "camp_enrollment_shap_beeswarm.png"
BEESWARM_NONCAMP_PNG    = OUTPUT_DIR / "noncamp_enrollment_shap_beeswarm.png"
ROC_PR_PREFIX_CAMP      = OUTPUT_DIR / "camp_enrollment"
ROC_PR_PREFIX_NONCAMP   = OUTPUT_DIR / "noncamp_enrollment"
MODEL_FILE_CAMP         = MODEL_DIR  / "camp_enrollment_model.pkl"
MODEL_FILE_NONCAMP      = MODEL_DIR  / "noncamp_enrollment_model.pkl"


# ─── PRETTY-LABEL HELPERS ──────────────────────────────────────────────────
def prettify_col(name: str) -> str:
    """
    Turn a snake-case feature name into a human-friendly label.
    """
    # nearest-X distances
    if name.startswith("dist_nearest_") and name.endswith("_m"):
        feat = name[len("dist_nearest_"):-2].replace("_", " ")
        return f"Nearest {feat} distance (m)"
    # generic distance_
    if name.startswith("distance_"):
        feat = name[len("distance_"):].replace("_", " ")
        return f"Distance to {feat} (m)"
    # conflicts
    if name.startswith("conflict_") and "_within_" in name:
        kinds, kms = name[len("conflict_"):].split("_within_")
        return f"{kinds.replace('_',' ').title()} events within {kms.replace('km',' km')}"
    # amenities
    if name.startswith("amenity_") and "_within_" in name:
        amen, kms = name[len("amenity_"):].split("_within_")
        return f"{amen.replace('_',' ').title()} within {kms.replace('km',' km')}"
    # Baghdad dist
    if name == "dist_to_baghdad_m":
        return "Distance to Baghdad (m)"
    # latitude/longitude
    if name == "lat":
        return "Latitude"
    if name == "lon":
        return "Longitude"
    # age bounds
    if name == "age_low":
        return "Age lower bound (years)"
    if name == "age_high":
        return "Age upper bound (years)"
    # expenditures
    if name == "exp_total":
        return "Total expenditure"
    if name == "exp_per_cap":
        return "Expenditure per capita"
    if name.endswith("_exp"):
        return name.replace("_exp", " expenditure").replace("_"," ").title()
    # logs
    if name.endswith("_log"):
        base = name[:-4].replace("_", " ")
        return f"{base.title()} (log1p)"
    # dependency
    if name == "dependency_ratio":
        return "Dependency ratio"
    # household size & debt
    if name == "num_hh_member":
        return "Household size"
    if name == "how_much_debt":
        return "Debt amount"
    # boolean flags → title case
    if name in {"child_work","no_food","sex"}:
        return name.replace("_"," ").title()
    # fallback
    return name.replace("_", " ").title()


def rename_for_plot(df: pd.DataFrame) -> pd.DataFrame:
    """Rename columns of df according to prettify_col(), leaving order intact."""
    return df.rename(columns={c: prettify_col(c) for c in df.columns})


# ─── THE REST OF YOUR SCRIPT (UNCHANGED) ──────────────────────────────────
def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    if "survey_weight_x" in df:
        df = df.rename(columns={"survey_weight_x": "survey_weight"})
    df = df.drop(columns=["survey_weight_y"], errors="ignore")
    drop_ids = [
        "id_ind","id","camp_last_updated","arrival_date_idp_camp","return_date_returnee",
        "age_mid","date_assessment","drop_out",
        "cereals","nuts_seed","milk_dairy","meat_fish_eggs","vegetables","fruits",
        "oil_fats","sweets","spices_condiments"
    ]
    df = df.drop(columns=[c for c in drop_ids if c in df], errors="ignore")

    num_cols = [
        "survey_weight","lat","lon","capacity","current_population",
        "rent_exp","food_exp","medical_exp","days_since_arrival","days_since_return",
        "child_distress_number","distance_clinic","distance_hospital","num_hh_member"
    ]
    for c in num_cols:
        if c in df:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    bool_cols = [
        "displaced_again","r_n_a_cannot_affort_school_expen","r_n_a_distance_no_resources",
        "r_n_a_school_not_safe","child_dropout_school","selling_assets","borrow_debt",
        "pri_live_regular_employment","pri_live_irregular_employment","type_assistance_cash",
        "type_assistance_food","sufficient_water_drinking","sufficient_water_hygiene",
        "access_soap","handwashing_facility","no_food","hungry","informal_site",
        "encl_iss_leaks_light_rain","encl_iss_limited_ventilation","hh_member_distress",
        "health_needs_consult","health_needs_trauma","child_work_cs"
    ]
    for c in bool_cols:
        if c in df:
            df[c] = df[c].map({"yes":1,"no":0}).astype("boolean")

    cat_cols = [
        "district_mcna","camp_name","relationship","marital_status","sex",
        "difficulty_seeing","difficulty_hearing","difficulty_walking",
        "difficulty_remembering","difficulty_selfcare","difficulty_communicating",
        "health_issue","child_work_type","shelter_type_area_assessed","displace_status"
    ]
    for c in cat_cols:
        if c in df:
            df[c] = df[c].astype("category")

    if {"rent_exp","food_exp","medical_exp","num_hh_member"}.issubset(df.columns):
        df["exp_total"]   = df[["rent_exp","food_exp","medical_exp"]].sum(axis=1)
        df["exp_per_cap"] = df["exp_total"] / (df["num_hh_member"] + 1)
        for base in ["rent_exp","food_exp","medical_exp","exp_total"]:
            df[f"{base}_log"] = np.log1p(df[base])
    if {"child_distress_number","num_hh_member"}.issubset(df.columns):
        df["dependency_ratio"] = df["child_distress_number"] / (df["num_hh_member"] + 1)

    drop_txt = [c for c,d in df.dtypes.items() if d == "object"]
    return df.drop(columns=drop_txt, errors="ignore")


def target_encode(col, X_tr, y_tr, X_val, smoothing=30):
    prior = y_tr.mean()
    stats = (
        pd.concat([X_tr[col], y_tr.rename("t")], axis=1)
          .groupby(col, observed=False)["t"].agg(["count","mean"])
    )
    stats["te"] = (stats["count"]*stats["mean"] + smoothing*prior) / (stats["count"]+smoothing)

    # cast to object so fillna can insert floats
    enc_tr  = X_tr[col].astype(object).map(stats["te"]).fillna(prior).astype(float)
    enc_val = X_val[col].astype(object).map(stats["te"]).fillna(prior).astype(float)

    return enc_tr, enc_val



def shap_summary(model, X_val, label):
    explainer = shap.TreeExplainer(model)
    sv        = explainer.shap_values(X_val)
    arr       = sv[1] if isinstance(sv, list) else sv
    return pd.DataFrame(
        {f"{label}_signed": arr.mean(0),
         f"{label}_abs":    np.abs(arr).mean(0)},
        index=X_val.columns
    )


def shap_by_category(shap_vals, feat_names, raw_cats):
    df_sh = pd.DataFrame(shap_vals, columns=feat_names, index=raw_cats.index)
    parts = []
    for col in raw_cats.columns:
        te_col = f"{col}_te"
        tmp    = pd.concat([raw_cats[col], df_sh[te_col]], axis=1)
        tmp.columns = [col, "shap_value"]
        parts.append(tmp.groupby(col)["shap_value"].mean().rename(f"{te_col}_meanSHAP"))
    return pd.concat(parts, axis=1)


def train_lgbm(X_tr, y_tr, X_val, y_val, *,
               name: str, roc_pr_prefix: Path):
    sw = X_tr.pop("survey_weight").astype(float).values
    sw *= compute_sample_weight("balanced", y_tr)
    X_val = X_val.drop(columns=["survey_weight"], errors="ignore")

    model = LGBMClassifier(
        objective="binary", n_estimators=5000, learning_rate=0.05,
        num_leaves=31, subsample=0.8, colsample_bytree=0.8,
        random_state=42, early_stopping_rounds=100
    )
    model.fit(
        X_tr, y_tr,
        sample_weight=sw,
        eval_set=[(X_val, y_val)],
        eval_metric="auc"
    )

    preds = model.predict_proba(X_val)[:,1]
    auc   = roc_auc_score(y_val, preds)
    ap    = average_precision_score(y_val, preds)
    print(f"  {name}: AUROC={auc:.3f}, PR-AUC={ap:.3f}")

    fpr, tpr, _ = roc_curve(y_val, preds)
    plt.figure()
    plt.plot(fpr, tpr, label=f"AUC={auc:.3f}")
    plt.plot([0,1],[0,1],"--",linewidth=1)
    plt.xlabel("False Positive Rate"); plt.ylabel("True Positive Rate")
    plt.title(f"{name} ROC Curve")
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(f"{roc_pr_prefix}_roc_curve.png", dpi=300)
    plt.close()

    precision, recall, _ = precision_recall_curve(y_val, preds)
    plt.figure()
    plt.plot(recall, precision, label=f"AP={ap:.3f}")
    plt.xlabel("Recall"); plt.ylabel("Precision")
    plt.title(f"{name} Precision–Recall")
    plt.legend(loc="lower left")
    plt.tight_layout()
    plt.savefig(f"{roc_pr_prefix}_pr_curve.png", dpi=300)
    plt.close()

    return model


def save_beeswarm(shap_vals, X_val, out_file: Path, title: str):
    """
    Rename columns for readability, then plot the SHAP beeswarm.
    """
    X_pretty = rename_for_plot(X_val)
    shap.summary_plot(
        shap_vals, X_pretty,
        plot_type="dot", max_display=MAX_DISPLAY,
        show=False, color_bar=True
    )
    plt.title(title, fontsize=14)
    plt.tight_layout()
    plt.savefig(out_file, dpi=300, bbox_inches="tight")
    plt.close()


def main():
    # 1) load + target
    df = pd.read_csv(DATA_FILE)
    df["target"] = df["school_regular_attendance_formal"].map({"yes":1,"no":0})
    df = df.dropna(subset=["target"]).reset_index(drop=True)

    # 2) preprocess
    df = preprocess(df)

    # 3) split
    X = df.drop(columns=["target"])
    y = df["target"]
    X_tr, X_val, y_tr, y_val = train_test_split(
        X, y, stratify=y, test_size=0.3, random_state=42
    )

    # 4) camp subgroups
    X_tr["in_camp"]  = X_tr["camp_name"].notna()
    X_val["in_camp"] = X_val["camp_name"].notna()

    mask_tr_c   = X_tr["in_camp"]
    mask_val_c  = X_val["in_camp"]

    Xc_tr, yc_tr   = X_tr[ mask_tr_c].copy(), y_tr[ mask_tr_c].reset_index(drop=True)
    Xc_val, yc_val = X_val[mask_val_c].copy(), y_val[mask_val_c].reset_index(drop=True)
    Xnc_tr, ync_tr   = X_tr[~mask_tr_c].copy(), y_tr[~mask_tr_c].reset_index(drop=True)
    Xnc_val, ync_val = X_val[~mask_val_c].copy(), y_val[~mask_val_c].reset_index(drop=True)

    Xc_val_cats  = Xc_val[TE_COLS].copy()
    Xnc_val_cats = Xnc_val[TE_COLS].copy()

    # 5) drop camp-only
    for subset,is_camp in [(Xc_tr,True),(Xc_val,True),(Xnc_tr,False),(Xnc_val,False)]:
        subset.drop(columns=CAMP_ONLY_DROP + (["camp_name"] if is_camp else []),
                    errors="ignore", inplace=True)

    # 6) target-encode
    for col in TE_COLS:
        if col in Xc_tr:
            tr,val = target_encode(col, Xc_tr, yc_tr, Xc_val)
            Xc_tr[f"{col}_te"], Xc_val[f"{col}_te"] = tr, val
            Xc_tr.drop(columns=[col], inplace=True)
            Xc_val.drop(columns=[col], inplace=True)
        if col in Xnc_tr:
            tr,val = target_encode(col, Xnc_tr, ync_tr, Xnc_val)
            Xnc_tr[f"{col}_te"], Xnc_val[f"{col}_te"] = tr, val
            Xnc_tr.drop(columns=[col], inplace=True)
            Xnc_val.drop(columns=[col], inplace=True)

    # 7) train camp model
    print("\n=== CAMP MODEL ===")
    camp_model = train_lgbm(
        Xc_tr.reset_index(drop=True), yc_tr,
        Xc_val.reset_index(drop=True), yc_val,
        name="Camp Enrollment", roc_pr_prefix=ROC_PR_PREFIX_CAMP
    )

    # 8) train non-camp
    print("\n=== NON-CAMP MODEL ===")
    noncamp_model = train_lgbm(
        Xnc_tr.reset_index(drop=True), ync_tr,
        Xnc_val.reset_index(drop=True), ync_val,
        name="Non-Camp Enrollment", roc_pr_prefix=ROC_PR_PREFIX_NONCAMP
    )

    # 9) SHAP summary
    print("\n→ Computing SHAP summary tables…")
    Xc_nosw  = Xc_val.drop(columns=["survey_weight"], errors="ignore")
    Xnc_nosw = Xnc_val.drop(columns=["survey_weight"], errors="ignore")
    shap_c   = shap_summary(camp_model,   Xc_nosw,  "camp")
    shap_nc  = shap_summary(noncamp_model, Xnc_nosw, "noncamp")
    shap_df  = pd.concat([shap_c, shap_nc], axis=1).fillna(0)
    abs_cols = [c for c in shap_df if c.endswith("_abs")]
    shap_df  = shap_df.loc[shap_df[abs_cols].max(axis=1).sort_values(ascending=False).index]
    shap_df.to_csv(SHAP_SUMMARY_CSV)
    print(f"  • SHAP summary → {SHAP_SUMMARY_CSV}")

    # 10) SHAP by-category
    print("→ Computing SHAP-by-category…")
    arr_c  = shap.TreeExplainer(camp_model).shap_values(Xc_nosw)
    arr_c  = arr_c[1] if isinstance(arr_c, list) else arr_c
    arr_nc = shap.TreeExplainer(noncamp_model).shap_values(Xnc_nosw)
    arr_nc = arr_nc[1] if isinstance(arr_nc, list) else arr_nc

    cats_camp    = shap_by_category(arr_c,  Xc_nosw.columns,  Xc_val_cats)
    cats_noncamp = shap_by_category(arr_nc, Xnc_nosw.columns, Xnc_val_cats)
    cats_camp.to_csv(SHAP_BY_CAT_CAMP_CSV)
    cats_noncamp.to_csv(SHAP_BY_CAT_NONCAMP_CSV)
    print(f"  • SHAP by-category saved")

    # 11) SHAP beeswarm
    print("→ Generating SHAP beeswarm plots…")
    save_beeswarm(arr_c,  Xc_nosw,  BEESWARM_CAMP_PNG,    "Camp Model: Top 20 SHAP Features")
    save_beeswarm(arr_nc, Xnc_nosw, BEESWARM_NONCAMP_PNG, "Non-Camp Model: Top 20 SHAP Features")
    print(f"  • Beeswarm PNGs → {BEESWARM_CAMP_PNG}, {BEESWARM_NONCAMP_PNG}")

    # 12) save models
    joblib.dump(camp_model,   MODEL_FILE_CAMP)
    joblib.dump(noncamp_model, MODEL_FILE_NONCAMP)
    print("→ Models written.")
    print("Done.")


if __name__ == "__main__":
    main()
