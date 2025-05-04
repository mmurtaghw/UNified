#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
iraq_all_geospatial_only.py
───────────────────────────────────────────────
Train a single LightGBM model to predict regular formal‑school attendance
(target = 1) for school‑age children using ONLY spatial variables:

• OSM amenity counts / distances
• ACLED conflict‑event counts
• Distance to Baghdad

Outputs written under output/iraq_all_data_output/
Model weights written under weights/iraq_all_data/
"""
from pathlib import Path
import numpy as np
import pandas as pd
import shap, joblib, matplotlib.pyplot as plt

from lightgbm               import LGBMClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics         import (
    roc_auc_score, average_precision_score,
    roc_curve, precision_recall_curve
)
from sklearn.utils.class_weight import compute_sample_weight

# ─── ENSURE WEIGHTS & OUTPUT ROOTS EXIST ─────────────────────────────
Path("weights").mkdir(exist_ok=True)
Path("output_iraq").mkdir(exist_ok=True)

# ─── OUTPUT DIRS & FILENAMES ─────────────────────────────────────────
OUTPUT_DIR = Path("output_iraq/iraq_all_data_geospatial_output")
MODEL_DIR  = Path("weights/iraq_all_geospatial_data")
OUTPUT_DIR.mkdir(exist_ok=True)
MODEL_DIR.mkdir(exist_ok=True)

# ─────────────── configuration ─────────────
DATA_FILE   = Path("prepared_data_iraq/UNHCR_IRQ_2022_MCNA_school_age_children_with_spatial_features.csv")
MAX_DISPLAY = 20
RANDOM_SEED = 42

# ───────────── data loader ────────────────
def load_features() -> pd.DataFrame:
    """
    Load the prepared spatial-feature CSV and retain only:
      amenity_*, dist_nearest_*, conflict_*, dist_to_baghdad_m,
      survey_weight, plus target.
    """
    df = pd.read_csv(DATA_FILE)

    # map target
    df["target"] = df["school_regular_attendance_formal"].map({"yes":1,"no":0})
    df = df[df["target"].notna()].reset_index(drop=True)

    # harmonise survey_weight
    df = (
        df.rename(columns={"survey_weight_x":"survey_weight"})
          .drop(columns=[c for c in ["survey_weight_y"] if c in df], errors="ignore")
    )
    if "survey_weight" not in df:
        df["survey_weight"] = 1.0

    # select spatial columns
    keep = [
        c for c in df.columns
        if c.startswith("amenity_")
        or c.startswith("dist_nearest_")
        or c.startswith("conflict_")
    ]
    keep += ["dist_to_baghdad_m", "survey_weight", "target"]

    return df[keep].replace({"yes":1,"no":0,"TRUE":1,"FALSE":0})

# ──────────── modelling utils ─────────────
def train_lgbm(Xtr, ytr, Xval, yval, *, seed=RANDOM_SEED):
    """LightGBM with survey-weight × class-balance weighting + early stopping."""
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

    preds = model.predict_proba(Xval)[:,1]
    print(f"     AUROC = {roc_auc_score(yval, preds):.3f}   "
          f"PR-AUC = {average_precision_score(yval, preds):.3f}")
    return model, preds

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

def plot_curves(y_true, y_score, prefix="unified"):
    # ROC curve
    fpr, tpr, _ = roc_curve(y_true, y_score)
    plt.figure()
    plt.plot(fpr, tpr, label=f'AUC = {roc_auc_score(y_true, y_score):.3f}')
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.legend(loc="lower right")
    plt.tight_layout()
    roc_path = OUTPUT_DIR / f"roc_curve_{prefix}.png"
    plt.savefig(roc_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"   → {roc_path}")

    # Precision-Recall curve
    prec, rec, _ = precision_recall_curve(y_true, y_score)
    plt.figure()
    plt.plot(rec, prec, label=f"AP = {average_precision_score(y_true, y_score):.3f}")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curve")
    plt.legend(loc="lower left")
    plt.tight_layout()
    pr_path = OUTPUT_DIR / f"pr_curve_{prefix}.png"
    plt.savefig(pr_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"   → {pr_path}")

# ─────────────────── main ───────────────────
def main() -> None:
    # 1) load and prepare
    df = load_features()

    # 2) define X, y and split
    X = df.drop(columns="target")
    y = df["target"].astype(int)
    Xtr, Xval, ytr, yval = train_test_split(
        X, y, stratify=y, test_size=0.30, random_state=RANDOM_SEED
    )

    # 3) train unified model & get validation scores
    print("\n=== UNIFIED MODEL ===")
    model, val_preds = train_lgbm(Xtr.copy(), ytr, Xval.copy(), yval)

    # 4) SHAP summary
    Xval_nosw = Xval.drop(columns="survey_weight", errors="ignore")
    shap_df = shap_summary(model, Xval_nosw, "unified") \
                .sort_values("unified_abs", ascending=False)
    out_shap = OUTPUT_DIR / "shap_amenities_enrollment_unified.csv"
    shap_df.to_csv(out_shap)
    print(f"   → {out_shap}")

    # 5) beeswarm
    save_beeswarm(model, Xval_nosw, "unified")

    # 6) ROC & PR curves
    plot_curves(yval, val_preds, prefix="unified")

    # 7) save model
    model_path = MODEL_DIR / "lgbm_enroll_unified.pkl"
    joblib.dump(model, model_path)
    print(f"   → {model_path}\nDone.")

if __name__ == "__main__":
    main()
