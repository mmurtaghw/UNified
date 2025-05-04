#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
unified_uganda.py
────────────────────────────────
Train a brand-new LGBM classifier on the **entire** Uganda dataset—
household + panel rows, with spatial features.

Outputs under output/uga_full_model_output/ and model weights under weights/uga_model_full/:
• lgbm_enroll_uga_unified.pkl
• shap_amenities_enrollment_uga_unified.csv
• shap_beeswarm_unified_top20.png
• roc_curve_unified.png
• pr_curve_unified.png
• shap_values_full.csv
"""

from pathlib import Path
import numpy as np
import pandas as pd
import shap, joblib, matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedKFold, train_test_split
from lightgbm import LGBMClassifier
from sklearn.metrics import (
    roc_auc_score, average_precision_score,
    roc_curve, precision_recall_curve, auc
)
from sklearn.utils.class_weight import compute_sample_weight

# ─── make sure our folders exist ─────────────────────────────
Path("weights").mkdir(exist_ok=True)
Path("output").mkdir(exist_ok=True)

# ─── OUTPUT DIRS & FILENAMES ─────────────────────────────────
OUTPUT_DIR = Path("output/uga_full_model_output")
MODEL_DIR  = Path("weights/uga_model_full")
OUTPUT_DIR.mkdir(exist_ok=True)
MODEL_DIR.mkdir(exist_ok=True)

# ─── configuration ────────────────────────────────────────────
N_FOLDS     = 5
DATA_FILE   = Path("prepared_data_uganda/UNHCR_UGA_2018_spatial_features.csv")
MAX_DISPLAY = 20
RANDOM_SEED = 42

def load_features(tau: float = 1000.0) -> pd.DataFrame:
    df = pd.read_csv(DATA_FILE, low_memory=False)

    # map in_school → target
    df["target"] = (
        pd.to_numeric(df["in_school"], errors="coerce")
          .fillna(0).astype(int).clip(0,1)
    )

    df = df[df["target"].notna()].reset_index(drop=True)

    # survey_weight
    if "normalized_weight" in df:
        df.rename(columns={"normalized_weight":"survey_weight"}, inplace=True)
    elif "raw_weight" in df:
        df.rename(columns={"raw_weight":"survey_weight"}, inplace=True)
    else:
        df["survey_weight"] = 1.0
    df["survey_weight"] = pd.to_numeric(df["survey_weight"], errors="coerce").fillna(1.0)
    df["is_refugee"] = df["is_refugee"].astype(int)

    # select core features
    keep = [c for c in df.columns if (
        c.startswith("amenity_")
        or c.startswith("dist_nearest_")
        or c.startswith("conflict_")
        or c == "dist_to_kampala_m"
    )]
    keep += ["is_refugee","survey_weight","target"]

    df = df[keep].copy()
    to_num = df.columns.difference(["survey_weight"])
    df[to_num] = df[to_num].apply(pd.to_numeric, errors="coerce")

    # spatial transforms
    new_feats = {}
    for c in df.columns:
        if c.startswith("dist_nearest_"):
            new_feats[f"log_{c}"]   = np.log1p(df[c])
            new_feats[f"inv_{c}"]   = 1.0 / (1.0 + df[c])
            new_feats[f"gauss_{c}"] = np.exp(-df[c] / tau)
    amen2 = [c for c in df if c.startswith("amenity_") and c.endswith("_2km")]
    new_feats["amenity_count_2km"] = df[amen2].sum(axis=1)
    conf_cols = [c for c in df if c.startswith("conflict_")]
    new_feats["conflict_exposure"] = df[conf_cols].sum(axis=1)

    combos = [
        ("amenity_school_within_2km","amenity_clinic_within_2km"),
        ("amenity_school_within_2km","amenity_hospital_within_2km"),
        ("amenity_clinic_within_2km","amenity_hospital_within_2km"),
    ]
    for a,b in combos:
        if a in df.columns and b in df.columns:
            name = f"{a}_{b.split('_')[-2]}"
            new_feats[name] = df[a].fillna(0) * df[b].fillna(0)

    df = pd.concat([df, pd.DataFrame(new_feats, index=df.index)], axis=1)

    # legacy
    if "dist_to_kampala_m" in df.columns:
        df["dist_to_baghdad_m"] = df["dist_to_kampala_m"]

    return df

def plot_roc_pr(y_true, y_score, tag: str):
    # ROC
    fpr, tpr, _ = roc_curve(y_true, y_score)
    roc_auc = auc(fpr, tpr)
    plt.figure()
    plt.plot(fpr, tpr)
    plt.plot([0,1],[0,1],"--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"ROC Curve ({tag}) — AUC={roc_auc:.3f}")
    plt.tight_layout()
    roc_path = OUTPUT_DIR / f"roc_curve_{tag}.png"
    plt.savefig(roc_path, dpi=300)
    plt.close()

    # PR
    prec, rec, _ = precision_recall_curve(y_true, y_score)
    pr_auc = auc(rec, prec)
    plt.figure()
    plt.plot(rec, prec)
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title(f"Precision–Recall Curve ({tag}) — AUC={pr_auc:.3f}")
    plt.tight_layout()
    pr_path = OUTPUT_DIR / f"pr_curve_{tag}.png"
    plt.savefig(pr_path, dpi=300)
    plt.close()

def shap_summary(model, Xval, label):
    expl = shap.TreeExplainer(model)
    sv   = expl.shap_values(Xval)
    arr  = sv[1] if isinstance(sv, list) else sv
    df   = pd.DataFrame({
        f"{label}_signed": arr.mean(0),
        f"{label}_abs":    np.abs(arr).mean(0)
    }, index=Xval.columns)
    out_csv = OUTPUT_DIR / f"shap_amenities_enrollment_uga_{label}.csv"
    df.sort_values(f"{label}_abs", ascending=False).to_csv(out_csv)
    print(f"   → {out_csv}")
    return df

def save_beeswarm(model, Xval, tag):
    expl = shap.TreeExplainer(model)
    sv   = expl.shap_values(Xval)
    arr  = sv[1] if isinstance(sv, list) else sv
    shap.summary_plot(arr, Xval, plot_type="dot", max_display=MAX_DISPLAY, show=False)
    plt.title(f"Unified model – top {MAX_DISPLAY}")
    plt.tight_layout()
    bs_path = OUTPUT_DIR / f"shap_beeswarm_{tag}_top{MAX_DISPLAY}.png"
    plt.savefig(bs_path, dpi=300)
    plt.close()
    print(f"   → {bs_path}")

def cross_validate_and_train(X, y, sw):
    from sklearn.model_selection import StratifiedKFold
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_SEED)
    best_iters=[]
    for fold,(tr,va) in enumerate(skf.split(X,y),1):
        Xtr,Xva = X.iloc[tr],X.iloc[va]
        ytr,yva = y.iloc[tr],y.iloc[va]
        swtr    = sw[tr]
        m = LGBMClassifier(n_estimators=3000,learning_rate=0.03,num_leaves=64,
                           subsample=0.8,colsample_bytree=0.8,
                           early_stopping_rounds=30,random_state=RANDOM_SEED)
        m.fit(Xtr,ytr,sample_weight=swtr,eval_set=[(Xva,yva)],eval_metric="auc")
        best_iters.append(m.best_iteration_)
        print(f"Fold {fold}: AUROC={roc_auc_score(yva,m.predict_proba(Xva)[:,1]):.3f}")
    avg_iter = max(1,int(np.mean(best_iters)))
    print(f"Retraining on full data with n_estimators={avg_iter}")
    final = LGBMClassifier(n_estimators=avg_iter,learning_rate=0.03,
                           num_leaves=64,subsample=0.8,colsample_bytree=0.8,
                           random_state=RANDOM_SEED)
    final.fit(X,y,sample_weight=sw)
    model_path = MODEL_DIR / "lgbm_enroll_uga_unified.pkl"
    joblib.dump(final, model_path)
    print(f"   → {model_path}")
    return final

def main():
    df = load_features()
    X  = df.drop(columns="target")
    y  = df["target"]
    sw = compute_sample_weight(class_weight="balanced", y=y)

    model = cross_validate_and_train(X, y, sw)

    # hold‑out metrics & SHAP
    Xtr, Xva, ytr, yva = train_test_split(X,y,stratify=y,test_size=0.3,random_state=RANDOM_SEED)
    preds = model.predict_proba(Xva)[:,1]
    plot_roc_pr(yva, preds, "unified")
    shap_summary(model, Xva, "unified")
    save_beeswarm(model, Xva, "unified")

    # full‑data SHAP matrix
    expl = shap.TreeExplainer(model)
    sv   = expl.shap_values(X)
    arr  = sv[1] if isinstance(sv,list) else sv
    full_shap = pd.DataFrame(arr, columns=X.columns)
    full_path = OUTPUT_DIR / "shap_values_full.csv"
    full_shap.to_csv(full_path, index=False)
    print(f"   → {full_path}")

if __name__=="__main__":
    main()
