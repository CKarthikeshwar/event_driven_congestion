#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
====================================================================
PHASE 1 : Baseline Models  (CatBoost backend)
====================================================================
Reads pipeline_out/modeling_table.pkl (from clean.py) and trains:

  A) PER-EVENT TRIAGE
       - priority_high        -> classifier
       - requires_rerouting   -> classifier  (balanced: ~8% positive)
       - duration_min         -> regressor   (valid rows only)
     CatBoost native categoricals: cause, corridor, zone, station, veh_type.
     No OrdinalEncoder needed.

  B) SPATIO-TEMPORAL FORECASTER
       - police_station x 3h panel, zero-filled
       - targets: event count + impact_sum (severity)
       - features: calendar + lag_1/lag_8/lag_56 + roll_24h/roll_7d + area
       - area column passed as native CatBoost categorical
       - benchmarked against same-time-last-week naive baseline

All splits TIME-BASED (train past, test future).

NOTE: decisions.py uses the fitted_models.pkl saved here.
      The storage format changed from (model, enc, names)
      to (model, cat_cols, num_cols) for triage and
      (model, (FEATS, AREA_COL)) for the forecaster.
      Update score_triage() and expected_load_surface() in decisions.py
      accordingly (see comments at end of this file).

Usage:   python models.py
Requires: pandas, numpy, catboost, scikit-learn, matplotlib
"""
import os, pickle, warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from catboost import CatBoostClassifier, CatBoostRegressor
from sklearn.metrics import (roc_auc_score, average_precision_score, f1_score, accuracy_score, mean_absolute_error, mean_squared_error, r2_score)

warnings.filterwarnings("ignore")
OUTDIR   = "pipeline_out"
AREA_COL = "police_station"
FREQ     = "3h"
_BACKEND = "catboost"
fig_n    = 0

os.makedirs(OUTDIR, exist_ok=True)

# saves a figure to pipeline_out with sequential numbering
def save(fig, name):
    global fig_n; fig_n += 1 # fig_n is a global counter for sequentially numbered figures
    p = os.path.join(OUTDIR, f"model_{fig_n:02d}_{name}.png")
    fig.savefig(p, dpi=110, bbox_inches="tight"); plt.close(fig)
    print(f"   saved {p}")

# ──────────────────────────────────────────────────────────────────────
# Model factories
# ──────────────────────────────────────────────────────────────────────

# iterations - maximum number of trees to build
# learning_rate - % each tree contributes to the final prediction (smaller lr usually need more trees but generalize better)
# depth - maximum depth of each tree (higher depth can capture more complex patterns but may overfit)
# verbose=0 - suppresses training output (verbose = 100 would print every 100 iterations)

# suppose we have a binary classification problem where the positive class is rare (e.g., 8% of the data).
# In such cases, we can use the auto_class_weights="Balanced" parameter to automatically adjust the weights of the classes during training.
# This helps the model pay more attention to the minority class and improve its performance on imbalanced datasets. 
# This improves recal, f1 score and detectionf of rare events at the expense of some accuracy.

# creates and returns a CatBoostClassifier with default hyperparameters
def make_clf():
    return CatBoostClassifier(iterations=400, learning_rate=0.05, depth=6, verbose=100, use_best_model=True)


# same thing but for imbalanced datasets
def make_clf_balanced():
    # auto_class_weights handles the ~8% positive rate in requires_rerouting
    return CatBoostClassifier(iterations=400, learning_rate=0.05, depth=6, verbose=100, auto_class_weights="Balanced", use_best_model=True)

# creates a CatBoostRegressor which is used when predicting continous values
def make_reg():
    return CatBoostRegressor(iterations=500, learning_rate=0.05, depth=6, verbose=100, use_best_model=True)

# ──────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────

# extractsfeature importances from a trained CatBoost model and returns them as a pandas Series sorted by importance
def importances(model, names):
    """CatBoost always exposes feature_importances_ — no permutation fallback needed."""
    return pd.Series(model.feature_importances_, index=names).sort_values()

# creates a horizontal bar plot of feature importances and saves it to a file
def plot_importance(imp, title, fname):
    fig, ax = plt.subplots(figsize=(8, max(3, 0.4 * len(imp)))) # figure size is dynamic based on number of features
    ax.barh(imp.index, imp.values, color="#534AB7")
    ax.set_title(title); ax.set_xlabel("importance")
    save(fig, fname) # the save function defined earlier``

# splits a DataFrame into training and testing sets based on time
# we shouldn't use train_test_split beause for time series data, we want to train on the past and predict the future
def time_split(df, frac=0.8):
    df = df.sort_values("occurrence_ts") # sort incidents chronologically
    k  = int(len(df) * frac) # split index based on fraction (default 80% train, 20% test)
    return df.iloc[:k].copy(), df.iloc[k:].copy() # returns two DataFrames after splitting 

def build_X(df, cat_cols, num_cols):
    """Build a DataFrame CatBoost can consume natively.
    Cats as str (CatBoost reads raw strings), nums as float, NaNs filled."""
    return pd.concat([
        df[cat_cols].astype(str).fillna("NA"), # converts categorical columns to string and fills NaNs with "NA"
        df[num_cols].astype(float).fillna(-1) # converts numerical columns to float and fills NaNs with -1
    ], axis=1)

# ══════════════════════════════════════════════════════════════════════
# A) PER-EVENT TRIAGE
# ══════════════════════════════════════════════════════════════════════

# trains 3 models: priority_high (classification), requires_rerouting (classification), duration_min (regression)
def triage(tbl): # tbl is the modeling table loaded from pipeline_out/modeling_table.pkl
    print("\n" + "=" * 60 + f"\nA) PER-EVENT TRIAGE   (backend: {_BACKEND})\n" + "=" * 60)

    CAT = ["cause", "event_type", "corridor_base", "zone", "police_station", "veh_type"]
    NUM = ["hour", "dow", "month", "is_weekend", "latitude", "longitude", "concurrent_load"]
    # keep only the coloumns that exist 
    CAT = [c for c in CAT if c in tbl.columns] 
    NUM = [c for c in NUM if c in tbl.columns]

    df = tbl.dropna(subset=["occurrence_ts"]).copy() # remove rows without occurence timestamps
    df["is_weekend"] = df["is_weekend"].astype(float) # convert Bool to Float for CatBoost
    fitted = {} # dictionary to store the fitted models and their corresponding categorical and numerical columns

    # ── binary classifiers ────────────────────────────────────────────
    for target in ["priority_high", "requires_rerouting"]: # trains 2 separate classifiersa
        if target not in df.columns:
            continue
        d = df[df[target].notna()].copy() # remove rows where the target variable is NaN
        d[target] = d[target].astype(int) # convert the target variable to integer (0 or 1)
        if d[target].nunique() < 2: # ensure that there are at least 2 classes
            print(f"\n[{target}] only one class — skipped."); continue

        tr, te  = time_split(d) # split the data using the function we defined earlier
        # build feature matrices using our build_X function and then extract the target
        Xtr     = build_X(tr, CAT, NUM)
        Xte     = build_X(te, CAT, NUM)
        ytr, yte = tr[target].values, te[target].values

        # choose the classifier and train 
        m = make_clf_balanced() if target == "requires_rerouting" else make_clf()
        m.fit(Xtr, ytr, cat_features=CAT, eval_set=(Xte, yte), early_stopping_rounds=200) # early stopping to prevent overfitting, using the test set as the evaluation set for early stopping
        print("Best iteration for " + target + ":", m.get_best_iteration()) # finds the best iteration
 
        # predict probabilities and convert to labels using a threshold of 0.5
        proba = m.predict_proba(Xte)[:, 1]
        pred  = (proba >= 0.5).astype(int)

        # evaluation 
        print(f"\n[{target}]  train={len(tr)} test={len(te)}" f"  positive rate={yte.mean():.2f}")
        if len(np.unique(yte)) == 2:
            print(f"   ROC-AUC {roc_auc_score(yte, proba):.3f} | "
                   # ROC-AUC (Receiver Operating Characteristic - Area Under Curve) measures the model's ability to distinguish between classes. It ranges from 0 to 1, where 1 indicates perfect classification and 0.5 indicates random guessing.
                  f"PR-AUC  {average_precision_score(yte, proba):.3f} | " 
                  # Precision-Recall AUC (Area Under the Precision-Recall Curve) measures the model's ability to rank positive instances higher than negative ones. It ranges from 0 to 1, where 1 indicates perfect ranking and 0.5 indicates random ranking. It is more imp for rare-event detection than ROC-AUC.
                  f"F1 {f1_score(yte, pred):.3f} | " 
                  
                  # F1 score is the harmonic mean of precision(TP/TP+FP) and recall(TP/TP+FN), providing a balance between the two metrics. It ranges from 0 to 1, where 1 indicates perfect precision and recall, and 0 indicates the worst performance.
                  f"Acc {accuracy_score(yte, pred):.3f}") # correct predictions / total predictions

        # plot feature importances
        plot_importance(importances(m, CAT + NUM), f"Triage importance: {target}", f"triage_{target}")

        # Store (model, cat_cols, num_cols)
        # decisions.py uses this triple to reconstruct the feature matrix
        fitted[target] = (m, CAT, NUM)

    # ── duration regressor ────────────────────────────────────────────
    if "duration_valid" in df.columns and df["duration_valid"].sum() > 50: # need at least 50 valid rows
        d        = df[df["duration_valid"]].copy() # keep only valid rows
        tr, te   = time_split(d)
        Xtr      = build_X(tr, CAT, NUM)
        Xte      = build_X(te, CAT, NUM)
        ytr, yte = tr["duration_min"].values, te["duration_min"].values

        m    = make_reg() # create a regression model
        m.fit(Xtr, ytr, cat_features=CAT, eval_set=(Xte, yte), early_stopping_rounds=200) # train the model
        print("Best iteration for duration_min:", m.get_best_iteration())
        pred = m.predict(Xte) # predict
        base = np.full_like(yte, np.median(ytr)) # naive baseline: predict the median of the training set for all test samples

        print(f"\n[duration_min]  train={len(tr)} test={len(te)} (valid rows only)")
        print(f"   MAE {mean_absolute_error(yte, pred):.1f} min  "
              f"(naive-median {mean_absolute_error(yte, base):.1f}) | " # MAE wrt naive baseline
              f"R2 {r2_score(yte, pred):.3f}") # measures how much variance is expained (1 is perfect, 0 means no better than predicting the mean, negative means worse than predicting the mean)
        plot_importance(importances(m, CAT + NUM), "Triage importance: duration", "triage_duration")
        fitted["duration_min"] = (m, CAT, NUM) # store model 
    else:
        print("\n[duration_min] too few valid rows — skipping.")

    return fitted

# ══════════════════════════════════════════════════════════════════════
# B) SPATIO-TEMPORAL FORECASTER
# ══════════════════════════════════════════════════════════════════════
def forecaster(tbl):
    print("\n" + "=" * 60 + f"\nB) SPATIO-TEMPORAL FORECASTER   (backend: {_BACKEND})\n" + "=" * 60)

    ev = tbl.dropna(subset=["occurrence_ts", AREA_COL]).copy() # remove rows without occurence timestamps or area information
    ev["tbin"] = ev["occurrence_ts"].dt.floor(FREQ) # create time bins by flooring the occurrence timestamps to the nearest 3 hours (or whatever FREQ is set to)
    # aggregate event counts and impact scores by area and time bin
    agg = (ev.groupby([AREA_COL, "tbin"]).agg(count=("id", "size"), impact_sum=("impact_score", "sum")).reset_index()) 

    # zero-fill the full area × time grid
    areas = agg[AREA_COL].unique() # get all areas
    tidx  = pd.date_range(agg["tbin"].min(), agg["tbin"].max(), freq=FREQ) # get all time bins
    # create a MultiIndex grid of all area-time combinations and convert it to a DataFrame
    grid  = (pd.MultiIndex.from_product([areas, tidx], names=[AREA_COL, "tbin"]).to_frame(index=False))
    # merge the aggregated counts and impact scores onto the grid, filling missing values with 0
    panel = grid.merge(agg, on=[AREA_COL, "tbin"], how="left")
    panel[["count", "impact_sum"]] = panel[["count", "impact_sum"]].fillna(0.0)

    # lag / rolling — all in bin units (1 bin = 3h)
    panel = panel.sort_values([AREA_COL, "tbin"]).reset_index(drop=True) # sort the panel by area then time
    g = panel.groupby(AREA_COL)["count"] # group by area
    for lag_bins, lag_name in [(1, "lag_1"), (8, "lag_8"), (56, "lag_56")]:
    # count 1 hr ago, 1 day ago, 1 week ago (in 3h bins)
        panel[lag_name] = g.shift(lag_bins)
    # current count is removed and only past count remains
    shifted          = panel.groupby(AREA_COL)["count"].shift(1)
    panel["roll_24h"] = (shifted.groupby(panel[AREA_COL]).rolling(8,  min_periods=1).mean().reset_index(level=0, drop=True)) # rolling mean of the past 8 bins (24 hours) excluding the current bin, with a minimum of 1 bin to avoid NaNs at the start of each area group
    panel["roll_7d"]  = (shifted.groupby(panel[AREA_COL]).rolling(56, min_periods=1).mean().reset_index(level=0, drop=True)) # rolling mean of the past 56 bins (7 days) excluding the current bin, with a minimum of 1 bin to avoid NaNs at the start of each area group

    panel["hour"]       = panel["tbin"].dt.hour
    panel["dow"]        = panel["tbin"].dt.dayofweek
    panel["month"]      = panel["tbin"].dt.month
    panel["is_weekend"] = (panel["tbin"].dt.dayofweek >= 5).astype(float)
    panel               = panel.dropna(subset=["lag_56"])   # drop warm-up as the first 7 days have no lag_56
    
    # final features
    FEATS = ["hour", "dow", "month", "is_weekend", "lag_1", "lag_8", "lag_56", "roll_24h", "roll_7d"]

    def make_fc_X(df):
        """Feature matrix: numeric FEATS + area as native categorical."""
        X             = df[FEATS].astype(float).copy()
        X[AREA_COL]   = df[AREA_COL].astype(str).values # area is kept as a categorical variable so that CatBoost can handle it natively without needing one-hot encoding or ordinal encoding
        return X

    # time based train/test split
    cut = panel["tbin"].quantile(0.8) 
    tr  = panel[panel["tbin"] <= cut].copy()
    te  = panel[panel["tbin"] >  cut].copy()

    # crerate X,Y for test and train using the above function
    Xtr      = make_fc_X(tr)
    Xte      = make_fc_X(te)
    ytr, yte = tr["count"].values, te["count"].values

    # ── count model ───────────────────────────────────────────────────
    m = make_reg()
    # train the model
    m.fit(Xtr, ytr, cat_features=[AREA_COL])
    # predict the test set and clip negative predictions to 0
    pred  = m.predict(Xte).clip(min=0)
    # naive baseline: predict the count from the same time last week (lag_56)
    naive = te["lag_56"].values             

    print(f"\nPanel: {len(panel)} station-bins | train {len(tr)} / test {len(te)}")
    print(f"   MODEL  MAE {mean_absolute_error(yte, pred):.3f}  " # MAE wrt acutal test values
          f"RMSE {mean_squared_error(yte, pred)**0.5:.3f}")       # RMSE wrt actual test values (punihses larger mistakes more)
    print(f"   NAIVE  MAE {mean_absolute_error(yte, naive):.3f}  "# MAE wrt naive baseline
          f"RMSE {mean_squared_error(yte, naive)**0.5:.3f}   (same time last week)") # RMSE wrt naive baseline 
    # gain wrt naive baseline, expressed as a percentage improvement in MAE (positive gain means the model is better than the naive baseline, and negative means it's worse)
    gain = (1 - mean_absolute_error(yte, pred) / max(mean_absolute_error(yte, naive), 1e-9)) * 100
    print(f"   -> model beats naive by {gain:.1f}% MAE")

    plot_importance(importances(m, FEATS + [AREA_COL]), "Forecaster feature importance", "forecaster_importance")

    # ── severity model (impact_sum) ───────────────────────────────────
    ytr_imp, yte_imp = tr["impact_sum"].values, te["impact_sum"].values
    m_impact = make_reg()
    m_impact.fit(Xtr, ytr_imp, cat_features=[AREA_COL]) # train the 2nd model to predict impact_sum
    pred_imp = m_impact.predict(Xte).clip(min=0)
    print(f"\n[impact_sum]  MAE {mean_absolute_error(yte_imp, pred_imp):.2f}  " # MAE wrt actual test values
          f"R2 {r2_score(yte_imp, pred_imp):.3f}") # R2 score wrt actual test values

    # ── actual vs predicted plot ──────────────────────────────────────
    busiest = ev[AREA_COL].value_counts().idxmax()
    sub     = te[te[AREA_COL] == busiest].sort_values("tbin")
    if len(sub) > 10:
        sp = m.predict(make_fc_X(sub)).clip(min=0)
        fig, ax = plt.subplots(figsize=(11, 4))
        ax.plot(sub["tbin"], sub["count"], label="actual", lw=1)
        ax.plot(sub["tbin"], sp,           label="predicted", lw=1.2, alpha=0.8)
        ax.set_title(f"Forecast vs actual — {busiest} (test window)")
        ax.set_ylabel("events / 3h"); ax.legend()
        save(fig, "forecast_vs_actual")

    # Store (model, (FEATS, AREA_COL)) so decisions.py can reconstruct X
    fc_meta = (FEATS, AREA_COL)
    return (m, m_impact), fc_meta

# ══════════════════════════════════════════════════════════════════════
def main():
    # locate cleaned data
    pkl = os.path.join(OUTDIR, "modeling_table.pkl")
    # if the file isn't found, exit the program with a message to run clean.py first
    if not os.path.exists(pkl):
        raise SystemExit("Run clean.py first.")
    # load the modeling table from the pickle file
    tbl = pd.read_pickle(pkl)
    print(f"Loaded modeling table: {tbl.shape}")

    # train the triage and forecaster models
    triage_models                  = triage(tbl)
    (fc_count, fc_impact), fc_meta = forecaster(tbl)

    # save all trained models
    with open(os.path.join(OUTDIR, "fitted_models.pkl"), "wb") as f:
        # stores info like decision trees and feature importances for the triage models, and the forecaster models along with their metadata (feature names and area column) in a pickle file for later use
        pickle.dump({
            "triage":              triage_models,       # {target: (model, CAT, NUM)}
            "forecaster":         (fc_count,   fc_meta),
            "forecaster_severity":(fc_impact,  fc_meta),
        }, f)
    print(f"\nSaved -> {OUTDIR}/fitted_models.pkl")

    # ── decisions.py compatibility note ───────────────────────────────
    # score_triage() currently expects (model, enc, names).
    # New format is (model, cat_cols, num_cols). Update it to:
    #
    #   for target, (model, cat_cols, num_cols) in triage.items():
    #       X = pd.concat([
    #               event_df[cat_cols].astype(str).fillna("NA"),
    #               event_df[num_cols].astype(float).fillna(-1)
    #           ], axis=1)
    #       proba = model.predict_proba(X)[:, 1]
    #
    # expected_load_surface() currently expects (model, enc).
    # New format is (model, (FEATS, AREA_COL)). Update it to:
    #
    #   model, (FEATS, AREA_COL) = forecaster
    #   X = panel[FEATS].astype(float).copy()
    #   X[AREA_COL] = panel[AREA_COL].astype(str).values
    #   panel["pred"] = model.predict(X).clip(min=0)

if __name__ == "__main__":
    main()