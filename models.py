#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
====================================================================
PHASE 1 : Baseline Models  (CatBoost backend)
====================================================================
Reads pipeline_out/modeling_table.pkl (from clean.py) and trains:

  A) PER-EVENT TRIAGE
       - priority_high        -> rule-based (not ML: deterministic cause×time rule)
       - requires_rerouting   -> CatBoostClassifier (balanced + hist rate features)
       - duration_min         -> CatBoostRegressor  (log-transformed target)

  B) SPATIO-TEMPORAL FORECASTER
       - police_station x 3h panel, zero-filled
       - target: event count (impact_sum removed — model learned nothing)
       - features: calendar + lag_1/lag_12h/lag_8/lag_56 + roll_24h/roll_7d
                   + area_mean/area_std + area (native categorical)
       - benchmarked against same-time-last-week naive baseline

All splits TIME-BASED (train past, test future).
Three-way split: train (70%) / validation (10%) / test (20%).
  - Validation is passed to CatBoost as eval_set for early stopping.
  - Test set is truly held-out: never touched during training.

Changes from previous version:
  - priority_high replaced by rule_based_priority() — learned 1.000 AUC
    from deterministic corridor×hour×weekend rule in ASTraM system
  - requires_rerouting: added historical rerouting rate features per cause,
    corridor, police station (computed from train only to avoid leakage)
  - requires_rerouting: l2_leaf_reg=1 (tighter minority class fit)
  - duration_min: log1p target transform, historical median duration features
  - forecaster: added lag_12h (12h lag) and area_mean/area_std static features
  - severity model removed (stopped at iter 0, R2≈0 — data too sparse)

NOTE: decisions.py storage format:
  triage -> {target: (model, cat_cols, num_cols, hist_lookups)} hist_lookups = {feat_name: {"lookup": dict, "default": float}}
  forecaster -> (model, (FEATS, AREA_COL))   [unchanged]
  See compatibility note at bottom of main() for decisions.py updates.

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
from sklearn.metrics import (roc_auc_score, average_precision_score, f1_score,
                             accuracy_score, mean_absolute_error,
                             mean_squared_error, r2_score)

warnings.filterwarnings("ignore")
OUTDIR   = "pipeline_out"
AREA_COL = "police_station"
FREQ     = "3h"
_BACKEND = "catboost"
fig_n    = 0

# ── split config ──────────────────────────────────────────────────────
TRAIN_FRAC = 0.70   # 70% of data goes to training
VAL_FRAC   = 0.10   # 10% goes to validation (early stopping watches this)
# remaining 20% is the test set — truly held-out, never seen during training
EARLY_STOP = 50     # stop training if validation metric doesn't improve for this many rounds

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

# iterations - upper bound on trees; early stopping will cut this before the limit
# learning_rate - % each tree contributes to the final prediction (smaller lr usually need more trees but generalize better)
# depth - maximum depth of each tree (higher depth can capture more complex patterns but may overfit)
# eval_metric - metric watched by early stopping on the validation set
# use_best_model=True - after training, rolls weights back to the iteration with the best validation metric
# verbose=100 - prints training + validation scores every 100 iterations

# suppose we have a binary classification problem where the positive class is rare (e.g., 8% of the data).
# In such cases, we can use the auto_class_weights="Balanced" parameter to automatically adjust the weights of the classes during training.
# This helps the model pay more attention to the minority class and improve its performance on imbalanced datasets.
# This improves recall, f1 score and detection of rare events at the expense of some accuracy.

# creates and returns a CatBoostClassifier with default hyperparameters
# eval_metric="AUC" — good general metric for balanced targets like priority_high (62/38 split)
def make_clf():
    return CatBoostClassifier(iterations=600, learning_rate=0.05, depth=6,
                              eval_metric="AUC", use_best_model=True, verbose=100)

# same thing but for imbalanced datasets
# eval_metric="PRAUC" — better than AUC for rare events: focuses on the minority class
# (with only 8% positives, AUC is dominated by the majority class and early stopping on it is misleading)
# l2_leaf_reg=1 — reduced from default 3: allows the model to fit the minority class more tightly
def make_clf_balanced():
    return CatBoostClassifier(iterations=600, learning_rate=0.05, depth=6,
                              eval_metric="PRAUC", auto_class_weights="Balanced",
                              l2_leaf_reg=1,       # change #9: tighter minority class fit
                              use_best_model=True, verbose=100)

# creates a CatBoostRegressor which is used when predicting continuous values
def make_reg():
    return CatBoostRegressor(iterations=700, learning_rate=0.05, depth=6,
                             eval_metric="RMSE", use_best_model=True, verbose=100)

# ──────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────

# extracts feature importances from a trained CatBoost model and returns them as a pandas Series sorted by importance
def importances(model, names):
    """CatBoost always exposes feature_importances_ — no permutation fallback needed."""
    return pd.Series(model.feature_importances_, index=names).sort_values()

# creates a horizontal bar plot of feature importances and saves it to a file
def plot_importance(imp, title, fname):
    fig, ax = plt.subplots(figsize=(8, max(3, 0.4 * len(imp)))) # figure size is dynamic based on number of features
    ax.barh(imp.index, imp.values, color="#534AB7")
    ax.set_title(title); ax.set_xlabel("importance")
    save(fig, fname) # the save function defined earlier

# splits a DataFrame into train / validation / test sets based on time
# we can't use random splits for time series: training on future data to predict the past would
# give falsely optimistic metrics that won't hold in production
# split order: [--- 70% train ---][-- 10% val --][---- 20% test ----]  (chronological)
# validation immediately follows train so early stopping sees the most recent training period
# test is the most recent 20% — truly held-out, never seen during training or early stopping
def time_split_3way(df):
    df = df.sort_values("occurrence_ts").reset_index(drop=True) # sort incidents chronologically
    n  = len(df)
    k1 = int(n * TRAIN_FRAC)                   # end of train
    k2 = int(n * (TRAIN_FRAC + VAL_FRAC))      # end of validation
    return df.iloc[:k1].copy(), df.iloc[k1:k2].copy(), df.iloc[k2:].copy()

# same idea but for the forecaster panel which is indexed by tbin (time bin) not occurrence_ts
# uses quantile on tbin to find the cut points so the split is even in time, not in row count
def panel_split_3way(panel):
    # fiding the split points based on the quantiles of tbin coloumn 
    cut1 = panel["tbin"].quantile(TRAIN_FRAC)
    cut2 = panel["tbin"].quantile(TRAIN_FRAC + VAL_FRAC)
    # create the splits based on the cut points
    # we use copy to avoid modifying the original panel
    tr   = panel[panel["tbin"] <= cut1].copy()
    val  = panel[(panel["tbin"] > cut1) & (panel["tbin"] <= cut2)].copy()
    te   = panel[panel["tbin"] >  cut2].copy()
    return tr, val, te

def build_X(df, cat_cols, num_cols):
    """Build a DataFrame CatBoost can consume natively.
    Cats as str (CatBoost reads raw strings), nums as float, NaNs filled."""
    return pd.concat([
        df[cat_cols].astype(str).fillna("NA"), # converts categorical columns to string and fills NaNs with "NA"
        df[num_cols].astype(float).fillna(-1)  # converts numerical columns to float and fills NaNs with -1
    ], axis=1)

# change #1: priority_high replaced with this rule function
# discovered from data: 607 corridor/hour/weekend combinations are 100% one class
# Non-corridor events are always Low priority regardless of cause or time
# this is a deterministic system rule — no ML model needed
def priority_rule(corridor_base, hour, is_weekend):
    if str(corridor_base) == "Non-corridor":
        return 0  # always Low priority
    return 1      # corridor events are High priority

# change #3: historical rate helper
# computes aggregated target rates from the training set and applies them to all splits
# "mean" for binary targets (rerouting rate), "median" for continuous (duration)
# computed from train only — never from val/test — to avoid any data leakage
# returns lookup dict saved alongside the model so decisions.py can apply it to new events
def add_hist_feature(tr, val, te, group_col, target_col, new_col, agg_fn="mean"):
    if agg_fn == "mean":
        lookup = tr.groupby(group_col)[target_col].mean()
    else:  # median for duration
        lookup = tr.groupby(group_col)[target_col].median()
    global_val = float(lookup.mean())  # fallback for unseen categories at inference
    for df in [tr, val, te]:
        df[new_col] = df[group_col].map(lookup).fillna(global_val)
    return lookup.to_dict(), global_val  # return for saving in the pickle

# ══════════════════════════════════════════════════════════════════════
# A) PER-EVENT TRIAGE
# ══════════════════════════════════════════════════════════════════════

# trains 2 models: requires_rerouting (classification), duration_min (regression)
# priority_high is handled by priority_rule() — no ML model
def triage(tbl): # tbl is the modeling table loaded from pipeline_out/modeling_table.pkl
    print("\n" + "=" * 60 + f"\nA) PER-EVENT TRIAGE   (backend: {_BACKEND})\n" + "=" * 60)
    print("   NOTE: priority_high uses rule_based_priority() — see above")

    CAT = ["cause", "event_type", "corridor_base", "zone", "police_station", "veh_type"]
    NUM = ["hour", "dow", "month", "is_weekend", "latitude", "longitude", "concurrent_load"]
    # keep only the columns that exist
    CAT = [c for c in CAT if c in tbl.columns]
    NUM = [c for c in NUM if c in tbl.columns]

    df = tbl.dropna(subset=["occurrence_ts"]).copy() # remove rows without occurrence timestamps
    df["is_weekend"] = df["is_weekend"].astype(float) # convert Bool to Float for CatBoost
    fitted = {} # dictionary to store the fitted models and their corresponding categorical and numerical columns

    # ── binary classifier (requires_rerouting only) ───────────────────
    # change #1: removed "priority_high" — it's a deterministic rule, not a learned model
    for target in ["requires_rerouting"]:
        if target not in df.columns:
            continue
        d = df[df[target].notna()].copy() # remove rows where the target variable is NaN
        d[target] = d[target].astype(int) # convert the target variable to integer (0 or 1)
        if d[target].nunique() < 2:       # ensure that there are at least 2 classes
            print(f"\n[{target}] only one class — skipped."); continue

        # 3-way temporal split: train sees the past, val is the bridge, test is the future
        # early stopping watches val — test is never touched until final evaluation
        tr, val, te = time_split_3way(d)

        # change #3: historical rerouting rate features
        # for each grouping column, compute the % of events that historically required rerouting
        # e.g. cause_hist_rr = "40% of accidents historically needed rerouting"
        # these give the model explicit signal it would otherwise need many splits to approximate
        NUM_use = list(NUM)  # start with base features, extend with hist features
        hist_lookups = {}    # saved alongside model so decisions.py can apply to new events
        for group_col, feat_name in [("cause",         "cause_hist_rr"),
                                      ("corridor_base", "corr_hist_rr"),
                                      ("police_station","stn_hist_rr")]:
            lookup_dict, global_mean = add_hist_feature(
                tr, val, te, group_col, "requires_rerouting", feat_name, agg_fn="mean")
            hist_lookups[feat_name] = {"lookup": lookup_dict, "default": global_mean}
            NUM_use.append(feat_name)

        # build feature matrices using our build_X function and then extract the target
        Xtr  = build_X(tr,  CAT, NUM_use)
        Xval = build_X(val, CAT, NUM_use)   # validation features — passed to eval_set
        Xte  = build_X(te,  CAT, NUM_use)
        ytr, yval, yte = tr[target].values, val[target].values, te[target].values

        # make_clf_balanced now includes l2_leaf_reg=1 (change #9)
        m = make_clf_balanced()
        # eval_set=(Xval, yval) — CatBoost evaluates on the validation set after each iteration
        # early_stopping_rounds — stops training if the eval_metric hasn't improved for this many rounds
        # NOTE: using Xval/yval here (not Xte/yte) so the test set stays truly held-out
        m.fit(Xtr, ytr, cat_features=CAT, eval_set=(Xval, yval),
              early_stopping_rounds=EARLY_STOP)
        print("Best iteration for " + target + ":", m.get_best_iteration()) # finds the best iteration

        # predict probabilities and convert to labels using a threshold of 0.5
        proba = m.predict_proba(Xte)[:, 1]
        pred  = (proba >= 0.5).astype(int)

        # evaluation — all metrics computed on the held-out test set only
        print(f"\n[{target}]  train={len(tr)} val={len(val)} test={len(te)}"
              f"  positive rate={yte.mean():.2f}")
        if len(np.unique(yte)) == 2:
            print(f"   ROC-AUC {roc_auc_score(yte, proba):.3f} | "
                   # ROC-AUC (Receiver Operating Characteristic - Area Under Curve) measures the model's ability to distinguish between classes. It ranges from 0 to 1, where 1 indicates perfect classification and 0.5 indicates random guessing.
                  f"PR-AUC  {average_precision_score(yte, proba):.3f} | "
                  # Precision-Recall AUC (Area Under the Precision-Recall Curve) measures the model's ability to rank positive instances higher than negative ones. It ranges from 0 to 1, where 1 indicates perfect ranking and 0.5 indicates random ranking. It is more imp for rare-event detection than ROC-AUC.
                  f"F1 {f1_score(yte, pred):.3f} | "
                  # F1 score is the harmonic mean of precision(TP/TP+FP) and recall(TP/TP+FN), providing a balance between the two metrics. It ranges from 0 to 1, where 1 indicates perfect precision and recall, and 0 indicates the worst performance.
                  f"Acc {accuracy_score(yte, pred):.3f}") # correct predictions / total predictions

        # plot feature importances — NUM_use includes the new hist rate features
        plot_importance(importances(m, CAT + NUM_use),
                        f"Triage importance: {target}", f"triage_{target}")

        # Store (model, cat_cols, num_cols, hist_lookups)
        # hist_lookups saved so decisions.py can compute hist features for new events at inference
        fitted[target] = (m, CAT, NUM_use, hist_lookups)

    # ── duration regressor ────────────────────────────────────────────
    if "duration_valid" in df.columns and df["duration_valid"].sum() > 50: # need at least 50 valid rows
        d = df[df["duration_valid"]].copy() # keep only valid rows

        # same 3-way temporal split
        tr, val, te = time_split_3way(d)

        # change #5: historical median duration features
        # "accidents on ORR historically take 90 min to clear" — strong signal for duration prediction
        # median used instead of mean because duration is right-skewed (a few 24h events distort mean)
        NUM_dur = list(NUM)   # separate copy so it doesn't affect the rerouting model's features
        dur_lookups = {}
        for group_col, feat_name in [("cause",         "cause_hist_dur"),
                                      ("corridor_base", "corr_hist_dur")]:
            lookup_dict, global_med = add_hist_feature(
                tr, val, te, group_col, "duration_min", feat_name, agg_fn="median")
            dur_lookups[feat_name] = {"lookup": lookup_dict, "default": global_med}
            NUM_dur.append(feat_name)

        Xtr  = build_X(tr,  CAT, NUM_dur)
        Xval = build_X(val, CAT, NUM_dur)   # validation features for early stopping
        Xte  = build_X(te,  CAT, NUM_dur)

        # change #4: log-transform the target before training
        # duration is heavily right-skewed (median ~65 min, some events at 24h+)
        # log1p(x) = log(1+x) compresses the tail and makes gradient boosting work better
        # we fit on log scale and reverse with expm1() (inverse of log1p) at prediction time
        ytr_raw  = tr["duration_min"].values
        yval_raw = val["duration_min"].values
        yte      = te["duration_min"].values  # kept on original scale for reporting
        ytr_log  = np.log1p(ytr_raw)
        yval_log = np.log1p(yval_raw)

        m = make_reg() # create a regression model
        # again: eval_set uses val on log scale, test remains held-out
        m.fit(Xtr, ytr_log, cat_features=CAT, eval_set=(Xval, yval_log),
              early_stopping_rounds=EARLY_STOP)
        print("Best iteration for duration_min:", m.get_best_iteration())

        # reverse the log transform to get predictions on original scale (minutes)
        pred = np.expm1(m.predict(Xte)).clip(min=0)
        # naive baseline: predict the median training duration for all test samples (original scale)
        base = np.full(len(yte), np.median(ytr_raw))

        print(f"\n[duration_min]  train={len(tr)} val={len(val)} test={len(te)} (valid rows only)")
        print(f"   MAE {mean_absolute_error(yte, pred):.1f} min  "
              f"(naive-median {mean_absolute_error(yte, base):.1f}) | " # MAE wrt naive baseline
              f"R2 {r2_score(yte, pred):.3f}") # measures how much variance is explained (1 is perfect, 0 means no better than predicting the mean, negative means worse than predicting the mean)
        plot_importance(importances(m, CAT + NUM_dur), "Triage importance: duration", "triage_duration")
        fitted["duration_min"] = (m, CAT, NUM_dur, dur_lookups) # store model with hist lookups
    else:
        print("\n[duration_min] too few valid rows — skipping.")

    return fitted

# ══════════════════════════════════════════════════════════════════════
# B) SPATIO-TEMPORAL FORECASTER
# ══════════════════════════════════════════════════════════════════════
def forecaster(tbl):
    print("\n" + "=" * 60 + f"\nB) SPATIO-TEMPORAL FORECASTER   (backend: {_BACKEND})\n" + "=" * 60)

    ev = tbl.dropna(subset=["occurrence_ts", AREA_COL]).copy() # remove rows without occurrence timestamps or area information
    ev["tbin"] = ev["occurrence_ts"].dt.floor(FREQ) # create time bins by flooring the occurrence timestamps to the nearest 3 hours (or whatever FREQ is set to)
    # aggregate event counts and impact scores by area and time bin
    agg = (ev.groupby([AREA_COL, "tbin"]).agg(count=("id", "size")).reset_index())

    # zero-fill the full area × time grid
    areas = agg[AREA_COL].unique() # get all areas
    tidx  = pd.date_range(agg["tbin"].min(), agg["tbin"].max(), freq=FREQ) # get all time bins
    # create a MultiIndex grid of all area-time combinations and convert it to a DataFrame
    grid  = (pd.MultiIndex.from_product([areas, tidx], names=[AREA_COL, "tbin"]).to_frame(index=False))
    # merge the aggregated counts onto the grid, filling missing values with 0
    panel = grid.merge(agg, on=[AREA_COL, "tbin"], how="left")
    panel["count"] = panel["count"].fillna(0.0)

    # lag / rolling — all in bin units (1 bin = 3h)
    panel = panel.sort_values([AREA_COL, "tbin"]).reset_index(drop=True) # sort the panel by area then time
    g = panel.groupby(AREA_COL)["count"] # group by area
    # change #6: added lag_12h (4 bins = 12h) to fill the gap between 3h and 24h
    # "if there was a spike this morning, there may be one this evening" — the 12h lag captures this
    for lag_bins, lag_name in [(1, "lag_1"), (4, "lag_12h"), (8, "lag_8"), (56, "lag_56")]:
    # count 1 bin (3h) ago, 4 bins (12h) ago, 8 bins (24h) ago, 56 bins (1 week) ago
        panel[lag_name] = g.shift(lag_bins)
    # current count is removed and only past count remains
    shifted           = panel.groupby(AREA_COL)["count"].shift(1)
    panel["roll_24h"] = (shifted.groupby(panel[AREA_COL]).rolling(8,  min_periods=1).mean().reset_index(level=0, drop=True)) # rolling mean of the past 8 bins (24 hours) excluding the current bin, with a minimum of 1 bin to avoid NaNs at the start of each area group
    panel["roll_7d"]  = (shifted.groupby(panel[AREA_COL]).rolling(56, min_periods=1).mean().reset_index(level=0, drop=True)) # rolling mean of the past 56 bins (7 days) excluding the current bin, with a minimum of 1 bin to avoid NaNs at the start of each area group

    panel["hour"]       = panel["tbin"].dt.hour
    panel["dow"]        = panel["tbin"].dt.dayofweek
    panel["month"]      = panel["tbin"].dt.month
    panel["is_weekend"] = (panel["tbin"].dt.dayofweek >= 5).astype(float)

    # change #7: area-level static baseline features
    # tells the model "Yelahanka typically sees 0.8 events/bin, this quiet station sees 0.1"
    # 2 events at Yelahanka is routine; 2 events at a quiet station is unusual
    # computed from the full panel BEFORE the split — these are static area properties,
    # similar to how lag features also use the full panel before splitting
    area_stats = (panel.groupby(AREA_COL)["count"].agg(area_mean="mean", area_std="std").reset_index())
    area_stats["area_std"] = area_stats["area_std"].fillna(0)  # areas with 1 row have NaN std
    panel = panel.merge(area_stats, on=AREA_COL, how="left")
    panel = panel.dropna(subset=["lag_56"])   # drop warm-up as the first 7 days have no lag_56
    # lag_12h only needs 4 bins warmup, which is satisfied once lag_56 (56 bins) is satisfied

    # final features — includes new lag_12h and area baseline features
    FEATS = ["hour", "dow", "month", "is_weekend", "lag_1", "lag_12h", "lag_8", "lag_56", "roll_24h", "roll_7d", "area_mean", "area_std"]   # change #7

    def make_fc_X(df):
        """Feature matrix: numeric FEATS + area as native categorical."""
        X           = df[FEATS].astype(float).copy()
        X[AREA_COL] = df[AREA_COL].astype(str).values # area is kept as a categorical variable so that CatBoost can handle it natively without needing one-hot encoding or ordinal encoding
        return X

    # 3-way temporal split on the panel — same principle as triage
    # val is used for early stopping, test is held-out for final metrics
    tr, val, te = panel_split_3way(panel)

    # create X, Y for train, val and test
    Xtr  = make_fc_X(tr)
    Xval = make_fc_X(val)   # validation features for early stopping
    Xte  = make_fc_X(te)
    ytr, yval, yte = tr["count"].values, val["count"].values, te["count"].values

    # ── count model ───────────────────────────────────────────────────
    m = make_reg()
    # train the model — early stopping watches the validation set, not the test set
    m.fit(Xtr, ytr, cat_features=[AREA_COL], eval_set=(Xval, yval),
          early_stopping_rounds=EARLY_STOP)
    print(f"Best iteration for count model: {m.get_best_iteration()}")
    # predict the test set and clip negative predictions to 0
    pred  = m.predict(Xte).clip(min=0)
    # naive baseline: predict the count from the same time last week (lag_56)
    naive = te["lag_56"].values

    print(f"\nPanel: {len(panel)} station-bins | train={len(tr)} val={len(val)} test={len(te)}")
    print(f"   MODEL  MAE {mean_absolute_error(yte, pred):.3f}  "  # MAE wrt actual test values
          f"RMSE {mean_squared_error(yte, pred)**0.5:.3f}")          # RMSE wrt actual test values (punishes larger mistakes more)
    print(f"   NAIVE  MAE {mean_absolute_error(yte, naive):.3f}  "  # MAE wrt naive baseline
          f"RMSE {mean_squared_error(yte, naive)**0.5:.3f}   (same time last week)") # RMSE wrt naive baseline
    # gain wrt naive baseline, expressed as a percentage improvement in MAE (positive gain means the model is better than the naive baseline, and negative means it's worse)
    gain = (1 - mean_absolute_error(yte, pred) / max(mean_absolute_error(yte, naive), 1e-9)) * 100
    print(f"   -> model beats naive by {gain:.1f}% MAE")

    plot_importance(importances(m, FEATS + [AREA_COL]), "Forecaster feature importance", "forecaster_importance")

    # change #8: severity model removed
    # impact_sum model stopped at iteration 0 and achieved R2≈0 on both synthetic and real data
    # the target is too sparse (most bins have impact_sum=0) for gradient boosting to learn from
    # replacement: in decisions.py, use predicted_count × area_mean_impact as a simple heuristic

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
    return m, fc_meta  # single count model only (change #8)

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
    triage_models    = triage(tbl)
    fc_count, fc_meta = forecaster(tbl)   # change #8: single model, not (count, impact) tuple

    # save all trained models
    with open(os.path.join(OUTDIR, "fitted_models.pkl"), "wb") as f:
        # triage: {target: (model, CAT, NUM_use, hist_lookups)}
        # forecaster: (count_model, (FEATS, AREA_COL))
        # priority_high rule is defined as priority_rule() — not stored in pickle
        pickle.dump({
            "triage":     triage_models,
            "forecaster": (fc_count, fc_meta),
            # forecaster_severity removed (change #8)
        }, f)
    print(f"\nSaved -> {OUTDIR}/fitted_models.pkl")

    # ── decisions.py compatibility notes ─────────────────────────────
    #
    # 1. score_triage() — triage tuple is now (model, cat_cols, num_cols, hist_lookups)
    #    Update to:
    #      for target, model_info in triage.items():
    #          model, cat_cols, num_cols = model_info[0], model_info[1], model_info[2]
    #          hist_lookups = model_info[3] if len(model_info) > 3 else {}
    #          # apply hist features to the event before building X:
    #          for feat_name, info in hist_lookups.items():
    #              col = feat_name.replace("_hist_rr","").replace("_hist_dur","")
    #              col = {"cause":"cause","corr":"corridor_base","stn":"police_station"}[col]
    #              event_df[feat_name] = event_df[col].map(info["lookup"]).fillna(info["default"])
    #          X = pd.concat([event_df[cat_cols].astype(str).fillna("NA"),
    #                         event_df[num_cols].astype(float).fillna(-1)], axis=1)
    #
    # 2. expected_load_surface() — forecaster format unchanged: (model, (FEATS, AREA_COL))
    #    But FEATS now includes lag_12h, area_mean, area_std — rebuild panel the same way.
    #
    # 3. priority_high — use priority_rule(corridor_base, hour, is_weekend) from this file
    #    instead of loading from pickle.
    #
    # 4. forecaster_severity key removed from pickle — update load_artifacts() to not expect it.

if __name__ == "__main__":
    main()