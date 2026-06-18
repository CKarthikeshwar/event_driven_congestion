#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
====================================================================
 ASTraM — PHASE 1 : Baseline Models
====================================================================
Reads pipeline_out/modeling_table.pkl (from astram_clean.py) and trains the
two baseline models in the pipeline, both gradient-boosted:

  A) PER-EVENT TRIAGE  (answers "how bad is THIS incident?")
       - priority_high           -> classifier
       - requires_rerouting      -> classifier
       - duration_min            -> regressor (only on valid rows)
     Trained ONLY on event attributes (cause, place, time, vehicle) so there
     is no leakage from the impact score.

  B) SPATIO-TEMPORAL FORECASTER  (answers "where & when will it be bad?")
       - aggregates events into a police_station x hour panel (zeros filled)
       - target = event count per bin (impact_sum also available)
       - features = calendar + lagged history (1h / 24h / 168h + rolling means)
       - compared against a naive "same hour last week" baseline.

All splits are TIME-BASED (train on the past, test on the future) — never
random — because this is forecasting.

Outputs: metrics to console, plots + fitted models to ./pipeline_out/.
Usage:   python astram_models.py
Requires: pandas, numpy, scikit-learn, matplotlib  (lightgbm optional)
"""
import os, pickle, warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.preprocessing import OrdinalEncoder
from sklearn.metrics import (roc_auc_score, average_precision_score, f1_score,
                             accuracy_score, mean_absolute_error,
                             mean_squared_error, r2_score)
from sklearn.inspection import permutation_importance

warnings.filterwarnings("ignore")
OUTDIR = "pipeline_out"
AREA_COL, FREQ = "police_station", "1h"
fig_n = 0

def save(fig, name):
    global fig_n; fig_n += 1
    p = os.path.join(OUTDIR, f"model_{fig_n:02d}_{name}.png")
    fig.savefig(p, dpi=110, bbox_inches="tight"); plt.close(fig)
    print(f"   saved {p}")

# ----------------------------------------------------------------------
# Model factory: LightGBM if available, else sklearn HistGradientBoosting
# ----------------------------------------------------------------------
try:
    from lightgbm import LGBMClassifier, LGBMRegressor
    _BACKEND = "lightgbm"
    def make_clf(): return LGBMClassifier(n_estimators=400, learning_rate=0.05,
                                          num_leaves=48, subsample=0.8,
                                          colsample_bytree=0.8, verbose=-1)
    def make_reg(): return LGBMRegressor(n_estimators=500, learning_rate=0.05,
                                         num_leaves=48, subsample=0.8,
                                         colsample_bytree=0.8, verbose=-1)
except ImportError:
    from sklearn.ensemble import (HistGradientBoostingClassifier as _HC,
                                  HistGradientBoostingRegressor as _HR)
    _BACKEND = "sklearn-histgbm"
    def make_clf(): return _HC(max_iter=400, learning_rate=0.05)
    def make_reg(): return _HR(max_iter=500, learning_rate=0.05)

def importances(model, X, y, names):
    """Native importances if exposed, else permutation importance."""
    if hasattr(model, "feature_importances_"):
        return pd.Series(model.feature_importances_, index=names).sort_values()
    r = permutation_importance(model, X, y, n_repeats=5, random_state=0, n_jobs=-1)
    return pd.Series(r.importances_mean, index=names).sort_values()

def plot_importance(imp, title, fname):
    fig, ax = plt.subplots(figsize=(8, max(3, 0.4 * len(imp))))
    ax.barh(imp.index, imp.values, color="#534AB7")
    ax.set_title(title); ax.set_xlabel("importance")
    save(fig, fname)

# ----------------------------------------------------------------------
# Encoding helper (fit on train, apply to both)
# ----------------------------------------------------------------------
def encode(train, test, cat_cols, num_cols):
    enc = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1,
                         encoded_missing_value=-2)
    tr_cat = enc.fit_transform(train[cat_cols].astype("object"))
    te_cat = enc.transform(test[cat_cols].astype("object"))
    tr = np.hstack([tr_cat, train[num_cols].to_numpy(dtype=float)])
    te = np.hstack([te_cat, test[num_cols].to_numpy(dtype=float)])
    return np.nan_to_num(tr, nan=-2), np.nan_to_num(te, nan=-2), enc, cat_cols + num_cols

def time_split(df, frac=0.8):
    df = df.sort_values("occurrence_ts")
    k = int(len(df) * frac)
    return df.iloc[:k].copy(), df.iloc[k:].copy()

# ======================================================================
# A) PER-EVENT TRIAGE
# ======================================================================
def triage(tbl):
    print("\n" + "=" * 60 + f"\nA) PER-EVENT TRIAGE   (backend: {_BACKEND})\n" + "=" * 60)
    CAT = ["cause", "event_type", "corridor_base", "zone", "police_station", "veh_type"]
    NUM = ["hour", "dow", "month", "is_weekend", "latitude", "longitude", "concurrent_load"]
    CAT = [c for c in CAT if c in tbl.columns]
    NUM = [c for c in NUM if c in tbl.columns]
    df = tbl.dropna(subset=["occurrence_ts"]).copy()
    df["is_weekend"] = df["is_weekend"].astype(float)
    fitted = {}

    # ---- two binary classifiers ----
    for target in ["priority_high", "requires_rerouting"]:
        if target not in df.columns:
            continue
        d = df[df[target].notna()].copy()
        d[target] = d[target].astype(int)
        if d[target].nunique() < 2:
            print(f"\n[{target}] only one class present - skipped."); continue
        tr, te = time_split(d)
        Xtr, Xte, enc, names = encode(tr, te, CAT, NUM)
        ytr, yte = tr[target].values, te[target].values
        m = make_clf(); m.fit(Xtr, ytr)
        proba = m.predict_proba(Xte)[:, 1]; pred = (proba >= 0.5).astype(int)
        print(f"\n[{target}]  train={len(tr)} test={len(te)}  positive rate={yte.mean():.2f}")
        if len(np.unique(yte)) == 2:
            print(f"   ROC-AUC {roc_auc_score(yte,proba):.3f} | "
                  f"PR-AUC {average_precision_score(yte,proba):.3f} | "
                  f"F1 {f1_score(yte,pred):.3f} | Acc {accuracy_score(yte,pred):.3f}")
        plot_importance(importances(m, Xte, yte, names),
                        f"Triage importance: {target}", f"triage_{target}")
        fitted[target] = (m, enc, names)

    # ---- duration regressor (valid rows only) ----
    if "duration_valid" in df.columns and df["duration_valid"].sum() > 50:
        d = df[df["duration_valid"]].copy()
        tr, te = time_split(d)
        Xtr, Xte, enc, names = encode(tr, te, CAT, NUM)
        ytr, yte = tr["duration_min"].values, te["duration_min"].values
        m = make_reg(); m.fit(Xtr, ytr); pred = m.predict(Xte)
        base = np.full_like(yte, np.median(ytr))
        print(f"\n[duration_min]  train={len(tr)} test={len(te)} (valid rows only)")
        print(f"   MAE {mean_absolute_error(yte,pred):.1f} min  "
              f"(naive-median MAE {mean_absolute_error(yte,base):.1f}) | "
              f"R2 {r2_score(yte,pred):.3f}")
        plot_importance(importances(m, Xte, yte, names),
                        "Triage importance: duration", "triage_duration")
        fitted["duration_min"] = (m, enc, names)
    else:
        print("\n[duration_min] too few valid rows to model "
              "(expected on real data ~ <5%); skipping.")
    return fitted

# ======================================================================
# B) SPATIO-TEMPORAL FORECASTER
# ======================================================================
def forecaster(tbl):
    print("\n" + "=" * 60 + f"\nB) SPATIO-TEMPORAL FORECASTER   (backend: {_BACKEND})\n" + "=" * 60)
    ev = tbl.dropna(subset=["occurrence_ts", AREA_COL]).copy()
    ev["tbin"] = ev["occurrence_ts"].dt.floor(FREQ)
    agg = (ev.groupby([AREA_COL, "tbin"])
             .agg(count=("id", "size"), impact_sum=("impact_score", "sum"))
             .reset_index())

    # full area x time grid (zero-fill empty bins — most station-hours have none)
    areas = agg[AREA_COL].unique()
    tidx = pd.date_range(agg["tbin"].min(), agg["tbin"].max(), freq=FREQ)
    grid = (pd.MultiIndex.from_product([areas, tidx], names=[AREA_COL, "tbin"])
              .to_frame(index=False))
    panel = grid.merge(agg, on=[AREA_COL, "tbin"], how="left")
    panel[["count", "impact_sum"]] = panel[["count", "impact_sum"]].fillna(0.0)

    # lag / rolling features, computed PER AREA (causal: only past bins)
    panel = panel.sort_values([AREA_COL, "tbin"]).reset_index(drop=True)
    g = panel.groupby(AREA_COL)["count"]
    for L in (1, 24, 168):
        panel[f"lag_{L}"] = g.shift(L)
    shifted = panel.groupby(AREA_COL)["count"].shift(1)
    panel["roll24"]  = (shifted.groupby(panel[AREA_COL])
                        .rolling(24,  min_periods=1).mean().reset_index(level=0, drop=True))
    panel["roll168"] = (shifted.groupby(panel[AREA_COL])
                        .rolling(168, min_periods=1).mean().reset_index(level=0, drop=True))

    panel["hour"]       = panel["tbin"].dt.hour
    panel["dow"]        = panel["tbin"].dt.dayofweek
    panel["month"]      = panel["tbin"].dt.month
    panel["is_weekend"] = (panel["tbin"].dt.dayofweek >= 5).astype(float)
    panel = panel.dropna(subset=["lag_168"])               # drop warm-up window

    FEATS = ["hour", "dow", "month", "is_weekend",
             "lag_1", "lag_24", "lag_168", "roll24", "roll168"]
    # encode area as a categorical feature too
    enc = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)

    cut = panel["tbin"].quantile(0.8)
    tr = panel[panel["tbin"] <= cut].copy()
    te = panel[panel["tbin"] >  cut].copy()
    area_tr = enc.fit_transform(tr[[AREA_COL]].astype("object"))
    area_te = enc.transform(te[[AREA_COL]].astype("object"))
    Xtr = np.hstack([tr[FEATS].to_numpy(float), area_tr])
    Xte = np.hstack([te[FEATS].to_numpy(float), area_te])
    ytr, yte = tr["count"].values, te["count"].values

    m = make_reg(); m.fit(Xtr, ytr); pred = m.predict(Xte).clip(min=0)
    naive = te["lag_168"].values                            # same hour last week
    print(f"\nPanel: {len(panel)} station-hours | train {len(tr)} / test {len(te)}")
    print(f"   MODEL  MAE {mean_absolute_error(yte,pred):.3f}  "
          f"RMSE {mean_squared_error(yte,pred)**0.5:.3f}")
    print(f"   NAIVE  MAE {mean_absolute_error(yte,naive):.3f}  "
          f"RMSE {mean_squared_error(yte,naive)**0.5:.3f}   (same hour last week)")
    gain = (1 - mean_absolute_error(yte, pred) / max(mean_absolute_error(yte, naive), 1e-9)) * 100
    print(f"   -> model beats naive by {gain:.1f}% MAE")

    plot_importance(importances(m, Xte, yte, FEATS + ["area"]),
                    "Forecaster feature importance", "forecaster_importance")

    # actual vs predicted for the busiest area over the test window
    busiest = ev[AREA_COL].value_counts().idxmax()
    sub = te[te[AREA_COL] == busiest].sort_values("tbin")
    if len(sub) > 10:
        sp = m.predict(np.hstack([sub[FEATS].to_numpy(float),
                                  enc.transform(sub[[AREA_COL]].astype("object"))])).clip(min=0)
        fig, ax = plt.subplots(figsize=(11, 4))
        ax.plot(sub["tbin"], sub["count"], label="actual", lw=1)
        ax.plot(sub["tbin"], sp, label="predicted", lw=1.2, alpha=0.8)
        ax.set_title(f"Forecast vs actual - {busiest} (test window)")
        ax.set_ylabel("events / hour"); ax.legend()
        save(fig, "forecast_vs_actual")
    return m, enc

# ======================================================================
def main():
    pkl = os.path.join(OUTDIR, "modeling_table.pkl")
    if not os.path.exists(pkl):
        raise SystemExit("Run astram_clean.py first to create modeling_table.pkl")
    tbl = pd.read_pickle(pkl)
    print(f"Loaded modeling table: {tbl.shape}")
    triage_models = triage(tbl)
    fc_model, fc_enc = forecaster(tbl)
    with open(os.path.join(OUTDIR, "fitted_models.pkl"), "wb") as f:
        pickle.dump({"triage": triage_models, "forecaster": (fc_model, fc_enc)}, f)
    print(f"\nSaved fitted models -> {OUTDIR}/fitted_models.pkl")

if __name__ == "__main__":
    main()