#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
====================================================================
 ASTraM — ADD-ON : Post-Event Learning Loop   (closes gap #3)
====================================================================
Standalone. Does NOT modify any existing file — it imports the Phase-2 logic
(astram_decisions) and adds the feedback machinery that was only conceptual
before: log every prediction, record the real outcome when it arrives, measure
drift against a frozen baseline, and retrain when a trigger fires.

Flow:
    triage a new event  -> log_prediction(...)         (prediction stored)
    event later resolves -> record_outcome(...)        (ground truth stored)
    periodically         -> evaluate() + maybe_retrain()

State files (all in pipeline_out/):
    predictions_log.csv        - append-only prediction + outcome ledger
    model_baseline_metrics.json- metrics snapshot taken at last (re)train

Retrain triggers (either one):
    * >= RETRAIN_AFTER_N newly-labelled rows since the last train, OR
    * a tracked metric has dropped more than DRIFT_TOL below baseline.

Retraining simply re-runs Phases 0+1 (astram_clean.py, astram_models.py) as a
subprocess and refreshes the baseline — no bespoke training code to drift from.

Usage (demo):  python astram_feedback_loop.py
Requires: pandas, numpy, scikit-learn  (no extra/heavy deps)
"""
import os, json, glob, subprocess, sys, datetime as dt
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, accuracy_score, mean_absolute_error

import decisions as D

OUTDIR          = "pipeline_out"
LOG_PATH        = os.path.join(OUTDIR, "predictions_log.csv")
BASELINE_PATH   = os.path.join(OUTDIR, "model_baseline_metrics.json")
RETRAIN_AFTER_N = 200      # retrain once this many new labelled rows accumulate
DRIFT_TOL       = 0.05     # retrain if a metric falls > 5 points below baseline

PRED_COLS = ["logged_at", "event_id",
             "pred_priority", "pred_priority_proba",
             "pred_reroute",  "pred_reroute_proba", "pred_duration",
             "actual_priority", "actual_reroute", "actual_duration",
             "outcome_recorded"]

# ----------------------------------------------------------------------
# ledger helpers
# ----------------------------------------------------------------------
def _load_log():
    if os.path.exists(LOG_PATH):
        return pd.read_csv(LOG_PATH)
    return pd.DataFrame(columns=PRED_COLS)

def _save_log(df):
    df.to_csv(LOG_PATH, index=False)

def log_prediction(event_id, event, triage):
    """Store a model prediction for one incoming event (outcome unknown yet)."""
    p = D.score_triage(pd.DataFrame([event]), triage)
    row = {c: np.nan for c in PRED_COLS}
    row.update(dict(
        logged_at=dt.datetime.now().isoformat(timespec="seconds"),
        event_id=event_id,
        pred_priority=int(D._priority_rule(event.get("corridor_base"), event.get("hour"), event.get("is_weekend"))),
        pred_priority_proba=float(D._priority_rule(event.get("corridor_base"), event.get("hour"), event.get("is_weekend"))),
        pred_reroute=int(p.get("requires_rerouting", [0])[0]),
        pred_reroute_proba=float(p.get("requires_rerouting_proba", [np.nan])[0]),
        pred_duration=float(p.get("duration_min", [np.nan])[0]),
        outcome_recorded=False))
    log = pd.concat([_load_log(), pd.DataFrame([row])], ignore_index=True)
    _save_log(log)
    return row

def record_outcome(event_id, actual_priority=None, actual_reroute=None,
                   actual_duration=None):
    """Fill in the ground truth once the event has actually resolved."""
    log = _load_log()
    m = log["event_id"].astype(str) == str(event_id)
    if not m.any():
        return False
    if actual_priority is not None: log.loc[m, "actual_priority"] = int(actual_priority)
    if actual_reroute  is not None: log.loc[m, "actual_reroute"]  = int(actual_reroute)
    if actual_duration is not None: log.loc[m, "actual_duration"] = float(actual_duration)
    log.loc[m, "outcome_recorded"] = True
    _save_log(log)
    return True

# ----------------------------------------------------------------------
# evaluation + drift
# ----------------------------------------------------------------------
def evaluate(log=None):
    """Score logged predictions against recorded outcomes."""
    log = _load_log() if log is None else log
    done = log[log["outcome_recorded"] == True]
    out = {"n_labelled": int(len(done))}
    if len(done) == 0:
        return out
    # priority
    d = done.dropna(subset=["actual_priority", "pred_priority"])
    if len(d):
        out["priority_acc"] = round(accuracy_score(d["actual_priority"], d["pred_priority"]), 3)
        if d["actual_priority"].nunique() == 2:
            out["priority_auc"] = round(roc_auc_score(d["actual_priority"],
                                                      d["pred_priority_proba"]), 3)
    # rerouting
    d = done.dropna(subset=["actual_reroute", "pred_reroute"])
    if len(d):
        out["reroute_acc"] = round(accuracy_score(d["actual_reroute"], d["pred_reroute"]), 3)
        if d["actual_reroute"].nunique() == 2:
            out["reroute_auc"] = round(roc_auc_score(d["actual_reroute"],
                                                     d["pred_reroute_proba"]), 3)
    # duration
    d = done.dropna(subset=["actual_duration", "pred_duration"])
    if len(d):
        out["duration_mae"] = round(mean_absolute_error(d["actual_duration"],
                                                        d["pred_duration"]), 1)
    return out

def save_baseline(metrics):
    with open(BASELINE_PATH, "w") as f:
        json.dump(metrics, f, indent=2)

def load_baseline():
    if os.path.exists(BASELINE_PATH):
        with open(BASELINE_PATH) as f:
            return json.load(f)
    return None

def detect_drift(current, baseline=None, tol=DRIFT_TOL):
    """A metric has drifted if an accuracy/AUC fell > tol below baseline."""
    baseline = baseline or load_baseline()
    if not baseline:
        return False, ["no baseline yet"]
    flags = []
    for k in ("priority_auc", "priority_acc", "reroute_auc", "reroute_acc"):
        if k in current and k in baseline and current[k] < baseline[k] - tol:
            flags.append(f"{k}: {baseline[k]} -> {current[k]}")
    return (len(flags) > 0), flags

# ----------------------------------------------------------------------
# retrain trigger
# ----------------------------------------------------------------------
def _n_new_labelled_since_baseline():
    b = load_baseline()
    seen = b.get("n_labelled", 0) if b else 0
    return max(int(evaluate().get("n_labelled", 0)) - seen, 0)

def maybe_retrain(dry_run=True):
    """Decide and (optionally) execute a retrain. Returns (did_retrain, why)."""
    current = evaluate()
    drifted, flags = detect_drift(current)
    n_new = _n_new_labelled_since_baseline()
    reasons = []
    if n_new >= RETRAIN_AFTER_N:
        reasons.append(f"{n_new} new labelled rows (>= {RETRAIN_AFTER_N})")
    if drifted:
        reasons.append("performance drift: " + "; ".join(flags))
    if not reasons:
        return False, ["no trigger met"]
    if dry_run:
        return False, ["WOULD retrain — " + " | ".join(reasons)]
    data = [d for d in sorted(glob.glob("*.xlsx")) + sorted(glob.glob("*.csv"))
            if "modeling_table" not in d and "predictions_log" not in d]
    if not data:
        return False, ["trigger met but no raw dataset found to retrain on"]
    subprocess.run([sys.executable, "clean.py", data[0]], check=True)
    subprocess.run([sys.executable, "models.py"], check=True)
    save_baseline(evaluate())                       # refresh baseline post-train
    return True, ["retrained — " + " | ".join(reasons)]

# ----------------------------------------------------------------------
# DEMO — simulate the loop on the real table (stream events, reveal outcomes)
# ----------------------------------------------------------------------
def main():
    tbl, triage, forecaster = D.load_artifacts()
    if os.path.exists(LOG_PATH):
        os.remove(LOG_PATH)                          # fresh demo
    feat = ["cause", "event_type", "corridor_base", "zone", "police_station",
            "veh_type", "hour", "dow", "month", "is_weekend",
            "latitude", "longitude", "concurrent_load"]

    # treat the most recent 300 events as a "live stream" we predict, then learn from
    stream = (tbl.dropna(subset=["occurrence_ts"])
                 .sort_values("occurrence_ts").tail(300).reset_index(drop=True))
    print(f"Streaming {len(stream)} recent events through the loop...")

    for i, r in stream.iterrows():                   # 1) predict + log
        log_prediction(r.get("id", i), {k: r.get(k) for k in feat}, triage)
    for i, r in stream.iterrows():                   # 2) outcomes arrive
        record_outcome(r.get("id", i),
                       actual_priority=int(bool(r.get("priority_high"))),
                       actual_reroute=int(bool(r.get("requires_rerouting"))),
                       actual_duration=r.get("duration_min")
                           if pd.notna(r.get("duration_min")) else None)

    metrics = evaluate()                             # 3) measure
    print("\nLive performance vs recorded outcomes:")
    print(json.dumps(metrics, indent=2))

    if load_baseline() is None:                      # first run -> set baseline
        save_baseline(metrics)
        print("\nBaseline snapshot saved.")

    did, why = maybe_retrain(dry_run=True)           # 4) retrain decision
    print(f"\nRetrain decision: {'RETRAIN' if did else 'hold'} — {'; '.join(why)}")
    print(f"\nLedger -> {LOG_PATH}\nBaseline -> {BASELINE_PATH}")

if __name__ == "__main__":
    main()