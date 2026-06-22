#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
====================================================================
PHASE 2 : Decision Layer  (offline, dataset-only)
====================================================================
Turns the Phase-1 model outputs into the three recommendations the problem
statement asks for. Uses ONLY the provided Astram data — no external sources.

  1. MANPOWER       : forecaster's expected load per area/shift  -> officer
                      roster, allocated greedily under a fixed headcount with
                      a minimum-coverage floor (provably optimal for the
                      concave "impact covered" objective; no solver needed).
  2. BARRICADING    : which causes/junctions historically drive rerouting +
                      high priority -> a standing choke-point list, plus a
                      per-event recommend_barricade() decision.
  3. DIVERSION      : corridor adjacency derived from event coordinates
                      (centroids + haversine) -> ranked alternate corridors
                      for a blocked one, preferring near + lightly-loaded.

Also: handle_event() — the integrated per-incident flow (triage -> barricade
-> diversion) that the Phase-3 dashboard will call.

Inputs : pipeline_out/modeling_table.pkl , pipeline_out/fitted_models.pkl
Outputs: recommendations to console + plots/CSVs in ./pipeline_out/
Usage  : python astram_decisions.py
Requires: pandas, numpy, matplotlib   (no internet, no extra solvers)
"""
import os, pickle, warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# priority_high is no longer a trained model — it's a deterministic rule
# discovered from data: Non-corridor events are always Low, corridor events always High
from models import priority_rule as _priority_rule

warnings.filterwarnings("ignore")
OUTDIR = "pipeline_out"
AREA_COL, FREQ = "police_station", "3h"   # FREQ must match models.py (was "1h")

# ---- operational config (tune to the real control room) ---------------
SHIFTS = {"Morning": list(range(6, 14)),
          "Evening": list(range(14, 22)),
          "Night":   [22, 23, 0, 1, 2, 3, 4, 5]}
TOTAL_OFFICERS   = 120     # pool to distribute per shift
MIN_COVERAGE     = 2       # floor for any area that has expected load
DIVERSION_K      = 4       # alternates to suggest per blocked corridor

fig_n = 0
def save(fig, name):
    global fig_n; fig_n += 1
    p = os.path.join(OUTDIR, f"decide_{fig_n:02d}_{name}.png")
    fig.savefig(p, dpi=110, bbox_inches="tight"); plt.close(fig)
    print(f"   saved {p}")

#shift identfier
def shift_of(hour):
    for s, hrs in SHIFTS.items():
        if hour in hrs:
            return s
    return "Night"

# ======================================================================
# Load artifacts
# ======================================================================

# load the modeling table and the fitted models (triage + forecaster) from disk
def load_artifacts():
    tbl = pd.read_pickle(os.path.join(OUTDIR, "modeling_table.pkl"))
    with open(os.path.join(OUTDIR, "fitted_models.pkl"), "rb") as f:
        models = pickle.load(f)
    return tbl, models["triage"], models["forecaster"]

# ======================================================================
# Rebuild the forecaster panel (same construction as Phase 1) and score it
# ======================================================================
def build_panel(tbl):
    # remove rows without timestamp or area
    ev = tbl.dropna(subset=["occurrence_ts", AREA_COL]).copy()
    # create time bins (3h)
    ev["tbin"] = ev["occurrence_ts"].dt.floor(FREQ)
    # count events per area per tbin
    agg = (ev.groupby([AREA_COL, "tbin"]).agg(count=("id", "size")).reset_index())
    # get all areas
    areas = agg[AREA_COL].unique()
    # get all time bins in the range
    tidx = pd.date_range(agg["tbin"].min(), agg["tbin"].max(), freq=FREQ)
    # create a complete panel of all area-tbin combinations, merge with agg to get counts
    panel = (pd.MultiIndex.from_product([areas, tidx], names=[AREA_COL, "tbin"]).to_frame(index=False).merge(agg, on=[AREA_COL, "tbin"], how="left"))
    # fill missing with 0
    panel["count"] = panel["count"].fillna(0.0)
    # sort chronologically so that each area has its hitory in time order
    panel = panel.sort_values([AREA_COL, "tbin"]).reset_index(drop=True)
    # create lag and rolling features by area — same as models.py
    g = panel.groupby(AREA_COL)["count"]
    # lag features in bin units (1 bin = 3h) — must match models.py exactly
    for lag_bins, lag_name in [(1, "lag_1"), (4, "lag_12h"), (8, "lag_8"), (56, "lag_56")]:
        panel[lag_name] = g.shift(lag_bins)
    # rolling averages
    shifted = panel.groupby(AREA_COL)["count"].shift(1)
    # rolling means in bin units (8 bins = 24h, 56 bins = 7d) 
    panel["roll_24h"] = (shifted.groupby(panel[AREA_COL]).rolling(8,  min_periods=1).mean().reset_index(level=0, drop=True))
    panel["roll_7d"]  = (shifted.groupby(panel[AREA_COL]).rolling(56, min_periods=1).mean().reset_index(level=0, drop=True))
    # time features
    panel["hour"]       = panel["tbin"].dt.hour
    panel["dow"]        = panel["tbin"].dt.dayofweek
    panel["month"]      = panel["tbin"].dt.month
    panel["is_weekend"] = (panel["tbin"].dt.dayofweek >= 5).astype(float)
    # area-level static features (mean and variance) — same as models.py
    area_stats = (panel.groupby(AREA_COL)["count"].agg(area_mean="mean", area_std="std").reset_index())
    area_stats["area_std"] = area_stats["area_std"].fillna(0)
    panel = panel.merge(area_stats, on=AREA_COL, how="left")
    return panel.dropna(subset=["lag_56"])   # drop warmup (was lag_168 with 1h bins)

def expected_load_surface(tbl, forecaster):
    """Score the panel with the model, then collapse to a recurring weekly
    profile: expected events per (area, dow, shift), severity-weighted."""
    # new format: (model, (FEATS, AREA_COL)) — no encoder, CatBoost handles natively
    # load the forecaster model and its feature list
    model, (FEATS, fc_area_col) = forecaster
    panel = build_panel(tbl)
    # build X exactly as in models.py: numeric FEATS + area as string categorical
    X = panel[FEATS].astype(float).copy()
    X[fc_area_col] = panel[fc_area_col].astype(str).values
    # predict and clip to non-negative
    panel["pred"] = model.predict(X).clip(min=0)
    # convert hours to shifts
    panel["shift"] = panel["hour"].map(shift_of)

    # expected events per area per (dow, shift) = mean over weeks
    # find no of unique dates per area/dow/shift to get the denominator for the mean — clip lower to 1 to avoid div by zero for rare combos
    n_dates = (panel.assign(_date=panel["tbin"].dt.date)
           .groupby([AREA_COL, "dow", "shift"])["_date"]
           .nunique()
           .clip(lower=1))
    # averave predicted incidents across weeks
    prof = (panel.groupby([AREA_COL, "dow", "shift"])["pred"].sum() / n_dates)
    prof = prof.reset_index(name="expected_events") # makes a DataFrame again from a pandas series

    # severity weight: areas with worse average impact need more presence
    # severity = avg impact score of incidents in that area
    sev = tbl.groupby(AREA_COL)["impact_score"].mean()
    # values are normalized 
    sev_w = (sev / sev.mean()).to_dict()
    # expected load = expected events x severity weight (relative to citywide average) 
    prof["demand"] = prof.apply(
        lambda r: r["expected_events"] * sev_w.get(r[AREA_COL], 1.0), axis=1)
    return prof

# ======================================================================
# 1) MANPOWER ALLOCATION
# ======================================================================

# converts demand score for each area into actual no of officers
def allocate_manpower(demand, total=TOTAL_OFFICERS, min_cov=MIN_COVERAGE):
    """Min-coverage floor to active areas (highest demand first), then split the
    remaining pool in proportion to expected severity-load, made exact with the
    largest-remainder method. Scale-invariant and fully explainable."""
    # keep only areas with positive demand (active areas)
    active = {a: d for a, d in demand.items() if d > 0}
    alloc = {a: 0 for a in demand} # start with zero officers
    if not active:
        return alloc
    # first sort by demand and allocate min officers(const) to each active area 
    for a in sorted(active, key=lambda x: -active[x]):            # min coverage
        if sum(alloc.values()) + min_cov <= total:
            alloc[a] = min_cov
        else:
            break
    remaining = total - sum(alloc.values()) # remaining officers
    floored = [a for a in active if alloc[a] > 0] # only areas that got min coverage
    tot_d = sum(active[a] for a in floored) 
    if remaining <= 0 or tot_d <= 0:
        return alloc
    shares = {a: remaining * active[a] / tot_d for a in floored}  # calculate proportional allocation for remaining officers
    for a, s in shares.items():
        alloc[a] += int(np.floor(s)) # allocate the integer part 
    leftover = remaining - sum(int(np.floor(s)) for s in shares.values()) # find leftover officers after flooring
    # sort the fractions based on descening order and then allocate one leftover officer to each until we run out of officers (called largest remainder method)
    for a in sorted(shares, key=lambda x: shares[x] - np.floor(shares[x]),
                    reverse=True)[:leftover]:                     # largest remainder
        alloc[a] += 1
    return alloc

# 
def manpower_roster(prof, dow, shift):
    sub = prof[(prof["dow"] == dow) & (prof["shift"] == shift)] # filter coloumns 
    demand = dict(zip(sub[AREA_COL], sub["demand"])) # build demand dictionary
    alloc = allocate_manpower(demand) # call the above function 
    # create a df
    roster = (pd.DataFrame({"area": list(demand), "expected_events(demand)": [demand[a] for a in demand]})
              .assign(officers=lambda d: d["area"].map(alloc)) # add officer allocation 
              .query("officers > 0") # remove 0 allocation areas
              .sort_values("officers", ascending=False)# sort by officers
              .reset_index(drop=True)) # reset no rows
    return roster

# ======================================================================
# 2) BARRICADING
# ======================================================================
def choke_point_list(tbl, by="junction"):
    """Standing choke points: rank places by volume x rerouting-rate x impact."""
    # choose the location coloumn to group by, default is junction, if not present use corridor_base
    key = by if by in tbl.columns and tbl[by].notna().any() else "corridor_base"
    d = tbl.dropna(subset=[key]) # remove rows with missing locations  
    g = d.groupby(key).agg( # group by location (junction or corridor_base)
        events=("id", "size"), # count of events
        reroute_rate=("requires_rerouting", "mean"), # mean of requires_rerouting for each location
        high_pri_rate=("priority_high", "mean"), # mean of priority_high
        mean_impact=("impact_score", "mean")) # mean impact score
    # barricade score = no of events * (0.5 + reroute_rate) * (mean_impact / 100)
    g["barricade_score"] = (g["events"] * (0.5 + g["reroute_rate"].fillna(0))* (g["mean_impact"] / 100)).round(1)
    return g.sort_values("barricade_score", ascending=False), key # sort descending 

# which casuses are most likely to require rerouting (and high priority)
def cause_rerouting_rates(tbl):
    return (tbl.groupby("cause") # group by cause
              .agg(reroute_rate=("requires_rerouting", "mean"), # compute reroute_rate
                   high_pri_rate=("priority_high", "mean"), # high priority rate
                   n=("id", "size")) # count indices
              .sort_values("reroute_rate", ascending=False)) # sort by reroute_rate descending

def recommend_barricade(event, triage=None, cause_rates=None):
    """Per-event decision. Uses triage predictions if available, else the raw
    event flags. Returns (decision, level, reason)."""
    # load existing flags
    pred_reroute = event.get("requires_rerouting")
    pred_high    = event.get("priority_high")
    # if triage is provided, use it to score the event and override the flags
    if triage is not None:
        p = score_triage(pd.DataFrame([event]), triage)
        pred_reroute = bool(p.get("requires_rerouting", [pred_reroute])[0])
        # priority_high from rule — no longer a trained model
        pred_high = bool(_priority_rule(event.get("corridor_base"),
                                        event.get("hour"),
                                        event.get("is_weekend")))
    # check if the cause is prone to rerouting (>=10% historically)
    cause = event.get("cause")
    cause_prone = (cause_rates is not None and cause in cause_rates.index
                   and cause_rates.loc[cause, "reroute_rate"] >= 0.10)
    # main decision logic: if predicted reroute OR (predicted high AND cause historically prone) -> barricade, else monitor
    if pred_reroute or (pred_high and cause_prone):
        level = "Full closure + diversion" if pred_reroute and pred_high else "Partial lane barricade"
        reason = (f"cause='{cause}' "
                  f"(predicted reroute={pred_reroute}, high-priority={pred_high})")
        return "BARRICADE", level, reason
    return "MONITOR", "None", f"low predicted disruption (cause='{cause}')"

# ======================================================================
# 3) DIVERSION  (corridor adjacency from coordinates, no maps)
# ======================================================================
# compute dist btwn 2 gps coordinates (lat/lon) using haversine formula, returns km
def _haversine(lat1, lon1, lat2, lon2):
    r = 6371.0
    p1, p2 = np.radians(lat1), np.radians(lat2)
    dphi = np.radians(lat2 - lat1); dl = np.radians(lon2 - lon1)
    a = np.sin(dphi/2)**2 + np.cos(p1)*np.cos(p2)*np.sin(dl/2)**2
    return 2 * r * np.arcsin(np.sqrt(a))

# build a summary table for each corridor: median lat/lon, no of events, mean impact, events/day
def corridor_geo(tbl):
    d = tbl.dropna(subset=["latitude", "longitude", "corridor_base"])
    d = d[~d["corridor_base"].str.contains("Non-corridor", case=False, na=False)]
    # median lat long are used to represent the corridor
    geo = d.groupby("corridor_base").agg(
        lat=("latitude", "median"), lon=("longitude", "median"),
        events=("id", "size"), mean_impact=("impact_score", "mean"))
    geo["events_per_day"] = geo["events"] / max(
        (tbl["occurrence_ts"].max() - tbl["occurrence_ts"].min()).days, 1)
    return geo

def suggest_diversion(blocked, geo, k=DIVERSION_K):
    if blocked not in geo.index: # checks existence of corridor
        return pd.DataFrame()
    b = geo.loc[blocked] # get the row corresponding to the blocked corridor
    others = geo.drop(index=blocked).copy() # consider all other coridors
    others["dist_km"] = _haversine(b["lat"], b["lon"], others["lat"], others["lon"]) # find dist
    # prefer near AND lightly loaded
    others["d_norm"] = others["dist_km"] / others["dist_km"].max() #normalize by dividing withmax
    others["l_norm"] = others["events_per_day"] / others["events_per_day"].max() # normalize load
    others["pick_score"] = (0.6 * others["d_norm"] + 0.4 * others["l_norm"]) # ranking formula 
    cols = ["dist_km", "events_per_day", "mean_impact", "pick_score"]
    return others.sort_values("pick_score")[cols].head(k).round(2)

# ======================================================================
# Triage scoring
# ======================================================================
# maps hist feature names back to the source column they were computed from
_HIST_COL_MAP = {
    "cause_hist_rr":  "cause",
    "corr_hist_rr":   "corridor_base",
    "stn_hist_rr":    "police_station",
    "cause_hist_dur": "cause",
    "corr_hist_dur":  "corridor_base",
}

def score_triage(event_df, triage):
    out = {}
    for target, model_info in triage.items():
        model        = model_info[0]
        cat_cols     = model_info[1]
        num_cols     = model_info[2]
        # hist_lookups present in new format (4-tuple); absent in old format (3-tuple)
        hist_lookups = model_info[3] if len(model_info) > 3 else {}

        event_df = event_df.copy()
        # apply historical rate features before building X
        for feat_name, info in hist_lookups.items():
            grp_col = _HIST_COL_MAP.get(feat_name, feat_name)
            event_df[feat_name] = (event_df[grp_col].astype(str)
                                   .map(info["lookup"])
                                   .fillna(info["default"]))

        # build X as DataFrame — CatBoost reads raw strings for cat columns
        X = pd.concat([
                event_df[cat_cols].astype(str).fillna("NA"),
                event_df[num_cols].astype(float).fillna(-1)
            ], axis=1)

        if hasattr(model, "predict_proba"):
            probas = model.predict_proba(X)[:, 1]
            out[target]              = (probas >= 0.5).astype(int)
            out[target + "_proba"]   = probas
        else:
            out[target] = model.predict(X)
    return out

# ======================================================================
# Integrated per-incident flow
# ======================================================================
def handle_event(event, artifacts):
    tbl, triage, forecaster = artifacts
    cause_rates = cause_rerouting_rates(tbl)
    geo = corridor_geo(tbl)
    pred = score_triage(pd.DataFrame([event]), triage)
    # add priority_high from rule — not a trained model anymore
    pred["priority_high"]       = [_priority_rule(event.get("corridor_base"),
                                                   event.get("hour"),
                                                   event.get("is_weekend"))]
    pred["priority_high_proba"] = pred["priority_high"]
    decision, level, reason = recommend_barricade({**event,
                     "requires_rerouting": bool(pred.get("requires_rerouting_proba", [0])[0] >= 0.5),
                     "priority_high": pred["priority_high"][0]},
                    triage=None,   # <-- skip internal score_triage
                    cause_rates=cause_rates)
    result = {"triage": {k: (v[0] if hasattr(v, "__len__") else v)
                         for k, v in pred.items()},
              "barricade": {"decision": decision, "level": level, "reason": reason}}
    if decision == "BARRICADE":
        cb = event.get("corridor_base")
        result["diversion"] = suggest_diversion(cb, geo).to_dict("index") if cb else {}
    return result

# ======================================================================
# DEMO / report
# ======================================================================
def main():
    tbl, triage, forecaster = load_artifacts()
    artifacts = (tbl, triage, forecaster)
    print(f"Loaded modeling table {tbl.shape}, triage targets: {list(triage)}")

    # ---- 1. MANPOWER ----
    print("\n" + "=" * 60 + "\n1) MANPOWER ALLOCATION\n" + "=" * 60)
    prof = expected_load_surface(tbl, forecaster)
    DOW, SHIFT = 0, "Morning"     # Monday morning, as an example
    roster = manpower_roster(prof, DOW, SHIFT)
    print(f"\nRoster — {['Mon','Tue','Wed','Thu','Fri','Sat','Sun'][DOW]} {SHIFT} "
          f"(pool={TOTAL_OFFICERS}):")
    print(roster.head(12).to_string(index=False))
    roster.to_csv(os.path.join(OUTDIR, "manpower_roster_example.csv"), index=False)
    if len(roster):
        fig, ax = plt.subplots(figsize=(9, max(3, 0.4 * len(roster.head(12)))))
        r = roster.head(12)[::-1]
        ax.barh(r["area"].astype(str), r["officers"], color="#1D9E75")
        ax.set_title(f"Officers allocated — {SHIFT} shift (example)")
        ax.set_xlabel("officers")
        save(fig, "manpower_allocation")

    # ---- 2. BARRICADING ----
    print("\n" + "=" * 60 + "\n2) BARRICADING\n" + "=" * 60)
    chokes, key = choke_point_list(tbl)
    print(f"\nTop standing choke points (by {key}):")
    print(chokes.head(10).round(2).to_string())
    chokes.to_csv(os.path.join(OUTDIR, "choke_points.csv"))
    cr = cause_rerouting_rates(tbl)
    print("\nRerouting rate by cause (barricade-prone if >= 0.10):")
    print(cr.head(8).round(3).to_string())
    fig, ax = plt.subplots(figsize=(9, 5))
    top = chokes.head(12)["barricade_score"][::-1]
    ax.barh(top.index.astype(str), top.values, color="#D85A30")
    ax.set_title(f"Barricade priority ({key})"); ax.set_xlabel("barricade score")
    save(fig, "choke_points")

    # ---- 3. DIVERSION ----
    print("\n" + "=" * 60 + "\n3) DIVERSION\n" + "=" * 60)
    geo = corridor_geo(tbl)
    blocked = geo.sort_values("events", ascending=False).index[0]
    alts = suggest_diversion(blocked, geo)
    print(f"\nIf '{blocked}' is blocked, suggested alternates (near + lighter load):")
    print(alts.to_string())
    if len(geo) > 2:
        fig, ax = plt.subplots(figsize=(7, 7))
        ax.scatter(geo["lon"], geo["lat"], s=30, c="#888780", label="corridors")
        ax.scatter(geo.loc[blocked, "lon"], geo.loc[blocked, "lat"], s=120,
                   c="#E24B4A", label="blocked")
        for a in alts.index:
            ax.scatter(geo.loc[a, "lon"], geo.loc[a, "lat"], s=90,
                       c="#1D9E75", marker="^")
            ax.annotate(a, (geo.loc[a, "lon"], geo.loc[a, "lat"]), fontsize=7)
        ax.set_title(f"Diversion alternates for {blocked}")
        ax.set_xlabel("longitude"); ax.set_ylabel("latitude"); ax.legend()
        save(fig, "diversion_map")

    # ---- 4. INTEGRATED EVENT FLOW ----
    print("\n" + "=" * 60 + "\n4) END-TO-END: one incoming event\n" + "=" * 60)
    sample = tbl.dropna(subset=["cause", "corridor_base"]).iloc[0].to_dict()
    ev = {k: sample.get(k) for k in
          ["cause", "event_type", "corridor_base", "zone", "police_station",
           "veh_type", "hour", "dow", "month", "is_weekend",
           "latitude", "longitude", "concurrent_load"]}
    import json
    print("Incoming event:", {k: ev[k] for k in ["cause", "corridor_base", "police_station", "hour"]})
    print(json.dumps(handle_event(ev, artifacts), indent=2, default=str))
    print(f"\nAll recommendations + plots written to ./{OUTDIR}/")

if __name__ == "__main__":
    main()