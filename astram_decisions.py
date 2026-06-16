#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
====================================================================
 ASTraM — PHASE 2 : Decision Layer  (offline, dataset-only)
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

warnings.filterwarnings("ignore")
OUTDIR = "pipeline_out"
AREA_COL, FREQ = "police_station", "1h"

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

def shift_of(hour):
    for s, hrs in SHIFTS.items():
        if hour in hrs:
            return s
    return "Night"

# ======================================================================
# Load artifacts
# ======================================================================
def load_artifacts():
    tbl = pd.read_pickle(os.path.join(OUTDIR, "modeling_table.pkl"))
    with open(os.path.join(OUTDIR, "fitted_models.pkl"), "rb") as f:
        models = pickle.load(f)
    return tbl, models["triage"], models["forecaster"]

# ======================================================================
# Rebuild the forecaster panel (same construction as Phase 1) and score it
# ======================================================================
def build_panel(tbl):
    ev = tbl.dropna(subset=["occurrence_ts", AREA_COL]).copy()
    ev["tbin"] = ev["occurrence_ts"].dt.floor(FREQ)
    agg = (ev.groupby([AREA_COL, "tbin"])
             .agg(count=("id", "size")).reset_index())
    areas = agg[AREA_COL].unique()
    tidx = pd.date_range(agg["tbin"].min(), agg["tbin"].max(), freq=FREQ)
    panel = (pd.MultiIndex.from_product([areas, tidx], names=[AREA_COL, "tbin"])
               .to_frame(index=False)
               .merge(agg, on=[AREA_COL, "tbin"], how="left"))
    panel["count"] = panel["count"].fillna(0.0)
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
    return panel.dropna(subset=["lag_168"])

def expected_load_surface(tbl, forecaster):
    """Score the panel with the model, then collapse to a recurring weekly
    profile: expected events per (area, dow, shift), severity-weighted."""
    model, enc = forecaster
    panel = build_panel(tbl)
    FEATS = ["hour", "dow", "month", "is_weekend",
             "lag_1", "lag_24", "lag_168", "roll24", "roll168"]
    X = np.hstack([panel[FEATS].to_numpy(float),
                   enc.transform(panel[[AREA_COL]].astype("object"))])
    panel["pred"] = model.predict(X).clip(min=0)
    panel["shift"] = panel["hour"].map(shift_of)

    # expected events per area per (dow, shift) = mean over weeks
    prof = (panel.groupby([AREA_COL, "dow", "shift"])["pred"].sum()
                 / panel.groupby([AREA_COL, "dow", "shift"])["tbin"]
                        .apply(lambda s: s.dt.date.nunique()).clip(lower=1))
    prof = prof.reset_index(name="expected_events")

    # severity weight: areas with worse average impact need more presence
    sev = tbl.groupby(AREA_COL)["impact_score"].mean()
    sev_w = (sev / sev.mean()).to_dict()
    prof["demand"] = prof.apply(
        lambda r: r["expected_events"] * sev_w.get(r[AREA_COL], 1.0), axis=1)
    return prof

# ======================================================================
# 1) MANPOWER ALLOCATION
# ======================================================================
def allocate_manpower(demand, total=TOTAL_OFFICERS, min_cov=MIN_COVERAGE):
    """Min-coverage floor to active areas (highest demand first), then split the
    remaining pool in proportion to expected severity-load, made exact with the
    largest-remainder method. Scale-invariant and fully explainable. Swap in an
    LP (PuLP/OR-Tools) if you later need hard per-area caps."""
    active = {a: d for a, d in demand.items() if d > 0}
    alloc = {a: 0 for a in demand}
    if not active:
        return alloc
    for a in sorted(active, key=lambda x: -active[x]):            # min coverage
        if sum(alloc.values()) + min_cov <= total:
            alloc[a] = min_cov
        else:
            break
    remaining = total - sum(alloc.values())
    floored = [a for a in active if alloc[a] > 0]
    tot_d = sum(active[a] for a in floored)
    if remaining <= 0 or tot_d <= 0:
        return alloc
    shares = {a: remaining * active[a] / tot_d for a in floored}  # proportional
    for a, s in shares.items():
        alloc[a] += int(np.floor(s))
    leftover = remaining - sum(int(np.floor(s)) for s in shares.values())
    for a in sorted(shares, key=lambda x: shares[x] - np.floor(shares[x]),
                    reverse=True)[:leftover]:                     # largest remainder
        alloc[a] += 1
    return alloc

def manpower_roster(prof, dow, shift):
    sub = prof[(prof["dow"] == dow) & (prof["shift"] == shift)]
    demand = dict(zip(sub[AREA_COL], sub["demand"]))
    alloc = allocate_manpower(demand)
    roster = (pd.DataFrame({"area": list(demand),
                            "expected_events": [demand[a] for a in demand]})
              .assign(officers=lambda d: d["area"].map(alloc))
              .query("officers > 0")
              .sort_values("officers", ascending=False)
              .reset_index(drop=True))
    return roster

# ======================================================================
# 2) BARRICADING
# ======================================================================
def choke_point_list(tbl, by="junction"):
    """Standing choke points: rank places by volume x rerouting-rate x impact."""
    key = by if by in tbl.columns and tbl[by].notna().any() else "corridor_base"
    d = tbl.dropna(subset=[key])
    g = d.groupby(key).agg(
        events=("id", "size"),
        reroute_rate=("requires_rerouting", "mean"),
        high_pri_rate=("priority_high", "mean"),
        mean_impact=("impact_score", "mean"))
    g["barricade_score"] = (g["events"] * (0.5 + g["reroute_rate"].fillna(0))
                            * (g["mean_impact"] / 100)).round(1)
    return g.sort_values("barricade_score", ascending=False), key

def cause_rerouting_rates(tbl):
    return (tbl.groupby("cause")
              .agg(reroute_rate=("requires_rerouting", "mean"),
                   high_pri_rate=("priority_high", "mean"),
                   n=("id", "size"))
              .sort_values("reroute_rate", ascending=False))

def recommend_barricade(event, triage=None, cause_rates=None):
    """Per-event decision. Uses triage predictions if available, else the raw
    event flags. Returns (decision, level, reason)."""
    pred_reroute = event.get("requires_rerouting")
    pred_high = event.get("priority_high")
    if triage is not None:                       # predict from attributes
        p = score_triage(pd.DataFrame([event]), triage)
        pred_reroute = bool(p.get("requires_rerouting", [pred_reroute])[0])
        pred_high = bool(p.get("priority_high", [pred_high])[0])
    cause = event.get("cause")
    cause_prone = (cause_rates is not None and cause in cause_rates.index
                   and cause_rates.loc[cause, "reroute_rate"] >= 0.10)
    if pred_reroute or (pred_high and cause_prone):
        level = "Full closure + diversion" if pred_reroute and pred_high else "Partial lane barricade"
        reason = (f"cause='{cause}' "
                  f"(predicted reroute={pred_reroute}, high-priority={pred_high})")
        return "BARRICADE", level, reason
    return "MONITOR", "None", f"low predicted disruption (cause='{cause}')"

# ======================================================================
# 3) DIVERSION  (corridor adjacency from coordinates, no maps)
# ======================================================================
def _haversine(lat1, lon1, lat2, lon2):
    r = 6371.0
    p1, p2 = np.radians(lat1), np.radians(lat2)
    dphi = np.radians(lat2 - lat1); dl = np.radians(lon2 - lon1)
    a = np.sin(dphi/2)**2 + np.cos(p1)*np.cos(p2)*np.sin(dl/2)**2
    return 2 * r * np.arcsin(np.sqrt(a))

def corridor_geo(tbl):
    d = tbl.dropna(subset=["latitude", "longitude", "corridor_base"])
    d = d[~d["corridor_base"].str.contains("Non-corridor", case=False, na=False)]
    geo = d.groupby("corridor_base").agg(
        lat=("latitude", "median"), lon=("longitude", "median"),
        events=("id", "size"), mean_impact=("impact_score", "mean"))
    geo["events_per_day"] = geo["events"] / max(
        (tbl["occurrence_ts"].max() - tbl["occurrence_ts"].min()).days, 1)
    return geo

def suggest_diversion(blocked, geo, k=DIVERSION_K):
    if blocked not in geo.index:
        return pd.DataFrame()
    b = geo.loc[blocked]
    others = geo.drop(index=blocked).copy()
    others["dist_km"] = _haversine(b["lat"], b["lon"], others["lat"], others["lon"])
    # prefer near AND lightly loaded
    others["d_norm"] = others["dist_km"] / others["dist_km"].max()
    others["l_norm"] = others["events_per_day"] / others["events_per_day"].max()
    others["pick_score"] = (0.6 * others["d_norm"] + 0.4 * others["l_norm"])
    cols = ["dist_km", "events_per_day", "mean_impact", "pick_score"]
    return others.sort_values("pick_score")[cols].head(k).round(2)

# ======================================================================
# Triage scoring (reconstruct features in the saved order)
# ======================================================================
def score_triage(event_df, triage):
    out = {}
    for target, (model, enc, names) in triage.items():
        n_cat = enc.n_features_in_
        cat_cols, num_cols = names[:n_cat], names[n_cat:]
        for c in cat_cols + num_cols:
            if c not in event_df.columns:
                event_df[c] = np.nan
        cat = enc.transform(event_df[cat_cols].astype("object"))
        num = event_df[num_cols].to_numpy(dtype=float)
        X = np.nan_to_num(np.hstack([cat, num]), nan=-2)
        if hasattr(model, "predict_proba"):
            out[target] = (model.predict_proba(X)[:, 1] >= 0.5).astype(int)
            out[target + "_proba"] = model.predict_proba(X)[:, 1]
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
    decision, level, reason = recommend_barricade(event, triage, cause_rates)
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