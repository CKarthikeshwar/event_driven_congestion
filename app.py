#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
====================================================================
 ASTraM — PHASE 3 : Operations Console  (Streamlit dashboard)
====================================================================
A single-page console that ties the whole pipeline together for a live demo /
prototype link. Four tabs:
    1. Overview & Map   - KPIs + incident map (Folium if online, else matplotlib)
    2. Event Triage     - type in an incoming event -> impact + barricade + diversion
    3. Manpower         - officer roster for a chosen day/shift
    4. Barricading      - standing choke-points, cause rates, diversion explorer

Design notes
------------
* MODEL + LOGIC use ONLY the Astram dataset (offline-safe, contest-compliant).
* The MAP is the only internet-touching part and it is optional: on a deployed
  host it shows tiles, on an air-gapped machine it falls back to a plain plot.
* Self-healing: if the .pkl artifacts are missing or fail to unpickle (e.g.
  library-version drift on the deploy host), it rebuilds them from the raw data.

Run locally : streamlit run app.py
Deploy      : push repo (this file + astram_*.py + pipeline_out/*.pkl + data) to
              GitHub -> Streamlit Community Cloud -> main file = app.py
Requires    : streamlit, pandas, numpy, matplotlib, scikit-learn, lightgbm
              (optional: folium, streamlit-folium for the live map)
"""
import os, glob, subprocess, sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import streamlit as st

import astram_decisions as D   # Phase-2 logic (which imports Phase-0/1 helpers)

OUTDIR = "pipeline_out"
DOW_NAMES = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]

st.set_page_config(page_title="ASTraM Congestion Console",
                   page_icon="🚦", layout="wide")

# ----------------------------------------------------------------------
# Self-healing artifact loader
# ----------------------------------------------------------------------
def _rebuild():
    """Regenerate modeling_table.pkl + fitted_models.pkl from the raw dataset."""
    data = (sorted(glob.glob("*.xlsx")) + sorted(glob.glob("*.csv")))
    data = [d for d in data if "modeling_table" not in d]
    if not data:
        st.error("No raw dataset (.xlsx/.csv) found to rebuild artifacts."); st.stop()
    with st.spinner(f"First-run setup: building models from {data[0]} ..."):
        subprocess.run([sys.executable, "astram_clean.py", data[0]], check=True)
        subprocess.run([sys.executable, "astram_models.py"], check=True)

@st.cache_resource(show_spinner="Loading models and scoring the city...")
def prepare():
    pkls = [os.path.join(OUTDIR, "modeling_table.pkl"),
            os.path.join(OUTDIR, "fitted_models.pkl")]
    try:
        if not all(os.path.exists(p) for p in pkls):
            _rebuild()
        tbl, triage, forecaster = D.load_artifacts()
    except Exception:
        _rebuild()
        tbl, triage, forecaster = D.load_artifacts()
    prof = D.expected_load_surface(tbl, forecaster)
    chokes, choke_key = D.choke_point_list(tbl)
    cause_rates = D.cause_rerouting_rates(tbl)
    geo = D.corridor_geo(tbl)
    return dict(tbl=tbl, triage=triage, forecaster=forecaster, prof=prof,
                chokes=chokes, choke_key=choke_key, cause_rates=cause_rates, geo=geo)

A = prepare()
tbl = A["tbl"]

def opts(colname):
    if colname not in tbl.columns:
        return []
    return sorted(x for x in tbl[colname].dropna().unique())

# ----------------------------------------------------------------------
# Map helper (Folium if available + online, else matplotlib scatter)
# ----------------------------------------------------------------------
def render_map(points, value="impact_score", max_pts=2500):
    pts = points.dropna(subset=["latitude", "longitude"])
    if len(pts) > max_pts:
        pts = pts.sample(max_pts, random_state=0)
    try:
        import folium
        from streamlit_folium import st_folium
        c = [pts["latitude"].median(), pts["longitude"].median()]
        m = folium.Map(location=c, zoom_start=11, tiles="OpenStreetMap")
        vmax = max(pts[value].max(), 1) if value in pts else 1
        for _, r in pts.iterrows():
            v = r.get(value, 1)
            folium.CircleMarker(
                [r["latitude"], r["longitude"]], radius=3,
                color=None, fill=True,
                fill_color=f"#{int(220):02x}{int(80+150*(1-v/vmax)):02x}40",
                fill_opacity=0.6,
                popup=f"{r.get('cause','?')} | impact {v:.0f}").add_to(m)
        st_folium(m, height=480, use_container_width=True)
        return
    except Exception:
        st.caption("Live map tiles unavailable — showing offline plot.")
        fig, ax = plt.subplots(figsize=(8, 7))
        sc = ax.scatter(pts["longitude"], pts["latitude"],
                        c=pts[value] if value in pts else None,
                        cmap="inferno", s=10, alpha=0.6)
        if value in pts:
            fig.colorbar(sc, ax=ax, label=value)
        ax.set_xlabel("longitude"); ax.set_ylabel("latitude")
        st.pyplot(fig)

# ======================================================================
# HEADER
# ======================================================================
st.title("ASTraM - Event-Driven Congestion Console")
st.caption("Forecast impact, allocate manpower, trigger barricading & diversion - "
           "all from the Astram event dataset.")

tab_map, tab_triage, tab_manpower, tab_barricade = st.tabs(
    ["Overview & Map", "Event Triage", "Manpower", "Barricading & Diversion"])

# ----------------------------------------------------------------------
# TAB 1 - OVERVIEW & MAP
# ----------------------------------------------------------------------
with tab_map:
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Events", f"{len(tbl):,}")
    if "event_type" in tbl:
        unplanned = (tbl["event_type"] == "unplanned").mean() * 100
        c2.metric("Unplanned", f"{unplanned:.0f}%")
    c3.metric("Median impact", f"{tbl['impact_score'].median():.0f}")
    if "requires_rerouting" in tbl:
        c4.metric("Need rerouting", f"{tbl['requires_rerouting'].mean()*100:.0f}%")
    if "occurrence_ts" in tbl:
        span = (tbl["occurrence_ts"].max() - tbl["occurrence_ts"].min()).days
        c5.metric("Days of data", f"{span}")

    st.divider()
    left, right = st.columns([1, 3])
    with left:
        causes = st.multiselect("Filter by cause", opts("cause"))
        hi_only = st.checkbox("High-impact only (>= 60)")
    view = tbl.copy()
    if causes:
        view = view[view["cause"].isin(causes)]
    if hi_only:
        view = view[view["impact_score"] >= 60]
    with right:
        st.markdown(f"**Incident map** - {len(view):,} events shown, coloured by impact")
        render_map(view)

# ----------------------------------------------------------------------
# TAB 2 - EVENT TRIAGE  (the live demo)
# ----------------------------------------------------------------------
with tab_triage:
    st.subheader("Assess an incoming event")
    st.caption("Fill in what's reported; the model predicts severity and the "
               "rules decide barricading + diversion.")
    a, b, c = st.columns(3)
    ev = {}
    ev["cause"]          = a.selectbox("Cause", opts("cause"))
    ev["corridor_base"]  = a.selectbox("Corridor", opts("corridor_base"))
    ev["police_station"] = b.selectbox("Police station", opts("police_station"))
    ev["zone"]           = b.selectbox("Zone", opts("zone") or ["NA"])
    ev["veh_type"]       = c.selectbox("Vehicle type", opts("veh_type") or ["NA"])
    ev["event_type"]     = c.selectbox("Event type", opts("event_type") or ["unplanned"])
    h, d2 = st.columns(2)
    ev["hour"] = h.slider("Hour of day", 0, 23, 18)
    dow_label  = d2.selectbox("Day of week", DOW_NAMES, index=0)
    ev["dow"]  = DOW_NAMES.index(dow_label)
    ev["month"] = 3
    ev["is_weekend"] = float(ev["dow"] >= 5)
    g = A["geo"]
    if ev["corridor_base"] in g.index:
        ev["latitude"], ev["longitude"] = g.loc[ev["corridor_base"], ["lat", "lon"]]
    ev.setdefault("latitude", tbl["latitude"].median())
    ev.setdefault("longitude", tbl["longitude"].median())
    ev["concurrent_load"] = 1

    if st.button("Assess event", type="primary"):
        res = D.handle_event(ev, (tbl, A["triage"], A["forecaster"]))
        tri = res["triage"]
        m1, m2, m3 = st.columns(3)
        if "priority_high_proba" in tri:
            m1.metric("P(High priority)", f"{float(tri['priority_high_proba']):.0%}")
        if "requires_rerouting_proba" in tri:
            m2.metric("P(needs rerouting)", f"{float(tri['requires_rerouting_proba']):.0%}")
        if "duration_min" in tri:
            m3.metric("Est. clear time", f"{float(tri['duration_min']):.0f} min")

        bar = res["barricade"]
        if bar["decision"] == "BARRICADE":
            st.error(f"BARRICADE - {bar['level']}\n\n{bar['reason']}")
        else:
            st.success(f"MONITOR - {bar['reason']}")

        if res.get("diversion"):
            st.markdown("**Suggested diversion (near + lighter load):**")
            st.dataframe(pd.DataFrame(res["diversion"]).T, width="stretch")

# ----------------------------------------------------------------------
# TAB 3 - MANPOWER
# ----------------------------------------------------------------------
with tab_manpower:
    st.subheader("Officer allocation by shift")
    cc1, cc2, cc3, cc4 = st.columns(4)
    dow_l   = cc1.selectbox("Day", DOW_NAMES, key="mp_dow")
    shift   = cc2.selectbox("Shift", list(D.SHIFTS), key="mp_shift")
    total   = cc3.slider("Officers available", 20, 400, D.TOTAL_OFFICERS, 10)
    min_cov = cc4.slider("Min per area", 0, 6, D.MIN_COVERAGE)

    prof = A["prof"]
    sub = prof[(prof["dow"] == DOW_NAMES.index(dow_l)) & (prof["shift"] == shift)]
    demand = dict(zip(sub[D.AREA_COL], sub["demand"]))
    alloc = D.allocate_manpower(demand, total=total, min_cov=min_cov)
    roster = (pd.DataFrame({"area": list(demand),
                            "expected_load": [round(demand[a], 2) for a in demand]})
              .assign(officers=lambda x: x["area"].map(alloc))
              .query("officers > 0").sort_values("officers", ascending=False)
              .reset_index(drop=True))
    lft, rgt = st.columns([1, 1])
    with lft:
        st.dataframe(roster, width="stretch", height=430)
        st.caption(f"Total deployed: {int(roster['officers'].sum())} / {total}")
    with rgt:
        if len(roster):
            fig, ax = plt.subplots(figsize=(6, max(3, 0.35 * len(roster))))
            r = roster[::-1]
            ax.barh(r["area"].astype(str), r["officers"], color="#1D9E75")
            ax.set_xlabel("officers"); ax.set_title(f"{dow_l} - {shift}")
            st.pyplot(fig)

# ----------------------------------------------------------------------
# TAB 4 - BARRICADING & DIVERSION
# ----------------------------------------------------------------------
with tab_barricade:
    st.subheader(f"Standing choke-points (by {A['choke_key']})")
    chokes = A["chokes"].head(15).reset_index()
    cL, cR = st.columns([1, 1])
    with cL:
        st.dataframe(chokes.round(2), width="stretch", height=420)
    with cR:
        top = A["chokes"].head(12)["barricade_score"][::-1]
        fig, ax = plt.subplots(figsize=(6, 5))
        ax.barh(top.index.astype(str), top.values, color="#D85A30")
        ax.set_xlabel("barricade score")
        st.pyplot(fig)

    st.divider()
    st.markdown("**Rerouting rate by cause** (>= 0.10 => barricade-prone)")
    st.dataframe(A["cause_rates"].round(3), width="stretch")

    st.divider()
    st.subheader("Diversion explorer")
    blocked = st.selectbox("If this corridor is blocked...",
                           sorted(A["geo"].index.tolist()))
    alts = D.suggest_diversion(blocked, A["geo"])
    if len(alts):
        st.dataframe(alts.reset_index(), width="stretch")
    else:
        st.info("No alternates available for this corridor in the data.")