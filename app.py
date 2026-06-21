#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
====================================================================
 PHASE 3 : Operations Console  (Streamlit dashboard)
====================================================================
A single-page console that ties the whole pipeline together for a live demo /
prototype link. Five tabs:
    1. Overview & Map      - KPIs + incident map (Folium if online, else matplotlib)
    2. Event Triage        - type in an incoming event -> impact + barricade + diversion
    3. Manpower            - officer roster for a chosen day/shift (proportional or LP)
    4. Barricading         - standing choke-points, cause rates, diversion explorer
    5. Deep ST Forecast    - ConvLSTM spatial heatmap (exploratory add-on)

Design notes
------------
* MODEL + LOGIC use ONLY the Astram dataset (offline-safe, contest-compliant).
* The MAP is the only internet-touching part and it is optional: on a deployed
  host it shows tiles, on an air-gapped machine it falls back to a plain plot.
* Self-healing: if the .pkl artifacts are missing or fail to unpickle (e.g.
  library-version drift on the deploy host), it rebuilds them from the raw data.
* LP allocator: uses PuLP/CBC if installed, falls back to SciPy, then proportional.
* Deep forecaster: runs as a subprocess when triggered; results persist as PNG.

Run locally : streamlit run app.py
Deploy      : push repo to GitHub -> Streamlit Community Cloud -> main file = app.py
Requires    : streamlit, pandas, numpy, matplotlib, catboost, scikit-learn
              (optional: folium, streamlit-folium for the live map)
              (optional: pulp or scipy for LP allocation)
              (optional: torch for the deep forecaster)
"""
import os, glob, subprocess, sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import streamlit as st

import decisions as D   # Phase-2 logic

# LP allocator — optional, falls back to proportional if module missing
try:
    from astram_lp_allocator import lp_allocate, _covered as _lp_covered
    _HAS_LP = True
except ImportError:
    _HAS_LP = False

OUTDIR    = "pipeline_out"
DOW_NAMES = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]

st.set_page_config(page_title="Congestion Console", page_icon="🚦", layout="wide")

# ----------------------------------------------------------------------
# Self-healing artifact loader
# ----------------------------------------------------------------------
def _rebuild():
    """Regenerate modeling_table.pkl + fitted_models.pkl from the raw dataset."""
    data = (sorted(glob.glob("*.xlsx")) + sorted(glob.glob("*.csv")))
    data = [d for d in data if "modeling_table" not in d and "predictions_log" not in d]
    if not data:
        st.error("No raw dataset (.xlsx/.csv) found to rebuild artifacts."); st.stop()
    with st.spinner(f"First-run setup: building models from {data[0]} ..."):
        subprocess.run([sys.executable, "astram_clean.py",  data[0]], check=True)
        subprocess.run([sys.executable, "models.py"], check=True)

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
    prof      = D.expected_load_surface(tbl, forecaster)
    chokes, choke_key = D.choke_point_list(tbl)
    cause_rates       = D.cause_rerouting_rates(tbl)
    geo               = D.corridor_geo(tbl)
    return dict(tbl=tbl, triage=triage, forecaster=forecaster, prof=prof,
                chokes=chokes, choke_key=choke_key,
                cause_rates=cause_rates, geo=geo)

A   = prepare()
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
st.title("Event-Driven Congestion Console")
st.caption("Forecast impact, allocate manpower, trigger barricading & diversion — "
           "all from the Astram event dataset.")

tab_map, tab_triage, tab_manpower, tab_barricade, tab_deep = st.tabs(
    ["Overview & Map", "Event Triage", "Manpower",
     "Barricading & Diversion", "Deep ST Forecast"])

# ----------------------------------------------------------------------
# TAB 1 — OVERVIEW & MAP
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
        causes  = st.multiselect("Filter by cause", opts("cause"))
        hi_only = st.checkbox("High-impact only (>= 60)")
    view = tbl.copy()
    if causes:
        view = view[view["cause"].isin(causes)]
    if hi_only:
        view = view[view["impact_score"] >= 60]
    with right:
        st.markdown(f"**Incident map** — {len(view):,} events shown, coloured by impact")
        render_map(view)

# ----------------------------------------------------------------------
# TAB 2 — EVENT TRIAGE
# ----------------------------------------------------------------------
with tab_triage:
    st.subheader("Assess an incoming event")
    st.caption("Fill in what's reported; the model predicts severity and the "
               "rules decide barricading + diversion.")
    a, b, c = st.columns(3)
    ev = {}
    ev["cause"]          = a.selectbox("Cause",          opts("cause"))
    ev["corridor_base"]  = a.selectbox("Corridor",       opts("corridor_base"))
    ev["police_station"] = b.selectbox("Police station", opts("police_station"))
    ev["zone"]           = b.selectbox("Zone",           opts("zone") or ["NA"])
    ev["veh_type"]       = c.selectbox("Vehicle type",   opts("veh_type") or ["NA"])
    ev["event_type"]     = c.selectbox("Event type",     opts("event_type") or ["unplanned"])
    h, d2 = st.columns(2)
    ev["hour"]       = h.slider("Hour of day", 0, 23, 18)
    dow_label        = d2.selectbox("Day of week", DOW_NAMES, index=0)
    ev["dow"]        = DOW_NAMES.index(dow_label)
    ev["month"]      = 3
    ev["is_weekend"] = float(ev["dow"] >= 5)
    g = A["geo"]
    if ev["corridor_base"] in g.index:
        ev["latitude"], ev["longitude"] = g.loc[ev["corridor_base"], ["lat", "lon"]]
    ev.setdefault("latitude",  tbl["latitude"].median())
    ev.setdefault("longitude", tbl["longitude"].median())
    ev["concurrent_load"] = 1

    if st.button("Assess event", type="primary"):
        res = D.handle_event(ev, (tbl, A["triage"], A["forecaster"]))
        tri = res["triage"]
        m1, m2, m3 = st.columns(3)
        if "priority_high_proba" in tri:
            m1.metric("Priority",
                      "High" if float(tri["priority_high_proba"]) >= 0.5 else "Low",
                      f"rule-based")
        if "requires_rerouting_proba" in tri:
            m2.metric("P(needs rerouting)", f"{float(tri['requires_rerouting_proba']):.0%}")
        if "duration_min" in tri:
            m3.metric("Est. clear time", f"{float(tri['duration_min']):.0f} min")

        bar = res["barricade"]
        if bar["decision"] == "BARRICADE":
            st.error(f"BARRICADE — {bar['level']}\n\n{bar['reason']}")
        else:
            st.success(f"MONITOR — {bar['reason']}")

        if res.get("diversion"):
            st.markdown("**Suggested diversion (near + lighter load):**")
            st.dataframe(pd.DataFrame(res["diversion"]).T, width="stretch")

# ----------------------------------------------------------------------
# TAB 3 — MANPOWER  (proportional or LP)
# ----------------------------------------------------------------------
with tab_manpower:
    st.subheader("Officer allocation by shift")

    # allocation method selector
    method_options = ["Proportional (default)"]
    if _HAS_LP:
        method_options.append("LP Optimal (ILP/CBC)")
    method = st.radio("Allocation method", method_options, horizontal=True,
                      help="LP Optimal finds the provably-optimal allocation under "
                           "the concave coverage objective using integer programming.")

    cc1, cc2, cc3, cc4 = st.columns(4)
    dow_l   = cc1.selectbox("Day",   DOW_NAMES,         key="mp_dow")
    shift   = cc2.selectbox("Shift", list(D.SHIFTS),    key="mp_shift")
    total   = cc3.slider("Officers available", 20, 400, D.TOTAL_OFFICERS, 10)
    min_cov = cc4.slider("Min per area",       0, 6,    D.MIN_COVERAGE)

    prof = A["prof"]
    sub  = prof[(prof["dow"] == DOW_NAMES.index(dow_l)) & (prof["shift"] == shift)]
    demand = dict(zip(sub[D.AREA_COL], sub["demand"]))

    # run chosen allocator
    if method.startswith("LP") and _HAS_LP:
        alloc, backend = lp_allocate(demand, total=total, min_cover=min_cov)
        st.caption(f"Solver: **{backend}** — allocation is provably optimal "
                   f"for the concave coverage objective.")
        # compute demand covered for both methods (for comparison)
        cap  = max(sum(d for d in demand.values() if d > 0) / total, 1e-9)
        prop = D.allocate_manpower(demand, total=total, min_cov=min_cov)
        cov_lp   = _lp_covered(demand, alloc, cap)
        cov_prop = _lp_covered(demand, prop,  cap)
        col_a, col_b = st.columns(2)
        col_a.metric("Demand covered (LP)",           f"{cov_lp:.2f}")
        col_b.metric("Demand covered (proportional)", f"{cov_prop:.2f}",
                     delta=f"{cov_lp - cov_prop:+.2f} from LP")
    else:
        alloc = D.allocate_manpower(demand, total=total, min_cov=min_cov)

    roster = (pd.DataFrame({"area":           list(demand),
                            "expected_load":  [round(demand[a], 2) for a in demand]})
              .assign(officers=lambda x: x["area"].map(alloc))
              .query("officers > 0")
              .sort_values("officers", ascending=False)
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
            ax.set_xlabel("officers")
            ax.set_title(f"{dow_l} — {shift} "
                         f"({'LP' if method.startswith('LP') else 'proportional'})")
            st.pyplot(fig)

    # LP vs proportional comparison table (only when LP is active)
    if method.startswith("LP") and _HAS_LP:
        with st.expander("LP vs proportional comparison"):
            cmp = (pd.DataFrame({
                    "area":         list(demand),
                    "demand":       [round(demand[a], 2) for a in demand],
                    "LP":           [alloc[a]  for a in demand],
                    "proportional": [prop[a]   for a in demand]})
                   .query("LP > 0 or proportional > 0")
                   .sort_values("demand", ascending=False)
                   .reset_index(drop=True))
            st.dataframe(cmp, width="stretch")

# ----------------------------------------------------------------------
# TAB 4 — BARRICADING & DIVERSION
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
    st.markdown("**Rerouting rate by cause** (>= 0.10 ⇒ barricade-prone)")
    st.dataframe(A["cause_rates"].round(3), width="stretch")

    st.divider()
    st.subheader("Diversion explorer")
    blocked = st.selectbox("If this corridor is blocked…",
                           sorted(A["geo"].index.tolist()))
    alts = D.suggest_diversion(blocked, A["geo"])
    if len(alts):
        st.dataframe(alts.reset_index(), width="stretch")
    else:
        st.info("No alternates available for this corridor in the data.")

# ----------------------------------------------------------------------
# TAB 5 — DEEP SPATIO-TEMPORAL FORECAST (ConvLSTM add-on)
# ----------------------------------------------------------------------
with tab_deep:
    st.subheader("Deep Spatio-Temporal Forecast (ConvLSTM)")
    st.caption(
        "An exploratory ConvLSTM rasterises the city into a 10×10 grid and learns "
        "spatial diffusion of incidents across cells. The CatBoost panel forecaster "
        "is the primary production model — this is presented as an architectural "
        "extension for denser / longer datasets."
    )

    heatmap_path = os.path.join(OUTDIR, "deep_forecaster_heatmaps.png")
    model_path   = os.path.join(OUTDIR, "deep_forecaster.pt")

    if os.path.exists(heatmap_path):
        # results already exist — just show them
        st.image(heatmap_path, width="stretch")
        st.markdown(
            "**Left:** mean actual incident density per grid cell (test window).  \n"
            "**Right:** ConvLSTM predicted density.  \n"
            "Brighter cells = more incidents. The model learns *where* incidents "
            "cluster spatially, not just *when* — picking up diffusion patterns "
            "invisible to the per-area panel model."
        )
        if os.path.exists(model_path):
            st.caption(f"Saved weights → `{model_path}`")

        # re-run button in case user wants fresh results
        with st.expander("Re-run the ConvLSTM (overwrites existing results)"):
            if st.button("Re-train ConvLSTM", key="retrain_deep"):
                with st.spinner("Training ConvLSTM on city grid (~2 min)..."):
                    result = subprocess.run(
                        [sys.executable, "astram_deep_forecaster.py"],
                        capture_output=True, text=True
                    )
                if result.returncode == 0:
                    st.success("Done! Refresh to see updated heatmaps.")
                    st.code(result.stdout[-800:])
                    st.rerun()
                else:
                    st.error("Training failed.")
                    st.code(result.stderr[-600:])
    else:
        # not run yet
        st.info(
            "The deep forecaster has not been run yet. "
            "Click below to train the ConvLSTM on the incident grid. "
            "This takes approximately 2–3 minutes."
        )
        col1, col2 = st.columns([1, 3])
        with col1:
            run_deep = st.button("Run ConvLSTM", type="primary", key="run_deep")
        with col2:
            st.caption("Requires PyTorch. The result is saved as a PNG and "
                       "displayed here automatically once complete.")

        if run_deep:
            with st.spinner("Building city raster and training ConvLSTM (~2 min)..."):
                result = subprocess.run(
                    [sys.executable, "astram_deep_forecaster.py"],
                    capture_output=True, text=True
                )
            if result.returncode == 0:
                st.success("Training complete!")
                st.code(result.stdout)
                st.rerun()
            else:
                st.error("Training failed. Make sure PyTorch is installed: "
                         "`pip install torch`")
                st.code(result.stderr[-800:])

        # show architecture explanation even before running
        with st.expander("How the ConvLSTM works"):
            st.markdown("""
**Architecture**

1. The city's lat/long bounding box is divided into a 10×10 raster.
2. For each 3-hour time bin, incidents are counted per cell — producing a spatial *frame*.
3. A ConvLSTM is trained to predict the next frame from the past 8 frames,
   learning both temporal patterns (time-of-day, day-of-week) and
   spatial diffusion (incidents on one road affecting adjacent cells).
4. Compared against a *persistence* baseline (predict the last frame as the next).

**Why we keep CatBoost as primary**

On 5 months of sparse data (~8,000 events across 150 days), the 10×10 grid 
has only ~0.024 incidents per cell per 3-hour bin. The CatBoost panel model 
operates on 55 named police-station areas with rich lag features, giving it 
more signal per training example. The ConvLSTM needs denser, longer data to 
show its architectural advantages — it is presented here as a pathway for 
future extension.
            """)