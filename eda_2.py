#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
====================================================================
 ASTraM Event-Driven Congestion — Preliminary Data Analysis (EDA)
====================================================================
Hackathon theme: forecast event-related traffic impact and recommend
optimal manpower, barricading and diversion plans.

What this script does
---------------------
1. Loads the anonymized Astram event dataset (.xlsx or .csv).
2. Repairs the garbled Kannada text (UTF-8 read as Windows-1252 -> mojibake).
3. Cleans the data (string "NULL" -> NaN, parses dates, builds features).
4. Runs a full EDA and saves ~20 charts as PNGs to ./eda_outputs/.
5. Writes a plain-text summary report + an actionable hotspot table (CSV).

Usage
-----
    python traffic_eda.py "Astram event data_anonymized.xlsx"
    python traffic_eda.py                # uses DEFAULT_PATH below

Requires: pandas, numpy, matplotlib, seaborn, openpyxl  (ftfy optional)
    pip install pandas numpy matplotlib seaborn openpyxl ftfy
"""

import os
import sys
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")            # save to file, no GUI needed
import matplotlib.pyplot as plt
import seaborn as sns

warnings.filterwarnings("ignore")
sns.set_theme(style="whitegrid")
plt.rcParams.update({"figure.dpi": 110, "savefig.bbox": "tight", "font.size": 10})

DEFAULT_PATH = "Astram event data_anonymized.xlsx"
OUTDIR = "eda_outputs"
os.makedirs(OUTDIR, exist_ok=True)
_fig_n = 0


def save(fig, name):
    """Save a figure with an auto-incrementing numeric prefix."""
    global _fig_n
    _fig_n += 1
    path = os.path.join(OUTDIR, f"{_fig_n:02d}_{name}.png")
    fig.savefig(path)
    plt.close(fig)
    print(f"   saved  {path}")


# ----------------------------------------------------------------------
# 1. KANNADA MOJIBAKE REPAIR
# ----------------------------------------------------------------------
# The text was originally UTF-8 Kannada but got decoded as Windows-1252,
# producing sequences like 'à²¸à²®'. We reverse it: re-encode to cp1252
# bytes, then decode as UTF-8. ftfy is used as a fallback when present.
try:
    import ftfy
    _HAS_FTFY = True
except ImportError:
    _HAS_FTFY = False

_MOJIBAKE_FLAGS = ("Ã", "Â", "à", "á", "â", "Ÿ", "€", "š", "›", "œ", "¬", "³", "²")


def fix_mojibake(text):
    if not isinstance(text, str) or text == "":
        return text
    if not any(flag in text for flag in _MOJIBAKE_FLAGS):
        return text  # already clean ASCII / English
    for enc in ("cp1252", "latin-1"):
        try:
            repaired = text.encode(enc, errors="strict").decode("utf-8", errors="strict")
            if repaired and "\ufffd" not in repaired:
                return repaired
        except (UnicodeEncodeError, UnicodeDecodeError):
            pass
    if _HAS_FTFY:
        return ftfy.fix_text(text)
    # last resort: lossy cp1252 pass
    try:
        return text.encode("cp1252", errors="replace").decode("utf-8", errors="replace")
    except Exception:
        return text


# ----------------------------------------------------------------------
# 2. LOAD + CLEAN
# ----------------------------------------------------------------------
def load(path):
    print(f"Loading: {path}")
    if path.lower().endswith((".xlsx", ".xls", ".xlsm")):
        df = pd.read_excel(path, engine="openpyxl")
    else:
        # try a few encodings for csv
        for enc in ("utf-8", "utf-8-sig", "cp1252", "latin-1"):
            try:
                df = pd.read_csv(path, encoding=enc)
                break
            except Exception:
                continue
    print(f"   shape: {df.shape[0]} rows x {df.shape[1]} cols")
    return df


def clean(df):
    df = df.copy()
    df.columns = [str(c).strip() for c in df.columns]

    # treat the literal string "NULL" (and blanks) as missing
    df = df.replace(["NULL", "null", "", "[]", "None"], np.nan)

    # repair text columns. Detect by actual content (isinstance str) so it works
    # whether pandas reports the column as object / string / str dtype.
    text_cols = [c for c in df.columns
                 if df[c].map(lambda x: isinstance(x, str)).any()]
    for c in text_cols:
        df[c] = df[c].map(fix_mojibake)

    # parse all datetime-ish columns
    date_like = [c for c in df.columns
                 if any(k in c.lower() for k in
                        ("date", "datetime", "time", "modified", "created", "closed", "resolved"))]
    for c in date_like:
        df[c] = pd.to_datetime(df[c], errors="coerce", utc=True).dt.tz_localize(None)

    # numeric coordinates
    for c in ("latitude", "longitude", "endlatitude", "endlongitude",
              "resolved_latitude", "resolved_longitude"):
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
            df.loc[df[c] == 0, c] = np.nan  # 0 is a placeholder, not a real coord

    return df


def col(df, *candidates):
    """Return the first matching column name (case-insensitive contains)."""
    low = {c.lower(): c for c in df.columns}
    for cand in candidates:
        if cand.lower() in low:
            return low[cand.lower()]
    for cand in candidates:
        for c in df.columns:
            if cand.lower() in c.lower():
                return c
    return None


def add_features(df):
    df = df.copy()
    start = col(df, "start_datetime", "start_date", "created_date")
    end = col(df, "closed_date", "resolved_date", "end_datetime", "modified_date")

    if start:
        s = df[start]
        df["_start"] = s
        df["hour"] = s.dt.hour
        df["dayofweek"] = s.dt.day_name()
        df["month"] = s.dt.to_period("M").astype(str)
        df["date"] = s.dt.date
        df["is_weekend"] = s.dt.dayofweek >= 5
        df["is_peak"] = s.dt.hour.isin([8, 9, 10, 17, 18, 19, 20])

    if start and end:
        dur = (df[end] - df[start]).dt.total_seconds() / 60.0
        dur = dur.where((dur > 0) & (dur < 7 * 24 * 60))  # drop negatives & >1 week
        df["duration_min"] = dur

    return df


# ----------------------------------------------------------------------
# 3. ANALYSIS + CHARTS
# ----------------------------------------------------------------------
def order_counts(series, top=None):
    vc = series.dropna().value_counts()
    return vc.head(top) if top else vc


def run(df):
    C = lambda *a: col(df, *a)
    print("\n=== DATA QUALITY ===")
    miss = (df.isna().mean() * 100).round(1).sort_values(ascending=False)
    print("Missing % (top 15):")
    print(miss.head(15).to_string())

    # ---- missingness bar -------------------------------------------------
    fig, ax = plt.subplots(figsize=(9, 7))
    nonempty = miss[miss > 0].head(25)[::-1]
    ax.barh(nonempty.index, nonempty.values, color="#c0392b")
    ax.set_xlabel("% missing"); ax.set_title("Column completeness (top missing)")
    save(fig, "missingness")

    # ---- planned vs unplanned -------------------------------------------
    et = C("event_type")
    if et:
        fig, ax = plt.subplots(figsize=(6, 5))
        vc = order_counts(df[et])
        ax.bar(vc.index, vc.values, color=["#2980b9", "#e67e22"])
        for i, v in enumerate(vc.values):
            ax.text(i, v, f"{v}\n({v/vc.sum()*100:.0f}%)", ha="center", va="bottom")
        ax.set_title("Planned vs Unplanned events"); ax.set_ylabel("count")
        save(fig, "planned_vs_unplanned")

    # ---- event cause ----------------------------------------------------
    cause = C("event_cause")
    if cause:
        fig, ax = plt.subplots(figsize=(9, 5))
        vc = order_counts(df[cause])
        sns.barplot(x=vc.values, y=vc.index, ax=ax, palette="viridis")
        ax.set_title("Event cause distribution"); ax.set_xlabel("count")
        save(fig, "event_cause")

        # cause x event_type stacked
        if et:
            ct = pd.crosstab(df[cause], df[et])
            fig, ax = plt.subplots(figsize=(9, 5))
            ct.plot(kind="barh", stacked=True, ax=ax, colormap="Set2")
            ax.set_title("Event cause by event type"); ax.set_xlabel("count")
            save(fig, "cause_by_type")

    # ---- temporal: hour of day ------------------------------------------
    if "hour" in df.columns:
        fig, ax = plt.subplots(figsize=(10, 4.5))
        hourly = df["hour"].value_counts().sort_index()
        bars = ax.bar(hourly.index, hourly.values, color="#16a085")
        for h in [8, 9, 10, 17, 18, 19, 20]:
            if h in hourly.index:
                bars[list(hourly.index).index(h)].set_color("#e74c3c")
        ax.set_title("Events by hour of day (red = peak hours)")
        ax.set_xlabel("hour"); ax.set_ylabel("count"); ax.set_xticks(range(0, 24))
        save(fig, "by_hour")

    # ---- temporal: day of week ------------------------------------------
    if "dayofweek" in df.columns:
        order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
        fig, ax = plt.subplots(figsize=(9, 4.5))
        dow = df["dayofweek"].value_counts().reindex(order).fillna(0)
        sns.barplot(x=dow.index, y=dow.values, ax=ax, palette="crest")
        ax.set_title("Events by day of week"); ax.set_ylabel("count")
        plt.xticks(rotation=30)
        save(fig, "by_dayofweek")

    # ---- temporal: monthly trend ----------------------------------------
    if "month" in df.columns:
        fig, ax = plt.subplots(figsize=(9, 4.5))
        m = df["month"].value_counts().sort_index()
        ax.plot(m.index, m.values, marker="o", color="#8e44ad")
        ax.set_title("Monthly event trend"); ax.set_ylabel("count")
        plt.xticks(rotation=45)
        save(fig, "monthly_trend")

    # ---- hour x dayofweek heatmap (when to staff) -----------------------
    if "hour" in df.columns and "dayofweek" in df.columns:
        order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
        piv = (df.pivot_table(index="dayofweek", columns="hour", values=C("id") or df.columns[0],
                              aggfunc="count").reindex(order))
        fig, ax = plt.subplots(figsize=(12, 4.5))
        sns.heatmap(piv, cmap="rocket_r", ax=ax, cbar_kws={"label": "events"})
        ax.set_title("Demand heatmap: day of week x hour (manpower planning)")
        save(fig, "heatmap_day_hour")

    # ---- spatial: corridor / zone / police station / junction -----------
    for key, fname, title in [
        (C("corridor"), "by_corridor", "Events by corridor"),
        (C("zone"), "by_zone", "Events by zone"),
        (C("police_station", "police_sta"), "by_police_station", "Events by police station"),
        (C("junction"), "by_junction", "Events by junction (top 15)"),
    ]:
        if key and df[key].notna().any():
            vc = order_counts(df[key], top=15)
            fig, ax = plt.subplots(figsize=(9, max(4, 0.4 * len(vc))))
            sns.barplot(x=vc.values, y=vc.index, ax=ax, palette="mako")
            ax.set_title(title); ax.set_xlabel("count")
            save(fig, fname)

    # ---- spatial scatter (geographic spread) ----------------------------
    lat, lng = C("latitude"), C("longitude")
    if lat and lng and df[lat].notna().sum() > 5:
        fig, ax = plt.subplots(figsize=(7, 7))
        hue = df[cause] if cause else None
        sns.scatterplot(x=df[lng], y=df[lat], hue=hue, s=25, alpha=0.6, ax=ax)
        ax.set_title("Geographic spread of events"); ax.set_xlabel("longitude"); ax.set_ylabel("latitude")
        if hue is not None:
            ax.legend(title="cause", bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=7)
        save(fig, "geo_scatter")

        # density hexbin
        sub = df[[lat, lng]].dropna()
        fig, ax = plt.subplots(figsize=(7, 7))
        hb = ax.hexbin(sub[lng], sub[lat], gridsize=25, cmap="inferno", mincnt=1)
        fig.colorbar(hb, ax=ax, label="events")
        ax.set_title("Incident density (hexbin hotspots)")
        ax.set_xlabel("longitude"); ax.set_ylabel("latitude")
        save(fig, "geo_density")

    # ---- priority -------------------------------------------------------
    pri = C("priority")
    if pri:
        fig, ax = plt.subplots(figsize=(6, 5))
        vc = order_counts(df[pri])
        ax.pie(vc.values, labels=vc.index, autopct="%1.0f%%",
               colors=sns.color_palette("Set1"), startangle=90)
        ax.set_title("Priority split")
        save(fig, "priority_pie")

        if cause:
            ct = pd.crosstab(df[cause], df[pri], normalize="index") * 100
            fig, ax = plt.subplots(figsize=(9, 5))
            ct.plot(kind="barh", stacked=True, ax=ax, colormap="coolwarm")
            ax.set_title("Priority mix by event cause (%)"); ax.set_xlabel("%")
            save(fig, "priority_by_cause")

    # ---- rerouting / barricading signal ---------------------------------
    rr = C("requires_rerouting", "requires_r")
    if rr:
        fig, ax = plt.subplots(figsize=(6, 4.5))
        vc = order_counts(df[rr])
        ax.bar(vc.index.astype(str), vc.values, color=["#27ae60", "#c0392b"])
        ax.set_title("Requires rerouting / diversion?"); ax.set_ylabel("count")
        save(fig, "rerouting")

    # ---- vehicle type ---------------------------------------------------
    vt = C("veh_type")
    if vt and df[vt].notna().any():
        fig, ax = plt.subplots(figsize=(8, 4.5))
        vc = order_counts(df[vt], top=12)
        sns.barplot(x=vc.values, y=vc.index, ax=ax, palette="flare")
        ax.set_title("Vehicle type involved"); ax.set_xlabel("count")
        save(fig, "vehicle_type")

    # ---- duration / resolution time (manpower demand) -------------------
    if "duration_min" in df.columns and df["duration_min"].notna().sum() > 5:
        d = df["duration_min"].dropna()
        print("\n=== RESOLUTION TIME (minutes) ===")
        print(d.describe().round(1).to_string())

        fig, ax = plt.subplots(figsize=(9, 4.5))
        ax.hist(d.clip(upper=d.quantile(0.95)), bins=40, color="#2c3e50")
        ax.set_title("Event duration / time-to-clear (clipped at 95th pct)")
        ax.set_xlabel("minutes"); ax.set_ylabel("count")
        save(fig, "duration_hist")

        if cause:
            fig, ax = plt.subplots(figsize=(10, 5))
            order = df.groupby(cause)["duration_min"].median().sort_values(ascending=False).index
            sns.boxplot(data=df[df["duration_min"] < d.quantile(0.95)],
                        x="duration_min", y=cause, order=order, ax=ax, palette="Spectral")
            ax.set_title("Time-to-clear by cause (which events tie up resources longest)")
            ax.set_xlabel("minutes")
            save(fig, "duration_by_cause")

    # ---- status ---------------------------------------------------------
    st = C("status")
    if st:
        fig, ax = plt.subplots(figsize=(6, 4.5))
        vc = order_counts(df[st])
        sns.barplot(x=vc.index, y=vc.values, ax=ax, palette="Blues_d")
        ax.set_title("Event status"); ax.set_ylabel("count")
        save(fig, "status")

    # ------------------------------------------------------------------
    # ACTIONABLE HOTSPOT TABLE  (corridor x cause ranked by volume)
    # ------------------------------------------------------------------
    corr = C("corridor")
    if corr and cause:
        hot = (df.groupby([corr, cause]).size().reset_index(name="events")
                 .sort_values("events", ascending=False))
        if "duration_min" in df.columns:
            med = (df.groupby([corr, cause])["duration_min"].median()
                     .reset_index(name="median_clear_min"))
            hot = hot.merge(med, on=[corr, cause], how="left")
        hot.to_csv(os.path.join(OUTDIR, "hotspot_ranking.csv"), index=False)
        print("\n=== TOP 10 HOTSPOTS (corridor x cause) ===")
        print(hot.head(10).to_string(index=False))

    # ------------------------------------------------------------------
    # TEXT SUMMARY REPORT
    # ------------------------------------------------------------------
    lines = ["ASTraM EVENT-DRIVEN CONGESTION — EDA SUMMARY", "=" * 48,
             f"Total events: {len(df)}",
             f"Date range : {df['_start'].min()}  ->  {df['_start'].max()}" if "_start" in df else "",
             ]
    if et:
        for k, v in df[et].value_counts(normalize=True).items():
            lines.append(f"  {k:<12}: {v*100:.1f}%")
    if cause:
        lines.append("\nTop causes:")
        for k, v in order_counts(df[cause]).head(5).items():
            lines.append(f"  {k:<18}: {v}")
    if "is_peak" in df.columns:
        lines.append(f"\nShare during peak hours (8-10,17-20): {df['is_peak'].mean()*100:.1f}%")
    if "duration_min" in df.columns and df["duration_min"].notna().any():
        lines.append(f"Median time-to-clear: {df['duration_min'].median():.0f} min "
                     f"(90th pct {df['duration_min'].quantile(0.9):.0f} min)")
    report = "\n".join([l for l in lines if l])
    with open(os.path.join(OUTDIR, "summary_report.txt"), "w", encoding="utf-8") as f:
        f.write(report)
    print("\n" + report)
    print(f"\nAll charts + report written to ./{OUTDIR}/")


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_PATH
    if not os.path.exists(path):
        sys.exit(f"File not found: {path}\nPass the path: python traffic_eda.py <file.xlsx>")
    df = load(path)
    df = clean(df)
    df = add_features(df)
    run(df)


if __name__ == "__main__":
    main()