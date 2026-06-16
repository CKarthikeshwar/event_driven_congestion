#!/usr/bin/env python3
"""
Preliminary EDA for the Bengaluru parking-violation dataset (plot-first).

What it does
------------
Loads the violations file, cleans it (parses the JSON-array columns, fixes
NULLs, coerces dates and lat/long), then writes a directory of figures plus a
couple of CSV/TXT summaries. Designed so the *graphs* carry the analysis.

It produces:
  data quality   -> missingness, duplicates, lat/long validity
  distributions  -> violation type, vehicle type, offence codes, stations,
                    validation status, data_sent
  temporal       -> daily volume, hour-of-day, day-of-week, hour x DOW heatmap
  spatial        -> hexbin density, top grid-cell hotspots, folium heatmap (HTML)
  enforcement    -> officer / device concentration (the patrol-bias check)
  cross-tabs     -> vehicle x violation, station x violation, rejection rates

Usage
-----
    pip install pandas numpy matplotlib openpyxl
    # optional but recommended:
    pip install seaborn folium

    python parking_eda.py "jan to may police violation_ano.xlsx"
    python parking_eda.py data.csv --outdir eda_outputs

Nothing about this script assumes a particular header spelling: it resolves
columns by fuzzy matching and prints what it found, so adjust CANONICAL below
if anything is mis-resolved.
"""

from __future__ import annotations

import argparse
import ast
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")  # headless / save-to-file
import matplotlib.pyplot as plt

try:
    import seaborn as sns

    sns.set_theme(style="whitegrid")
    HAVE_SNS = True
except Exception:
    HAVE_SNS = False

try:
    import folium
    from folium.plugins import HeatMap

    HAVE_FOLIUM = True
except Exception:
    HAVE_FOLIUM = False


# --------------------------------------------------------------------------- #
# Config
# --------------------------------------------------------------------------- #

# Rough bounding box for Bengaluru; points outside this are flagged as bad geo.
BLR_BBOX = dict(lat_min=12.6, lat_max=13.3, lon_min=77.3, lon_max=77.9)

# Canonical column name -> list of candidate header spellings (case-insensitive).
# First exact match wins; otherwise a substring fallback is tried.
CANONICAL = {
    "id": ["id"],
    "lat": ["latitude", "lat"],
    "lon": ["longitude", "long", "lon", "lng"],
    "location": ["location", "address"],
    "vehicle_number": ["vehicle_number", "vehicle_no", "vehicle_num"],
    "vehicle_type": ["vehicle_type"],
    "description": ["description"],
    "violation_type": ["violation_type", "violation_types"],
    "offence_code": ["offence_code", "offence_codes", "offense_code"],
    "created_date": ["created_date", "created_at", "created"],
    "closed_date": ["closed_date", "closed_at"],
    "modified_date": ["modified_date", "modified_at", "updated_date"],
    "device_id": ["device_id"],
    "created_by": ["created_by", "user_id", "officer_id"],
    "center_code": ["center_code", "centre_code"],
    "police_station": ["police_station", "police_sta", "station"],
    "data_sent": ["data_sent"],
    "junction_name": ["junction_name", "junction"],
    "action_taken": ["action_taken"],
    "validation_status": ["validation_status", "validation"],
    "validation_timestamp": ["validation_timestamp", "validated_at"],
}

NULLISH = {"NULL", "null", "None", "none", "NaN", "nan", "", "NA", "N/A"}


# --------------------------------------------------------------------------- #
# Small helpers
# --------------------------------------------------------------------------- #

def resolve_columns(df: pd.DataFrame) -> dict:
    """Map canonical names to the actual columns present in df."""
    lower = {c.lower().strip(): c for c in df.columns}
    resolved, unresolved = {}, []
    for canon, cands in CANONICAL.items():
        hit = None
        for cand in cands:  # exact (case-insensitive) first
            if cand.lower() in lower:
                hit = lower[cand.lower()]
                break
        if hit is None:  # substring fallback on the most distinctive token
            token = cands[0].lower().split("_")[0]
            for lc, orig in lower.items():
                if token in lc:
                    hit = orig
                    break
        if hit is not None:
            resolved[canon] = hit
        else:
            unresolved.append(canon)
    print("Resolved columns:")
    for k, v in resolved.items():
        print(f"  {k:22s} -> {v}")
    if unresolved:
        print("Could NOT resolve (analyses using these are skipped):")
        for u in unresolved:
            print(f"  {u}")
    return resolved


def parse_listish(val):
    """Turn '[112,104]' / '[\"WRONG PARKING\"]' into a Python list."""
    if isinstance(val, list):
        return val
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return np.nan
    s = str(val).strip()
    if s in NULLISH:
        return np.nan
    for parser in (ast.literal_eval, json.loads):
        try:
            out = parser(s)
            return out if isinstance(out, list) else [out]
        except Exception:
            continue
    # last resort: strip brackets/quotes and split on comma
    s2 = s.strip("[]").replace('"', "").replace("'", "")
    return [t.strip() for t in s2.split(",") if t.strip()]


def first_or_nan(x):
    if isinstance(x, list) and x:
        return x[0]
    if isinstance(x, str):
        return x
    return np.nan


def has_time_component(s: pd.Series) -> bool:
    """True if a datetime series carries meaningful time-of-day variation."""
    s = s.dropna()
    if s.empty:
        return False
    return s.dt.hour.nunique() > 1 or s.dt.minute.nunique() > 1


def save_fig(fig, outdir: Path, name: str):
    path = outdir / name
    fig.tight_layout()
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {name}")


def barh_counts(series, title, outdir, fname, top=None, color="#377AB7"):
    vc = series.value_counts(dropna=False)
    if top:
        vc = vc.head(top)
    vc = vc.iloc[::-1]  # largest at top after barh
    fig, ax = plt.subplots(figsize=(8, max(3, 0.4 * len(vc) + 1)))
    ax.barh([str(i) for i in vc.index], vc.values, color=color)
    ax.set_title(title)
    ax.set_xlabel("count")
    for i, v in enumerate(vc.values):
        ax.text(v, i, f" {v:,}", va="center", fontsize=8)
    save_fig(fig, outdir, fname)


# --------------------------------------------------------------------------- #
# Load + clean
# --------------------------------------------------------------------------- #

def load(path: Path, sheet) -> pd.DataFrame:
    if path.suffix.lower() in (".xlsx", ".xlsm", ".xls"):
        df = pd.read_excel(path, sheet_name=sheet)
    else:
        df = pd.read_csv(path)
    print(f"Loaded {path.name}: {df.shape[0]:,} rows x {df.shape[1]} cols\n")
    return df


def clean(df: pd.DataFrame, col: dict) -> pd.DataFrame:
    df = df.copy()

    # NULL-strings -> NaN across object/string columns
    obj_cols = df.select_dtypes(include=["object", "str"]).columns
    df[obj_cols] = df[obj_cols].apply(
        lambda s: s.map(lambda v: np.nan if str(v).strip() in NULLISH else v)
    )

    # list-ish columns
    for key in ("violation_type", "offence_code"):
        if key in col:
            df[col[key]] = df[col[key]].map(parse_listish)

    # primary violation type, normalized
    if "violation_type" in col:
        df["violation_primary"] = (
            df[col["violation_type"]].map(first_or_nan).str.upper().str.strip()
        )

    # number of offences per record
    if "offence_code" in col:
        df["num_offences"] = df[col["offence_code"]].map(
            lambda x: len(x) if isinstance(x, list) else np.nan
        )

    # dates
    for key in ("created_date", "closed_date", "modified_date",
                "validation_timestamp"):
        if key in col:
            df[col[key]] = pd.to_datetime(df[col[key]], errors="coerce", utc=True)

    # lat / lon numeric + validity flag
    if "lat" in col and "lon" in col:
        df[col["lat"]] = pd.to_numeric(df[col["lat"]], errors="coerce")
        df[col["lon"]] = pd.to_numeric(df[col["lon"]], errors="coerce")
        df["geo_valid"] = (
            df[col["lat"]].between(BLR_BBOX["lat_min"], BLR_BBOX["lat_max"])
            & df[col["lon"]].between(BLR_BBOX["lon_min"], BLR_BBOX["lon_max"])
        )

    return df


def pick_intraday_time(df, col):
    """Choose the timestamp column with real time-of-day variation."""
    for key in ("created_date", "validation_timestamp", "modified_date"):
        if key in col and pd.api.types.is_datetime64_any_dtype(df[col[key]]):
            if has_time_component(df[col[key]]):
                print(f"Using '{col[key]}' for intraday (hour/DOW) analysis.\n")
                return col[key]
    print("No column has time-of-day variation; intraday plots skipped.\n")
    return None


# --------------------------------------------------------------------------- #
# Analyses
# --------------------------------------------------------------------------- #

def quality_report(df, col, outdir, log):
    print("== Data quality ==")
    # missingness
    miss = df.isna().mean().sort_values() * 100
    fig, ax = plt.subplots(figsize=(8, max(3, 0.3 * len(miss) + 1)))
    ax.barh(miss.index, miss.values, color="#C0504D")
    ax.set_xlabel("% missing")
    ax.set_title("Missing values by column")
    save_fig(fig, outdir, "01_missingness.png")

    # df.duplicated() can't hash list-valued columns; stringify them first
    df_hash = df.apply(
        lambda s: s.map(lambda v: str(v) if isinstance(v, list) else v)
    )
    n_dupe = df_hash.duplicated().sum()
    log.append(f"Total rows: {len(df):,}")
    log.append(f"Exact duplicate rows: {n_dupe:,}")
    if "id" in col:
        log.append(f"Duplicate ids: {df[col['id']].duplicated().sum():,}")
    if "geo_valid" in df:
        bad = (~df["geo_valid"]).sum()
        log.append(f"Out-of-Bengaluru / missing geo: {bad:,} "
                   f"({bad / len(df):.1%})")
    print(f"  duplicates={n_dupe:,}  (see summary.txt)\n")


def distributions(df, col, outdir):
    print("== Distributions ==")
    if "violation_primary" in df:
        barh_counts(df["violation_primary"], "Violation type",
                    outdir, "02_violation_type.png")
    if "vehicle_type" in col:
        barh_counts(df[col["vehicle_type"]], "Vehicle type",
                    outdir, "03_vehicle_type.png", color="#4F8A5B")
    if "offence_code" in col:
        codes = df[col["offence_code"]].dropna().explode().astype(str)
        barh_counts(codes, "Offence codes (exploded)",
                    outdir, "04_offence_codes.png", color="#8064A2")
    if "police_station" in col:
        barh_counts(df[col["police_station"]], "Top 20 police stations",
                    outdir, "05_police_stations.png", top=20, color="#E69138")
    if "validation_status" in col:
        barh_counts(df[col["validation_status"]], "Validation status",
                    outdir, "06_validation_status.png", color="#3D85C6")
    if "data_sent" in col:
        barh_counts(df[col["data_sent"]].astype(str), "data_sent flag",
                    outdir, "07_data_sent.png", color="#999999")
    print()


def temporal(df, col, time_col, outdir, log):
    if time_col is None:
        return
    print("== Temporal ==")
    t = df[time_col].dropna()
    log.append(f"Date range: {t.min()}  ->  {t.max()}")

    # daily volume
    daily = t.dt.date.value_counts().sort_index()
    fig, ax = plt.subplots(figsize=(11, 3.5))
    ax.plot(pd.to_datetime(daily.index), daily.values, lw=1)
    ax.set_title("Violations per day")
    ax.set_ylabel("count")
    save_fig(fig, outdir, "08_daily_volume.png")

    # hour of day
    fig, ax = plt.subplots(figsize=(9, 3.5))
    hr = t.dt.hour.value_counts().sort_index()
    ax.bar(hr.index, hr.values, color="#377AB7")
    ax.set_xticks(range(0, 24))
    ax.set_title("Violations by hour of day")
    ax.set_xlabel("hour")
    save_fig(fig, outdir, "09_hour_of_day.png")

    # day of week
    order = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
    dow = t.dt.dayofweek.value_counts().sort_index()
    fig, ax = plt.subplots(figsize=(7, 3.5))
    ax.bar([order[i] for i in dow.index], dow.values, color="#4F8A5B")
    ax.set_title("Violations by day of week")
    save_fig(fig, outdir, "10_day_of_week.png")

    # hour x DOW heatmap
    piv = (
        pd.crosstab(t.dt.dayofweek, t.dt.hour)
        .reindex(range(7))
        .reindex(columns=range(24))
        .fillna(0)
    )
    fig, ax = plt.subplots(figsize=(11, 4))
    im = ax.imshow(piv.values, aspect="auto", cmap="viridis")
    ax.set_yticks(range(7))
    ax.set_yticklabels(order)
    ax.set_xticks(range(24))
    ax.set_xticklabels(range(24))
    ax.set_xlabel("hour")
    ax.set_title("Violation intensity: day of week x hour")
    fig.colorbar(im, ax=ax, label="count")
    save_fig(fig, outdir, "11_hour_dow_heatmap.png")
    print()


def spatial(df, col, outdir, log):
    if "lat" not in col or "lon" not in col:
        return
    print("== Spatial ==")
    g = df[df.get("geo_valid", True)].copy()
    lat, lon = col["lat"], col["lon"]

    # hexbin density
    fig, ax = plt.subplots(figsize=(8, 8))
    hb = ax.hexbin(g[lon], g[lat], gridsize=60, cmap="inferno", mincnt=1)
    ax.set_title("Violation density (hexbin)")
    ax.set_xlabel("longitude")
    ax.set_ylabel("latitude")
    fig.colorbar(hb, ax=ax, label="count")
    save_fig(fig, outdir, "12_hexbin_density.png")

    # top grid-cell hotspots (~110 m cells via 3-decimal rounding)
    g["cell_lat"] = g[lat].round(3)
    g["cell_lon"] = g[lon].round(3)
    grp = (
        g.groupby(["cell_lat", "cell_lon"])
        .size()
        .reset_index(name="count")
        .sort_values("count", ascending=False)
    )
    if "location" in col:
        names = (
            g.groupby(["cell_lat", "cell_lon"])[col["location"]]
            .agg(lambda s: s.dropna().mode().iat[0] if not s.dropna().empty else "")
        )
        grp = grp.merge(names.rename("location"), on=["cell_lat", "cell_lon"])
    grp.head(50).to_csv(outdir / "top_hotspots.csv", index=False)
    log.append(f"Top hotspot cell: {grp.iloc[0].to_dict()}")
    print(f"  wrote top_hotspots.csv  (top cell has {grp['count'].max():,} hits)")

    # folium interactive heatmap
    if HAVE_FOLIUM:
        m = folium.Map(location=[g[lat].median(), g[lon].median()],
                       zoom_start=12, tiles="cartodbpositron")
        HeatMap(g[[lat, lon]].values.tolist(), radius=9, blur=7).add_to(m)
        m.save(str(outdir / "hotspot_heatmap.html"))
        print("  wrote hotspot_heatmap.html (open in a browser)")
    else:
        print("  folium not installed -> skipped interactive heatmap")
    print()


def enforcement_bias(df, col, outdir, log):
    """Patrol-based data over-represents wherever officers actually went."""
    print("== Enforcement / selection bias ==")
    for key, fname, title in (
        ("created_by", "13_officer_concentration.png", "Violations per officer (top 25)"),
        ("device_id", "14_device_concentration.png", "Violations per device (top 25)"),
    ):
        if key not in col:
            continue
        vc = df[col[key]].value_counts()
        share_top10 = vc.head(10).sum() / vc.sum()
        log.append(f"{key}: {vc.size:,} unique, top-10 produce {share_top10:.1%}")
        barh_counts(df[col[key]], title, outdir, fname, top=25, color="#A64D79")
        print(f"  {key}: {vc.size:,} unique, top-10 share={share_top10:.1%}")
    print()


def cross_tabs(df, col, outdir):
    print("== Cross-tabs ==")
    # vehicle x violation
    if "vehicle_type" in col and "violation_primary" in df:
        ct = pd.crosstab(df[col["vehicle_type"]], df["violation_primary"])
        fig, ax = plt.subplots(figsize=(9, max(4, 0.4 * len(ct) + 1)))
        im = ax.imshow(ct.values, aspect="auto", cmap="magma")
        ax.set_xticks(range(ct.shape[1]))
        ax.set_xticklabels(ct.columns, rotation=30, ha="right")
        ax.set_yticks(range(ct.shape[0]))
        ax.set_yticklabels(ct.index)
        ax.set_title("Vehicle type x violation type")
        fig.colorbar(im, ax=ax, label="count")
        save_fig(fig, outdir, "15_vehicle_x_violation.png")

    # station x violation (stacked, top stations)
    if "police_station" in col and "violation_primary" in df:
        top_st = df[col["police_station"]].value_counts().head(15).index
        sub = df[df[col["police_station"]].isin(top_st)]
        ct = pd.crosstab(sub[col["police_station"]], sub["violation_primary"])
        ct = ct.loc[ct.sum(1).sort_values().index]
        fig, ax = plt.subplots(figsize=(9, 6))
        ct.plot(kind="barh", stacked=True, ax=ax, colormap="tab20")
        ax.set_title("Top 15 stations x violation type")
        ax.set_xlabel("count")
        save_fig(fig, outdir, "16_station_x_violation.png")

    # rejection rate by station
    if "police_station" in col and "validation_status" in col:
        vs = df[col["validation_status"]].astype(str).str.lower()
        tmp = df.assign(_rej=(vs == "rejected").astype(float))
        rate = (
            tmp.groupby(col["police_station"])["_rej"].mean()
            .sort_values(ascending=False).head(20)
        )
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.barh(rate.index[::-1], (rate.values[::-1] * 100), color="#CC0000")
        ax.set_xlabel("% rejected")
        ax.set_title("Rejection rate by station (top 20)")
        save_fig(fig, outdir, "17_rejection_rate.png")
    print()


def column_summary(df, outdir):
    rows = []
    for c in df.columns:
        s = df[c]
        # list-valued columns (parsed offence_code / violation_type) are
        # unhashable -> convert lists to tuples so unique()/nunique() work.
        hashable = s.map(lambda v: tuple(v) if isinstance(v, list) else v)
        ex = hashable.dropna().unique()[:3]
        rows.append({
            "column": c,
            "dtype": str(s.dtype),
            "n_unique": hashable.nunique(dropna=True),
            "pct_missing": round(s.isna().mean() * 100, 1),
            "examples": " | ".join(map(lambda x: str(x)[:25], ex)),
        })
    pd.DataFrame(rows).to_csv(outdir / "column_summary.csv", index=False)
    print("Wrote column_summary.csv")


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #

def main():
    ap = argparse.ArgumentParser(description="Plot-first EDA for parking data.")
    ap.add_argument("path", help="path to the .xlsx or .csv file")
    ap.add_argument("--sheet", default=0, help="sheet name/index for xlsx")
    ap.add_argument("--outdir", default="eda_outputs")
    args = ap.parse_args()

    path = Path(args.path)
    if not path.exists():
        sys.exit(f"File not found: {path}")
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    df = load(path, args.sheet)
    col = resolve_columns(df)
    print()
    df = clean(df, col)
    time_col = pick_intraday_time(df, col)

    log = []
    column_summary(df, outdir)
    quality_report(df, col, outdir, log)
    distributions(df, col, outdir)
    temporal(df, col, time_col, outdir, log)
    spatial(df, col, outdir, log)
    enforcement_bias(df, col, outdir, log)
    cross_tabs(df, col, outdir)

    (outdir / "summary.txt").write_text("\n".join(log))
    print(f"\nText summary -> {outdir / 'summary.txt'}")
    print(f"All figures + CSVs are in: {outdir.resolve()}")


if __name__ == "__main__":
    main()