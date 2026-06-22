#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
====================================================================
PHASE 0 : Clean -> Modeling Table (+ Event Impact Score)
====================================================================
Turns the raw anonymized Astram export into ONE trustworthy, model-ready
event-level table. This is the foundation every downstream model depends on.

It does, in order:
  1. Load (.xlsx/.csv) and repair Kannada mojibake (UTF-8 read as cp1252).
  2. NULL/placeholder -> NaN, parse timestamps, coerce coordinates.
  3. TIMEZONE: convert to IST (configurable) + a diagnostic so YOU can
     verify which timestamp column actually reflects event occurrence.
  4. Normalize the messy cause / corridor / boolean taxonomies.
  5. DURATION: compute robustly, winsorize, flag the (few) valid rows.
  6. IMPACT SCORE: a composite 0-100 severity proxy (priority + rerouting
     + duration + local concurrent load, weighted by vehicle & corridor).
  7. Save modeling_table.pkl (typed) + .csv (inspection) + diagnostics.

Usage:
    python astram_clean.py "Astram event data_anonymized - Astram event data_anonymizedb40ac87.csv"

Output dir: ./pipeline_out/
Requires: pandas, numpy, openpyxl   (ftfy optional)
"""
import os, sys, warnings
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# ----------------------------------------------------------------------
# CONFIG 
# ----------------------------------------------------------------------
OUTDIR            = "pipeline_out"
OCCURRENCE_COL    = "start_datetime"   # column treated as "when it happened"
FALLBACK_OCC_COL  = "modified_datetime"     # used where OCCURRENCE_COL is missing
SHIFT_TO_IST      = False               # stored tz looks like UTC(+0); shift +5:30
TARGET_TZ         = "Asia/Kolkata"
DURATION_CAP_MIN  = 24 * 60            # treat > 24h as invalid (data artifact)
DROP_TEST_ROWS    = True               # drop rows whose cause is 'test_demo' 
END_COL_PRIORITY = ["closed_datetime", "resolved_datetime", "modified_datetime"]

# Impact Score weights (sum ~1.0). Duration is down-weighted: it is ~96% missing.
W_PRIORITY, W_REROUTE, W_DURATION, W_DENSITY = 0.45, 0.30, 0.10, 0.15
DUR_NORM_CAP_MIN  = 180                # minutes that maps to "full" duration impact
PRIORITY_LOW_VAL  = 0.35              # Low priority still carries some impact

VEH_MULT = {  # heavier / larger vehicles block more road
    "heavy_vehicle": 1.30, "truck": 1.30, "bmtc_bus": 1.30, "ksrtc_bus": 1.30,
    "private_bus": 1.20, "lcv": 1.15, "private_car": 1.00, "taxi": 1.00,
    "auto": 0.90, "others": 1.00,
}
os.makedirs(OUTDIR, exist_ok=True)

# ----------------------------------------------------------------------
# Mojibake repair (same logic proven in the EDA script)
# ----------------------------------------------------------------------
try:
    import ftfy; _HAS_FTFY = True
except ImportError:
    _HAS_FTFY = False
_FLAGS = ("Ã", "Â", "à", "á", "â", "Ÿ", "€", "š", "›", "œ")

def fix_mojibake(t):
    if not isinstance(t, str) or not any(f in t for f in _FLAGS):
        return t
    for enc in ("cp1252", "latin-1"):
        try:
            r = t.encode(enc, "strict").decode("utf-8", "strict")
            if r and "\ufffd" not in r:
                return r
        except (UnicodeEncodeError, UnicodeDecodeError):
            pass
    return ftfy.fix_text(t) if _HAS_FTFY else t

# ----------------------------------------------------------------------
# Load + base clean
# ----------------------------------------------------------------------

# checkif the file is an Excel file or a CSV file and load it accordingly
# if it is a CSV file, try different encodings until one works (does not fix the broken kannada text)
def load(path):
    if path.lower().endswith((".xlsx", ".xls", ".xlsm")):
        df = pd.read_excel(path, engine="openpyxl")
    else:
        for enc in ("utf-8", "utf-8-sig", "cp1252", "latin-1"):
            try:
                df = pd.read_csv(path, encoding=enc); break
            except Exception:
                continue
    # strip whitespace from column names
    df.columns = [str(c).strip() for c in df.columns]
    print(f"Loaded {df.shape[0]} rows x {df.shape[1]} cols")
    return df

# coloumn finder - that tries to find the best match for a given list of candidate column names in a dataframe
def col(df, *cands):
    low = {c.lower(): c for c in df.columns}
    # exact match first, then substring match
    for c in cands:
        if c.lower() in low:
            return low[c.lower()]
    for c in cands:
        for k in df.columns:
            if c.lower() in k.lower():
                return k
    return None

def base_clean(df):
    # replace common NULL/placeholder values with NaN
    df = df.replace(["NULL", "null", "", "[]", "None", "nan"], np.nan)
    # repair every text column if it has broken text (like kannada mojibake)
    for c in df.columns:                       
        if df[c].map(lambda x: isinstance(x, str)).any():
            df[c] = df[c].map(fix_mojibake)
    # coerce coordinates to numeric, treat 0 as NaN
    for c in ("latitude", "longitude", "endlatitude", "endlongitude", "resolved_latitude", "resolved_longitude"):
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
            df.loc[df[c] == 0, c] = np.nan 
    return df

# ----------------------------------------------------------------------
# Timezone handling + diagnostic
# ----------------------------------------------------------------------

# convert a coloumn of date/time values and convert to proper timestamps, optonally shifting to IST and removing timezone info (making them naive timestamps in IST)
def to_local(series):
    dt = pd.to_datetime(series, errors="coerce", utc=True)  # naive read as UTC
    if SHIFT_TO_IST:
        dt = dt.dt.tz_convert(TARGET_TZ)
    return dt.dt.tz_localize(None)

def diagnose_timestamps(df):
    """Print hour-of-day spread (in IST) for each candidate timestamp so you can
    confirm which one looks like real traffic (morning + evening peaks)."""
    
    print("\n--- TIMESTAMP DIAGNOSTIC (hour-of-day share, IST) ---")
    print("Pick the column whose peaks look like real traffic, set OCCURRENCE_COL.")

    cands = [c for c in (OCCURRENCE_COL, FALLBACK_OCC_COL, "modified_date", "modified_on", "created_date", "closed_datetime", "closed_date") if col(df, c)]
    for name in dict.fromkeys(cands):
        real = col(df, name)
        h = to_local(df[real]).dt.hour.dropna()
        if h.empty:
            continue
        share = (h.value_counts(normalize=True).sort_index() * 100).round(1)
        bars = " ".join(f"{hr:02d}:{share.get(hr,0):>4.1f}" for hr in range(0, 24, 3))
        print(f"  {real:<16} | {bars}")

# ----------------------------------------------------------------------
# Taxonomy normalization
# ----------------------------------------------------------------------

# convert different spellings of the same cause into a normalized form
CAUSE_MAP = {
    "water logging": "water_logging", "waterlogging": "water_logging",
    "pot holes": "pot_holes", "potholes": "pot_holes", "pothole": "pot_holes",
    "tree fall": "tree_fall", "treefall": "tree_fall",
    "road conditions": "road_conditions",
    "vehicle breakdown": "vehicle_breakdown",
    "fog / low visibility": "fog", "fog/low visibility": "fog",
    "low visibility": "fog", "fog": "fog", "debris": "debris",
}
def norm_cause(x):
    if not isinstance(x, str):
        return x
    k = x.strip().lower().replace("-", " ")
    k2 = k.replace(" ", "_")
    return CAUSE_MAP.get(k, CAUSE_MAP.get(k2, k2))


def strip_corridor_suffix(x):
    """'Bellary Road 1' -> 'Bellary Road' ; keeps a coarse corridor key."""
    if not isinstance(x, str):
        return x
    parts = x.strip().rsplit(" ", 1)
    return parts[0] if len(parts) == 2 and parts[1].isdigit() else x.strip()

# convert various boolean-like columns to actual booleans
def to_bool(s):
    return s.astype(str).str.strip().str.lower().map(
        {"true": True, "yes": True, "1": True, "t": True,
         "false": False, "no": False, "0": False, "f": False})

# ----------------------------------------------------------------------
# Build the modeling table
# ----------------------------------------------------------------------
def build(df):
    C = lambda *a: col(df, *a)
    # creates a new dataframe and gradually fills it with cleaned coloumns
    out = pd.DataFrame()
    # -- unique row ID (if not present) ---
    out["id"] = df[C("id")] if C("id") else np.arange(len(df))

    # --- occurrence timestamp (IST) ---
    # finds the main occurence timestamp and uses a fallback one if it is missing 
    occ = to_local(df[C(OCCURRENCE_COL)]) if C(OCCURRENCE_COL) else pd.NaT
    if C(FALLBACK_OCC_COL):
        occ = occ.fillna(to_local(df[C(FALLBACK_OCC_COL)]))
    out["occurrence_ts"] = occ # final occurrence timestamp in IST (naT if missing/invalid)

    # --- categorical / spatial fields (normalized) ---
    # normalizes categorical fields like cause and corridor, and extracts spatial features like police station and junction
    cause_c = C("event_cause", "cause")
    out["cause"] = df[cause_c].map(norm_cause) if cause_c else np.nan
    out["event_type"]     = df[C("event_type")] if C("event_type") else np.nan
    corr_c = C("corridor")
    out["corridor"]       = df[corr_c] if corr_c else np.nan                         # Bellary Road 1, Bellary Road 2, etc.
    out["corridor_base"]  = out["corridor"].map(strip_corridor_suffix)               # Bellary Road
    out["zone"]           = df[C("zone")] if C("zone") else np.nan
    out["police_station"] = df[C("police_station", "police_sta")] if C("police_station", "police_sta") else np.nan
    out["junction"]       = df[C("junction")] if C("junction") else np.nan
    out["veh_type"]       = df[C("veh_type")] if C("veh_type") else np.nan
    out["status"]         = df[C("status")] if C("status") else np.nan
    out["latitude"]       = df[C("latitude")]  if C("latitude")  else np.nan
    out["longitude"]      = df[C("longitude")] if C("longitude") else np.nan

    # --- operational labels (booleans) ---
    # changes priority from high or low to high priority true or false
    pr_c = C("priority")
    out["priority_high"] = (df[pr_c].astype(str).str.strip().str.lower() == "high") if pr_c else np.nan
    # changes requires_rerouting from true/false or yes/no to boolean
    rr_c = C("requires_rerouting", "requires_r")
    out["requires_rerouting"] = to_bool(df[rr_c]) if rr_c else np.nan

    # --- duration (robust) ---
    end_c = next((C(x) for x in END_COL_PRIORITY if C(x)), None)
    if C(OCCURRENCE_COL) and end_c:
        dur = (to_local(df[end_c]) - out["occurrence_ts"]).dt.total_seconds() / 60.0
        cap = DURATION_CAP_MIN if end_c != col(df, "modified_datetime") else 480
        out["duration_min"] = dur.where((dur > 0) & (dur < cap)) # if durations are too long or negative, treat as invalid (data artifact) 
        out["duration_valid"] = out["duration_min"].notna()
    else:
        out["duration_min"], out["duration_valid"] = np.nan, False

    # --- temporal features (IST) ---
    # extract useful temporal features from the timestamp
    ts = out["occurrence_ts"]
    out["hour"]       = ts.dt.hour
    out["dow"]        = ts.dt.dayofweek
    out["dow_name"]   = ts.dt.day_name()
    out["month"]      = ts.dt.month
    out["date"]       = ts.dt.date
    out["is_weekend"] = ts.dt.dayofweek >= 5
    out["is_peak"]    = ts.dt.hour.isin([8, 9, 10, 17, 18, 19, 20])

    # --- local concurrent load (density proxy): events in same station-hour ---
    key = out["police_station"].fillna("NA").astype(str) + "|" + \
          out["date"].astype(str) + "|" + out["hour"].astype("Int64").astype(str)
    out["concurrent_load"] = key.map(key.value_counts())

    # there are exactly 3 rows where the event cause is test_demo, which is a test row and should be dropped from the modeling table
    if DROP_TEST_ROWS:
        before = len(out)
        out = out[out["cause"] != "test_demo"].copy()
        if before - len(out):
            print(f"Dropped {before-len(out)} test_demo rows")

    return out

# ----------------------------------------------------------------------
# Impact Score
# ----------------------------------------------------------------------
def add_impact_score(df):
    # if priority_high is True, impact score is 1.0, if priority_high is False or NaN, impact score is PRIORITY_LOW_VAL
    pr = np.where(df["priority_high"] == True, 1.0, np.where(df["priority_high"] == False, PRIORITY_LOW_VAL, PRIORITY_LOW_VAL))
    # if requires_rerouting is True, impact score is 1.0, else 0.0
    rr = np.where(df["requires_rerouting"] == True, 1.0, 0.0)
    # duration is normalized to a 0-1 scale by dividing it with the duration norm cap and then clipping it ( clip is min (max(0, x), 1) )) if a value exceeds to the cap then it has already contributed enough to the impact score, if it is missing then it is treated as neutral (0.30)
    dur_norm = (df["duration_min"] / DUR_NORM_CAP_MIN).clip(0, 1).fillna(0.30)  # neutral if missing
    # if concurrent_load is missing, treat it as 1 (neutral)
    dens = df["concurrent_load"].fillna(1)
    # normalize concurrent load to a 0-1 scale by dividing it with the 95th percentile of the concurrent load and then clipping it 
    dens_norm = (dens / max(dens.quantile(0.95), 1)).clip(0, 1)

    # basic impact score as a weighted sum 
    base = (W_PRIORITY * pr + W_REROUTE * rr + W_DURATION * dur_norm + W_DENSITY * dens_norm)        # 0..1

    # vehicle type multiplier
    veh_mult  = df["veh_type"].map(VEH_MULT).fillna(1.0).values
    # corridor multiplier: if corridor_base contains "Non-corridor" (case-insensitive), multiplier is 1.0, else 1.10 (corridors are slightly more impactful)
    corr_mult = np.where(df["corridor_base"].fillna("Non-corridor").str.contains("Non-corridor", case=False), 1.0, 1.10)

    raw = base * veh_mult * corr_mult
    df["impact_raw"]   = raw
    # scale it to 0-100 and round to 1 decimal place for easy reading
    df["impact_score"] = (raw / raw.max() * 100).round(1)          # 0..100, easy to read
    return df

# ----------------------------------------------------------------------
def main():
    path = sys.argv[1] if len(sys.argv) > 1 else "Astram event data_anonymized - Astram event data_anonymizedb40ac87.csv"
    if not os.path.exists(path):
        sys.exit(f"File not found: {path}")
    df = base_clean(load(path))
    diagnose_timestamps(df)
    tbl = add_impact_score(build(df))

    # save
    pkl = os.path.join(OUTDIR, "modeling_table.pkl")
    csv = os.path.join(OUTDIR, "modeling_table.csv")
    tbl.to_pickle(pkl)
    tbl.to_csv(csv, index=False)

    # diagnostics
    print(f"\nModeling table: {tbl.shape[0]} rows x {tbl.shape[1]} cols")
    print(f"  occurrence range : {tbl['occurrence_ts'].min()} -> {tbl['occurrence_ts'].max()}")
    print(f"  duration valid   : {int(tbl['duration_valid'].sum())} rows "
          f"({tbl['duration_valid'].mean()*100:.1f}%)")
    print("\nImpact score distribution:")
    print(tbl["impact_score"].describe().round(1).to_string())
    print("\nMean impact by cause (top 10):")
    print(tbl.groupby("cause")["impact_score"].agg(["mean", "count"])
             .sort_values("mean", ascending=False).head(10).round(1).to_string())
    print(f"\nSaved -> {pkl}\n         {csv}")

if __name__ == "__main__":
    main()