#!/usr/bin/env python3
"""
Timestamp pair diagnostic — ASTraM dataset
Answers:
  1. What does each timestamp column actually contain?
  2. Does start + end form a clean duration pair?
  3. Can modified_datetime fill in for a missing end?
  4. Are created + closed / resolved usable pairs?
Run: python ts_diagnostic.py "Astram event data_anonymized.xlsx"
"""
import sys, warnings
import numpy as np
import pandas as pd
warnings.filterwarnings("ignore")

PATH = sys.argv[1] if len(sys.argv) > 1 else "Astram event data_anonymized - Astram event data_anonymizedb40ac87.csv"
for enc in ("utf-8", "utf-8-sig", "cp1252", "latin-1"):
    try:
        df = pd.read_csv(PATH, encoding=enc)
        break
    except (UnicodeDecodeError, Exception):
        continue
df = df.replace(["NULL","null","","[]","None"], np.nan)
print(f"Loaded {len(df)} rows\n")

# ── 1. parse all six timestamp columns ──────────────────────────────────
TS_COLS = ["start_datetime","end_datetime","modified_datetime",
           "created_date","closed_datetime","resolved_datetime"]
ts = {}
for c in TS_COLS:
    col = next((x for x in df.columns if x.lower()==c.lower()), None)
    if col:
        ts[c] = pd.to_datetime(df[col], errors="coerce", utc=True).dt.tz_localize(None)

# ── 2. coverage + spread ─────────────────────────────────────────────────
print("=" * 55)
print("COVERAGE & SPREAD (raw, no timezone correction)")
print("=" * 55)
for name, s in ts.items():
    pct  = s.notna().mean() * 100
    rng  = f"{s.min().date()} -> {s.max().date()}" if s.notna().any() else "all null"
    print(f"  {name:<22} {pct:5.1f}% filled   {rng}")

# ── 3. are start & created_date the same? ───────────────────────────────
if "start_datetime" in ts and "created_date" in ts:
    both   = ts["start_datetime"].notna() & ts["created_date"].notna()
    diff_s = (ts["start_datetime"] - ts["created_date"]).dt.total_seconds().abs()
    same   = (diff_s[both] < 60).mean() * 100      # within 1 minute = "same"
    print(f"\nstart_datetime vs created_date — {same:.1f}% of rows differ by < 1 min "
          f"({'effectively identical' if same > 90 else 'different columns'})")

# ── 4. helper: evaluate a start→end pair ────────────────────────────────
def pair_report(start_name, end_name, label):
    s = ts.get(start_name); e = ts.get(end_name)
    if s is None or e is None:
        print(f"  {label:<40} column missing"); return None
    both  = s.notna() & e.notna()
    dur   = (e - s).dt.total_seconds() / 60.0      # minutes
    pos   = (dur[both] > 0).mean()   * 100
    ok    = ((dur[both] > 0) & (dur[both] < 24*60)).mean() * 100   # 0-24h
    med   = dur[both][dur[both] > 0].median()
    p95   = dur[both][dur[both] > 0].quantile(0.95)
    cov   = both.mean() * 100
    print(f"\n  {label}")
    print(f"    both non-null  : {cov:.1f}%")
    print(f"    end > start    : {pos:.1f}%  of paired rows")
    print(f"    0 < dur < 24h  : {ok:.1f}%  of paired rows  <- usable for modeling")
    print(f"    median dur     : {med:.0f} min  |  95th pct : {p95:.0f} min")
    return dur[both & (dur > 0) & (dur < 24*60)]

print("\n" + "=" * 55)
print("PAIR QUALITY (duration = end - start)")
print("=" * 55)

d1 = pair_report("start_datetime",  "end_datetime",      "start  -> end")
d2 = pair_report("start_datetime",  "closed_datetime",   "start  -> closed")
d3 = pair_report("start_datetime",  "resolved_datetime", "start  -> resolved")
d4 = pair_report("start_datetime",  "modified_datetime", "start  -> modified  [fallback?]")
d5 = pair_report("created_date",    "closed_datetime",   "created -> closed")
d6 = pair_report("created_date",    "resolved_datetime", "created -> resolved")

# ── 5. verdict ───────────────────────────────────────────────────────────
print("\n" + "=" * 55)
print("VERDICT")
print("=" * 55)

pairs = {
    "start -> end"      : d1,
    "start -> closed"   : d2,
    "start -> resolved" : d3,
    "created -> closed" : d5,
    "created -> resolved":d6,
}
best_name, best_ser, best_cov = None, None, 0
for name, ser in pairs.items():
    if ser is not None and len(ser) > best_cov:
        best_cov = len(ser); best_name = name; best_ser = ser

if best_name:
    print(f"\n  Best pair for duration  : {best_name}")
    print(f"  Usable rows             : {best_cov}")
    if d4 is not None and best_ser is not None:
        corr = pd.concat([best_ser, d4], axis=1).dropna().corr().iloc[0,1]
        print(f"  start->modified corr with best pair: {corr:.2f}"
              f"  ({'good fallback' if corr > 0.6 else 'weak fallback'})")

print("\n  Recommendation for OCCURRENCE_COL in astram_clean.py:")
if "start_datetime" in ts and "created_date" in ts:
    both = ts["start_datetime"].notna() & ts["created_date"].notna()
    diff = (ts["start_datetime"]-ts["created_date"]).dt.total_seconds().abs()
    if (diff[both] < 60).mean() > 0.9:
        print("  start_datetime = created_date (same field). Use start_datetime.")
        print("  Neither reflects true occurrence — they reflect logging time.")
        print("  For modeling: keep as OCCURRENCE_COL but don't over-interpret hour.")
    else:
        print("  start_datetime and created_date differ — start_datetime is likely")
        print("  closer to true occurrence. Use start_datetime as OCCURRENCE_COL.")

print("\n  Recommendation for duration in astram_clean.py:")
if best_name:
    end_col = best_name.split("->")[1].strip().replace(" ","_")
    # map to real column names
    col_map = {"end":"end_datetime","closed":"closed_datetime",
               "resolved":"resolved_datetime","modified":"modified_datetime"}
    real_end = col_map.get(end_col, end_col+"_datetime")
    print(f"  Set END_COL = '{real_end}'  (gives {best_cov} usable durations)")
    if d4 is not None and best_ser is not None and real_end != "modified_datetime":
        print(f"  modified_datetime as secondary fallback: "
              f"{'yes' if corr > 0.6 else 'not reliable'}")