import pandas as pd

tbl = pd.read_pickle("pipeline_out/modeling_table.pkl")
combo = (tbl["corridor_base"].astype(str) + "|" + 
         tbl["hour"].astype(str) + "|" + 
         tbl["is_weekend"].astype(str))
result = (tbl.groupby(combo)["priority_high"]
            .agg(["mean", "count"])
            .query("mean >= 0.95 or mean <= 0.05")
            .sort_values("count", ascending=False))
print(f"Combinations that are near-100% one class: {len(result)}")
print(result.head(20).to_string())