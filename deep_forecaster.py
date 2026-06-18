#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
====================================================================
 ASTraM — ADD-ON : Deep Spatio-Temporal Forecaster (ConvLSTM)
====================================================================
Standalone. Does NOT modify existing files. The "Tier-3 / future-work" option
from the plan, built because you asked: a ConvLSTM that learns spatial diffusion
of incidents over a gridded map, instead of the per-area GBM panel.

How it works:
  * Bin the city's lat/long bounding box into a GRID x GRID raster.
  * For each time bin, build a frame = incident counts per cell (a heatmap).
  * Train a ConvLSTM to predict the NEXT frame from the past T frames, so it
    captures both where incidents are AND how they spread between cells.
  * Compare to a persistence baseline (next frame == last frame).

Honest note: on this dataset (sparse, ~5 months) the LightGBM panel forecaster
from Phase 1 is competitive and far cheaper. Keep GBM as primary; present this
as "we also explored a deep spatio-temporal model." It needs PyTorch.

Usage:  python astram_deep_forecaster.py
Requires: torch, pandas, numpy, matplotlib
"""
import os, sys
import numpy as np
import pandas as pd

OUTDIR = "pipeline_out"
GRID   = 10        # GRID x GRID raster over the city
FREQ   = "3h"      # time-bin width (coarser than 1h to densify the frames)
SEQ_T  = 8         # past frames used to predict the next one
HIDDEN = 16
EPOCHS = 8
SEED   = 0

def load_table():
    p = os.path.join(OUTDIR, "modeling_table.pkl")
    if not os.path.exists(p):
        sys.exit("Run astram_clean.py first to create modeling_table.pkl")
    return pd.read_pickle(p)

# ----------------------------------------------------------------------
# Build the frame tensor: (n_bins, GRID, GRID) incident counts
# ----------------------------------------------------------------------
def build_frames(tbl):
    d = tbl.dropna(subset=["occurrence_ts", "latitude", "longitude"]).copy()
    lat0, lat1 = d["latitude"].quantile([0.01, 0.99])
    lon0, lon1 = d["longitude"].quantile([0.01, 0.99])
    d = d[(d.latitude.between(lat0, lat1)) & (d.longitude.between(lon0, lon1))]
    d["gi"] = np.clip(((d.latitude - lat0) / (lat1 - lat0) * GRID).astype(int), 0, GRID - 1)
    d["gj"] = np.clip(((d.longitude - lon0) / (lon1 - lon0) * GRID).astype(int), 0, GRID - 1)
    d["tbin"] = d["occurrence_ts"].dt.floor(FREQ)

    tidx = pd.date_range(d.tbin.min(), d.tbin.max(), freq=FREQ)
    tpos = {t: i for i, t in enumerate(tidx)}
    frames = np.zeros((len(tidx), GRID, GRID), dtype=np.float32)
    for _, r in d.iterrows():
        frames[tpos[r.tbin], r.gi, r.gj] += 1.0
    return frames

def make_sequences(frames, T):
    X, Y = [], []
    for i in range(len(frames) - T):
        X.append(frames[i:i + T])      # (T, G, G)
        Y.append(frames[i + T])        # (G, G)
    X = np.asarray(X)[:, :, None]       # (N, T, 1, G, G)
    Y = np.asarray(Y)[:, None]          # (N, 1, G, G)
    return X, Y

# ----------------------------------------------------------------------
# ConvLSTM
# ----------------------------------------------------------------------
def build_and_train(X, Y):
    import torch
    import torch.nn as nn
    torch.manual_seed(SEED)

    class ConvLSTMCell(nn.Module):
        def __init__(self, in_ch, hid, k=3):
            super().__init__()
            self.hid = hid
            self.conv = nn.Conv2d(in_ch + hid, 4 * hid, k, padding=k // 2)
        def forward(self, x, h, c):
            i, f, o, g = torch.chunk(self.conv(torch.cat([x, h], 1)), 4, 1)
            c = torch.sigmoid(f) * c + torch.sigmoid(i) * torch.tanh(g)
            h = torch.sigmoid(o) * torch.tanh(c)
            return h, c

    class ConvLSTM(nn.Module):
        def __init__(self, hid=HIDDEN):
            super().__init__()
            self.cell = ConvLSTMCell(1, hid)
            self.head = nn.Conv2d(hid, 1, 1)
        def forward(self, x):                       # x: (B,T,1,G,G)
            B, T, _, G, _ = x.shape
            h = x.new_zeros(B, self.cell.hid, G, G)
            c = torch.zeros_like(h)
            for t in range(T):
                h, c = self.cell(x[:, t], h, c)
            return torch.relu(self.head(h))          # (B,1,G,G) nonneg

    k = int(len(X) * 0.8)                            # temporal split
    Xtr, Ytr = torch.tensor(X[:k]), torch.tensor(Y[:k])
    Xte, Yte = torch.tensor(X[k:]), torch.tensor(Y[k:])
    model = ConvLSTM()
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    lossf = nn.MSELoss()

    bs = 32
    for ep in range(EPOCHS):
        model.train(); perm = torch.randperm(len(Xtr)); tot = 0.0
        for b in range(0, len(Xtr), bs):
            idx = perm[b:b + bs]
            opt.zero_grad()
            loss = lossf(model(Xtr[idx]), Ytr[idx])
            loss.backward(); opt.step(); tot += loss.item() * len(idx)
        print(f"   epoch {ep+1}/{EPOCHS}  train MSE {tot/len(Xtr):.4f}")

    model.eval()
    with torch.no_grad():
        pred = model(Xte).numpy()
    return pred, Yte.numpy(), Xte.numpy(), model

# ----------------------------------------------------------------------
def main():
    np.random.seed(SEED)
    try:
        import torch  # noqa
    except ImportError:
        sys.exit("PyTorch not installed. `pip install torch` to run the deep model.")

    tbl = load_table()
    frames = build_frames(tbl)
    print(f"Frames: {frames.shape} (time x {GRID} x {GRID}); "
          f"mean incidents/cell/bin = {frames.mean():.3f}")
    X, Y = make_sequences(frames, SEQ_T)
    print(f"Sequences: {X.shape} -> {Y.shape}; training ConvLSTM...")

    pred, true, Xte, model = build_and_train(X, Y)

    mae_model = np.abs(pred - true).mean()
    persistence = Xte[:, -1]                          # last input frame
    mae_persist = np.abs(persistence - true).mean()
    print("\nTest frame-level MAE (incidents/cell):")
    print(f"   ConvLSTM     {mae_model:.4f}")
    print(f"   persistence  {mae_persist:.4f}   (predict 'same as last bin')")
    gain = (1 - mae_model / max(mae_persist, 1e-9)) * 100
    print(f"   -> ConvLSTM vs persistence: {gain:+.1f}% MAE")

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        avg_true = true.mean(0)[0]; avg_pred = pred.mean(0)[0]
        fig, ax = plt.subplots(1, 2, figsize=(10, 4.5))
        for a, img, t in zip(ax, [avg_true, avg_pred], ["actual avg", "predicted avg"]):
            im = a.imshow(img, cmap="inferno"); a.set_title(t); a.axis("off")
            fig.colorbar(im, ax=a, fraction=0.046)
        fig.suptitle("Deep ST forecaster — mean incident heatmap (test)")
        out = os.path.join(OUTDIR, "deep_forecaster_heatmaps.png")
        fig.savefig(out, dpi=110, bbox_inches="tight"); plt.close(fig)
        print(f"   saved {out}")
    except Exception as e:
        print("   (plot skipped:", e, ")")

    import torch
    torch.save(model.state_dict(), os.path.join(OUTDIR, "deep_forecaster.pt"))
    print(f"\nSaved -> {OUTDIR}/deep_forecaster.pt")
    print("Note: keep the LightGBM panel as primary; this is the deep-ST exploration.")

if __name__ == "__main__":
    main()