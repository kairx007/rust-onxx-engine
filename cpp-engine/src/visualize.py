"""
src/visualize.py
────────────────
Visualises raw FI-2010 LOB data: mid-price, bid-ask spread, and market depth.

Usage (run from project root):
    python src/visualize.py

Outputs:
    plots/01_bid_ask_spread.png
    plots/02_market_depth_heatmap.png
    plots/03_order_flow_imbalance.png

Fixes vs. original:
    1. Auto path resolution — works from project root or notebooks/.
    2. FI-2010 column layout corrected:
         After dropping the index col, the layout is:
           col 0  = Ask Price 1  (best ask)
           col 1  = Ask Vol 1
           col 2  = Bid Price 1  (best bid)
           col 3  = Bid Vol 1
           col 4  = Ask Price 2  ...
         So ask PRICES are even cols {0,4,8,...} and ask VOLUMES are {1,5,9,...}
              bid PRICES are odd  cols {2,6,10,...} and bid VOLUMES are {3,7,11,...}
         The original used col 2 as "Best Bid Price" which is actually Ask Vol 1.
    3. Plot title fixed — data is RAW (not z-score normalized).
    4. Mid-price and spread added to Plot 1 (more informative than just ask/bid lines).
    5. Volume heatmap now correctly selects ask+bid volume columns separately.
    6. OFI (Order Flow Imbalance) plot added — a key microstructure signal.
    7. Plots saved to disk instead of just plt.show() (useful in headless envs).
    8. plt.close() called after each figure to avoid memory leaks.
"""

import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# ── Config ────────────────────────────────────────────────────────────────────
NROWS     = 2000
PLOT_DIR  = "plots"
os.makedirs(PLOT_DIR, exist_ok=True)

# ── Path resolution ───────────────────────────────────────────────────────────
file_path = "data/raw/FI2010_train.csv"
if not os.path.exists(file_path):
    file_path = "../data/raw/FI2010_train.csv"

print(f"⏳ Loading data from {file_path}...")

# ── Load & clean ──────────────────────────────────────────────────────────────
df = pd.read_csv(file_path, skiprows=1, header=None, nrows=NROWS)
df = df.iloc[:, 1:]               # drop garbage index column
df.columns = range(df.shape[1])   # reset to 0-indexed

print(f"✅ Data loaded. Shape: {df.shape}")

# ── FI-2010 column layout (after dropping index col) ─────────────────────────
# Level k  (0-indexed):
#   Ask Price k : col 4k
#   Ask Vol   k : col 4k+1
#   Bid Price k : col 4k+2
#   Bid Vol   k : col 4k+3
#
# FIX 2: original used col 2 as "Best Bid Price" — that is actually Ask Vol 1.
ask_price_cols = [4 * k     for k in range(10)]   # [0, 4,  8, 12, ...]
ask_vol_cols   = [4 * k + 1 for k in range(10)]   # [1, 5,  9, 13, ...]
bid_price_cols = [4 * k + 2 for k in range(10)]   # [2, 6, 10, 14, ...]
bid_vol_cols   = [4 * k + 3 for k in range(10)]   # [3, 7, 11, 15, ...]

best_ask = df.iloc[:, ask_price_cols[0]]           # col 0
best_bid = df.iloc[:, bid_price_cols[0]]           # col 2
mid_price = (best_ask + best_bid) / 2.0
spread    = best_ask - best_bid

print(f"\n🧐 Sanity check (row 0):")
print(f"   Best Ask Price : {best_ask.iloc[0]:.4f}")
print(f"   Best Bid Price : {best_bid.iloc[0]:.4f}")
print(f"   Mid Price      : {mid_price.iloc[0]:.4f}")
print(f"   Spread         : {spread.iloc[0]:.6f}")

N_PLOT = 500   # number of ticks to show in time-series plots

# ══════════════════════════════════════════════════════════════════════════════
# PLOT 1 — Best Ask / Bid prices + mid-price + spread
# FIX 3: title corrected (data is raw, not normalized)
# FIX 4: mid-price and spread subplot added
# ══════════════════════════════════════════════════════════════════════════════
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 8), sharex=True)
fig.suptitle("Limit Order Book — Best Ask / Bid & Spread (Raw Prices)", fontsize=14)

ax1.plot(best_ask.iloc[:N_PLOT],  label="Best Ask (Sellers)", color="red",   linewidth=1)
ax1.plot(best_bid.iloc[:N_PLOT],  label="Best Bid (Buyers)",  color="green", linewidth=1)
ax1.plot(mid_price.iloc[:N_PLOT], label="Mid-Price",          color="blue",  linewidth=1, linestyle="--")
ax1.set_ylabel("Price")
ax1.legend(fontsize=9)
ax1.grid(True, alpha=0.3)

ax2.fill_between(range(N_PLOT), spread.iloc[:N_PLOT], alpha=0.4, color="orange", label="Bid-Ask Spread")
ax2.set_ylabel("Spread")
ax2.set_xlabel("Time (Ticks)")
ax2.legend(fontsize=9)
ax2.grid(True, alpha=0.3)

plt.tight_layout()
out1 = os.path.join(PLOT_DIR, "01_bid_ask_spread.png")
plt.savefig(out1, dpi=150, bbox_inches="tight")
plt.close()
print(f"\n💾 Saved → {out1}")

# ══════════════════════════════════════════════════════════════════════════════
# PLOT 2 — Market Depth Heatmap (ask + bid volumes across 10 levels)
# FIX 5: volume columns correctly separated into ask and bid grids
# ══════════════════════════════════════════════════════════════════════════════
ask_vol_data = df.iloc[:1000, ask_vol_cols].T.values   # (10, 1000)
bid_vol_data = df.iloc[:1000, bid_vol_cols].T.values   # (10, 1000)

fig, (ax_ask, ax_bid) = plt.subplots(2, 1, figsize=(15, 8), sharex=True)
fig.suptitle("Market Depth Heatmap — Volume at Each LOB Level", fontsize=14)

sns.heatmap(ask_vol_data, cmap="Reds",   robust=True, ax=ax_ask,
            yticklabels=[f"Ask L{i+1}" for i in range(10)])
ax_ask.set_title("Ask Side (Sell Orders)")
ax_ask.set_ylabel("Level")

sns.heatmap(bid_vol_data, cmap="Greens", robust=True, ax=ax_bid,
            yticklabels=[f"Bid L{i+1}" for i in range(10)])
ax_bid.set_title("Bid Side (Buy Orders)")
ax_bid.set_ylabel("Level")
ax_bid.set_xlabel("Time (Ticks)")

plt.tight_layout()
out2 = os.path.join(PLOT_DIR, "02_market_depth_heatmap.png")
plt.savefig(out2, dpi=150, bbox_inches="tight")
plt.close()
print(f"💾 Saved → {out2}")

# ══════════════════════════════════════════════════════════════════════════════
# PLOT 3 — Order Flow Imbalance (OFI)
# FIX 6: new plot — OFI is a core microstructure signal mentioned in the roadmap
#
# OFI = (Best Bid Volume - Best Ask Volume) / (Best Bid Volume + Best Ask Volume)
# Range: [-1, +1]
#   +1 → all volume on the buy side  (bullish pressure)
#   -1 → all volume on the sell side (bearish pressure)
# ══════════════════════════════════════════════════════════════════════════════
best_ask_vol = df.iloc[:, ask_vol_cols[0]].astype(float)
best_bid_vol = df.iloc[:, bid_vol_cols[0]].astype(float)
total_vol    = best_bid_vol + best_ask_vol
ofi          = (best_bid_vol - best_ask_vol) / total_vol.replace(0, np.nan)

fig, ax = plt.subplots(figsize=(15, 4))
ax.plot(ofi.iloc[:N_PLOT], color="purple", linewidth=0.8, alpha=0.8)
ax.axhline(0, color="black", linewidth=0.8, linestyle="--")
ax.fill_between(range(N_PLOT), ofi.iloc[:N_PLOT], 0,
                where=(ofi.iloc[:N_PLOT] > 0), color="green", alpha=0.3, label="Buy pressure")
ax.fill_between(range(N_PLOT), ofi.iloc[:N_PLOT], 0,
                where=(ofi.iloc[:N_PLOT] < 0), color="red",   alpha=0.3, label="Sell pressure")
ax.set_title("Order Flow Imbalance (OFI) — Best Level")
ax.set_xlabel("Time (Ticks)")
ax.set_ylabel("OFI")
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)

plt.tight_layout()
out3 = os.path.join(PLOT_DIR, "03_order_flow_imbalance.png")
plt.savefig(out3, dpi=150, bbox_inches="tight")
plt.close()
print(f"💾 Saved → {out3}")

print("\n✅ All plots saved to plots/")