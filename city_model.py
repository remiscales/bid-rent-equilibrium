"""
Singapore Monocentric vs Polycentric City Model
================================================
Reproduces the HDB resale price gradient analysis from the interactive app.
Requires: pandas, numpy, matplotlib, scipy
Install: pip install pandas numpy matplotlib scipy
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from scipy import stats

# ── Data ──────────────────────────────────────────────────────────────────────
# Realistic estimates based on known Singapore HDB spatial patterns
# dist_cbd: straight-line km from Raffles Place
# dist_regional: km to nearest designated regional centre
# regional: whether the town itself is a regional centre

towns = [
    {"name": "Queenstown",    "dist_cbd": 3.2,  "price": 598, "regional": False, "nearest_rc": "Jurong East",  "dist_rc": 11.2},
    {"name": "Buona Vista",   "dist_cbd": 4.1,  "price": 572, "regional": False, "nearest_rc": "Jurong East",  "dist_rc": 10.4},
    {"name": "Toa Payoh",     "dist_cbd": 4.4,  "price": 555, "regional": False, "nearest_rc": "Tampines",     "dist_rc": 14.2},
    {"name": "Bishan",        "dist_cbd": 7.5,  "price": 530, "regional": False, "nearest_rc": "Tampines",     "dist_rc": 11.0},
    {"name": "Marine Parade", "dist_cbd": 5.2,  "price": 548, "regional": False, "nearest_rc": "Tampines",     "dist_rc": 13.5},
    {"name": "Clementi",      "dist_cbd": 8.2,  "price": 520, "regional": False, "nearest_rc": "Jurong East",  "dist_rc": 6.3},
    {"name": "Ang Mo Kio",    "dist_cbd": 9.1,  "price": 495, "regional": False, "nearest_rc": "Tampines",     "dist_rc": 9.8},
    {"name": "Serangoon",     "dist_cbd": 8.8,  "price": 500, "regional": False, "nearest_rc": "Tampines",     "dist_rc": 10.2},
    {"name": "Bedok",         "dist_cbd": 11.4, "price": 468, "regional": False, "nearest_rc": "Tampines",     "dist_rc": 7.2},
    {"name": "Tampines",      "dist_cbd": 18.5, "price": 460, "regional": True,  "nearest_rc": "Tampines",     "dist_rc": 0.0},
    {"name": "Jurong East",   "dist_cbd": 14.2, "price": 442, "regional": True,  "nearest_rc": "Jurong East",  "dist_rc": 0.0},
    {"name": "Hougang",       "dist_cbd": 13.0, "price": 450, "regional": False, "nearest_rc": "Tampines",     "dist_rc": 6.8},
    {"name": "Pasir Ris",     "dist_cbd": 22.0, "price": 415, "regional": False, "nearest_rc": "Tampines",     "dist_rc": 4.2},
    {"name": "Yishun",        "dist_cbd": 20.3, "price": 398, "regional": False, "nearest_rc": "Woodlands",    "dist_rc": 6.1},
    {"name": "Choa Chu Kang", "dist_cbd": 18.8, "price": 410, "regional": False, "nearest_rc": "Jurong East",  "dist_rc": 6.5},
    {"name": "Woodlands",     "dist_cbd": 25.6, "price": 375, "regional": True,  "nearest_rc": "Woodlands",    "dist_rc": 0.0},
    {"name": "Sembawang",     "dist_cbd": 26.8, "price": 362, "regional": False, "nearest_rc": "Woodlands",    "dist_rc": 2.8},
    {"name": "Bukit Panjang", "dist_cbd": 16.5, "price": 430, "regional": False, "nearest_rc": "Jurong East",  "dist_rc": 5.8},
    {"name": "Sengkang",      "dist_cbd": 16.2, "price": 435, "regional": False, "nearest_rc": "Tampines",     "dist_rc": 4.5},
    {"name": "Punggol",       "dist_cbd": 19.5, "price": 418, "regional": False, "nearest_rc": "Tampines",     "dist_rc": 4.8},
    {"name": "Jurong West",   "dist_cbd": 17.0, "price": 385, "regional": False, "nearest_rc": "Jurong East",  "dist_rc": 4.2},
    {"name": "Geylang",       "dist_cbd": 4.8,  "price": 490, "regional": False, "nearest_rc": "Tampines",     "dist_rc": 14.0},
    {"name": "Kallang",       "dist_cbd": 3.8,  "price": 560, "regional": False, "nearest_rc": "Tampines",     "dist_rc": 15.2},
    {"name": "Bukit Merah",   "dist_cbd": 4.0,  "price": 565, "regional": False, "nearest_rc": "Jurong East",  "dist_rc": 11.8},
    {"name": "Bukit Batok",   "dist_cbd": 14.8, "price": 422, "regional": False, "nearest_rc": "Jurong East",  "dist_rc": 4.1},
    {"name": "Tengah",        "dist_cbd": 17.5, "price": 390, "regional": False, "nearest_rc": "Jurong East",  "dist_rc": 5.0},
]

names      = [t["name"]      for t in towns]
dist_cbd   = np.array([t["dist_cbd"]   for t in towns])
dist_rc    = np.array([t["dist_rc"]    for t in towns])
price      = np.array([t["price"]      for t in towns])
is_rc      = np.array([t["regional"]   for t in towns])

# ── Model 1: Monocentric OLS ───────────────────────────────────────────────────
slope1, intercept1, r1, p1, se1 = stats.linregress(dist_cbd, price)
r2_mono = r1 ** 2
price_hat_mono = intercept1 + slope1 * dist_cbd

# ── Model 2: Polycentric OLS (dist_cbd + dist_rc) ─────────────────────────────
X2 = np.column_stack([np.ones(len(price)), dist_cbd, dist_rc])
coeffs2, residuals2, rank2, sv2 = np.linalg.lstsq(X2, price, rcond=None)
price_hat_poly = X2 @ coeffs2
ss_res2 = np.sum((price - price_hat_poly) ** 2)
ss_tot  = np.sum((price - price.mean()) ** 2)
r2_poly = 1 - ss_res2 / ss_tot

print("=" * 55)
print("  Singapore City Model — Regression Results")
print("=" * 55)
print(f"\nModel 1 — Monocentric (dist to CBD only)")
print(f"  Intercept : SGD ${intercept1:,.0f}k")
print(f"  Slope     : SGD ${slope1:,.1f}k per km from CBD")
print(f"  R²        : {r2_mono:.3f}")
print(f"  p-value   : {p1:.4f}")

print(f"\nModel 2 — Polycentric (dist to CBD + dist to nearest RC)")
print(f"  Intercept      : SGD ${coeffs2[0]:,.0f}k")
print(f"  β_CBD          : SGD ${coeffs2[1]:,.1f}k per km from CBD")
print(f"  β_RegionalCtr  : SGD ${coeffs2[2]:,.1f}k per km from RC")
print(f"  R²             : {r2_poly:.3f}")
print(f"  ΔR²            : +{r2_poly - r2_mono:.3f}")
print("=" * 55)

# ── Plotting ──────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 6))
fig.suptitle("Testing the Monocentric City Model Against Singapore's Spatial Structure",
             fontsize=13, fontweight="bold", y=1.01)

BLUE   = "#2a78d6"
RED    = "#e34948"
GREEN  = "#1baf7a"
GREY   = "#888780"

# ── Panel 1: Monocentric scatter + regression line ────────────────────────────
ax1 = axes[0]
x_line = np.linspace(dist_cbd.min() - 1, dist_cbd.max() + 1, 200)
y_line = intercept1 + slope1 * x_line

ax1.scatter(dist_cbd[~is_rc], price[~is_rc], color=BLUE,  s=70, zorder=3, label="Regular town")
ax1.scatter(dist_cbd[is_rc],  price[is_rc],  color=RED,   s=90, zorder=4, marker="D", label="Regional centre")
ax1.plot(x_line, y_line, color=BLUE, linewidth=2, label=f"Monocentric fit (R²={r2_mono:.2f})")

for i, t in enumerate(towns):
    if t["regional"] or t["name"] in ("Queenstown", "Woodlands", "Sembawang", "Tampines"):
        ax1.annotate(t["name"], (dist_cbd[i], price[i]),
                     textcoords="offset points", xytext=(5, 4),
                     fontsize=7.5, color=GREY)

ax1.set_xlabel("Distance from CBD — Raffles Place (km)", fontsize=10)
ax1.set_ylabel("Median HDB resale price (SGD $k)", fontsize=10)
ax1.set_title("Model 1: Monocentric", fontsize=11, fontweight="bold")
ax1.legend(fontsize=9)
ax1.set_ylim(320, 640)
ax1.grid(axis="y", alpha=0.3, linewidth=0.7)
ax1.spines[["top", "right"]].set_visible(False)

# ── Panel 2: Both models + residuals from polycentric ─────────────────────────
ax2 = axes[1]

# Polycentric fitted line: hold dist_rc fixed at its mean for the 2D slice
dist_rc_mean = dist_rc.mean()
y_poly_line = coeffs2[0] + coeffs2[1] * x_line + coeffs2[2] * dist_rc_mean

ax2.scatter(dist_cbd[~is_rc], price[~is_rc], color=BLUE, s=70, zorder=3, alpha=0.8)
ax2.scatter(dist_cbd[is_rc],  price[is_rc],  color=RED,  s=90, zorder=4, marker="D")

# Residual lines from polycentric fit
for i in range(len(towns)):
    col = RED if is_rc[i] else BLUE
    ax2.plot([dist_cbd[i], dist_cbd[i]], [price[i], price_hat_poly[i]],
             color=col, linewidth=0.8, alpha=0.4, zorder=2)

ax2.plot(x_line, y_line,      color=BLUE,  linewidth=2,    linestyle="-",  label=f"Monocentric fit (R²={r2_mono:.2f})")
ax2.plot(x_line, y_poly_line, color=GREEN, linewidth=2,    linestyle="--", label=f"Polycentric fit (R²={r2_poly:.2f}, dist_rc held at mean)")

for i, t in enumerate(towns):
    if t["regional"] or t["name"] in ("Queenstown", "Woodlands", "Sembawang"):
        ax2.annotate(t["name"], (dist_cbd[i], price[i]),
                     textcoords="offset points", xytext=(5, 4),
                     fontsize=7.5, color=GREY)

ax2.set_xlabel("Distance from CBD — Raffles Place (km)", fontsize=10)
ax2.set_ylabel("Median HDB resale price (SGD $k)", fontsize=10)
ax2.set_title("Model 2: Polycentric comparison", fontsize=11, fontweight="bold")
ax2.legend(fontsize=9)
ax2.set_ylim(320, 640)
ax2.grid(axis="y", alpha=0.3, linewidth=0.7)
ax2.spines[["top", "right"]].set_visible(False)

plt.tight_layout()
plt.savefig("preview.png", dpi=150, bbox_inches="tight")
print("\nPlot saved to preview.png")
plt.show()
