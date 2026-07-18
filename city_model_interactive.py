"""
Singapore Monocentric vs Polycentric City Model — Interactive GUI
=================================================================
Requires: numpy, matplotlib, scipy
Install:  python3 -m pip install numpy matplotlib scipy
Run:      python3 city_model_interactive.py
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.widgets import Button
from matplotlib.animation import FuncAnimation
from scipy import stats

# ── Colours ───────────────────────────────────────────────────────────────────
BLUE       = "#2a78d6"
RED        = "#e34948"
GREEN      = "#1baf7a"
GREY       = "#888780"
BG         = "#f8f8f6"
PANEL_BG   = "#ffffff"
BTN_ON     = "#2a78d6"
BTN_OFF    = "#e8e8e4"
BTN_TXT_ON = "white"
BTN_TXT_OF = "#444441"

# ── Data ──────────────────────────────────────────────────────────────────────
towns = [
    {"name": "Queenstown",    "dist_cbd": 3.2,  "price": 598, "regional": False, "dist_rc": 11.2},
    {"name": "Buona Vista",   "dist_cbd": 4.1,  "price": 572, "regional": False, "dist_rc": 10.4},
    {"name": "Toa Payoh",     "dist_cbd": 4.4,  "price": 555, "regional": False, "dist_rc": 14.2},
    {"name": "Bishan",        "dist_cbd": 7.5,  "price": 530, "regional": False, "dist_rc": 11.0},
    {"name": "Marine Parade", "dist_cbd": 5.2,  "price": 548, "regional": False, "dist_rc": 13.5},
    {"name": "Clementi",      "dist_cbd": 8.2,  "price": 520, "regional": False, "dist_rc":  6.3},
    {"name": "Ang Mo Kio",    "dist_cbd": 9.1,  "price": 495, "regional": False, "dist_rc":  9.8},
    {"name": "Serangoon",     "dist_cbd": 8.8,  "price": 500, "regional": False, "dist_rc": 10.2},
    {"name": "Bedok",         "dist_cbd": 11.4, "price": 468, "regional": False, "dist_rc":  7.2},
    {"name": "Tampines",      "dist_cbd": 18.5, "price": 460, "regional": True,  "dist_rc":  0.0},
    {"name": "Jurong East",   "dist_cbd": 14.2, "price": 442, "regional": True,  "dist_rc":  0.0},
    {"name": "Hougang",       "dist_cbd": 13.0, "price": 450, "regional": False, "dist_rc":  6.8},
    {"name": "Pasir Ris",     "dist_cbd": 22.0, "price": 415, "regional": False, "dist_rc":  4.2},
    {"name": "Yishun",        "dist_cbd": 20.3, "price": 398, "regional": False, "dist_rc":  6.1},
    {"name": "Choa Chu Kang", "dist_cbd": 18.8, "price": 410, "regional": False, "dist_rc":  6.5},
    {"name": "Woodlands",     "dist_cbd": 25.6, "price": 375, "regional": True,  "dist_rc":  0.0},
    {"name": "Sembawang",     "dist_cbd": 26.8, "price": 362, "regional": False, "dist_rc":  2.8},
    {"name": "Bukit Panjang", "dist_cbd": 16.5, "price": 430, "regional": False, "dist_rc":  5.8},
    {"name": "Sengkang",      "dist_cbd": 16.2, "price": 435, "regional": False, "dist_rc":  4.5},
    {"name": "Punggol",       "dist_cbd": 19.5, "price": 418, "regional": False, "dist_rc":  4.8},
    {"name": "Jurong West",   "dist_cbd": 17.0, "price": 385, "regional": False, "dist_rc":  4.2},
    {"name": "Geylang",       "dist_cbd": 4.8,  "price": 490, "regional": False, "dist_rc": 14.0},
    {"name": "Kallang",       "dist_cbd": 3.8,  "price": 560, "regional": False, "dist_rc": 15.2},
    {"name": "Bukit Merah",   "dist_cbd": 4.0,  "price": 565, "regional": False, "dist_rc": 11.8},
    {"name": "Bukit Batok",   "dist_cbd": 14.8, "price": 422, "regional": False, "dist_rc":  4.1},
    {"name": "Tengah",        "dist_cbd": 17.5, "price": 390, "regional": False, "dist_rc":  5.0},
]

names    = [t["name"]     for t in towns]
dist_cbd = np.array([t["dist_cbd"]  for t in towns])
dist_rc  = np.array([t["dist_rc"]   for t in towns])
price    = np.array([t["price"]     for t in towns])
is_rc    = np.array([t["regional"]  for t in towns])

# ── Regressions ───────────────────────────────────────────────────────────────
slope1, intercept1, r1, p1, _ = stats.linregress(dist_cbd, price)
r2_mono = r1 ** 2

X2 = np.column_stack([np.ones(len(price)), dist_cbd, dist_rc])
coeffs2, *_ = np.linalg.lstsq(X2, price, rcond=None)
price_hat_poly = X2 @ coeffs2
ss_res2 = np.sum((price - price_hat_poly) ** 2)
ss_tot  = np.sum((price - price.mean()) ** 2)
r2_poly = 1 - ss_res2 / ss_tot

x_line       = np.linspace(1, 29, 300)
y_mono_line  = intercept1 + slope1 * x_line
y_poly_line  = coeffs2[0] + coeffs2[1] * x_line + coeffs2[2] * dist_rc.mean()

print("=" * 55)
print("  Singapore City Model — Regression Results")
print("=" * 55)
print(f"\nModel 1 — Monocentric")
print(f"  Slope  : SGD ${slope1:,.1f}k per km")
print(f"  R²     : {r2_mono:.3f}   p = {p1:.4f}")
print(f"\nModel 2 — Polycentric")
print(f"  β_CBD  : SGD ${coeffs2[1]:,.1f}k per km")
print(f"  β_RC   : SGD ${coeffs2[2]:,.1f}k per km to nearest RC")
print(f"  R²     : {r2_poly:.3f}   ΔR² = +{r2_poly - r2_mono:.3f}")
print("=" * 55)

# ── Figure layout ─────────────────────────────────────────────────────────────
fig = plt.figure(figsize=(13, 7.5), facecolor=BG)
fig.canvas.manager.set_window_title("Singapore City Model")

# Main plot area
ax = fig.add_axes([0.08, 0.13, 0.86, 0.72], facecolor=PANEL_BG)

# Button axes
ax_btn_mono = fig.add_axes([0.08,  0.02, 0.18, 0.06])
ax_btn_poly = fig.add_axes([0.28,  0.02, 0.22, 0.06])

btn_mono = Button(ax_btn_mono, "Monocentric model",  color=BTN_ON,  hovercolor="#1a68c6")
btn_poly = Button(ax_btn_poly, "Polycentric comparison", color=BTN_OFF, hovercolor="#d8d8d4")

for btn, is_active in [(btn_mono, True), (btn_poly, False)]:
    btn.label.set_fontsize(10)
    btn.label.set_color(BTN_TXT_ON if is_active else BTN_TXT_OF)
    btn.ax.set_facecolor(BTN_ON if is_active else BTN_OFF)

# Stat boxes (right side of button row)
stat_ax = fig.add_axes([0.55, 0.01, 0.42, 0.08], facecolor=BG)
stat_ax.axis("off")
stat_text = stat_ax.text(0, 0.5, "", fontsize=10, va="center", color="#333")

# ── State ─────────────────────────────────────────────────────────────────────
state = {"view": "mono", "anim": None, "poly_alpha": 0.0}

# ── Draw helpers ──────────────────────────────────────────────────────────────
def label_towns(ax_):
    for i, t in enumerate(towns):
        if t["regional"] or t["name"] in ("Queenstown", "Woodlands", "Sembawang", "Tampines", "Kallang"):
            ax_.annotate(t["name"], (dist_cbd[i], price[i]),
                         textcoords="offset points", xytext=(6, 4),
                         fontsize=8, color=GREY)

def update_stats(view):
    if view == "mono":
        stat_text.set_text(
            f"Model: Monocentric     R² = {r2_mono:.2f}     "
            f"Price drop per km: –${abs(slope1*1000):,.0f}"
        )
    else:
        stat_text.set_text(
            f"Model: Polycentric     R² = {r2_poly:.2f}  (+{r2_poly-r2_mono:.2f})     "
            f"β_CBD: –${abs(coeffs2[1]*1000):,.0f}/km     β_RC: –${abs(coeffs2[2]*1000):,.0f}/km"
        )
    fig.canvas.draw_idle()

def draw_base():
    ax.clear()
    ax.set_facecolor(PANEL_BG)

    # Scatter
    ax.scatter(dist_cbd[~is_rc], price[~is_rc],
               color=BLUE, s=75, zorder=3, label="Regular town", alpha=0.9)
    ax.scatter(dist_cbd[is_rc], price[is_rc],
               color=RED, s=100, zorder=4, marker="D", label="Regional centre")

    # Monocentric line (always shown)
    ax.plot(x_line, y_mono_line, color=BLUE, linewidth=2.2,
            label=f"Monocentric fit  R²={r2_mono:.2f}", zorder=2)

    label_towns(ax)

    ax.set_xlim(0, 30)
    ax.set_ylim(310, 650)
    ax.set_xlabel("Distance from CBD — Raffles Place (km)", fontsize=10, color="#444")
    ax.set_ylabel("Median HDB resale price (SGD $k)", fontsize=10, color="#444")
    ax.tick_params(colors="#666", labelsize=9)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"${int(v)}k"))
    ax.grid(axis="y", alpha=0.25, linewidth=0.7, color="#ccc")
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color("#ccc")

    title = ("Model 1: Monocentric gradient"
             if state["view"] == "mono"
             else "Model 1 vs Model 2: Monocentric + Polycentric")
    ax.set_title(title, fontsize=12, fontweight="bold",
                 color="#222", pad=10)

    return ax

# ── Polycentric animation ──────────────────────────────────────────────────────
FRAMES = 30

def animate_to_poly(_frame):
    alpha = (_frame + 1) / FRAMES
    state["poly_alpha"] = alpha

    draw_base()

    # Residual lines fading in
    for i in range(len(towns)):
        col = RED if is_rc[i] else BLUE
        ax.plot([dist_cbd[i], dist_cbd[i]],
                [price[i], price_hat_poly[i]],
                color=col, linewidth=0.9, alpha=0.35 * alpha, zorder=2)

    # Polycentric line drawing in
    n_pts = max(2, int(alpha * len(x_line)))
    ax.plot(x_line[:n_pts], y_poly_line[:n_pts],
            color=GREEN, linewidth=2.2, linestyle="--",
            alpha=alpha, label=f"Polycentric fit  R²={r2_poly:.2f}", zorder=5)

    # Legend
    handles = [
        mpatches.Patch(color=BLUE,  label="Regular town"),
        mpatches.Patch(color=RED,   label="Regional centre"),
        plt.Line2D([0], [0], color=BLUE,  lw=2,   label=f"Monocentric fit  R²={r2_mono:.2f}"),
        plt.Line2D([0], [0], color=GREEN, lw=2, linestyle="--",
                   label=f"Polycentric fit  R²={r2_poly:.2f}  (+{r2_poly-r2_mono:.2f})"),
    ]
    ax.legend(handles=handles, fontsize=8.5, framealpha=0.9,
              loc="upper right", edgecolor="#ddd")

    fig.canvas.draw_idle()

def draw_mono_static():
    draw_base()
    handles = [
        mpatches.Patch(color=BLUE, label="Regular town"),
        mpatches.Patch(color=RED,  label="Regional centre"),
        plt.Line2D([0], [0], color=BLUE, lw=2,
                   label=f"Monocentric fit  R²={r2_mono:.2f}"),
    ]
    ax.legend(handles=handles, fontsize=8.5, framealpha=0.9,
              loc="upper right", edgecolor="#ddd")
    fig.canvas.draw_idle()

def draw_poly_static():
    draw_base()
    for i in range(len(towns)):
        col = RED if is_rc[i] else BLUE
        ax.plot([dist_cbd[i], dist_cbd[i]],
                [price[i], price_hat_poly[i]],
                color=col, linewidth=0.9, alpha=0.35, zorder=2)
    ax.plot(x_line, y_poly_line, color=GREEN, linewidth=2.2, linestyle="--",
            label=f"Polycentric fit  R²={r2_poly:.2f}", zorder=5)
    handles = [
        mpatches.Patch(color=BLUE,  label="Regular town"),
        mpatches.Patch(color=RED,   label="Regional centre"),
        plt.Line2D([0], [0], color=BLUE,  lw=2,   label=f"Monocentric fit  R²={r2_mono:.2f}"),
        plt.Line2D([0], [0], color=GREEN, lw=2, linestyle="--",
                   label=f"Polycentric fit  R²={r2_poly:.2f}  (+{r2_poly-r2_mono:.2f})"),
    ]
    ax.legend(handles=handles, fontsize=8.5, framealpha=0.9,
              loc="upper right", edgecolor="#ddd")
    fig.canvas.draw_idle()

# ── Tooltip on hover ──────────────────────────────────────────────────────────
annot = ax.annotate("", xy=(0, 0), xytext=(12, 12),
                    textcoords="offset points",
                    bbox=dict(boxstyle="round,pad=0.4", fc="white", ec="#ccc", lw=0.8),
                    fontsize=9, color="#222", zorder=10)
annot.set_visible(False)

def on_hover(event):
    if event.inaxes != ax:
        annot.set_visible(False)
        fig.canvas.draw_idle()
        return
    for i in range(len(towns)):
        dx = (dist_cbd[i] - event.xdata) / 2
        dy = (price[i]    - event.ydata) / 40
        if dx*dx + dy*dy < 0.5:
            tag = " ★ Regional centre" if is_rc[i] else ""
            annot.set_text(f"{names[i]}{tag}\n{dist_cbd[i]:.1f} km from CBD\nSGD ${price[i]}k")
            annot.xy = (dist_cbd[i], price[i])
            annot.set_visible(True)
            fig.canvas.draw_idle()
            return
    annot.set_visible(False)
    fig.canvas.draw_idle()

fig.canvas.mpl_connect("motion_notify_event", on_hover)

# ── Button callbacks ───────────────────────────────────────────────────────────
def switch_mono(event):
    if state["view"] == "mono":
        return
    state["view"] = "mono"
    if state["anim"]:
        state["anim"].event_source.stop()
    btn_mono.ax.set_facecolor(BTN_ON)
    btn_mono.label.set_color(BTN_TXT_ON)
    btn_poly.ax.set_facecolor(BTN_OFF)
    btn_poly.label.set_color(BTN_TXT_OF)
    draw_mono_static()
    update_stats("mono")

def switch_poly(event):
    if state["view"] == "poly":
        return
    state["view"] = "poly"
    btn_mono.ax.set_facecolor(BTN_OFF)
    btn_mono.label.set_color(BTN_TXT_OF)
    btn_poly.ax.set_facecolor(BTN_ON)
    btn_poly.label.set_color(BTN_TXT_ON)
    update_stats("poly")
    anim = FuncAnimation(fig, animate_to_poly, frames=FRAMES,
                         interval=30, repeat=False)
    state["anim"] = anim
    fig.canvas.draw_idle()

btn_mono.on_clicked(switch_mono)
btn_poly.on_clicked(switch_poly)

# ── Initial draw ──────────────────────────────────────────────────────────────
draw_mono_static()
update_stats("mono")

fig.text(0.5, 0.965,
         "Testing the Monocentric City Model Against Singapore's Spatial Structure",
         ha="center", fontsize=13, fontweight="bold", color="#111")
fig.text(0.5, 0.945,
         "HDB resale flat prices vs distance from Raffles Place (CBD) — realistic estimates",
         ha="center", fontsize=9, color=GREY)

plt.savefig("singapore_city_model.png", dpi=150, bbox_inches="tight",
            facecolor=BG)
print("Static preview saved → singapore_city_model.png")
plt.show()
