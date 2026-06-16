import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch

plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "text.latex.preamble": r"\usepackage{amsmath}",
})

C_PRE  = "#2a9d8f"
C_J    = "#264653"
C_POST = "#e76f51"
C_FWD  = "#457b9d"
C_BWD  = "#e63946"
C_TD   = "#f4a261"

FS = 20

fig, ax = plt.subplots(figsize=(22, 6.5))

ax.set_aspect('equal')

ax.set_ylim(3.5, 9.9)
ax.set_xlim(0, 22)
ax.axis("off")

# ── Positions ──────────────────────────────────────────────────────────────
cx, cy   = 11.0, 5.8
pre_x    = 6.0
pre_ys   = [7.5, 5.8, 4.1]
post_x   = 16.0
post_ys  = [7.5, 5.8, 4.1]
r_small  = 0.48
r_large  = 0.92

# ── Shared top anchor for both side panels ─────────────────────────────────
top_y = pre_ys[0] + r_small + 0.05   # aligns with top of top neuron

# ── Draw outlined circle ───────────────────────────────────────────────────
def draw_circle(ax, x, y, r, ec, label, label_dy=0):
    ax.add_patch(plt.Circle((x, y), r, facecolor="none", edgecolor=ec,
                             linewidth=2.2, zorder=3))
    ax.text(x, y + label_dy, label, ha="center", va="center",
            fontsize=FS, color=ec, fontweight="bold", zorder=4)

draw_circle(ax, cx, cy, r_large, C_J, r"$j$", label_dy=0.55)

g_base = cy - 0.45
g_amp  = 0.52
gx = np.linspace(cx - r_large + 0.12, cx + r_large - 0.12, 300)
gy = g_base + g_amp * np.exp(-0.5 * ((gx - cx) / 0.27) ** 2)
ax.plot(gx, gy, color=C_J, linewidth=2, zorder=5)
ax.plot([cx, cx], [g_base, g_base + g_amp],
        color=C_J, linewidth=1.4, linestyle=":", zorder=6)
ax.text(cx + 0.13, g_base + g_amp + 0.07, r"$\mu_j^t$",
        ha="left", va="bottom", fontsize=FS, color=C_J, zorder=6)
ax.text(cx, cy - r_large - 0.30,
        r"$v^t_j \sim \mathcal{N}(\mu^t_j,\, \sigma^t_j)$",
        ha="center", va="top", fontsize=FS, color=C_J, style="italic")

for y, lbl in zip(pre_ys,  [r"$i_1$", r"$i_2$", r"$i_3$"]):
    draw_circle(ax, pre_x,  y, r_small, C_PRE,  lbl)
for y, lbl in zip(post_ys, [r"$k_1$", r"$k_2$", r"$k_3$"]):
    draw_circle(ax, post_x, y, r_small, C_POST, lbl)

# ── Curved arrow ───────────────────────────────────────────────────────────
def curved_arrow(ax, x1, y1, x2, y2, rad, color, lw=1.8, ls="-"):
    ax.add_patch(FancyArrowPatch(
        (x1, y1), (x2, y2),
        connectionstyle=f"arc3,rad={rad}",
        arrowstyle="-|>", mutation_scale=16,
        color=color, linewidth=lw, linestyle=ls, zorder=2))

RAD = 0.25
for y in pre_ys:
    curved_arrow(ax, pre_x + r_small, y, cx - r_large, cy,  RAD, C_FWD)
    curved_arrow(ax, cx - r_large, cy, pre_x + r_small, y,  RAD, C_BWD, ls="--")
for y in post_ys:
    curved_arrow(ax, cx + r_large, cy, post_x - r_small, y,  RAD, C_FWD)
    curved_arrow(ax, post_x - r_small, y, cx + r_large, cy,  RAD, C_BWD, ls="--")

# ── TD error ───────────────────────────────────────────────────────────────
td_y = 9.1
curved_arrow(ax, cx, td_y, cx, cy + r_large + 0.05, 0.0, C_TD, lw=2.2, ls="--")
ax.text(cx + 0.25, (cy + r_large + td_y) / 2, r"$\delta_t$",
        ha="left", va="center", fontsize=FS, color=C_TD, fontweight="bold")
ax.text(cx, td_y + 0.10,
        r"TD error \quad $\delta_t = R_{t+1} + \gamma V_{t+1} - V_t$",
        ha="center", va="bottom", fontsize=FS, color=C_TD,
        bbox=dict(boxstyle="round,pad=0.35", fc="white", ec=C_TD, lw=1.5))

# ── LEFT PANEL — starts at top_y ──────────────────────────────────────────
lx = 2.8
row = 0.60   # vertical step between lines

ax.text(lx, top_y,
        r"\textbf{Perturbation trace}",
        ha="center", va="top", fontsize=FS, color=C_J)
ax.text(lx, top_y - 1*row,
        r"$\xi_j^t = \alpha\,\xi_j^{t-1} + \varepsilon_j^t$",
        ha="center", va="top", fontsize=FS, color=C_J)
ax.text(lx, top_y - 2*row,
        r"$\varepsilon_j^t \sim \mathcal{N}(0,\,\sigma^2)$",
        ha="center", va="top", fontsize=FS, color=C_J)

ax.text(lx, top_y - 3.2*row,
        r"\textbf{1-step backprop gradient}",
        ha="center", va="top", fontsize=FS, color=C_BWD)
ax.text(lx, top_y - 4.2*row,
        r"$\nabla^{\mathrm{1\text{-}step}} ="
        r"\; W_{ji} \cdot \psi_j^t \cdot"
        r"\dfrac{(1-\alpha^2)\xi_j^t}{(\sigma_j^t)^2}$",
        ha="center", va="top", fontsize=FS, color=C_BWD)

left_bottom = top_y - 5.5*row   # approx bottom of dfrac

# ── RIGHT PANEL — starts at same top_y ────────────────────────────────────
legend_items = [
    (C_PRE,  "-",  r"Pre-synaptic neurons"),
    (C_J,    "-",  r"Neuron $j$ (membrane voltage)"),
    (C_POST, "-",  r"Post-synaptic neurons"),
    (C_FWD,  "-",  r"Forward signal / eligibility trace"),
    (C_BWD,  "--", r"Local gradient (backward)"),
    (C_TD,   "--", r"TD error broadcast"),
]

rx_sw = 17.3
rx_tx = 17.9
leg_row = 0.60

ax.text(rx_sw + 1.5, top_y,
        r"\textbf{Legend}",
        ha="center", va="top", fontsize=FS, color="#333333")

for i, (color, ls, label) in enumerate(legend_items):
    yi = top_y - (i + 1) * leg_row
    ax.plot([rx_sw, rx_sw + 0.5], [yi, yi],
            color=color, linewidth=2.5, linestyle=ls, solid_capstyle="round")
    ax.text(rx_tx, yi, label,
            ha="left", va="center", fontsize=FS, color="#333333")

right_bottom = top_y - (len(legend_items) + 1) * leg_row

# ── Trim ylim to actual content — no empty space ───────────────────────────
diag_bottom = cy - r_large - 0.30 - 0.55   # distribution text bottom
content_bottom = min(left_bottom, right_bottom, diag_bottom)

ax.set_title(r"Distributed e-prop: local gradient flow",
             fontsize=FS, pad=12, color="#1d1d1d")

os.makedirs("img", exist_ok=True)
plt.savefig("img/distributed_eprop_diagram.pdf", bbox_inches="tight",
            facecolor=fig.get_facecolor())
plt.savefig("img/distributed_eprop_diagram.png", dpi=300, bbox_inches="tight",
            facecolor=fig.get_facecolor())