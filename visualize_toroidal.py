import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, Rectangle, Circle

# ── Matplotlib / LaTeX style ────────────────────────────────────────────────
plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "text.latex.preamble": r"\usepackage{amsmath}",
})

# ── Colors (similar palette) ────────────────────────────────────────────────
C_PRE   = "#2a9d8f"
C_POST  = "#e76f51"
C_J     = "#264653"
C_CONN  = "#457b9d"
C_WRAP  = "#e63946"
C_BOX   = "#333333"
FS = 18

# ── RNG ─────────────────────────────────────────────────────────────────────
rng = np.random.default_rng(7)

# ── Toroidal helpers ────────────────────────────────────────────────────────
def wrap01(x):
    """Wrap coordinate to [0, 1)."""
    return x % 1.0

def torus_delta(a, b):
    """
    Minimal signed delta on 1D torus from a to b in [-0.5, 0.5).
    Useful for showing shortest wrap direction.
    """
    d = (b - a + 0.5) % 1.0 - 0.5
    return d

def sample_toroidal_gaussian_partners(j_xy, K, sigma, pre_shape):
    """
    Sample K presynaptic partners on a 2D torus around j_xy (in [0,1)^2),
    using Gaussian offsets, then map to nearest pre grid indices.
    Returns unique indices (may be <K if duplicates appear).
    """
    H, W = pre_shape
    js = []
    attempts = 0
    while len(js) < K and attempts < 50 * K:
        attempts += 1
        dx, dy = rng.normal(0.0, sigma, size=2)
        x = wrap01(j_xy[0] + dx)
        y = wrap01(j_xy[1] + dy)

        # nearest grid index
        ix = int(np.round(x * (W - 1)))
        iy = int(np.round(y * (H - 1)))
        js.append((iy, ix))
        js = list(dict.fromkeys(js))  # unique preserve order

    return js

# ── Coordinate mapping to figure space ──────────────────────────────────────
def sheet_to_fig(xy01, origin, size):
    """
    Map (x,y) in [0,1]^2 to figure coords given origin (x0,y0) and size (w,h).
    """
    x0, y0 = origin
    w, h = size
    x = x0 + xy01[0] * w
    y = y0 + xy01[1] * h
    return x, y

def grid_index_to_xy01(idx, shape):
    """Convert (row,col) -> normalized (x,y) in [0,1] using cell centers."""
    H, W = shape
    iy, ix = idx
    x = ix / (W - 1) if W > 1 else 0.5
    y = iy / (H - 1) if H > 1 else 0.5
    return np.array([x, y])

# ── Drawing primitives ──────────────────────────────────────────────────────
def draw_sheet(ax, origin, size, color, title):
    x0, y0 = origin
    w, h = size
    ax.add_patch(Rectangle((x0, y0), w, h, fill=False, lw=2.0, ec=color, zorder=1))
    ax.text(x0 + w/2, y0 + h + 0.35, title, ha="center", va="bottom",
            fontsize=FS, color=color, fontweight="bold")

def draw_points_grid(ax, origin, size, shape, color, r=0.05, alpha=0.6):
    H, W = shape
    for iy in range(H):
        for ix in range(W):
            xy01 = grid_index_to_xy01((iy, ix), shape)
            x, y = sheet_to_fig(xy01, origin, size)
            ax.add_patch(Circle((x, y), r, fc=color, ec="none", alpha=alpha, zorder=2))

def curved_arrow(ax, x1, y1, x2, y2, rad, color, lw=1.8, ls="-", ms=14, z=3):
    ax.add_patch(FancyArrowPatch(
        (x1, y1), (x2, y2),
        connectionstyle=f"arc3,rad={rad}",
        arrowstyle="-|>",
        mutation_scale=ms,
        color=color,
        linewidth=lw,
        linestyle=ls,
        zorder=z
    ))

def draw_wrapped_connection(ax, pre_xy01, post_xy01,
                           pre_origin, pre_size, post_origin, post_size,
                           color=C_CONN, wrap_color=C_WRAP, lw=1.9, rad=0.0,
                           show_wrap=True):
    """
    Draw connection from pre to post. If pre->post shortest torus delta crosses border,
    optionally show wrap with dashed red segment cues.
    NOTE: Here we visualize wrap based on minimal deltas separately for x,y
    just to annotate crossing; the actual arrow is drawn from pre point to post point
    in figure coordinates (across gap).
    """
    # actual endpoints in fig coords
    x1, y1 = sheet_to_fig(pre_xy01, pre_origin, pre_size)
    x2, y2 = sheet_to_fig(post_xy01, post_origin, post_size)
    curved_arrow(ax, x1, y1, x2, y2, rad=rad, color=color, lw=lw, ls="-", ms=15, z=4)

    if not show_wrap:
        return

    # Detect wrap in each dimension on torus (in [0,1)). If |delta| is near 0.5,
    # means it likely wrapped. More robust: check whether naive diff differs from torus diff.
    dx_naive = post_xy01[0] - pre_xy01[0]
    dy_naive = post_xy01[1] - pre_xy01[1]
    dx_t = torus_delta(pre_xy01[0], post_xy01[0])
    dy_t = torus_delta(pre_xy01[1], post_xy01[1])

    wrapped_x = np.sign(dx_naive) != np.sign(dx_t) and abs(dx_naive) > 0.5
    wrapped_y = np.sign(dy_naive) != np.sign(dy_t) and abs(dy_naive) > 0.5

    if wrapped_x or wrapped_y:
        # small wrap indicator near the pre point (dashed red tick)
        tick = 0.28
        ax.plot([x1 - tick, x1 + tick], [y1, y1],
                color=wrap_color, lw=2.2, ls="--", zorder=5)
        ax.plot([x2 - tick, x2 + tick], [y2, y2],
                color=wrap_color, lw=2.2, ls="--", zorder=5)

# ── Main figure ─────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(22, 7.0))
ax.set_aspect("equal")
ax.axis("off")
ax.set_xlim(0, 22)
ax.set_ylim(0, 10)

# Layout: left panel (text), then pre sheet, gap, post sheet, right legend
# Sheets
pre_origin  = (6.0, 1.3)
post_origin = (13.2, 1.3)
sheet_size  = (5.2, 7.2)

# Layer shapes (edit)
pre_shape  = (9, 9)
post_shape = (9, 9)

draw_sheet(ax, pre_origin, sheet_size, C_PRE,  r"\textbf{Pre layer (sheet)}")
draw_sheet(ax, post_origin, sheet_size, C_POST, r"\textbf{Post layer (sheet)}")

# light grid points
draw_points_grid(ax, pre_origin,  sheet_size, pre_shape,  C_PRE,  r=0.055, alpha=0.35)
draw_points_grid(ax, post_origin, sheet_size, post_shape, C_POST, r=0.055, alpha=0.35)

# Choose a postsynaptic neuron j (in post grid) and highlight
j_idx = (1, 8)  # (row, col) in post layer
j_xy01 = grid_index_to_xy01(j_idx, post_shape)
jx, jy = sheet_to_fig(j_xy01, post_origin, sheet_size)

ax.add_patch(Circle((jx, jy), 0.18, fc="none", ec=C_J, lw=2.6, zorder=6))
ax.text(jx, jy + 0.42, r"$j$", ha="center", va="center",
        fontsize=FS, color=C_J, fontweight="bold", zorder=7)

# Sample presynaptic partners using toroidal Gaussian around j_xy01
K = 14
sigma = 0.18
partners = sample_toroidal_gaussian_partners(j_xy01, K=K, sigma=sigma, pre_shape=pre_shape)

# Draw connections
for t, idx in enumerate(partners):
    pre_xy01 = grid_index_to_xy01(idx, pre_shape)
    px, py = sheet_to_fig(pre_xy01, pre_origin, sheet_size)

    # highlight used presynaptic neuron
    ax.add_patch(Circle((px, py), 0.10, fc=C_PRE, ec="none", alpha=0.9, zorder=6))

    # small radial variation to reduce overlap
    rad = 0.12 * np.sin(2*np.pi * (t / max(1, len(partners))))
    draw_wrapped_connection(
        ax,
        pre_xy01=pre_xy01,
        post_xy01=j_xy01,
        pre_origin=pre_origin, pre_size=sheet_size,
        post_origin=post_origin, post_size=sheet_size,
        color=C_CONN, wrap_color=C_WRAP, lw=1.8, rad=rad,
        show_wrap=True
    )

# Highlight j in post layer
ax.add_patch(Circle((jx, jy), 0.11, fc=C_POST, ec="none", alpha=0.95, zorder=7))

# ── LEFT PANEL (math / sampling rule) ─────────────────────────────────────
lx   = 2.7
FS_L = 14        # smaller font for formula lines only
top_y = 9.0
row   = 0.85     # more breathing room between lines

ax.text(lx, top_y,
        r"\textbf{Toroidal Gaussian connectivity}",
        ha="center", va="top", fontsize=16, color=C_BOX)

ax.text(lx, top_y - 1.0*row,
        r"Sample offsets:",
        ha="center", va="top", fontsize=FS_L, color=C_BOX)
ax.text(lx, top_y - 1.55*row,
        r"$(\Delta x,\Delta y)\sim\mathcal{N}(0,\sigma^2 I)$",
        ha="center", va="top", fontsize=FS_L, color=C_BOX)

ax.text(lx, top_y - 2.6*row,
        r"Wrap on torus:",
        ha="center", va="top", fontsize=FS_L, color=C_BOX)
ax.text(lx, top_y - 3.15*row,
        r"$x' = (x+\Delta x)\bmod 1$",
        ha="center", va="top", fontsize=FS_L, color=C_BOX)
ax.text(lx, top_y - 3.65*row,
        r"$y' = (y+\Delta y)\bmod 1$",
        ha="center", va="top", fontsize=FS_L, color=C_BOX)

ax.text(lx, top_y - 4.7*row,
        r"Map to grid:",
        ha="center", va="top", fontsize=FS_L, color=C_BOX)
ax.text(lx, top_y - 5.25*row,
        r"$(i_x,i_y)=\mathrm{nearestGrid}(x',y')$",
        ha="center", va="top", fontsize=FS_L, color=C_BOX)

ax.text(lx, top_y - 6.4*row,
        r"\textbf{Shown:} partners of neuron $j$",
        ha="center", va="top", fontsize=FS_L, color=C_J)
ax.text(lx, top_y - 6.95*row,
        rf"$K={K},\quad \sigma={sigma:.2f}$",
        ha="center", va="top", fontsize=FS_L, color=C_J)

# ── RIGHT LEGEND ────────────────────────────────────────────────────────────
rx_sw = 19.0
rx_tx = 19.6
leg_y = 9.2
leg_row = 0.62

ax.text(rx_sw + 1.1, leg_y, r"\textbf{Legend}",
        ha="center", va="top", fontsize=FS, color=C_BOX)

legend_items = [
    (C_PRE,  "-",  r"Pre neurons"),
    (C_POST, "-",  r"Post neurons"),
    (C_J,    "-",  r"Highlighted $j$"),
    (C_CONN, "-",  r"Connection"),
    (C_WRAP, "--", r"Wrap indicator"),
]

for i, (color, ls, label) in enumerate(legend_items):
    yi = leg_y - (i + 1) * leg_row
    ax.plot([rx_sw, rx_sw + 0.55], [yi, yi], color=color, lw=2.8, ls=ls,
            solid_capstyle="round")
    ax.text(rx_tx, yi, label, ha="left", va="center",
            fontsize=FS, color=C_BOX)

# Title
ax.set_title(r"Toroidal connectivity between two layers (pre $\rightarrow$ post)",
             fontsize=FS, pad=12, color="#1d1d1d")

# Save
os.makedirs("img", exist_ok=True)
plt.savefig("img/toroidal_connectivity_two_layers.pdf", bbox_inches="tight",
            facecolor=fig.get_facecolor())
plt.savefig("img/toroidal_connectivity_two_layers.png", dpi=300, bbox_inches="tight",
            facecolor=fig.get_facecolor())
# plt.show()