import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Circle

# ── Matplotlib / LaTeX style ────────────────────────────────────────────────
plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "text.latex.preamble": r"\usepackage{amsmath}",
})

# ── Input-encoding parameters ───────────────────────────────────────────────
VISIBLE_RANGE = 5      # 5x5 local viewport centered on Pac-Man
NUM_CHANNELS  = 3      # RGB observation channels
OBS_SCALE     = 2      # input resolution multiplier (2 -> Pac-Man 10x10, 4 -> snake 20x20)
P_MAX         = 0.5    # maximum firing probability (50%)
K             = 30     # Bernoulli trials per neuron per step (WAIT_INC)
GRID          = VISIBLE_RANGE * OBS_SCALE

# ── Colors ──────────────────────────────────────────────────────────────────
C_BOX   = "#333333"
CH_COL  = ["#e63946", "#2a9d8f", "#457b9d"]      # R, G, B channel accents
CH_NAME = ["Red channel", "Green channel", "Blue channel"]

# entity -> rgb (matches get_local_observation in the agent code)
WALL    = (0.3, 0.3, 0.3)
FOOD    = (0.0, 0.5, 0.0)
CAPSULE = (0.0, 0.5, 0.5)
PACMAN  = (1.0, 1.0, 0.0)
GHOST   = (1.0, 0.0, 0.0)
SCARED  = (0.0, 0.0, 1.0)
EMPTY   = (0.0, 0.0, 0.0)

# ── A representative 5x5 local observation (Pac-Man centered at (2,2)) ───────
scene = [
    [WALL,  EMPTY,  FOOD,    FOOD,   WALL   ],
    [FOOD,  EMPTY, GHOST,   FOOD,   CAPSULE],
    [FOOD,  EMPTY,  PACMAN,  FOOD,   FOOD   ],
    [WALL,  FOOD,  FOOD,    EMPTY,  FOOD   ],
    [WALL,  WALL,  FOOD,    SCARED, FOOD   ],
]
obs5 = np.array(scene, dtype=np.float32)                 # (5,5,3)

# Upscale with nearest-neighbour: each cell -> OBS_SCALE x OBS_SCALE block
obs = np.repeat(np.repeat(obs5, OBS_SCALE, axis=0), OBS_SCALE, axis=1)

# Firing probability per input neuron: p = intensity * P_MAX
fire_prob = obs * P_MAX                                   # in [0, P_MAX]

# ── Figure ──────────────────────────────────────────────────────────────────
fig = plt.figure(figsize=(20, 6.4))
gs = fig.add_gridspec(1, 4, width_ratios=[1, 1, 1, 1], wspace=0.16)

# Panel 0: semantic viewport (RGB)
ax0 = fig.add_subplot(gs[0, 0])
ax0.set_aspect("equal"); ax0.axis("off")
ax0.set_xlim(-0.5, VISIBLE_RANGE - 0.5)
ax0.set_ylim(-0.5, VISIBLE_RANGE - 0.5)
ax0.invert_yaxis()
for r in range(VISIBLE_RANGE):
    for c in range(VISIBLE_RANGE):
        ax0.add_patch(Rectangle((c - 0.5, r - 0.5), 1, 1,
                                facecolor=tuple(obs5[r, c]),
                                edgecolor="#dddddd", lw=0.8, zorder=1))
ax0.set_title(r"Local view  ($5\times5\times3$)", fontsize=20, color=C_BOX, pad=12)

# Panels 1-3: per-channel Poisson input grids (circles)
def draw_channel(ax, ch):
    ax.set_aspect("equal"); ax.axis("off")
    ax.set_xlim(-0.7, GRID - 0.3)
    ax.set_ylim(-0.7, GRID - 0.3)
    ax.invert_yaxis()
    col = CH_COL[ch]
    # block boundaries (one viewport cell = OBS_SCALE x OBS_SCALE neurons)
    for br in range(VISIBLE_RANGE):
        for bc in range(VISIBLE_RANGE):
            ax.add_patch(Rectangle((bc * OBS_SCALE - 0.5, br * OBS_SCALE - 0.5),
                                   OBS_SCALE, OBS_SCALE, fill=False,
                                   ec="#cfcfcf", lw=1.0, zorder=3))
    for r in range(GRID):
        for c in range(GRID):
            p = float(fire_prob[r, c, ch])     # 0 .. P_MAX
            frac = p / P_MAX                    # 0 .. 1
            ax.add_patch(Circle((c, r), 0.46, fc="#f4f4f4",
                                ec="#e8e8e8", lw=0.5, zorder=1))
            if frac > 0:
                ax.add_patch(Circle((c, r), 0.14 + 0.32 * frac, fc=col,
                                    ec="none", alpha=0.3 + 0.7 * frac, zorder=2))
    ax.set_title(CH_NAME[ch], fontsize=20, color=col, pad=12)

for i in range(NUM_CHANNELS):
    draw_channel(fig.add_subplot(gs[0, i + 1]), i)

# Title + caption
fig.suptitle(
    r"Poisson rate-coded input: each $5\times5$ viewport cell expands to a "
    r"$" + f"{OBS_SCALE}" + r"\times" + f"{OBS_SCALE}" + r"$ block of neurons per channel",
    fontsize=20, color="#1d1d1d", y=1.03)


os.makedirs("img", exist_ok=True)
fig.savefig("img/poisson_input_encoding.pdf", bbox_inches="tight", facecolor="white")
fig.savefig("img/poisson_input_encoding.png", dpi=300, bbox_inches="tight", facecolor="white")
# plt.show()