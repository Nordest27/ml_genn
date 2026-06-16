import os
import numpy as np
import matplotlib.pyplot as plt

# Parameters from Bellec et al.
v_th = 1.0
gamma_pd = 0.3

# Normalized membrane potential relative to threshold
x = np.linspace(-1.5 * v_th, 1.5 * v_th, 1000)

# Heaviside spike function
spike = (x >= 0).astype(float)

# Surrogate derivative
psi = (gamma_pd / v_th) * np.maximum(
    0,
    1 - np.abs(x / v_th)
)

# Figure
fig, axs = plt.subplots(1, 2, figsize=(10, 4))

# --------------------------------------------------
# Left: spike generation
# --------------------------------------------------
axs[0].plot(x, spike, linewidth=3)

axs[0].axvline(
    0,
    linestyle="--",
    alpha=0.7,
    label="Threshold"
)

axs[0].set_title("Spike Generation Function")
axs[0].set_xlabel(r"$v_j^t - A_j^t$")
axs[0].set_ylabel(r"$z_j^t$")
axs[0].set_ylim(-0.05, 1.1)
axs[0].grid(True, alpha=0.3)
axs[0].legend()

axs[0].annotate(
    "No spike",
    xy=(-0.7, 0),
    xytext=(-1.1, 0.25),
    arrowprops=dict()
)

axs[0].annotate(
    "Spike emitted",
    xy=(0.7, 1),
    xytext=(0.3, 0.6),
    arrowprops=dict()
)

# --------------------------------------------------
# Right: surrogate derivative
# --------------------------------------------------
axs[1].plot(
    x,
    psi,
    linewidth=3,
    label=r"Pseudo-derivative $\psi_j^t$"
)

axs[1].axvline(
    0,
    linestyle="--",
    alpha=0.7,
    label="Threshold"
)

axs[1].fill_between(
    x,
    psi,
    alpha=0.2
)

axs[1].set_title("Surrogate Gradient")
axs[1].set_xlabel(r"$v_j^t - A_j^t$")
axs[1].set_ylabel(r"$\psi_j^t$")
axs[1].grid(True, alpha=0.3)
axs[1].legend()

axs[1].annotate(
    "Highest learning signal\nnear threshold",
    xy=(0, gamma_pd / v_th),
    xytext=(0.35, 0.22),
    arrowprops=dict()
)

axs[1].annotate(
    "Gradient ≈ 0\nfar from threshold",
    xy=(1.2, 0),
    xytext=(0.8, 0.12),
    arrowprops=dict()
)

# --------------------------------------------------
# Save
# --------------------------------------------------
plt.tight_layout()

os.makedirs("img", exist_ok=True)

plt.savefig(
    "img/spiking_function_dynamics.pdf",
    bbox_inches="tight"
)

plt.savefig(
    "img/spiking_function_dynamics.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()

print("Saved to img/spiking_function_dynamics.pdf")
print("Saved to img/spiking_function_dynamics.png")