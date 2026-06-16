import numpy as np
import matplotlib.pyplot as plt

gamma = 0.5
lam = 0.8

delay = 10
T = 25

t = np.arange(T)

standard = (gamma * lam) ** t
decoupled = lam ** t

fig, ax = plt.subplots(figsize=(10, 6))

# traces
ax.plot(
    t,
    standard,
    lw=3,
    label=fr"Coupled TD($\lambda$): decay = $\gamma\lambda={gamma*lam:.1f}$"
)

ax.plot(
    t,
    decoupled,
    lw=3,
    label=fr"Decoupled: decay = $\lambda={lam:.1f}$"
)

# TD error arrival
ax.axvline(
    delay,
    ls="--",
    lw=2,
    alpha=0.8,
    label=fr"TD error arrives ($\delta_t$)"
)

# remaining credit at arrival
ax.scatter(delay, standard[delay], s=100, zorder=5)
ax.scatter(delay, decoupled[delay], s=100, zorder=5)

ax.annotate(
    fr"${standard[delay]:.4f}$",
    (delay, standard[delay]),
    xytext=(delay + 1, standard[delay] * 2)
)

ax.annotate(
    fr"${decoupled[delay]:.3f}$",
    (delay, decoupled[delay]),
    xytext=(delay + 1, decoupled[delay] * 1.3)
)

# shaded area = usable credit
ax.fill_between(
    t[:delay + 1],
    standard[:delay + 1],
    alpha=0.15
)

ax.fill_between(
    t[:delay + 1],
    decoupled[:delay + 1],
    alpha=0.15
)

ratio = decoupled[delay] / standard[delay]

ax.text(
    0.55,
    0.8,
    f"Credit available at reward arrival:\n"
    f"Coupled: {standard[delay]:.4f}\n"
    f"Decoupled: {decoupled[delay]:.4f}\n\n"
    f"Ratio ≈ {ratio:.0f}×",
    transform=ax.transAxes,
    bbox=dict()
)

ax.set_yscale("log")

ax.set_xlabel("Steps since action")
ax.set_ylabel("Credit trace magnitude")
ax.set_title(
    "Long-delay credit assignment with coupled and decoupled traces"
)

ax.grid(True, which="both", alpha=0.3)
ax.legend()
plt.tight_layout()
plt.savefig(
    "img/credit_trace_comparison.pdf",
    bbox_inches="tight"
)