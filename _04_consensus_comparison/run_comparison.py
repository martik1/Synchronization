"""
Runs the linear-consensus vs. Kuramoto-consensus comparison for adversarial
phase placements straddling the 0/360 deg wraparound boundary (e.g. clusters
near 5 deg and 356 deg), and saves the report figures to ./plots/.
"""
import os
import numpy as np
import matplotlib.pyplot as plt

from consensus import (
    make_adversarial_phases, circular_mean_deg, circular_error_deg,
    run_linear_consensus, run_kuramoto_consensus, kuramoto_order_parameter,
)

# ---- palette (validated categorical set, light-surface values) ----
BLUE = "#2a78d6"     # linear consensus
ORANGE = "#eb6834"   # kuramoto consensus
AQUA = "#1baf7a"     # cluster A (~5 deg)
VIOLET = "#4a3aa7"   # cluster B (~356 deg)
RED = "#e34948"      # naive arithmetic mean (wrong target)
MUTED = "#898781"
SECONDARY = "#52514e"
GRID = "#e1e0d9"

plt.rcParams.update({
    "figure.facecolor": "#fcfcfb",
    "axes.facecolor": "#fcfcfb",
    "axes.edgecolor": MUTED,
    "axes.labelcolor": "#0b0b0b",
    "text.color": "#0b0b0b",
    "xtick.color": SECONDARY,
    "ytick.color": SECONDARY,
    "grid.color": GRID,
    "font.size": 10,
    "axes.grid": True,
    "grid.linewidth": 0.6,
    "axes.spines.top": False,
    "axes.spines.right": False,
})

HERE = os.path.dirname(os.path.abspath(__file__))
PLOTS_DIR = os.path.join(HERE, "plots")
os.makedirs(PLOTS_DIR, exist_ok=True)

# ---- main scenario: adversarial clusters near 5 deg and 356 deg ----
N_PER_CLUSTER = 10
GAP_DEG = 5.0          # cluster centers at 5 deg and 355 deg
JITTER_DEG = 2.0
SEED = 7
LIN_ITERS = 150
KUR_ITERS = 150
LIN_EPS = 0.3 / (2 * N_PER_CLUSTER)
KUR_DT = 0.2
KUR_K = 1.0

phases0 = make_adversarial_phases(N_PER_CLUSTER, GAP_DEG, seed=SEED, jitter_deg=JITTER_DEG)
n = len(phases0)
cluster = np.where(phases0 < 180.0, "A", "B")  # A ~ near 5deg, B ~ near 356deg

true_mean = circular_mean_deg(phases0)
naive_mean = phases0.mean() % 360.0

lin_hist = run_linear_consensus(phases0, iters=LIN_ITERS, eps=LIN_EPS)
kur_hist = run_kuramoto_consensus(phases0, iters=KUR_ITERS, dt=KUR_DT, K=KUR_K)

lin_final_err = np.mean(np.abs(circular_error_deg(lin_hist[-1], true_mean)))
kur_final_err = np.mean(np.abs(circular_error_deg(kur_hist[-1], true_mean)))

print("=== Adversarial scenario: clusters near 5 deg and 356 deg ===")
print(f"n nodes: {n}  (cluster A near {GAP_DEG:.0f} deg, cluster B near {360-GAP_DEG:.0f} deg)")
print(f"True circular-mean consensus target: {true_mean:.3f} deg")
print(f"Naive arithmetic mean of raw readings: {naive_mean:.3f} deg")
print(f"Linear consensus final value:   {lin_hist[-1, 0]:.3f} deg  (mean abs error {lin_final_err:.3f} deg)")
print(f"Kuramoto consensus final value: {kur_hist[-1, 0]:.3f} deg  (mean abs error {kur_final_err:.5f} deg)")

# ---------------------------------------------------------------
# Figure 1: trajectories, linear (top) vs kuramoto (bottom)
# ---------------------------------------------------------------
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(7, 6.4), sharex=True)

for ax, hist, title in [
    (ax1, lin_hist, "Linear consensus (raw, unwrapped phase values)"),
    (ax2, kur_hist, "Kuramoto consensus  (sin-coupled, circular by construction)"),
]:
    for i in range(n):
        color = AQUA if cluster[i] == "A" else VIOLET
        ax.plot(hist[:, i], color=color, lw=0.9, alpha=0.7)
    ax.axhline(true_mean, color=MUTED, ls="--", lw=1.4, label="True circular-mean target")
    ax.set_ylim(-10, 370)
    ax.set_ylabel("Phase (deg)")
    ax.set_title(title, fontsize=10, loc="left", color=SECONDARY)

ax1.axhline(naive_mean, color=RED, ls="--", lw=1.4, label="Linear consensus locks onto (naive arithmetic mean)")
ax2.set_xlabel("Iteration")

from matplotlib.lines import Line2D
cluster_handles = [
    Line2D([0], [0], color=AQUA, lw=2, label=f"Cluster A (~{GAP_DEG:.0f}°)"),
    Line2D([0], [0], color=VIOLET, lw=2, label=f"Cluster B (~{360-GAP_DEG:.0f}°)"),
]
ax1_handles, ax1_labels = ax1.get_legend_handles_labels()
ax1.legend(handles=ax1_handles + cluster_handles, loc="center right", fontsize=7)
ax2_handles, ax2_labels = ax2.get_legend_handles_labels()
ax2.legend(handles=ax2_handles + cluster_handles, loc="center right", fontsize=7)

fig.suptitle("Adversarial phase placement straddling the 0°/360° boundary", y=0.99, fontsize=12)
fig.tight_layout(rect=[0, 0, 1, 0.96])
fig.savefig(os.path.join(PLOTS_DIR, "01_trajectories.png"), dpi=200)
plt.close(fig)

# ---------------------------------------------------------------
# Figure 2: polar snapshots (initial, linear-final, kuramoto-final)
# ---------------------------------------------------------------
fig, axes = plt.subplots(1, 3, figsize=(10, 3.6), subplot_kw={"projection": "polar"})

def polar_scatter(ax, angles_deg, colors, title, target_deg=None, target_label=None):
    rad = np.deg2rad(angles_deg)
    ax.scatter(rad, np.ones_like(rad), c=colors, s=40, edgecolors="white", linewidths=0.6, zorder=3)
    if target_deg is not None:
        ax.plot([np.deg2rad(target_deg)] * 2, [0, 1], color=MUTED, ls="--", lw=1.3, zorder=2)
        ax.scatter([np.deg2rad(target_deg)], [1], marker="*", s=140, color=MUTED,
                   edgecolors="white", linewidths=0.6, zorder=4, label=target_label)
    ax.set_theta_zero_location("E")
    ax.set_theta_direction(1)
    ax.set_ylim(0, 1.15)
    ax.set_yticklabels([])
    ax.set_title(title, fontsize=10, pad=14)
    ax.grid(color=GRID, lw=0.6)

init_colors = [AQUA if c == "A" else VIOLET for c in cluster]
polar_scatter(axes[0], phases0, init_colors, "Initial (adversarial) phases",
              target_deg=true_mean, target_label="True mean")
polar_scatter(axes[1], lin_hist[-1], [BLUE] * n, "Linear consensus result",
              target_deg=true_mean, target_label="True mean")
polar_scatter(axes[2], kur_hist[-1], [ORANGE] * n, "Kuramoto consensus result",
              target_deg=true_mean, target_label="True mean")

handles = [
    Line2D([0], [0], marker="o", color="w", markerfacecolor=AQUA, markersize=8, label="Cluster A"),
    Line2D([0], [0], marker="o", color="w", markerfacecolor=VIOLET, markersize=8, label="Cluster B"),
    Line2D([0], [0], marker="*", color=MUTED, linestyle="--", markersize=11, label="True circular mean"),
]
fig.legend(handles=handles, loc="lower center", ncol=3, fontsize=8, frameon=False, bbox_to_anchor=(0.5, -0.18))
fig.suptitle("Where each protocol actually converges on the circle", fontsize=12, y=1.06)
fig.tight_layout()
fig.savefig(os.path.join(PLOTS_DIR, "02_polar_before_after.png"), dpi=200, bbox_inches="tight")
plt.close(fig)

# ---------------------------------------------------------------
# Figure 3: convergence error vs iteration (log scale)
# ---------------------------------------------------------------
lin_err_t = np.mean(np.abs(circular_error_deg(lin_hist, true_mean)), axis=1)
kur_err_t = np.mean(np.abs(circular_error_deg(kur_hist, true_mean)), axis=1)

fig, ax = plt.subplots(figsize=(6.4, 4.2))
ax.plot(lin_err_t, color=BLUE, lw=1.8, label="Linear consensus")
ax.plot(kur_err_t, color=ORANGE, lw=1.8, label="Kuramoto consensus")
ax.set_yscale("log")
ax.set_xlabel("Iteration")
ax.set_ylabel("Mean absolute error vs. true circular mean (deg)")
ax.set_title("Convergence toward the true consensus value", loc="left", fontsize=11)
ax.legend(fontsize=9)
fig.tight_layout()
fig.savefig(os.path.join(PLOTS_DIR, "03_error_convergence.png"), dpi=200)
plt.close(fig)

# ---------------------------------------------------------------
# Figure 4: Kuramoto order parameter r(t) (coherence -> 1 = phase sync)
# ---------------------------------------------------------------
r_t = kuramoto_order_parameter(kur_hist)
zoom = 60
fig, ax = plt.subplots(figsize=(6.4, 3.6))
ax.plot(r_t[:zoom], color=ORANGE, lw=1.8)
ax.set_ylim(r_t[:zoom].min() - 0.002, 1.001)
ax.set_xlabel("Iteration")
ax.set_ylabel(r"Kuramoto order parameter $r(t)$")
ax.set_title("Kuramoto consensus: phase coherence over time  (r=1 is full sync)",
             loc="left", fontsize=11)
fig.tight_layout()
fig.savefig(os.path.join(PLOTS_DIR, "04_kuramoto_order_parameter.png"), dpi=200)
plt.close(fig)

# ---------------------------------------------------------------
# Figure 5: adversarial sweep across the symmetric cluster half-angle
# ---------------------------------------------------------------
gaps = np.concatenate([np.arange(1, 80, 2), np.arange(80, 90, 1)])
lin_errs, kur_errs = [], []
for gap in gaps:
    p0 = make_adversarial_phases(N_PER_CLUSTER, gap, seed=SEED, jitter_deg=JITTER_DEG)
    tmean = circular_mean_deg(p0)
    lh = run_linear_consensus(p0, iters=LIN_ITERS, eps=LIN_EPS)
    kh = run_kuramoto_consensus(p0, iters=KUR_ITERS, dt=KUR_DT, K=KUR_K)
    lin_errs.append(np.mean(np.abs(circular_error_deg(lh[-1], tmean))))
    kur_errs.append(np.mean(np.abs(circular_error_deg(kh[-1], tmean))))

fig, ax = plt.subplots(figsize=(6.8, 4.4))
ax.plot(gaps, lin_errs, color=BLUE, lw=1.8, marker="o", ms=3.5, label="Linear consensus")
ax.plot(gaps, kur_errs, color=ORANGE, lw=1.8, marker="o", ms=3.5, label="Kuramoto consensus")
ax.set_yscale("log")
ax.set_xlabel(r"Cluster half-angle from 0$^\circ$ (deg)  —  clusters at $\pm$this value")
ax.set_ylabel("Final mean absolute error (deg, log scale)")
ax.set_title("Symmetric clusters straddling the wrap boundary, at every separation", loc="left", fontsize=10.5)
ax.legend(fontsize=9)
ax.annotate("approaching the antipodal\n(splay) equilibrium",
            xy=(87, kur_errs[-1]), xytext=(55, 3),
            fontsize=8, color=SECONDARY,
            arrowprops=dict(arrowstyle="->", color=SECONDARY, lw=0.8))
fig.tight_layout()
fig.savefig(os.path.join(PLOTS_DIR, "05_adversarial_sweep.png"), dpi=200)
plt.close(fig)

print(f"\nSaved 5 figures to {PLOTS_DIR}")
