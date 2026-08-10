"""
Quantifies the Cramer-Rao lower bound (CRLB) for estimating a common phase
from n noisy per-node measurements, and checks where linear consensus and
circular (Kuramoto) consensus stand relative to it.

Noise model: each node's reading is one draw from a von Mises(mu, kappa)
distribution -- the circular analogue of N(mu, sigma^2), correctly defined
on the circle. Its Fisher information gives a closed-form CRLB (see
consensus.crlb_deg2). The circular mean is the von Mises MLE for mu, which
is also the value Kuramoto consensus's sin-coupled dynamics settles onto
(verified empirically in run_comparison.py to <0.001 deg for the tight-
cluster case); the plain arithmetic mean is what linear consensus computes.
"""
import os
import numpy as np
import matplotlib.pyplot as plt
import scienceplots  # noqa: F401

from consensus import crlb_deg2, monte_carlo_trial_errors

plt.style.use(["science", "no-latex"])

BLUE = "#2a78d6"     # linear consensus
ORANGE = "#eb6834"   # circular / kuramoto consensus
MUTED = "#5a5a5a"    # CRLB reference

plt.rcParams.update({
    "figure.facecolor": "#fcfcfb",
    "axes.facecolor": "#fcfcfb",
    "font.size": 10,
    "axes.grid": True,
    "grid.color": "#e1e0d9",
    "grid.linewidth": 0.6,
})

HERE = os.path.dirname(os.path.abspath(__file__))
PLOTS_DIR = os.path.join(HERE, "plots")
os.makedirs(PLOTS_DIR, exist_ok=True)

rng = np.random.default_rng(42)
KAPPA = 15.0          # ~14.8 deg effective noise std per reading
TRIALS = 20000

# ---------------------------------------------------------------
# Figure 6: RMSE vs. number of nodes n, benign vs. adversarial mu
# ---------------------------------------------------------------
n_list = np.array([2, 4, 8, 16, 32, 64, 128, 256])

results = {}
for mu in (180.0, 0.0):
    lin_rmse, circ_rmse = [], []
    for n in n_list:
        lin_err, circ_err = monte_carlo_trial_errors(mu, KAPPA, int(n), TRIALS, rng)
        lin_rmse.append(np.sqrt(np.mean(lin_err ** 2)))
        circ_rmse.append(np.sqrt(np.mean(circ_err ** 2)))
    results[mu] = (np.array(lin_rmse), np.array(circ_rmse))

crlb_rmse = np.sqrt(crlb_deg2(n_list, KAPPA))

fig, axes = plt.subplots(1, 2, figsize=(8.5, 3.8), sharey=True)
titles = {180.0: r"Benign: $\mu=180^\circ$ (far from wrap boundary)",
          0.0: r"Adversarial: $\mu=0^\circ$ (astride wrap boundary)"}
for ax, mu in zip(axes, (180.0, 0.0)):
    lin_rmse, circ_rmse = results[mu]
    ax.plot(n_list, lin_rmse, "o-", color=BLUE, lw=1.6, ms=4, label="Linear consensus", zorder=2)
    ax.plot(n_list, circ_rmse, "s-", color=ORANGE, lw=2.4, ms=4, label="Kuramoto / circular mean", zorder=2)
    ax.plot(n_list, crlb_rmse, "--", color=MUTED, lw=1.6, label=r"CRLB (theoretical)", zorder=3)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Number of nodes $n$")
    ax.set_title(titles[mu], fontsize=9.5)
axes[0].set_ylabel("RMSE (deg, log scale)")
axes[1].legend(fontsize=7.5, loc="center right")
fig.suptitle(r"Estimator efficiency vs. Cramer-Rao bound ($\kappa=%d$ von Mises noise)" % KAPPA, fontsize=11)
fig.tight_layout(rect=[0, 0, 1, 0.94])
fig.savefig(os.path.join(PLOTS_DIR, "06_crlb_vs_n.png"), dpi=200)
plt.close(fig)

# ---------------------------------------------------------------
# Figure 7: bias and RMSE vs. true mean location mu (fixed n)
# ---------------------------------------------------------------
N_FIXED = 20
mu_list = np.linspace(0.0, 180.0, 37)
lin_bias, circ_bias, lin_rmse2, circ_rmse2 = [], [], [], []
for mu in mu_list:
    lin_err, circ_err = monte_carlo_trial_errors(mu, KAPPA, N_FIXED, TRIALS, rng)
    lin_bias.append(lin_err.mean())
    circ_bias.append(circ_err.mean())
    lin_rmse2.append(np.sqrt(np.mean(lin_err ** 2)))
    circ_rmse2.append(np.sqrt(np.mean(circ_err ** 2)))

crlb_rmse_fixed = np.sqrt(crlb_deg2(N_FIXED, KAPPA))

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(6.8, 6.2), sharex=True)

ax1.axhline(0, color=MUTED, lw=1.0)
ax1.plot(mu_list, lin_bias, color=BLUE, lw=1.8, label="Linear consensus")
ax1.plot(mu_list, circ_bias, color=ORANGE, lw=1.8, label="Kuramoto / circular mean")
ax1.set_ylabel("Bias (deg)")
ax1.set_title(r"Bias vs. true phase location $\mu$  ($n=%d$, $\kappa=%d$)" % (N_FIXED, KAPPA),
              fontsize=10.5, loc="left")
ax1.legend(fontsize=8)

ax2.plot(mu_list, lin_rmse2, color=BLUE, lw=1.8, label="Linear consensus", zorder=2)
ax2.plot(mu_list, circ_rmse2, color=ORANGE, lw=2.6, label="Kuramoto / circular mean", zorder=2)
ax2.axhline(crlb_rmse_fixed, color=MUTED, ls="--", lw=1.6, label="CRLB (theoretical)", zorder=3)
ax2.set_yscale("log")
ax2.set_xlabel(r"True phase $\mu$ (deg)")
ax2.set_ylabel("RMSE (deg, log scale)")
ax2.set_title("RMSE vs. CRLB -- the failure is bias-driven, not variance-driven",
              fontsize=10.5, loc="left")
ax2.legend(fontsize=8)

fig.tight_layout()
fig.savefig(os.path.join(PLOTS_DIR, "07_bias_and_crlb_vs_mu.png"), dpi=200)
plt.close(fig)

# ---------------------------------------------------------------
# Console summary
# ---------------------------------------------------------------
print("=== CRLB quantification (von Mises noise, kappa=%.1f) ===" % KAPPA)
print(f"CRLB RMSE at n={N_FIXED}: {crlb_rmse_fixed:.3f} deg")
print()
print(f"{'n':>5} | {'CRLB RMSE':>10} | {'linear RMSE (mu=180)':>21} | {'circ RMSE (mu=180)':>19} "
      f"| {'linear RMSE (mu=0)':>19} | {'circ RMSE (mu=0)':>17}")
for i, n in enumerate(n_list):
    print(f"{n:5d} | {crlb_rmse[i]:10.3f} | {results[180.0][0][i]:21.3f} | {results[180.0][1][i]:19.3f} "
          f"| {results[0.0][0][i]:19.3f} | {results[0.0][1][i]:17.3f}")

print(f"\nSaved figures 06 and 07 to {PLOTS_DIR}")
