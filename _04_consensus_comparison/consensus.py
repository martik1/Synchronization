"""
Linear (arithmetic) consensus vs. Kuramoto-style (phase) consensus.

Phases are quantities that live on a circle (mod 360 deg). Standard linear
consensus, x_i(t+1) = x_i(t) + eps * sum_j a_ij (x_j(t) - x_i(t)), assumes
a flat (non-wrapping) value space, so it has no notion that 356 deg and
5 deg are only 9 deg apart. Kuramoto consensus instead couples nodes through
sin(theta_j - theta_i), which is exactly 360-deg periodic, so it "sees" the
circle correctly. Adversarial placements that straddle the 0/360 deg
wraparound boundary are the case that separates the two.
"""
import numpy as np


def circular_mean_deg(angles_deg):
    """True consensus value for phases: the circular mean, in [0, 360)."""
    angles_rad = np.deg2rad(angles_deg)
    mean_rad = np.arctan2(np.mean(np.sin(angles_rad)), np.mean(np.cos(angles_rad)))
    return np.rad2deg(mean_rad) % 360.0


def circular_error_deg(angles_deg, target_deg):
    """Smallest-signed-angle distance of each node's phase to a target, in deg."""
    diff = (angles_deg - target_deg + 180.0) % 360.0 - 180.0
    return diff


def complete_graph_laplacian(n):
    A = np.ones((n, n)) - np.eye(n)
    D = np.diag(A.sum(axis=1))
    return D - A, A


def run_linear_consensus(x0_deg, iters=200, eps=None):
    """
    Discrete-time linear consensus on raw (unwrapped) phase values over a
    complete graph. x(t+1) = (I - eps*L) x(t). No knowledge of the 360 deg
    wraparound is used anywhere -- this is the naive, textbook algorithm.
    """
    n = len(x0_deg)
    L, _ = complete_graph_laplacian(n)
    if eps is None:
        eps = 1.0 / n  # keeps (I - eps*L) stable for a complete graph

    W = np.eye(n) - eps * L
    history = np.zeros((iters + 1, n))
    history[0] = x0_deg
    x = x0_deg.copy()
    for t in range(iters):
        x = W @ x
        history[t + 1] = x
    return history


def run_kuramoto_consensus(theta0_deg, iters=200, dt=0.5, K=1.0, omega=None):
    """
    Discrete-time (Euler) simulation of a leaderless Kuramoto phase model
    over a complete graph:
        dtheta_i/dt = omega_i + (K/n) * sum_j sin(theta_j - theta_i)
    With omega_i = 0 this is a pure phase-consensus / synchronization
    protocol, the circular analogue of linear consensus.
    """
    n = len(theta0_deg)
    theta = np.deg2rad(theta0_deg.copy())
    if omega is None:
        omega = np.zeros(n)

    history = np.zeros((iters + 1, n))
    history[0] = np.rad2deg(theta) % 360.0
    for t in range(iters):
        # sum_j sin(theta_j - theta_i) for all i, vectorized
        diff = theta[None, :] - theta[:, None]
        coupling = (K / n) * np.sin(diff).sum(axis=1)
        theta = theta + dt * (omega + coupling)
        history[t + 1] = np.rad2deg(theta) % 360.0
    return history


def kuramoto_order_parameter(history_deg):
    """r(t) = |mean_i exp(i theta_i(t))|, in [0, 1]. 1 = perfect phase sync."""
    rad = np.deg2rad(history_deg)
    z = np.mean(np.exp(1j * rad), axis=1)
    return np.abs(z)


def linear_order_parameter(history_deg):
    """
    Analogous coherence measure for the linear-consensus trajectory: 1 minus
    the (normalized) spread of raw values across nodes. Included only so the
    two protocols can be compared on a similar 0-1 coherence scale; unlike
    the Kuramoto order parameter it has no circular meaning.
    """
    spread = history_deg.std(axis=1)
    return 1.0 / (1.0 + spread)


def make_adversarial_phases(n_per_cluster, gap_deg, seed=0, jitter_deg=2.0):
    """
    Two tight clusters straddling the 0/360 deg boundary: one at
    `gap_deg` and one at `360 - gap_deg`. Small `gap_deg` (e.g. 5 deg,
    so clusters sit at 5 deg and 356/355 deg-ish) is the adversarial case:
    the clusters are close on the circle but far apart as raw numbers.
    """
    rng = np.random.default_rng(seed)
    c1 = gap_deg + rng.normal(0, jitter_deg, n_per_cluster)
    c2 = (360.0 - gap_deg) + rng.normal(0, jitter_deg, n_per_cluster)
    phases = np.concatenate([c1, c2]) % 360.0
    rng.shuffle(phases)
    return phases
