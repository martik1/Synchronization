"""
imports
"""
import numpy as np
from scipy.constants import c
"""
custom imports
"""
from synchronization.utility import Node, estimate_clock_offsets, estimate_clock_rates
from synchronization.protocols import sbs_protocol
from synchronization.plotting import plot_network

if __name__ == "__main__":
    np.random.seed(333)

    n_nodes = 15
    dim_1 = 500 # square
    tau_start = 2
    K = 500

    x_grid, y_grid = np.meshgrid(
        np.arange(0, dim_1),
        np.arange(0, dim_1)
    )

    # generate ntwork of nodes, index 0 becomes reference automatically
    network = [Node(is_reference = (i == 0), index = i, x_grid=x_grid, y_grid=y_grid) for i in range(n_nodes)]

    # collect data
    tau = tau_start
    matrices = []
    for _ in range(K):
        send_receive_matrix, tau_end = sbs_protocol(network, tau, 1)
        matrices.append(send_receive_matrix)
        tau = tau_end

    rho_hat = estimate_clock_rates(network, matrices)
    theta_hat, d_hat_sec = estimate_clock_offsets(network, matrices, rho_hat)

    d_hat_meters = d_hat_sec * c

    print("\n--- CLOCK RATES (RHO) ---")
    for i, node in enumerate(network):
        print(f"Node {i}: True={node.rho:.6f}, Est={rho_hat[i]:.6f}")

    print("\n--- CLOCK OFFSETS (THETA) ---")
    for i, node in enumerate(network):
        print(f"Node {i}: True={node.theta:.6f}, Est={theta_hat[i]:.6f}")

    print("\n--- DISTANCES (METERS) ---")
    for i in range(n_nodes):
        for j in range(i + 1, n_nodes):
            true_dist = np.linalg.norm(network[i].p - network[j].p)
            est_dist = d_hat_meters[i, j]
            print(f"Nodes {i}-{j}: True={true_dist:.2f}m, Est={est_dist:.2f}m")

    # plot the situation
    plot_network(network)