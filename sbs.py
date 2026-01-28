import numpy as np
import matplotlib.pyplot as plt
import scienceplots
plt.style.use(['science','ieee'])
from scipy.constants import c
import cvxpy as cp

class Node:
    def __init__(self, is_reference=False, index = 0,
                 theta=None, rho=None,
                 x_grid = np.ndarray, y_grid = np.ndarray):
        """
        Initializes a node instance.
        
        Args:
            is_reference (bool): If True, clock becomes ref. (theta=0, rho=0).
            index (int): Each node gets its unique indentifier.
            theta (float): offset. If None, randomized.
            rho (float): skew. If None, randomized.
            x_grid (np.ndarray): x dimension of grid
            y_grid (np.ndarray): y dimension of grid
        """
        self.is_reference = is_reference
        self.index = index
        self.x = np.random.uniform(x_grid.min(), x_grid.max())
        self.y = np.random.uniform(y_grid.min(), y_grid.max())
        self.p = np.array([self.x, self.y])
        
        if is_reference:
            self.theta = 0.0
            self.rho = 1
        else:
            # assign provided values or generate random ones
            self.theta = theta if theta is not None else np.random.uniform(0, 0.5)
            self.rho = rho if rho is not None else 1 + np.random.uniform(-1e-4, 1e-4)

    def transmit(self, tau):
        """
        Returns the reported clock time c(tau) based on tau.
        Formula: c(tau) = theta + rho * tau
        """
        if self.is_reference:
            return tau
        
        T_transmit = self.rho * tau + self.theta

        return T_transmit
    
    def receive(self, sender_node, tau_i):
        """
        Returns the receive timestamp or clock reading
        at node j(this one) for a message from node i at
        round k:
        T^k_ij = rho_j(tau^k_i + d_ij) + theta_j + eps^k_j
        """
        # hardcoded sanity check
        if self.index == sender_node.index:
            raise ValueError(f"Node {self.index} can not receive from itself")

        sigma = 1e-6
        p_i = sender_node.p
        d_ij = np.linalg.norm(p_i - self.p) / c # normed dist.
        epsilon = np.random.normal(0, sigma) # takes std. dev. not variance!

        T_receive = self.rho * (tau_i + d_ij) + self.theta + epsilon

        return T_receive

    def __repr__(self):
        return (f"Clock {self.index} (Ref={self.is_reference}"
                f"theta={self.theta:.4f}, rho={self.rho:.6f})")

def sbs_protocol(network, tau_start, step):
    size = len(network)
    send_receive_matrix = np.zeros((size, size))
    tau_current = tau_start

    for i, sender in enumerate(network):
        tau_i_local = sender.transmit(tau_current)       
        for j, receiver in enumerate(network):
            # skip, so diagonal stays as local time
            if i == j:
                send_receive_matrix[i, j] = tau_i_local
            else:
                # row i (sender), column j (receivers) -> T_receive
                send_receive_matrix[i, j] = receiver.receive(sender, tau_current)

        tau_current += step
    return send_receive_matrix, tau_current

def estimate_clock_rates(network, matrices):
    n = len(network)
    K = len(matrices)
    
    # pre-built numpy matrix for speed
    rows = []
    for k in range(K - 1):
        Mk, Mkp1 = matrices[k], matrices[k+1]
        for i in range(n):
            for j in range(n):
                if i == j: continue
                d_rec = Mkp1[i, j] - Mk[i, j]
                
                row = np.zeros(n)
                row[i] = d_rec
                row[j] = -(Mkp1[i, i] - Mk[i, i])
                rows.append(row)

    A_mat = np.array(rows)
    rho = cp.Variable(n)
    
    # obj: minimize residuals
    # constr: force rho[0] to be 1
    prob = cp.Problem(cp.Minimize(cp.sum_squares(A_mat @ rho)), 
                     [rho[0] == 1.0])
    
    prob.solve() 
    return rho.value

def estimate_clock_offsets(network, matrices, rho_hat):
    n = len(network)
    K = len(matrices)

    theta = cp.Variable(n)
    n_distances = n * (n - 1) // 2
    d_vec = cp.Variable(n_distances)

    # map (i, j) pairs to indices in d_vec
    pair_to_idx = {}
    idx = 0
    for i in range(n):
        for j in range(i + 1, n):
            pair_to_idx[tuple(sorted((i, j)))] = idx
            idx += 1

    A_rows = []
    h_vals = []

    for k in range(K):
        T = matrices[k]
        for i in range(n):
            for j in range(n):
                if i == j: continue
                
                rho_i, rho_j = rho_hat[i], rho_hat[j]
                rho_ji = rho_j / rho_i
                
                # h_ij_k = T_observed_at_j - (rho_j/rho_i)*T_sender_local
                h_ij_k = T[i, j] - rho_ji * T[i, i]
                h_vals.append(h_ij_k)
                
                # rho_j * d_ij + theta_j - rho_ji * theta_i = h_ij_k
                row_theta = np.zeros(n)
                row_theta[j] = 1
                row_theta[i] = -rho_ji
                
                row_d = np.zeros(n_distances)
                row_d[pair_to_idx[tuple(sorted((i, j)))]] = rho_j
                
                A_rows.append(cp.hstack([row_theta, row_d]))

    A_mat = cp.vstack(A_rows)
    h_vec = np.array(h_vals)
    
    objective = cp.Minimize(cp.sum_squares(A_mat @ cp.hstack([theta, d_vec]) - h_vec))
    
    constraints = [theta[0] == 0, d_vec >= 0]
    prob = cp.Problem(objective, constraints)
    prob.solve(solver = 'SCS', verbose = True) # ECOS ?

    # reconst. symmetric d matrix for print loop
    d_matrix = np.zeros((n, n))
    for (i, j), idx in pair_to_idx.items():
        d_matrix[i, j] = d_matrix[j, i] = d_vec.value[idx]

    return theta.value, d_matrix

def plot_network(network):
    plt.figure(figsize = (2,2))
    for node in network:
        if node.index == 0:
            plt.scatter(node.x, node.y,
                    c = "red", edgecolors = "black",
                    s = 6, linewidths=0.5)
        else:
            plt.scatter(node.x, node.y,
                    c = "blue", edgecolors = "black",
                    s = 4, linewidths=0.5)
        plt.text(node.x + 2, node.y + 2,
                 f"Node {node.index}", fontsize = 4)
    plt.xlabel("$x$")
    plt.ylabel("$y$")
    plt.grid(True)
    plt.show()
    
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