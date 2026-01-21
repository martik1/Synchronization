import numpy as np
import matplotlib.pyplot as plt
import scienceplots
plt.style.use(['science','ieee'])
from scipy.constants import c

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
        sigma = 1e-4
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
        print(f"Round {i + 1}: Node {sender.index} is broadcasting at time {tau_current}")
        
        for j, receiver in enumerate(network):
            # skip, so diagonal stays 0
            if i == j:
                send_receive_matrix[i, j] = tau_current
            T_receive = receiver.receive(sender, tau_current)
            print(f"Node {receiver.index} received packet {T_receive} sent from {sender.index} at {tau_current}")
            # row i (sender), column j (receivers)
            send_receive_matrix[i, j] = T_receive

        # increment time
        tau_current += step
    return send_receive_matrix, tau_current

def estimate_clock_rates(network, matrices):
    n = len(network)
    K = len(matrices)
    A = []
    b = []

    # iterate through consecutive rounds to get deltas
    for k in range(K - 1):
        M_k = matrices[k]
        M_kp1 = matrices[k+1]

        for i in range(n):     # sender index
            for j in range(n): # receiver index
                if i == j: continue

                # delta_rec: T_ij(k+1) - T_ij(k)
                delta_rec = M_kp1[i, j] - M_k[i, j]
                
                # delta_trans: T_i(k+1) - T_i(k) -> diagonal
                delta_trans = M_kp1[i, i] - M_k[i, i]

                # LS problem: A * rho = 0
                row = np.zeros(n)
                row[i] = delta_rec
                row[j] = -delta_trans
                A.append(row)
                b.append(0)

    constraint = np.zeros(n)
    constraint[0] = 1.0
    A.append(constraint)
    b.append(1.0)

    A = np.array(A)
    b = np.array(b)
    rho_hat, residuals, rank, s = np.linalg.lstsq(A, b, rcond=None)
    
    return rho_hat

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

    n_nodes = 10
    dim_1 = 500 # 50 by 50 square

    x_grid, y_grid = np.meshgrid(
        np.arange(0, dim_1),
        np.arange(0, dim_1)
    )

    # generate nodes, index 0 becomes reference automatically
    network = [Node(is_reference = (i == 0), index = i, x_grid=x_grid, y_grid=y_grid) for i in range(n_nodes)]

    tau = 2
    matrices = []
    K = 5
    for run in range(K):
        send_receive_matrix, tau_end = sbs_protocol(network, tau, 1)
        tau = tau_end
        matrices.append(send_receive_matrix)

    estimated_rhos = estimate_clock_rates(network, matrices)

    for i, node in enumerate(network):
        true_rate = node.rho 
        error = abs(true_rate - estimated_rhos[i])
        print(f"{i} | {true_rate:<12.6f} | {estimated_rhos[i]:<12.6f} | {error:<12.6e}")

    # plot the situation
    plot_network(network)