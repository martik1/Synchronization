import numpy as np
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