import numpy as np
from scipy.constants import c

class GlobalTime:
    def __init__(self):
        self.t = 0.0

class Node:
    def __init__(self, time_source, 
                 is_reference=False, is_stationary=True, 
                 index = 0,
                 f_mo = 10e6, 
                 sigma_alpha = 100e-6, sigma_phi = 10e-9,
                 f_c = 10e9, f_s = 1e6,
                 x_grid = np.ndarray, y_grid = np.ndarray):
        """
        Initializes a node instance as implemented in
        'WIRELESS DISTRIBUTED FREQUENCY AND PHASE SYNCHRONIZATION'.
        
        Args:
            is_reference (bool): If True, clock becomes ref. (alpha = 1).
            index (int): Each node gets its unique indentifier.
            x_grid (np.ndarray): x dimension of grid.
            y_grid (np.ndarray): y dimension of grid.
        """
        # general information
        self.is_reference = is_reference
        self.is_stationary = is_stationary
        self.index = index
        self.time_source = time_source
        
        # geometry
        self.p = np.array([
            np.random.uniform(x_grid.min(), x_grid.max()), 
            np.random.uniform(y_grid.min(), y_grid.max())
        ])

        if self.is_stationary:
            self.v_vec = np.array([0.0, 0.0])
        else:
            speed = np.random.uniform(1.0, 27.8)
            angle = np.random.uniform(0, 2 * np.pi)
            self.v_vec = np.array([speed * np.cos(angle), speed * np.sin(angle)])
        
        # setup
        if self.is_reference:
            self.alpha = 1
        else:
            self.alpha = np.random.normal(loc = 1.0, scale = sigma_alpha) # generate random drifts

        self.f_mo = f_mo # nominal f of main oscillator is identical
        self.f_mo_self = self.alpha * self.f_mo

        self.phi = np.random.normal(loc = 0, scale = sigma_phi) # random bias

        self.f_c = f_c # nominal carrier frequency
        self.f_c_self = f_c * self.alpha # true carrier frequency
        self.f_s_self = f_s * self.alpha # true sampling frequency

        # Phase offsets
        self.gamma_tx = np.random.uniform(0, 2 * np.pi)
        self.gamma_rx = np.random.uniform(0, 2 * np.pi)

    def tau_at(self, global_t):
        """Returns the local time tau for a given global time t."""
        return self.alpha * global_t + self.phi
    
    def _get_distance(self, node_j):
        """Returns distance to any other node."""
        p_j = node_j.p
        R_ij = np.linalg.norm(self.p - p_j)
        return R_ij
    
    def _get_rel_radial_v(self, node_j):
        """
        Calculates the relative velocity projected onto the LOS.
        Positive value implies nodes are moving away from each other.
        """
        # LOS vector from i (self) to j (sender)
        los_vec = node_j.p - self.p
        dist = np.linalg.norm(los_vec)
        
        if dist == 0: return 0.0
        
        unit_los = los_vec / dist
        
        # relative velocity vector (Source - Observer)
        rel_v_vec = node_j.v_vec - self.v_vec
        
        # scalar projection onto LOS
        v_radial = np.dot(rel_v_vec, unit_los)
        return v_radial

    def _get_doppler_shift(self, node_j):
        """
        Calculates fd = -2 * v_radial * fc / c
        Factor of 2 is added since we are considering a radar
        scenario.
        """
        v_r = self._get_rel_radial_v(node_j)
        # Using the nominal carrier frequency fc
        f_d = -2 * v_r * self.f_c / c
        return f_d 

    def transmit(self, global_t, T_p):
        """
        s_j^alpha = exp(j 2 pi f^alpha_j tau_j) * rect((tau_j - T_p/2)/T_p)
        Note: The rect function is 1 when 0 <= tau_j <= T_p.
        """
        tau_j = self.tau_at(global_t)
        
        # Check if the node's local clock is within the pulse duration
        if 0 <= tau_j <= T_p:
            # We use the drifted carrier frequency f_c_self
            # and include the hardware phase bias gamma_tx
            phase = 2 * np.pi * self.f_c_self * tau_j + self.gamma_tx
            return np.exp(1j * phase)
        
        return 0.0j

    def reset_buffer(self):
        self.rx_buffer = []

    def receive(self, sender, T_p):
        """
        Processes the incoming single-tone signal.
        """
        # 1. Physics: Signal propagation
        dist = self._get_distance(sender)
        tof = dist / c
        t_arrival = self.time_source.t
        t_emit = t_arrival - tof
        
        # 2. Capture the transmitted signal at the moment of emission
        w = sender.transmit(t_emit, T_p)
        
        if w == 0: return 0.0j
        
        # 3. Physics: Doppler shift in the channel
        f_d = self._get_doppler_shift(sender)
        channel_gain = np.exp(1j * 2 * np.pi * f_d * t_arrival)
        
        # 4. Hardware: Downconversion using the receiver's local clock
        # We mix with the conjugate of the local oscillator
        local_tau = self.tau_at(t_arrival)
        mixing_signal = np.exp(-1j * (2 * np.pi * self.f_c_self * local_tau + self.gamma_rx))
        
        r = w * channel_gain * mixing_signal 

        self.rx_buffer.append(r)

        return r

    def _est_sync_params(self, f_s):
        """
        Node looks at its buffer and find the freq
        and phase offset without knowledge of alpha.
        """
        samples = np.array(self.rx_buffer)
        N = len(samples)

        # Debugging print
        if N == 0:
            print(f"DEBUG: Node {self.index} buffer is empty! Check your loop.")
            return 0, 0, None, None # Prevent crash

        R_f = np.fft.fftshift(np.fft.fft(samples)) / f_s
        freqs = np.fft.fftshift(np.fft.fftfreq(N, d = 1/f_s))

        idx_max = np.argmax(np.abs(R_f))
        f_hat = freqs[idx_max]
        gamma_hat = np.angle(R_f[idx_max])

        return f_hat, gamma_hat, R_f, freqs

    import numpy as np
from scipy.constants import c

class GlobalTime:
    def __init__(self):
        self.t = 0.0

class Node:
    def __init__(self, time_source, 
                 is_reference=False, is_stationary=True, 
                 index = 0,
                 f_mo = 10e6, 
                 sigma_alpha = 100e-6, sigma_phi = 10e-9,
                 f_c = 10e9, f_s = 1e6,
                 x_grid = np.ndarray, y_grid = np.ndarray):
        """
        Initializes a node instance as implemented in
        'WIRELESS DISTRIBUTED FREQUENCY AND PHASE SYNCHRONIZATION'.
        
        Args:
            is_reference (bool): If True, clock becomes ref. (alpha = 1).
            index (int): Each node gets its unique indentifier.
            x_grid (np.ndarray): x dimension of grid.
            y_grid (np.ndarray): y dimension of grid.
        """
        # general information
        self.is_reference = is_reference
        self.is_stationary = is_stationary
        self.index = index
        self.time_source = time_source
        
        # geometry
        self.p = np.array([
            np.random.uniform(x_grid.min(), x_grid.max()), 
            np.random.uniform(y_grid.min(), y_grid.max())
        ])

        if self.is_stationary:
            self.v_vec = np.array([0.0, 0.0])
        else:
            speed = np.random.uniform(1.0, 27.8)
            angle = np.random.uniform(0, 2 * np.pi)
            self.v_vec = np.array([speed * np.cos(angle), speed * np.sin(angle)])
        
        # setup
        if self.is_reference:
            self.alpha = 1
        else:
            self.alpha = np.random.normal(loc = 1.0, scale = sigma_alpha) # generate random drifts

        self.f_mo = f_mo # nominal f of main oscillator is identical
        self.f_mo_self = self.alpha * self.f_mo

        self.phi = np.random.normal(loc = 0, scale = sigma_phi) # random bias

        self.f_c = f_c # nominal carrier frequency
        self.f_c_self = f_c * self.alpha # true carrier frequency
        self.f_s_self = f_s * self.alpha # true sampling frequency

        # Phase offsets
        self.gamma_tx = np.random.uniform(0, 2 * np.pi)
        self.gamma_rx = np.random.uniform(0, 2 * np.pi)

    def tau_at(self, global_t):
        """Returns the local time tau for a given global time t."""
        return self.alpha * global_t + self.phi
    
    def _get_distance(self, node_j):
        """Returns distance to any other node."""
        p_j = node_j.p
        R_ij = np.linalg.norm(self.p - p_j)
        return R_ij
    
    def _get_rel_radial_v(self, node_j):
        """
        Calculates the relative velocity projected onto the LOS.
        Positive value implies nodes are moving away from each other.
        """
        # LOS vector from i (self) to j (sender)
        los_vec = node_j.p - self.p
        dist = np.linalg.norm(los_vec)
        
        if dist == 0: return 0.0
        
        unit_los = los_vec / dist
        
        # relative velocity vector (Source - Observer)
        rel_v_vec = node_j.v_vec - self.v_vec
        
        # scalar projection onto LOS
        v_radial = np.dot(rel_v_vec, unit_los)
        return v_radial

    def _get_doppler_shift(self, node_j):
        """
        Calculates fd = -2 * v_radial * fc / c
        Factor of 2 is added since we are considering a radar
        scenario.
        """
        v_r = self._get_rel_radial_v(node_j)
        # Using the nominal carrier frequency fc
        f_d = -2 * v_r * self.f_c / c
        return f_d 

    def transmit(self, global_t, T_p):
        """
        s_j^alpha = exp(j 2 pi f^alpha_j tau_j) * rect((tau_j - T_p/2)/T_p)
        Note: The rect function is 1 when 0 <= tau_j <= T_p.
        """
        tau_j = self.tau_at(global_t)
        
        # Check if the node's local clock is within the pulse duration
        if 0 <= tau_j <= T_p:
            # We use the drifted carrier frequency f_c_self
            # and include the hardware phase bias gamma_tx
            phase = 2 * np.pi * self.f_c_self * tau_j + self.gamma_tx
            return np.exp(1j * phase)
        
        return 0.0j

    def reset_buffer(self):
        self.rx_buffer = []

    def receive(self, sender, T_p):
        """
        Processes the incoming single-tone signal.
        """
        # 1. Physics: Signal propagation
        dist = self._get_distance(sender)
        tof = dist / c
        t_arrival = self.time_source.t
        t_emit = t_arrival - tof
        
        # 2. Capture the transmitted signal at the moment of emission
        w = sender.transmit(t_emit, T_p)
        
        if w == 0: return 0.0j
        
        # 3. Physics: Doppler shift in the channel
        f_d = self._get_doppler_shift(sender)
        channel_gain = np.exp(1j * 2 * np.pi * f_d * t_arrival)
        
        # 4. Hardware: Downconversion using the receiver's local clock
        # We mix with the conjugate of the local oscillator
        local_tau = self.tau_at(t_arrival)
        mixing_signal = np.exp(-1j * (2 * np.pi * self.f_c_self * local_tau + self.gamma_rx))
        
        r = w * channel_gain * mixing_signal 

        self.rx_buffer.append(r)

        return r

    def _est_sync_params(self, f_s):
        """
        Node looks at its buffer and find the freq
        and phase offset without knowledge of alpha.
        """
        samples = np.array(self.rx_buffer)
        N = len(samples)

        # Debugging print
        if N == 0:
            print(f"DEBUG: Node {self.index} buffer is empty! Check your loop.")
            return 0, 0, None, None # Prevent crash

        R_f = np.fft.fftshift(np.fft.fft(samples)) / f_s
        freqs = np.fft.fftshift(np.fft.fftfreq(N, d = 1/f_s))

        idx_max = np.argmax(np.abs(R_f))
        f_hat = freqs[idx_max]
        gamma_hat = np.angle(R_f[idx_max])

        return f_hat, gamma_hat, R_f, freqs

    @staticmethod
    def solve_alphas_ls(nodes, measurement_log):
        """
        Solves for all alpha values in the network using Least Squares.
        
        Args:
            nodes: List of all Node instances.
            measurement_log: List of dicts or tuples: 
                             (receiver_idx, sender_idx, f_hat)
        """
        num_nodes = len(nodes)
        H = []
        B = []

        # 1. Constraint: Reference node alpha = 1
        # Find the index of the reference node
        ref_idx = next(i for i, n in enumerate(nodes) if n.is_reference)
        anchor_row = np.zeros(num_nodes)
        anchor_row[ref_idx] = 1.0
        H.append(anchor_row)
        B.append(1.0)

        # 2. Build rows from measurements
        for rx_idx, tx_idx, f_hat in measurement_log:
            row = np.zeros(num_nodes)
            fc = nodes[rx_idx].f_c
            
            # Equation: (f_hat + fc) * alpha_rx - (fc) * alpha_tx = 0
            row[rx_idx] = f_hat + fc
            row[tx_idx] = -fc
            
            H.append(row)
            B.append(0.0)

        H = np.array(H)
        B = np.array(B)

        # 3. Solve the overdetermined system
        # Returns [alpha_0, alpha_1, ..., alpha_N]
        alphas_hat, _, _, _ = np.linalg.lstsq(H, B, rcond=None)
        
        return alphas_hat

    def __repr__(self):
        status = "REF" if self.is_reference else "NODE"
        motion = "Stat" if self.is_stationary else f"Mov({np.linalg.norm(self.v_vec):.1f}m/s)"

        # Calculate current clock offset in microseconds for readability
        # (tau - t) * 1e6
        offset_us = (self.tau_at(self.time_source.t) - self.time_source.t) * 1e6

        # Calculate frequency drift in parts per million (ppm)
        # Most oscillator specs use ppm
        drift_ppm = (self.alpha - 1.0) * 1e6

        return (
            f"[{status} {self.index:02d}] "
            f"Pos:({self.p[0]:.1f}, {self.p[1]:.1f}) | {motion} | "
            f"Drift:{drift_ppm:+.2f}ppm | Offset:{offset_us:+.3f}µs | "
            f"PhaseTX:{np.degrees(self.gamma_tx):.1f}°"
        )