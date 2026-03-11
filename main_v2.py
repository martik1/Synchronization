from synchronization._01_utility.kenney_node_single_tone import Node, GlobalTime
import numpy as np
import matplotlib.pyplot as plt
from scipy.constants import c

def main():
    # 1. Initialization
    shared_clock = GlobalTime()
    T_p = 50e-6             # Increased pulse width to 50us for better FFT resolution
    f_s = 2e6               # Sampling frequency (2MHz)
    x_grid = y_grid = np.array([-500, 500]) 

    # Node Setup
    ref = Node(is_reference=True, is_stationary=True, time_source=shared_clock, 
               x_grid=x_grid, y_grid=y_grid, index=0)
    mob = Node(is_reference=False, is_stationary=False, time_source=shared_clock, 
               x_grid=x_grid, y_grid=y_grid, index=1)
    
    # Calculate physics-based distance and Time of Flight (ToF)
    distance = ref._get_distance(mob)
    tof_actual = distance / c 
    
    print(f"--- Scenario Setup ---")
    print(f"Distance: {distance:.2f} m")
    print(f"Expected ToF: {tof_actual*1e6:.3f} µs")
    print(mob) # View true alpha and velocity
    
    # 2. Simulation Time Window
    # Create a buffer around the expected arrival
    t_start = tof_actual - (0.5 * T_p)
    t_end = tof_actual + (1.5 * T_p)
    t_steps = np.arange(t_start, t_end, 1/f_s)
    
    rx_signal = []
    ref.reset_buffer()

    # 3. Transmission / Reception Loop
    for now in t_steps:
        shared_clock.t = now
        # Process the signal through the channel
        r = ref.receive(mob, T_p)
        rx_signal.append(r)

    rx_signal = np.array(rx_signal)

    # 4. Estimation Capabilities
    # A. FFT Based Sync Estimation (Inside the Node)
    f_hat, gamma_hat, R_fft, f_axis = ref._est_sync_params(f_s)
    
    # B. Least Squares Alpha Estimation (Network-wide logic)
    # Equation: (f_hat + fc) * alpha_i - (fc) * alpha_j = 0
    # We store the measurement to feed the static LS solver
    measurements = [(ref.index, mob.index, f_hat)]
    all_nodes = [ref, mob]
    
    alpha_estimates = Node.solve_alphas_ls(all_nodes, measurements)
    
    print(f"\n--- Estimation Results ---")
    print(f"FFT Freq Offset (f_hat): {f_hat:.2f} Hz")
    print(f"True Alpha (Mob):        {mob.alpha:.10f}")
    print(f"Estimated Alpha (Mob):   {alpha_estimates[1]:.10f}")
    print(f"Estimation Error (ppm):  {abs(mob.alpha - alpha_estimates[1])*1e6:.4f}")

    # 5. Visualization
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
    
    # Magnitude plot: Arrival profile
    ax1.plot(t_steps * 1e6, np.abs(rx_signal), color='blue', lw=2)
    ax1.axvline(tof_actual * 1e6, color='red', linestyle='--', label='True ToF')
    ax1.set_title(f"Received Pulse Magnitude (ToF: {tof_actual*1e6:.3f}µs)")
    ax1.set_ylabel("Magnitude")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # FFT Plot
    ax2.plot(f_axis / 1e3, np.abs(R_fft), color='green')
    ax2.axvline(f_hat / 1e3, color='orange', linestyle='--', label=f'Peak: {f_hat:.1f}Hz')
    ax2.set_title("Frequency Spectrum of Received Signal")
    ax2.set_xlabel("Frequency (kHz)")
    ax2.set_ylabel("Spectral Density")
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()