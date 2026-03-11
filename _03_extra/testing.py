import numpy as np
import matplotlib.pyplot as plt

# 1. Setup Parameters
fs = 1000          # Sampling frequency
T_total = 1.0      # Total time
T_window = 0.01     # Length of the "on" period
f0 = 50        # Frequency (offset from bin center to show leakage)
noise_power = 0.2 # Magnitude of the noise

t = np.linspace(0, T_total, int(fs * T_total), endpoint=False)

# 2. Create the components
window = np.zeros_like(t)
window[int(0.55*fs) : int(0.6*fs)] = 1.0  # Window is "on" from 0.4s to 0.6s
pure_tone = np.sin(2 * np.pi * f0 * t)
windowed_tone = pure_tone * window
noise = np.sqrt(noise_power) * np.random.normal(size=t.shape)
noisy_signal = windowed_tone + noise

# 3. Frequency Analysis (with Zero Padding for smoothness)
n_fft = 32768
freqs = np.fft.fftfreq(n_fft, 1/fs)
pos = (freqs >= 0) & (freqs <= 120) # Focus on 0-120Hz

def get_db(data):
    mag = np.abs(np.fft.fft(data, n=n_fft))
    return 20 * np.log10(mag / np.max(mag) + 1e-10)

# Create the Multi-plot
fig, axes = plt.subplots(3, 2, figsize=(12, 10))

# Row 1: The Window
axes[0,0].plot(t, window, 'orange'); axes[0,0].set_title("1. Rectangular Window (Time)")
axes[0,1].plot(freqs[pos], get_db(window)[pos], 'orange'); axes[0,1].set_title("Sinc Pattern (Freq)")

# Row 2: Windowed Tone
axes[1,0].plot(t, windowed_tone, 'blue'); axes[1,0].set_title("2. Windowed 50.5Hz Tone (Time)")
axes[1,1].plot(freqs[pos], get_db(windowed_tone)[pos], 'blue'); axes[1,1].set_title("Shifted Sinc (Freq)")

# Row 3: Noisy Signal
axes[2,0].plot(t, noisy_signal, 'red', alpha=0.7); axes[2,0].set_title("3. Tone + Noise (Time)")
axes[2,1].plot(freqs[pos], get_db(noisy_signal)[pos], 'red'); axes[2,1].set_title("Sinc + Noise Floor (Freq)")

plt.tight_layout()
plt.show()