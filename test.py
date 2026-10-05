import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks

#parametres

sample_rate = 1_000_000      # 1 MHz d'échantillonnage
duration = 0.02              # 20 ms
signal_frequency = 100_000   # signal à +100 kHz
signal_amplitude = 1.0
noise_amplitude = 0.25

# I = In phase
# Q = Quadrature (90°)

t = np.arange(0, duration, 1 / sample_rate)
signal = (
    signal_amplitude
    * np.exp(2j * np.pi * signal_frequency * t)
)
# Bruit complexe
noise = noise_amplitude * (
    np.random.randn(len(t))
    + 1j * np.random.randn(len(t))
)
# Signal reçu = signal + bruit
iq = signal + noise
# I et Q séparément
I = np.real(iq)
Q = np.imag(iq)
N = len(iq)
# Fenêtre pour réduire les effets de bord
window = np.hanning(N)
spectrum = np.fft.fftshift(
    np.fft.fft(iq * window)
)
frequencies = np.fft.fftshift(
    np.fft.fftfreq(N, 1 / sample_rate)
)
# Puissance
power = np.abs(spectrum) ** 2
# Conversion en dB
power_db = 10 * np.log10(power + 1e-12)
# Recherche des pics
peaks, properties = find_peaks(
    power_db,
    prominence=10
)
if len(peaks) > 0:
    strongest_peak = peaks[np.argmax(power_db[peaks])]
    detected_frequency = frequencies[strongest_peak]
    detected_power = power_db[strongest_peak]
else:
    detected_frequency = None
    detected_power = None

# estimation bruit

noise_floor = np.median(power_db)

print()
print("========== ANALYSE DU SIGNAL ==========")
print()

print(f"Fréquence d'échantillonnage : "
      f"{sample_rate / 1e6:.3f} MHz")

print(f"Nombre d'échantillons       : {N}")

print(f"Durée analysée              : "
      f"{duration * 1000:.2f} ms")

print(f"Niveau de bruit estimé      : "
      f"{noise_floor:.2f} dB")

if detected_frequency is not None:

    print()
    print("----- Signal détecté -----")

    print(f"Fréquence                  : "
          f"{detected_frequency / 1000:.2f} kHz")

    print(f"Puissance relative         : "
          f"{detected_power:.2f} dB")

    print(f"Rapport signal/bruit       : "
          f"{detected_power - noise_floor:.2f} dB")

else:

    print("Aucun signal détecté.")
print()
print("========================================")
plt.figure(figsize=(12, 6))
plt.plot(
    frequencies / 1000,
    power_db,
    linewidth=1
)
if detected_frequency is not None:
    plt.axvline(
        detected_frequency / 1000,
        color="red",
        linestyle="--",
        label=f"Signal : {detected_frequency / 1000:.1f} kHz"
    )
plt.xlabel("Fréquence (kHz)")
plt.ylabel("Puissance (dB)")
plt.title("Analyse spectrale I/Q")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()
