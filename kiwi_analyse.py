import subprocess
import glob
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile

KIWI_HOST = "g3sdr.com"
KIWI_PORT = 8073
FREQUENCY_KHZ = 7100
DURATION_SECONDS = 10
OUTPUT_DIR = "kiwi_data"
os.makedirs(OUTPUT_DIR, exist_ok=True)
print("Connexion au KiwiSDR...")
print(f"Serveur     : {KIWI_HOST}:{KIWI_PORT}")
print(f"Fréquence   : {FREQUENCY_KHZ} kHz")
print()
command = [
    sys.executable,
    "kiwirecorder.py",
    "--server",
    KIWI_HOST,
    "--port",
    str(KIWI_PORT),
    "--freq",
    str(FREQUENCY_KHZ),
    "--mode",
    "iq",
    "--kiwi-wav",
    "--tlimit",
    str(DURATION_SECONDS),
    "--filename",
    os.path.join(
        OUTPUT_DIR,
        "kiwi_iq.wav"
    ),
]
try:
    subprocess.run(
        command,
        check=True
    )
except subprocess.CalledProcessError:
    print("Erreur pendant la réception KiwiSDR.")
    sys.exit(1)
files = glob.glob(
    os.path.join(OUTPUT_DIR, "*.wav")
)
if not files:
    print("Aucun fichier I/Q trouvé.")
    sys.exit(1)
filename = files[0]
print()
print("Fichier reçu :", filename)
sample_rate, data = wavfile.read(filename)
print("Sample rate :", sample_rate)
print("Dimensions  :", data.shape)

if data.ndim != 2 or data.shape[1] < 2:
    print("Le fichier ne contient pas deux canaux I/Q.")
    sys.exit(1)
I = data[:, 0].astype(np.float64)
Q = data[:, 1].astype(np.float64)

# Signal complexe
iq = I + 1j * Q
iq = iq - np.mean(iq)
N = len(iq)
window = np.hanning(N)
spectrum = np.fft.fftshift(
    np.fft.fft(
        iq * window
    )
)
frequencies = np.fft.fftshift(
    np.fft.fftfreq(
        N,
        1 / sample_rate
    )
)
power = np.abs(spectrum) ** 2
power_db = 10 * np.log10(
    power + 1e-12
)
peak_index = np.argmax(power_db)
peak_frequency = frequencies[peak_index]
peak_power = power_db[peak_index]
noise_floor = np.median(power_db)
snr = peak_power - noise_floor
center_frequency_hz = FREQUENCY_KHZ * 1000
metadata = {
    "center_frequency_hz": center_frequency_hz,
    "sample_rate_hz": int(sample_rate),
    "samples": int(N),
    "duration_seconds": N / sample_rate,
    "peak_offset_hz": float(peak_frequency),
    "noise_floor_db": float(noise_floor),
    "peak_power_db": float(peak_power),
    "estimated_snr_db": float(snr),
}

print()
print("=" * 50)
print("ANALYSE RF")
print("=" * 50)
print(
    f"Fréquence centrale : "
    f"{center_frequency_hz / 1e6:.6f} MHz"
)
print(
    f"Sample rate        : "
    f"{sample_rate / 1000:.1f} kHz"
)
print(
    f"Échantillons       : "
    f"{N}"
)
print(
    f"Durée              : "
    f"{N / sample_rate:.2f} s"
)
print(
    f"Pic détecté        : "
    f"{peak_frequency / 1000:.2f} kHz "
    f"(relatif)"
)
print(
    f"Puissance du pic   : "
    f"{peak_power:.2f} dB"
)
print(
    f"Niveau de bruit    : "
    f"{noise_floor:.2f} dB"
)
print(
    f"SNR estimé         : "
    f"{snr:.2f} dB"
)
print("=" * 50)
plt.figure(figsize=(14, 7))
plt.plot(
    frequencies / 1000,
    power_db
)
plt.axvline(
    peak_frequency / 1000,
    color="red",
    linestyle="--",
    label=(
        f"Pic : "
        f"{peak_frequency / 1000:.2f} kHz"
    )
)
plt.xlabel(
    "Fréquence relative (kHz)"
)
plt.ylabel(
    "Puissance (dB)"
)
plt.title(
    f"Spectre RF réel — "
    f"KiwiSDR {FREQUENCY_KHZ} kHz"
)
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()
