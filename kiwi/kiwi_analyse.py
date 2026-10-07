import os
import sys
import json
import csv
import wave
import hashlib
import numpy as np
import matplotlib.pyplot as plt
from kiwi.wavreader import read_kiwi_iq_wav
FILENAME = "iq.wav"
CENTER_FREQUENCY_HZ = 7_100_000
OUTPUT_DIR = "kiwi_analysis"
os.makedirs(OUTPUT_DIR, exist_ok=True)
JSON_FILE = os.path.join(OUTPUT_DIR, "metadata.json")
CSV_FILE = os.path.join(OUTPUT_DIR, "blocks.csv")
SPECTRUM_FILE = os.path.join(OUTPUT_DIR, "spectrum.png")
SPECTROGRAM_FILE = os.path.join(OUTPUT_DIR, "spectrogram.png")
IQ_FILE = os.path.join(OUTPUT_DIR, "iq.npy")
FFT_SIZE = 4096
OVERLAP = 0.75
STEP = int(FFT_SIZE * (1 - OVERLAP))
PEAK_MIN_DISTANCE_HZ = 20
PEAK_THRESHOLD_DB = 8
BANDWIDTH_THRESHOLDS_DB = [3, 6, 10, 20]
def percentile(values, p):
    if len(values) == 0:
        return None
    return float(np.percentile(values, p))
def safe_float(value):
    if value is None:
        return None
    value = float(value)
    if not np.isfinite(value):
        return None
    return value
def rms(x):
    if len(x) == 0:
        return 0.0
    return float(np.sqrt(np.mean(np.abs(x) ** 2)))
def db10(x):
    return 10 * np.log10(np.maximum(x, 1e-20))
def db20(x):
    return 20 * np.log10(np.maximum(np.abs(x), 1e-20))
if not os.path.exists(FILENAME):
    print(f"Fichier introuvable : {FILENAME}")
    sys.exit(1)
print()
print("=" * 70)
print("KIWI-SDR IQ ANALYSER")
print("=" * 70)
print()
print("Fichier :", FILENAME)
print()
sha256 = hashlib.sha256()
with open(FILENAME, "rb") as f:
    while True:
        chunk = f.read(1024 * 1024)
        if not chunk:
            break
        sha256.update(chunk)
file_sha256 = sha256.hexdigest()
file_size = os.path.getsize(FILENAME)
print("Taille :", file_size, "octets")
print("SHA-256 :", file_sha256)
print()
print("=" * 70)
print("INSPECTION WAV / RIFF")
print("=" * 70)
wav_info = {}
try:
    with wave.open(FILENAME, "rb") as wf:
        wav_info = {
            "channels": wf.getnchannels(),
            "sample_rate": wf.getframerate(),
            "sample_width_bytes": wf.getsampwidth(),
            "frames": wf.getnframes(),
            "compression": wf.getcomptype(),
            "compression_name": wf.getcompname()
        }
        print("Channels :", wav_info["channels"])
        print("Sample rate WAV :", wav_info["sample_rate"])
        print("Sample width :", wav_info["sample_width_bytes"], "bytes")
        print("Frames WAV :", wav_info["frames"])
except Exception as e:
    print("Impossible de lire le header WAV :", e)
print()
print("=" * 70)
print("LECTURE IQ KIWI")
print("=" * 70)
try:
    timestamps, iq = read_kiwi_iq_wav(FILENAME)
except Exception as e:
    print("Erreur lecture Kiwi IQ :", e)
    sys.exit(1)
timestamps = np.asarray(timestamps, dtype=np.float64)
iq = np.asarray(iq, dtype=np.complex64)
print("Timestamps :", timestamps.shape, timestamps.dtype)
print("IQ :", iq.shape, iq.dtype)
if len(iq) == 0:
    print("Aucun échantillon IQ.")
    sys.exit(1)
timestamp_info = {
    "count": int(len(timestamps)),
    "dtype": str(timestamps.dtype),
    "first": safe_float(timestamps[0]) if len(timestamps) else None,
    "last": safe_float(timestamps[-1]) if len(timestamps) else None
}
timestamp_stats = {}
if len(timestamps) > 1:
    dt_raw = np.diff(timestamps)
    positive_dt = dt_raw[dt_raw > 0]
    median_dt = np.median(positive_dt) if len(positive_dt) else 0
    timestamp_stats = {
        "delta_mean": safe_float(np.mean(dt_raw)),
        "delta_median": safe_float(np.median(dt_raw)),
        "delta_min": safe_float(np.min(dt_raw)),
        "delta_max": safe_float(np.max(dt_raw)),
        "delta_std": safe_float(np.std(dt_raw)),
        "negative_deltas": int(np.sum(dt_raw < 0)),
        "zero_deltas": int(np.sum(dt_raw == 0)),
        "large_gaps": int(np.sum(dt_raw > median_dt * 2)) if median_dt > 0 else 0
    }
if len(timestamps) > 1:
    dt = np.diff(timestamps)
    positive_dt = dt[dt > 0]
    if len(positive_dt):
        median_dt = np.median(positive_dt)
        filtered_dt = positive_dt[positive_dt < median_dt * 10]
        sample_rate = 1.0 / np.median(filtered_dt) if len(filtered_dt) else 0
    else:
        sample_rate = 0
else:
    sample_rate = wav_info.get("sample_rate", 0)
if sample_rate <= 0:
    print("Sample rate invalide.")
    sys.exit(1)
duration = len(iq) / sample_rate
I = np.real(iq)
Q = np.imag(iq)
amplitude = np.abs(iq)
power_linear_time = amplitude ** 2
mean_I = np.mean(I)
mean_Q = np.mean(Q)
mean_power = np.mean(power_linear_time)
rms_amplitude = np.sqrt(mean_power)
peak_amplitude = np.max(amplitude)
crest_factor = peak_amplitude / max(rms_amplitude, 1e-20)
crest_factor_db = 20 * np.log10(max(crest_factor, 1e-20))
dc_amplitude = np.abs(np.mean(iq))
dc_db_relative = 20 * np.log10(max(dc_amplitude, 1e-20))
amplitude_percentiles = {
    "p01": percentile(amplitude, 1),
    "p05": percentile(amplitude, 5),
    "p25": percentile(amplitude, 25),
    "p50": percentile(amplitude, 50),
    "p75": percentile(amplitude, 75),
    "p95": percentile(amplitude, 95),
    "p99": percentile(amplitude, 99)
}
print()
print("=" * 70)
print("FFT GLOBALE")
print("=" * 70)
iq_centered = iq - np.mean(iq)
N = len(iq_centered)
window = np.hanning(N)
window_power = max(np.sum(window ** 2), 1e-20)
spectrum = np.fft.fftshift(np.fft.fft(iq_centered * window))
frequencies = np.fft.fftshift(np.fft.fftfreq(N, d=1 / sample_rate))
power_linear = np.abs(spectrum) ** 2 / window_power
power_db = db10(power_linear)
total_power = np.mean(np.abs(iq_centered) ** 2)
peak_index = np.argmax(power_db)
peak_offset_hz = frequencies[peak_index]
peak_frequency_hz = CENTER_FREQUENCY_HZ + peak_offset_hz
peak_power_db = power_db[peak_index]
noise_floor_db = np.median(power_db)
snr_db = peak_power_db - noise_floor_db
print("Puissance IQ moyenne :", f"{total_power:.6g}")
print("Pic principal :", f"{peak_frequency_hz:.3f} Hz")
print("SNR estimé :", f"{snr_db:.2f} dB")
candidate_indices = []
sorted_indices = np.argsort(power_db)[::-1]
min_distance_bins = max(1, int(PEAK_MIN_DISTANCE_HZ / max(sample_rate / N, 1e-20)))
for index in sorted_indices:
    if power_db[index] < noise_floor_db + PEAK_THRESHOLD_DB:
        break
    if all(abs(index - existing) >= min_distance_bins for existing in candidate_indices):
        candidate_indices.append(index)
    if len(candidate_indices) >= 20:
        break
peaks = []
for rank, index in enumerate(candidate_indices, start=1):
    offset = frequencies[index]
    absolute = CENTER_FREQUENCY_HZ + offset
    peaks.append({
        "rank": rank,
        "offset_hz": safe_float(offset),
        "frequency_hz": safe_float(absolute),
        "power_db": safe_float(power_db[index])
    })
bandwidths = {}
for threshold in BANDWIDTH_THRESHOLDS_DB:
    limit = peak_power_db - threshold
    indices = np.where(power_db >= limit)[0]
    if len(indices):
        low = frequencies[indices[0]]
        high = frequencies[indices[-1]]
        bandwidths[f"{threshold}_db"] = {
            "low_offset_hz": safe_float(low),
            "high_offset_hz": safe_float(high),
            "low_frequency_hz": safe_float(CENTER_FREQUENCY_HZ + low),
            "high_frequency_hz": safe_float(CENTER_FREQUENCY_HZ + high),
            "bandwidth_hz": safe_float(high - low)
        }
    else:
        bandwidths[f"{threshold}_db"] = None
frequency_step = abs(float(np.median(np.diff(frequencies)))) if len(frequencies) > 1 else 0.0
noise_reference_power = float(np.percentile(power_linear, 20))
signal_power_spectrum = np.maximum(power_linear - noise_reference_power, 0.0)
power_sum = float(np.sum(signal_power_spectrum))
occupied_bandwidths = {}
if power_sum > 0 and frequency_step > 0:
    cumulative_power = np.cumsum(signal_power_spectrum) / power_sum
    def occupied_bandwidth(percent):
        low_fraction = (1.0 - percent) / 2.0
        high_fraction = 1.0 - low_fraction
        low_index = int(np.searchsorted(cumulative_power, low_fraction))
        high_index = int(np.searchsorted(cumulative_power, high_fraction))
        low_index = max(0, min(low_index, len(frequencies) - 1))
        high_index = max(0, min(high_index, len(frequencies) - 1))
        return {
            "low_offset_hz": safe_float(frequencies[low_index]),
            "high_offset_hz": safe_float(frequencies[high_index]),
            "low_frequency_hz": safe_float(CENTER_FREQUENCY_HZ + frequencies[low_index]),
            "high_frequency_hz": safe_float(CENTER_FREQUENCY_HZ + frequencies[high_index]),
            "bandwidth_hz": safe_float(max(0.0, frequencies[high_index] - frequencies[low_index])),
            "contained_power_percent": percent * 100.0
        }
    occupied_bandwidths = {
        "90_percent": occupied_bandwidth(0.90),
        "95_percent": occupied_bandwidth(0.95),
        "99_percent": occupied_bandwidth(0.99)
    }
spectral_centroid_offset_hz = safe_float(np.sum(frequencies * signal_power_spectrum) / power_sum) if power_sum > 0 else None
spectral_rms_bandwidth_hz = safe_float(np.sqrt(np.sum(((frequencies - (spectral_centroid_offset_hz or 0.0)) ** 2) * signal_power_spectrum) / power_sum)) if power_sum > 0 else None
spectral_skewness = None
if power_sum > 0 and spectral_rms_bandwidth_hz is not None and spectral_rms_bandwidth_hz > 0:
    spectral_skewness = safe_float(np.sum(((frequencies - (spectral_centroid_offset_hz or 0.0)) ** 3) * signal_power_spectrum) / power_sum / spectral_rms_bandwidth_hz ** 3)
instantaneous_phase = np.angle(iq)
phase_step = np.angle(iq[1:] * np.conj(iq[:-1]))
instantaneous_frequency = phase_step * sample_rate / (2 * np.pi)
valid_if = instantaneous_frequency[np.isfinite(instantaneous_frequency)]
if_stats = {
    "mean_hz": safe_float(np.mean(valid_if)) if len(valid_if) else None,
    "std_hz": safe_float(np.std(valid_if)) if len(valid_if) else None,
    "median_hz": safe_float(np.median(valid_if)) if len(valid_if) else None,
    "p05_hz": percentile(valid_if, 5),
    "p95_hz": percentile(valid_if, 95)
}
amplitude_mean = np.mean(amplitude)
amplitude_std = np.std(amplitude)
amplitude_cv = amplitude_std / max(amplitude_mean, 1e-20)
amplitude_skewness = safe_float(np.mean(((amplitude - amplitude_mean) / max(amplitude_std, 1e-20)) ** 3))
phase_diff_std = safe_float(np.std(phase_step)) if len(phase_step) else None
plt.figure(figsize=(15, 8))
plt.plot(frequencies / 1000, power_db, linewidth=0.8)
for peak in peaks[:10]:
    plt.axvline(peak["offset_hz"] / 1000, color="red", alpha=0.25)
plt.axvline(peak_offset_hz / 1000, color="red", linestyle="--", linewidth=1.5, label=f"Pic principal {peak_offset_hz / 1000:.3f} kHz")
plt.xlabel("Fréquence relative (kHz)")
plt.ylabel("Puissance (dB)")
plt.title("Spectre IQ KiwiSDR")
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()
plt.savefig(SPECTRUM_FILE, dpi=160)
plt.close()
print()
print("=" * 70)
print("ANALYSE TEMPORELLE")
print("=" * 70)
block_results = []
spectrogram_blocks = []
spectrogram_times = []
hann = np.hanning(FFT_SIZE)
block_frequencies = np.fft.fftshift(np.fft.fftfreq(FFT_SIZE, d=1 / sample_rate))
block_window_power = max(np.sum(hann ** 2), 1e-20)
for start in range(0, len(iq) - FFT_SIZE + 1, STEP):
    block = iq[start:start + FFT_SIZE]
    block_centered = block - np.mean(block)
    block_fft = np.fft.fftshift(np.fft.fft(block_centered * hann))
    block_power = np.abs(block_fft) ** 2 / block_window_power
    block_db = db10(block_power)
    block_peak_index = np.argmax(block_db)
    block_peak_offset = block_frequencies[block_peak_index]
    block_peak_power = block_db[block_peak_index]
    block_noise = np.median(block_db)
    block_snr = block_peak_power - block_noise
    block_rms = rms(block)
    block_amp = np.abs(block)
    block_peak_amp = np.max(block_amp)
    block_crest = block_peak_amp / max(block_rms, 1e-20)
    block_power_sum = np.sum(block_power)
    block_centroid = np.sum(block_frequencies * block_power) / max(block_power_sum, 1e-20)
    block_rbw = np.sqrt(np.sum(((block_frequencies - block_centroid) ** 2) * block_power) / max(block_power_sum, 1e-20))
    block_results.append({
        "time_s": float(start / sample_rate),
        "peak_offset_hz": float(block_peak_offset),
        "peak_frequency_hz": float(CENTER_FREQUENCY_HZ + block_peak_offset),
        "peak_power_db": float(block_peak_power),
        "noise_floor_db": float(block_noise),
        "snr_db": float(block_snr),
        "rms": float(block_rms),
        "peak_amplitude": float(block_peak_amp),
        "crest_factor_db": float(20 * np.log10(max(block_crest, 1e-20))),
        "spectral_centroid_offset_hz": float(block_centroid),
        "spectral_rms_bandwidth_hz": float(block_rbw)
    })
    spectrogram_blocks.append(block_db)
    spectrogram_times.append(start / sample_rate)
if block_results:
    fieldnames = list(block_results[0].keys())
    with open(CSV_FILE, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(block_results)
block_peak_frequencies = np.array([x["peak_frequency_hz"] for x in block_results])
block_peak_powers = np.array([x["peak_power_db"] for x in block_results])
block_snrs = np.array([x["snr_db"] for x in block_results])
block_centroids = np.array([x["spectral_centroid_offset_hz"] for x in block_results])
block_bandwidths = np.array([x["spectral_rms_bandwidth_hz"] for x in block_results])
temporal_analysis = {}
if len(block_results):
    temporal_analysis = {
        "blocks": int(len(block_results)),
        "peak_frequency_mean_hz": safe_float(np.mean(block_peak_frequencies)),
        "peak_frequency_std_hz": safe_float(np.std(block_peak_frequencies)),
        "peak_frequency_min_hz": safe_float(np.min(block_peak_frequencies)),
        "peak_frequency_max_hz": safe_float(np.max(block_peak_frequencies)),
        "peak_power_mean_db": safe_float(np.mean(block_peak_powers)),
        "peak_power_std_db": safe_float(np.std(block_peak_powers)),
        "peak_power_min_db": safe_float(np.min(block_peak_powers)),
        "peak_power_max_db": safe_float(np.max(block_peak_powers)),
        "snr_mean_db": safe_float(np.mean(block_snrs)),
        "snr_std_db": safe_float(np.std(block_snrs)),
        "centroid_mean_offset_hz": safe_float(np.mean(block_centroids)),
        "centroid_std_hz": safe_float(np.std(block_centroids)),
        "rms_bandwidth_mean_hz": safe_float(np.mean(block_bandwidths)),
        "rms_bandwidth_std_hz": safe_float(np.std(block_bandwidths))
    }
print()
print("Création spectrogramme...")
if spectrogram_blocks:
    spectrogram = np.array(spectrogram_blocks)
    spectrogram_times = np.array(spectrogram_times)
    plt.figure(figsize=(15, 9))
    extent = [
        block_frequencies[0] / 1000,
        block_frequencies[-1] / 1000,
        spectrogram_times[0],
        spectrogram_times[-1]
    ]
    plt.imshow(spectrogram, aspect="auto", origin="lower", extent=extent, cmap="viridis")
    plt.colorbar(label="Puissance (dB)")
    plt.xlabel("Fréquence relative (kHz)")
    plt.ylabel("Temps (s)")
    plt.title("Spectrogramme IQ KiwiSDR")
    plt.tight_layout()
    plt.savefig(SPECTROGRAM_FILE, dpi=160)
    plt.close()
np.save(IQ_FILE, iq)
metadata = {
    "analysis": {
        "software": "kiwi_analyse.py",
        "version": "advanced-2.1",
        "fft_size": FFT_SIZE,
        "fft_overlap": OVERLAP,
        "fft_step": STEP,
        "frequency_resolution_hz": safe_float(sample_rate / N),
        "block_frequency_resolution_hz": safe_float(sample_rate / FFT_SIZE)
    },
    "file": {
        "filename": os.path.basename(FILENAME),
        "path": os.path.abspath(FILENAME),
        "size_bytes": int(file_size),
        "sha256": file_sha256
    },
    "wav": wav_info,
    "capture": {
        "source": "KiwiSDR",
        "mode": "IQ",
        "center_frequency_hz": CENTER_FREQUENCY_HZ,
        "center_frequency_mhz": CENTER_FREQUENCY_HZ / 1e6,
        "sample_rate_hz": safe_float(sample_rate),
        "samples": int(len(iq)),
        "duration_seconds": safe_float(duration),
        "iq_dtype": str(iq.dtype)
    },
    "timestamps": {
        **timestamp_info,
        "statistics": timestamp_stats
    },
    "iq_statistics": {
        "i_mean": safe_float(mean_I),
        "q_mean": safe_float(mean_Q),
        "i_rms": safe_float(rms(I)),
        "q_rms": safe_float(rms(Q)),
        "amplitude_mean": safe_float(amplitude_mean),
        "amplitude_std": safe_float(amplitude_std),
        "amplitude_cv": safe_float(amplitude_cv),
        "amplitude_min": safe_float(np.min(amplitude)),
        "amplitude_max": safe_float(np.max(amplitude)),
        "power_mean": safe_float(mean_power),
        "crest_factor": safe_float(crest_factor),
        "crest_factor_db": safe_float(crest_factor_db),
        "dc_amplitude": safe_float(dc_amplitude),
        "dc_relative_db": safe_float(dc_db_relative)
    },
    "amplitude_distribution": amplitude_percentiles,
    "phase_analysis": {
        "instantaneous_frequency": if_stats,
        "phase_step_std_rad": phase_diff_std
    },
    "modulation_indicators": {
        "amplitude_coefficient_of_variation": safe_float(amplitude_cv),
        "amplitude_skewness": amplitude_skewness,
        "frequency_deviation_p05_hz": safe_float(if_stats["p05_hz"]),
        "frequency_deviation_p95_hz": safe_float(if_stats["p95_hz"]),
        "frequency_deviation_peak_to_peak_hz": safe_float(if_stats["p95_hz"] - if_stats["p05_hz"]) if if_stats["p05_hz"] is not None and if_stats["p95_hz"] is not None else None
    },
    "rf_analysis": {
        "peak_offset_hz": safe_float(peak_offset_hz),
        "peak_frequency_hz": safe_float(peak_frequency_hz),
        "peak_power_db": safe_float(peak_power_db),
        "noise_floor_db": safe_float(noise_floor_db),
        "estimated_snr_db": safe_float(snr_db),
        "peaks": peaks,
        "bandwidths": bandwidths,
        "occupied_bandwidths": occupied_bandwidths,
        "occupied_bandwidth_noise_reference_percentile": 20,
        "occupied_bandwidth_signal_power": safe_float(power_sum),
        "spectral_centroid_offset_hz": spectral_centroid_offset_hz,
        "spectral_centroid_frequency_hz": safe_float(CENTER_FREQUENCY_HZ + spectral_centroid_offset_hz) if spectral_centroid_offset_hz is not None else None,
        "spectral_rms_bandwidth_hz": spectral_rms_bandwidth_hz,
        "spectral_skewness": spectral_skewness
    },
    "temporal_analysis": temporal_analysis,
    "outputs": {
        "spectrum": SPECTRUM_FILE,
        "spectrogram": SPECTROGRAM_FILE,
        "iq": IQ_FILE,
        "blocks_csv": CSV_FILE,
        "metadata": JSON_FILE
    }
}
with open(JSON_FILE, "w", encoding="utf-8") as f:
    json.dump(metadata, f, indent=2, ensure_ascii=False)
print()
print("=" * 70)
print("ANALYSE TERMINÉE")
print("=" * 70)
print()
print(f"Centre             : {CENTER_FREQUENCY_HZ / 1e6:.6f} MHz")
print(f"Sample rate        : {sample_rate:.3f} Hz")
print(f"Durée              : {duration:.3f} s")
print(f"Résolution FFT     : {sample_rate / N:.4f} Hz")
print(f"Pic principal      : {peak_frequency_hz / 1000:.3f} kHz")
print(f"Offset             : {peak_offset_hz / 1000:.3f} kHz")
print(f"SNR                : {snr_db:.2f} dB")
print(f"Bande -3 dB        : {bandwidths['3_db']['bandwidth_hz']:.2f} Hz" if bandwidths.get("3_db") else "Bande -3 dB        : inconnue")
print(f"Bande -10 dB       : {bandwidths['10_db']['bandwidth_hz']:.2f} Hz" if bandwidths.get("10_db") else "Bande -10 dB       : inconnue")
print(f"OBW 90 %           : {occupied_bandwidths['90_percent']['bandwidth_hz']:.2f} Hz" if occupied_bandwidths.get("90_percent") else "OBW 90 %           : inconnue")
print(f"OBW 95 %           : {occupied_bandwidths['95_percent']['bandwidth_hz']:.2f} Hz" if occupied_bandwidths.get("95_percent") else "OBW 95 %           : inconnue")
print(f"OBW 99 %           : {occupied_bandwidths['99_percent']['bandwidth_hz']:.2f} Hz" if occupied_bandwidths.get("99_percent") else "OBW 99 %           : inconnue")
print()
print("Fichiers :")
print("  ", JSON_FILE)
print("  ", CSV_FILE)
print("  ", SPECTRUM_FILE)
print("  ", SPECTROGRAM_FILE)
print("  ", IQ_FILE)
print()
print("Le fichier principal est :", JSON_FILE)
print("=" * 70)
