import os
import sys
import json
import csv
import struct
import wave
import hashlib
import numpy as np
import matplotlib.pyplot as plt
from kiwi.wavreader import read_kiwi_iq_wav

FILENAME = "iq.wav" #   remplacer par ton wav"
CENTER_FREQUENCY_HZ = 7_100_000 #   remplacer par la fréquence centrale souhaitée
OUTPUT_DIR = "kiwi_analysis" #   le directory dans lequel les fichiers produits seront stockés
os.makedirs(OUTPUT_DIR, exist_ok=True)
JSON_FILE = os.path.join(
    OUTPUT_DIR,
    "metadata.json"
)
CSV_FILE = os.path.join(
    OUTPUT_DIR,
    "blocks.csv"
)
SPECTRUM_FILE = os.path.join(
    OUTPUT_DIR,
    "spectrum.png"
)
SPECTROGRAM_FILE = os.path.join(
    OUTPUT_DIR,
    "spectrogram.png"
)
IQ_FILE = os.path.join(
    OUTPUT_DIR,
    "iq.npy"
)
FFT_SIZE = 4096
OVERLAP = 0.75
STEP = int(
    FFT_SIZE * (1 - OVERLAP)
)
PEAK_MIN_DISTANCE_HZ = 20
PEAK_THRESHOLD_DB = 8
BANDWIDTH_THRESHOLDS_DB = [
    3,
    6,
    10,
    20
]
def percentile(values, p):
    if len(values) == 0:
        return None
    return float(
        np.percentile(values, p)
    )
def safe_float(value):
    if value is None:
        return None
    value = float(value)
    if not np.isfinite(value):
        return None
    return value
def safe_int(value):
    if value is None:
        return None
    return int(value)
def rms(x):
    if len(x) == 0:
        return 0.0
    return float(
        np.sqrt(
            np.mean(
                np.abs(x) ** 2
            )
        )
    )
def db10(x):
    return 10 * np.log10(
        np.maximum(x, 1e-20)
    )
def db20(x):
    return 20 * np.log10(
        np.maximum(np.abs(x), 1e-20)
    )
if not os.path.exists(FILENAME):
    print(
        f"Fichier introuvable : {FILENAME}"
    )
    sys.exit(1)
print()
print("=" * 70)
print("KIWI-SDR IQ ANALYSER")
print("=" * 70)
print()
print(
    "Fichier :",
    FILENAME
)
print()
print("Calcul SHA-256...")
sha256 = hashlib.sha256()
with open(
    FILENAME,
    "rb"
) as f:
    while True:
        chunk = f.read(1024 * 1024)
        if not chunk:
            break
        sha256.update(chunk)
file_sha256 = sha256.hexdigest()
file_size = os.path.getsize(
    FILENAME
)
print(
    "Taille :",
    file_size,
    "octets"
)
print(
    "SHA-256 :",
    file_sha256
)
print()
print("=" * 70)
print("INSPECTION WAV / RIFF")
print("=" * 70)
wav_info = {}
try:
    with wave.open(
        FILENAME,
        "rb"
    ) as wf:
        wav_info = {
            "channels":
                wf.getnchannels(),
            "sample_rate":
                wf.getframerate(),
            "sample_width_bytes":
                wf.getsampwidth(),
            "frames":
                wf.getnframes(),
            "compression":
                wf.getcomptype(),
            "compression_name":
                wf.getcompname()
        }
        print(
            "Channels :",
            wav_info["channels"]
        )
        print(
            "Sample rate WAV :",
            wav_info["sample_rate"]
        )
        print(
            "Sample width :",
            wav_info["sample_width_bytes"],
            "bytes"
        )
        print(
            "Frames WAV :",
            wav_info["frames"]
        )
except Exception as e:
    print(
        "Impossible de lire le header WAV :",
        e
    )
print()
print("=" * 70)
print("LECTURE IQ KIWI")
print("=" * 70)
try:
    timestamps, iq = (
        read_kiwi_iq_wav(
            FILENAME
        )
    )
except Exception as e:
    print(
        "Erreur lecture Kiwi IQ :",
        e
    )
    sys.exit(1)
timestamps = np.asarray(
    timestamps,
    dtype=np.float64
)
iq = np.asarray(
    iq,
    dtype=np.complex64
)
print(
    "Timestamps :",
    timestamps.shape,
    timestamps.dtype
)
print(
    "IQ :",
    iq.shape,
    iq.dtype
)
if len(iq) == 0:
    print(
        "Aucun échantillon IQ."
    )
    sys.exit(1)
timestamp_info = {
    "count":
        int(len(timestamps)),
    "dtype":
        str(timestamps.dtype),
    "first":
        safe_float(
            timestamps[0]
        )
        if len(timestamps)
        else None,
    "last":
        safe_float(
            timestamps[-1]
        )
        if len(timestamps)
        else None
}
timestamp_stats = {}
if len(timestamps) > 1:
    dt = np.diff(
        timestamps
    )
    timestamp_stats = {
        "delta_mean":
            safe_float(
                np.mean(dt)
            ),
        "delta_median":
            safe_float(
                np.median(dt)
            ),
        "delta_min":
            safe_float(
                np.min(dt)
            ),
        "delta_max":
            safe_float(
                np.max(dt)
            ),
        "delta_std":
            safe_float(
                np.std(dt)
            ),
        "negative_deltas":
            int(
                np.sum(dt < 0)
            ),
        "zero_deltas":
            int(
                np.sum(dt == 0)
            ),
        "large_gaps":
            int(
                np.sum(
                    dt >
                    (
                        np.median(dt) * 2
                    )
                )
            )
    }
if len(timestamps) > 1:
    duration_timestamp = (
        timestamps[-1]
        - timestamps[0]
    )
    if duration_timestamp > 0:
        sample_rate = (
            (len(iq) - 1)
            / duration_timestamp
        )
    else:
        sample_rate = (
            wav_info.get(
                "sample_rate",
                0
            )
        )
else:
    sample_rate = (
        wav_info.get(
            "sample_rate",
            0
        )
    )
if sample_rate <= 0:
    print(
        "Sample rate invalide."
    )
    sys.exit(1)
duration = (
    len(iq)
    / sample_rate
)
I = np.real(iq)
Q = np.imag(iq)
amplitude = np.abs(iq)
power_linear = (
    amplitude ** 2
)
mean_I = np.mean(I)
mean_Q = np.mean(Q)
mean_power = np.mean(
    power_linear
)
rms_amplitude = np.sqrt(
    mean_power
)
peak_amplitude = np.max(
    amplitude
)
crest_factor = (
    peak_amplitude
    / max(
        rms_amplitude,
        1e-20
    )
)
crest_factor_db = (
    20
    * np.log10(
        max(
            crest_factor,
            1e-20
        )
    )
)
dc_amplitude = np.abs(
    np.mean(iq)
)
dc_db_relative = (
    20
    * np.log10(
        max(
            dc_amplitude,
            1e-20
        )
    )
)
print()
print("=" * 70)
print("FFT GLOBALE")
print("=" * 70)
iq_centered = (
    iq
    - np.mean(iq)
)
N = len(
    iq_centered
)
window = np.hanning(
    N
)
spectrum = np.fft.fftshift(
    np.fft.fft(
        iq_centered
        * window
    )
)
frequencies = np.fft.fftshift(
    np.fft.fftfreq(
        N,
        d=1 / sample_rate
    )
)
power = (
    np.abs(spectrum) ** 2
)
power_db = db10(
    power
)
peak_index = np.argmax(
    power_db
)
peak_offset_hz = (
    frequencies[
        peak_index
    ]
)
peak_frequency_hz = (
    CENTER_FREQUENCY_HZ
    + peak_offset_hz
)
peak_power_db = (
    power_db[
        peak_index
    ]
)
noise_floor_db = (
    np.median(
        power_db
    )
)
snr_db = (
    peak_power_db
    - noise_floor_db
)
print()
print(
    "Recherche des pics..."
)
candidate_indices = []
sorted_indices = np.argsort(
    power_db
)[::-1]
min_distance_bins = max(
    1,
    int(
        PEAK_MIN_DISTANCE_HZ
        /
        (
            sample_rate
            / N
        )
    )
)
for index in sorted_indices:
    if (
        power_db[index]
        <
        noise_floor_db
        + PEAK_THRESHOLD_DB
    ):
        break
    if all(
        abs(
            index
            - existing
        )
        >=
        min_distance_bins
        for existing
        in candidate_indices
    ):
        candidate_indices.append(
            index
        )
    if len(
        candidate_indices
    ) >= 20:
        break
peaks = []
for rank, index in enumerate(
    candidate_indices,
    start=1
):
    offset = (
        frequencies[index]
    )
    absolute = (
        CENTER_FREQUENCY_HZ
        + offset
    )
    peaks.append({
        "rank":
            rank,
        "offset_hz":
            safe_float(
                offset
            ),
        "frequency_hz":
            safe_float(
                absolute
            ),
        "power_db":
            safe_float(
                power_db[index]
            )
    })
print()
for peak in peaks:
    print(
        f"#{peak['rank']:02d} "
        f"{peak['frequency_hz'] / 1000:.3f} kHz "
        f""
        f"{peak['power_db']:.2f} dB"
    )
bandwidths = {}
for threshold in (
    BANDWIDTH_THRESHOLDS_DB
):
    limit = (
        peak_power_db
        - threshold
    )
    indices = np.where(
        power_db >= limit
    )[0]
    if len(indices):
        low = (
            frequencies[
                indices[0]
            ]
        )
        high = (
            frequencies[
                indices[-1]
            ]
        )
        bandwidth = (
            high - low
        )
        bandwidths[
            f"{threshold}_db"
        ] = {
            "low_offset_hz":
                safe_float(
                    low
                ),
            "high_offset_hz":
                safe_float(
                    high
                ),
            "low_frequency_hz":
                safe_float(
                    CENTER_FREQUENCY_HZ
                    + low
                ),
            "high_frequency_hz":
                safe_float(
                    CENTER_FREQUENCY_HZ
                    + high
                ),
            "bandwidth_hz":
                safe_float(
                    bandwidth
                )
        }
    else:
        bandwidths[
            f"{threshold}_db"
        ] = None
plt.figure(
    figsize=(15, 8)
)
plt.plot(
    frequencies / 1000,
    power_db,
    linewidth=0.8
)
for peak in peaks[:10]:
    plt.axvline(
        peak["offset_hz"] / 1000,
        color="red",
        alpha=0.25
    )
plt.axvline(
    peak_offset_hz / 1000,
    color="red",
    linestyle="--",
    linewidth=1.5,
    label=(
        f"Pic principal "
        f"{peak_offset_hz / 1000:.3f} kHz"
    )
)
plt.xlabel(
    "Fréquence relative (kHz)"
)
plt.ylabel(
    "Puissance (dB)"
)
plt.title(
    "Spectre IQ KiwiSDR"
)
plt.grid(
    True,
    alpha=0.3
)
plt.legend()
plt.tight_layout()
plt.savefig(
    SPECTRUM_FILE,
    dpi=160
)
plt.close()
print()
print("=" * 70)
print("ANALYSE TEMPORELLE")
print("=" * 70)
block_results = []
spectrogram_blocks = []
spectrogram_times = []
hann = np.hanning(
    FFT_SIZE
)
for start in range(
    0,
    len(iq) - FFT_SIZE + 1,
    STEP
):
    block = iq[
        start:
        start + FFT_SIZE
    ]
    block_centered = (
        block
        - np.mean(block)
    )
    block_fft = np.fft.fftshift(
        np.fft.fft(
            block_centered
            * hann
        )
    )
    block_power = (
        np.abs(
            block_fft
        ) ** 2
    )
    block_db = db10(
        block_power
    )
    block_peak_index = (
        np.argmax(
            block_db
        )
    )
    block_peak_offset = (
        frequencies[
            block_peak_index
        ]
    )
    block_peak_power = (
        block_db[
            block_peak_index
        ]
    )
    block_noise = (
        np.median(
            block_db
        )
    )
    block_snr = (
        block_peak_power
        - block_noise
    )
    block_rms = rms(
        block
    )
    block_amp = np.abs(
        block
    )
    block_peak_amp = (
        np.max(
            block_amp
        )
    )
    block_crest = (
        block_peak_amp
        /
        max(
            block_rms,
            1e-20
        )
    )
    block_results.append({
        "time_s":
            float(
                start / sample_rate
            ),
        "peak_offset_hz":
            float(
                block_peak_offset
            ),
        "peak_frequency_hz":
            float(
                CENTER_FREQUENCY_HZ
                + block_peak_offset
            ),
        "peak_power_db":
            float(
                block_peak_power
            ),
        "noise_floor_db":
            float(
                block_noise
            ),
        "snr_db":
            float(
                block_snr
            ),
        "rms":
            float(
                block_rms
            ),
        "peak_amplitude":
            float(
                block_peak_amp
            ),
        "crest_factor_db":
            float(
                20
                * np.log10(
                    max(
                        block_crest,
                        1e-20
                    )
                )
            )
    })
    spectrogram_blocks.append(
        block_db
    )
    spectrogram_times.append(
        start / sample_rate
    )
if block_results:
    fieldnames = list(
        block_results[0].keys()
    )
    with open(
        CSV_FILE,
        "w",
        newline="",
        encoding="utf-8"
    ) as f:
        writer = csv.DictWriter(
            f,
            fieldnames=fieldnames
        )
        writer.writeheader()
        writer.writerows(
            block_results
        )
block_peak_frequencies = np.array([
    x["peak_frequency_hz"]
    for x in block_results
])
block_peak_powers = np.array([
    x["peak_power_db"]
    for x in block_results
])
block_snrs = np.array([
    x["snr_db"]
    for x in block_results
])
temporal_analysis = {}
if len(block_results):
    temporal_analysis = {
        "blocks":
            int(
                len(block_results)
            ),
        "peak_frequency_mean_hz":
            safe_float(
                np.mean(
                    block_peak_frequencies
                )
            ),
        "peak_frequency_std_hz":
            safe_float(
                np.std(
                    block_peak_frequencies
                )
            ),
        "peak_frequency_min_hz":
            safe_float(
                np.min(
                    block_peak_frequencies
                )
            ),
        "peak_frequency_max_hz":
            safe_float(
                np.max(
                    block_peak_frequencies
                )
            ),
        "peak_power_mean_db":
            safe_float(
                np.mean(
                    block_peak_powers
                )
            ),
        "peak_power_std_db":
            safe_float(
                np.std(
                    block_peak_powers
                )
            ),
        "peak_power_min_db":
            safe_float(
                np.min(
                    block_peak_powers
                )
            ),
        "peak_power_max_db":
            safe_float(
                np.max(
                    block_peak_powers
                )
            ),
        "snr_mean_db":
            safe_float(
                np.mean(
                    block_snrs
                )
            ),
        "snr_std_db":
            safe_float(
                np.std(
                    block_snrs
                )
            )
    }
print()
print(
    "Création spectrogramme..."
)
if spectrogram_blocks:
    spectrogram = np.array(
        spectrogram_blocks
    )
    spectrogram_times = np.array(
        spectrogram_times
    )
    plt.figure(
        figsize=(15, 9)
    )
    extent = [
        frequencies[0] / 1000,
        frequencies[-1] / 1000,
        spectrogram_times[0],
        spectrogram_times[-1]
    ]
    plt.imshow(
        spectrogram,
        aspect="auto",
        origin="lower",
        extent=extent,
        cmap="viridis"
    )
    plt.colorbar(
        label="Puissance (dB)"
    )
    plt.xlabel(
        "Fréquence relative (kHz)"
    )
    plt.ylabel(
        "Temps (s)"
    )
    plt.title(
        "Spectrogramme IQ KiwiSDR"
    )
    plt.tight_layout()
    plt.savefig(
        SPECTROGRAM_FILE,
        dpi=160
    )
    plt.close()
np.save(
    IQ_FILE,
    iq
)
amplitude_percentiles = {
    "p01":
        percentile(
            amplitude,
            1
        ),
    "p05":
        percentile(
            amplitude,
            5
        ),
    "p25":
        percentile(
            amplitude,
            25
        ),
    "p50":
        percentile(
            amplitude,
            50
        ),
    "p75":
        percentile(
            amplitude,
            75
        ),
    "p95":
        percentile(
            amplitude,
            95
        ),
    "p99":
        percentile(
            amplitude,
            99
        )
}
phase = np.unwrap(
    np.angle(iq)
)
phase_diff = np.diff(
    phase
)
frequency_from_phase = (
    phase_diff
    * sample_rate
    /
    (2 * np.pi)
)
phase_frequency_stats = {
    "mean_hz":
        safe_float(
            np.mean(
                frequency_from_phase
            )
        ),
    "std_hz":
        safe_float(
            np.std(
                frequency_from_phase
            )
        ),
    "median_hz":
        safe_float(
            np.median(
                frequency_from_phase
            )
        ),
    "p05_hz":
        percentile(
            frequency_from_phase,
            5
        ),
    "p95_hz":
        percentile(
            frequency_from_phase,
            95
        )
}
metadata = {
    "analysis": {
        "software":
            "kiwi_analyse.py",
        "version":
            "advanced-1.0",
        "fft_size":
            FFT_SIZE,
        "fft_overlap":
            OVERLAP,
        "fft_step":
            STEP
    },
    "file": {
        "filename":
            os.path.basename(
                FILENAME
            ),
        "path":
            os.path.abspath(
                FILENAME
            ),
        "size_bytes":
            int(file_size),
        "sha256":
            file_sha256
    },
    "wav": wav_info,
    "capture": {
        "source":
            "KiwiSDR",
        "mode":
            "IQ",
        "center_frequency_hz":
            CENTER_FREQUENCY_HZ,
        "center_frequency_mhz":
            CENTER_FREQUENCY_HZ / 1e6,
        "sample_rate_hz":
            safe_float(
                sample_rate
            ),
        "samples":
            int(len(iq)),
        "duration_seconds":
            safe_float(
                duration
            ),
        "iq_dtype":
            str(iq.dtype)
    },
    "timestamps": {
        **timestamp_info,
        "statistics":
            timestamp_stats
    },
    "iq_statistics": {
        "i_mean":
            safe_float(
                mean_I
            ),
        "q_mean":
            safe_float(
                mean_Q
            ),
        "i_rms":
            safe_float(
                rms(I)
            ),
        "q_rms":
            safe_float(
                rms(Q)
            ),
        "amplitude_mean":
            safe_float(
                np.mean(
                    amplitude
                )
            ),
        "amplitude_rms":
            safe_float(
                rms_amplitude
            ),
        "amplitude_min":
            safe_float(
                np.min(
                    amplitude
                )
            ),
        "amplitude_max":
            safe_float(
                np.max(
                    amplitude
                )
            ),
        "power_mean":
            safe_float(
                mean_power
            ),
        "crest_factor":
            safe_float(
                crest_factor
            ),
        "crest_factor_db":
            safe_float(
                crest_factor_db
            ),
        "dc_amplitude":
            safe_float(
                dc_amplitude
            ),
        "dc_relative_db":
            safe_float(
                dc_db_relative
            )
    },
    "amplitude_distribution":
        amplitude_percentiles,
    "phase_analysis":
        phase_frequency_stats,
    "rf_analysis": {
        "peak_offset_hz":
            safe_float(
                peak_offset_hz
            ),
        "peak_frequency_hz":
            safe_float(
                peak_frequency_hz
            ),
        "peak_power_db":
            safe_float(
                peak_power_db
            ),
        "noise_floor_db":
            safe_float(
                noise_floor_db
            ),
        "estimated_snr_db":
            safe_float(
                snr_db
            ),
        "peaks":
            peaks,
        "bandwidths":
            bandwidths
    },
    "temporal_analysis":
        temporal_analysis,
    "outputs": {
        "spectrum":
            SPECTRUM_FILE,
        "spectrogram":
            SPECTROGRAM_FILE,
        "iq":
            IQ_FILE,
        "blocks_csv":
            CSV_FILE,
        "metadata":
            JSON_FILE
    }
}
with open(
    JSON_FILE,
    "w",
    encoding="utf-8"
) as f:
    json.dump(
        metadata,
        f,
        indent=2,
        ensure_ascii=False
    )
print()
print("=" * 70)
print("ANALYSE TERMINÉE")
print("=" * 70)
print()
print(
    f"Centre             : "
    f"{CENTER_FREQUENCY_HZ / 1e6:.6f} MHz"
)
print(
    f"Sample rate        : "
    f"{sample_rate:.3f} Hz"
)
print(
    f"Durée              : "
    f"{duration:.3f} s"
)
print(
    f"Pic principal      : "
    f"{peak_frequency_hz / 1000:.3f} kHz"
)
print(
    f"Offset             : "
    f"{peak_offset_hz / 1000:.3f} kHz"
)
print(
    f"SNR                : "
    f"{snr_db:.2f} dB"
)
print(
    f"Bande -3 dB        : "
    f"{bandwidths['3_db']['bandwidth_hz']:.2f} Hz"
    if bandwidths.get("3_db")
    else "Bande -3 dB : inconnue"
)
print(
    f"Bande -10 dB       : "
    f"{bandwidths['10_db']['bandwidth_hz']:.2f} Hz"
    if bandwidths.get("10_db")
    else "Bande -10 dB : inconnue"
)
print()
print(
    "Fichiers :"
)
print(
    "  ",
    JSON_FILE
)
print(
    "  ",
    CSV_FILE
)
print(
    "  ",
    SPECTRUM_FILE
)
print(
    "  ",
    SPECTROGRAM_FILE
)
print(
    "  ",
    IQ_FILE
)
print()
print(
    "Le fichier principal est :",
    JSON_FILE
)
print(
    "Le CSV contient l'évolution du signal dans le temps."
)
print("=" * 70)
