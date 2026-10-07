import os
import sys
import json
import numpy as np
import matplotlib.pyplot as plt

ANALYSIS_DIR = "kiwi_analysis"
IQ_FILE = os.path.join(ANALYSIS_DIR, "iq.npy")
METADATA_FILE = os.path.join(ANALYSIS_DIR, "metadata.json")
OUTPUT_DIR = os.path.join(ANALYSIS_DIR, "decoder")
DECODER_JSON = os.path.join(OUTPUT_DIR, "decoded.json")
SPECTRUM_FILE = os.path.join(OUTPUT_DIR, "decoder_spectrum.png")

os.makedirs(OUTPUT_DIR, exist_ok=True)

MIN_WIFI_SAMPLE_RATE = 20_000_000
WIFI_BAND_LOW_HZ = 2_400_000_000
WIFI_BAND_HIGH_HZ = 2_500_000_000

def safe_float(value):
    if value is None:
        return None
    value = float(value)
    return value if np.isfinite(value) else None

def db10(x):
    return 10.0 * np.log10(np.maximum(x, 1e-20))

def load_metadata():
    if not os.path.exists(METADATA_FILE):
        raise FileNotFoundError(f"Metadata introuvables : {METADATA_FILE}")
    with open(METADATA_FILE, "r", encoding="utf-8") as f:
        return json.load(f)

def load_iq():
    if not os.path.exists(IQ_FILE):
        raise FileNotFoundError(f"IQ introuvable : {IQ_FILE}")
    iq = np.load(IQ_FILE)
    iq = np.asarray(iq, dtype=np.complex64)
    if iq.ndim != 1:
        iq = iq.reshape(-1)
    if len(iq) == 0:
        raise ValueError("Le fichier IQ est vide.")
    return iq

def estimate_spectrum(iq, sample_rate):
    n = min(len(iq), max(4096, 1 << int(np.floor(np.log2(len(iq))))))
    if n < 2:
        return None, None
    segment = iq[:n]
    segment = segment - np.mean(segment)
    window = np.hanning(n)
    spectrum = np.fft.fftshift(np.fft.fft(segment * window))
    frequencies = np.fft.fftshift(np.fft.fftfreq(n, d=1.0 / sample_rate))
    power = np.abs(spectrum) ** 2 / max(np.sum(window ** 2), 1e-20)
    return frequencies, db10(power)

def detect_activity(frequencies, power_db):
    if frequencies is None or power_db is None or len(power_db) == 0:
        return {
            "detected": False,
            "occupied_bandwidth_hz": None,
            "peak_offset_hz": None,
            "peak_power_db": None,
            "noise_floor_db": None
        }

    peak_index = int(np.argmax(power_db))
    peak_power = float(power_db[peak_index])
    noise_floor = float(np.median(power_db))
    threshold = noise_floor + 10.0
    active = power_db >= threshold

    if not np.any(active):
        return {
            "detected": False,
            "occupied_bandwidth_hz": None,
            "peak_offset_hz": safe_float(frequencies[peak_index]),
            "peak_power_db": peak_power,
            "noise_floor_db": noise_floor
        }

    active_indices = np.where(active)[0]
    low = frequencies[active_indices[0]]
    high = frequencies[active_indices[-1]]

    return {
        "detected": True,
        "occupied_bandwidth_hz": safe_float(high - low),
        "active_low_offset_hz": safe_float(low),
        "active_high_offset_hz": safe_float(high),
        "peak_offset_hz": safe_float(frequencies[peak_index]),
        "peak_power_db": peak_power,
        "noise_floor_db": noise_floor,
        "snr_estimate_db": safe_float(peak_power - noise_floor)
    }

def detect_wifi_band(center_frequency):
    return WIFI_BAND_LOW_HZ <= center_frequency <= WIFI_BAND_HIGH_HZ

def analyze_ofdm(iq, sample_rate):
    if len(iq) < 2:
        return {
            "possible_ofdm": False,
            "cyclic_prefix_correlation": None
        }

    max_lag = min(int(sample_rate * 2e-6), len(iq) // 4)

    if max_lag < 1:
        return {
            "possible_ofdm": False,
            "cyclic_prefix_correlation": None
        }

    correlations = []

    for lag in range(1, max_lag + 1):
        x = iq[:-lag]
        y = iq[lag:]
        denominator = np.sqrt(
            np.sum(np.abs(x) ** 2) *
            np.sum(np.abs(y) ** 2)
        )

        if denominator <= 0:
            correlations.append(0.0)
        else:
            correlations.append(
                float(
                    np.abs(np.sum(x * np.conj(y))) /
                    denominator
                )
            )

    correlation = max(correlations) if correlations else 0.0

    return {
        "possible_ofdm": bool(correlation > 0.15),
        "cyclic_prefix_correlation": safe_float(correlation)
    }

def save_spectrum(frequencies, power_db, center_frequency):
    if frequencies is None or power_db is None:
        return

    plt.figure(figsize=(15, 8))
    absolute_mhz = (center_frequency + frequencies) / 1e6
    plt.plot(absolute_mhz, power_db, linewidth=0.8)
    plt.xlabel("Fréquence (MHz)")
    plt.ylabel("Puissance (dB)")
    plt.title("Spectre utilisé par le décodeur")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(SPECTRUM_FILE, dpi=160)
    plt.close()

def main():
    print("=" * 70)
    print("KIWI-SDR DECODER")
    print("=" * 70)

    try:
        metadata = load_metadata()
        iq = load_iq()
    except Exception as e:
        print(f"Erreur : {e}")
        sys.exit(1)

    capture = metadata.get("capture", {})
    center_frequency = float(
        capture.get("center_frequency_hz", 0)
    )
    sample_rate = float(
        capture.get("sample_rate_hz", 0)
    )

    duration = float(
        capture.get(
            "duration_seconds",
            len(iq) / sample_rate if sample_rate > 0 else 0
        )
    )

    if sample_rate <= 0:
        print("Sample rate invalide.")
        sys.exit(1)

    frequencies, power_db = estimate_spectrum(
        iq,
        sample_rate
    )

    activity = detect_activity(
        frequencies,
        power_db
    )

    wifi_band = detect_wifi_band(
        center_frequency
    )

    sufficient_sample_rate = (
        sample_rate >= MIN_WIFI_SAMPLE_RATE
    )

    ofdm = analyze_ofdm(
        iq,
        sample_rate
    )

    save_spectrum(
        frequencies,
        power_db,
        center_frequency
    )

    result = {
        "decoder": {
            "software": "decoder.py",
            "version": "0.1.0",
            "status": "analysis_only"
        },
        "input": {
            "iq_file": IQ_FILE,
            "metadata_file": METADATA_FILE,
            "samples": int(len(iq)),
            "sample_rate_hz": safe_float(sample_rate),
            "center_frequency_hz": safe_float(center_frequency),
            "duration_seconds": safe_float(duration)
        },
        "signal": {
            "activity": activity,
            "possible_ofdm": ofdm["possible_ofdm"],
            "cyclic_prefix_correlation": ofdm[
                "cyclic_prefix_correlation"
            ]
        },
        "wifi": {
            "center_frequency_in_2_4ghz_band": wifi_band,
            "sample_rate_sufficient_for_20mhz_channel": sufficient_sample_rate,
            "likely_decodable": bool(
                wifi_band and
                sufficient_sample_rate and
                ofdm["possible_ofdm"]
            ),
            "required_sample_rate_hz": MIN_WIFI_SAMPLE_RATE
        },
        "decoding": {
            "protocol": None,
            "standard": None,
            "channel": None,
            "channel_width_hz": None,
            "bssid": None,
            "ssid": None,
            "mcs": None,
            "spatial_streams": None,
            "phy_rate_mbps": None,
            "frames_decoded": 0
        },
        "outputs": {
            "spectrum": SPECTRUM_FILE,
            "decoded_metadata": DECODER_JSON
        }
    }

    with open(
        DECODER_JSON,
        "w",
        encoding="utf-8"
    ) as f:
        json.dump(
            result,
            f,
            indent=2,
            ensure_ascii=False
        )

    print()
    print(
        f"Fréquence centrale : "
        f"{center_frequency / 1e6:.6f} MHz"
    )
    print(
        f"Sample rate        : "
        f"{sample_rate / 1e6:.6f} MS/s"
    )
    print(
        f"Échantillons       : {len(iq)}"
    )
    print(
        f"Durée              : {duration:.3f} s"
    )
    print(
        f"Bande 2,4 GHz      : "
        f"{'oui' if wifi_band else 'non'}"
    )
    print(
        f"OFDM possible      : "
        f"{'oui' if ofdm['possible_ofdm'] else 'non'}"
    )
    print(
        f"Débit suffisant    : "
        f"{'oui' if sufficient_sample_rate else 'non'}"
    )
    print(
        f"Décodage possible  : "
        f"{'oui' if result['wifi']['likely_decodable'] else 'non'}"
    )
    print()
    print(
        f"Résultat : {DECODER_JSON}"
    )
    print("=" * 70)

if __name__ == "__main__":
    main()
