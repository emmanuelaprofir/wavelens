import importlib
import matplotlib.pyplot as plt

try:
    RtlSdr = importlib.import_module("rtlsdr").RtlSdr
except ModuleNotFoundError as exc:
    raise ModuleNotFoundError(
        "The 'pyrtlsdr' package is not installed. Install it with: pip install pyrtlsdr"
    ) from exc

sdr = RtlSdr()
sdr.sample_rate =2.048e6
sample_rate=sdr.sample_rate
sdr.center_freq = 100e6
center_freq = sdr.center_freq
sdr.gain = 0

plt.close()
NumberOfSamples = sample_rate/1
samples = sdr.read_samples(NumberOfSamples)
sdr.close()

print(samples[200:210])

plt.psd(samples, Fs=sample_rate/1e6, Fc=center_freq/1e6)
plt.xlabel('Frequency (MHz)')
plt.ylabel('Relative power (dB)')

del samples
plt.show()