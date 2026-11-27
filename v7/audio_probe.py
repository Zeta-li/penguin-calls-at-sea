"""Paired, known-SNR perturbations of the three published illustrative waveforms."""

from pathlib import Path
import wave

import numpy as np
import pandas as pd
from scipy.ndimage import uniform_filter1d
from scipy.signal import butter, hilbert, sosfiltfilt, stft, welch


HERE = Path(__file__).resolve().parent
BANDS = {"500-1100": (500, 1100), "3000-3600": (3000, 3600)}
SNR_DB = (-6, 0, 6, 12)
SEED = 72026
DRAWS = 32


def load_pcm(path):
    with wave.open(str(path), "rb") as f:
        assert f.getnchannels() == 1 and f.getsampwidth() == 2
        fs = f.getframerate()
        data = np.frombuffer(f.readframes(f.getnframes()), dtype="<i2").astype(float)
    data -= data.mean()
    return fs, data / np.sqrt(np.mean(data ** 2))


def views(waveform, fs):
    _, _, spectrum = stft(waveform, fs=fs, nperseg=256, noverlap=192,
                          boundary=None, padded=False)
    amplitude = np.abs(spectrum)
    envelope = uniform_filter1d(np.abs(hilbert(waveform)), size=round(fs * .010))
    return {"spectrogram": amplitude.ravel(), "spectrum": amplitude.mean(axis=1),
            "envelope": envelope}


def run():
    rng = np.random.default_rng(SEED)
    rows = []
    inventory = []
    for path in sorted((HERE / "audio").glob("*.wav")):
        call = path.stem.rsplit("_", 1)[-1]
        fs, clean = load_pcm(path)
        reference = views(clean, fs)
        freq, power = welch(clean, fs, nperseg=256)
        filters = {band: butter(4, limits, btype="band", fs=fs, output="sos")
                   for band, limits in BANDS.items()}
        for band, (lo, hi) in BANDS.items():
            fraction = power[(freq >= lo) & (freq <= hi)].sum() / power.sum()
            inventory.append((call, fs, len(clean), band, fraction))
        for draw in range(DRAWS):
            # The same white-noise realization is filtered into both frequency bands.
            white = rng.normal(size=len(clean))
            for band, filt in filters.items():
                noise = sosfiltfilt(filt, white)
                noise -= noise.mean()
                noise /= np.sqrt(np.mean(noise ** 2))
                for snr in SNR_DB:
                    received = clean + noise * 10 ** (-snr / 20)
                    for name, view in views(received, fs).items():
                        similarity = np.corrcoef(reference[name], view)[0, 1]
                        rows.append((call, draw, band, snr, name, similarity))
    assert len(inventory) == 6 and len(rows) == 3 * DRAWS * 2 * len(SNR_DB) * 3
    out = HERE / "results"
    out.mkdir(exist_ok=True)
    series = pd.DataFrame(rows, columns=["call", "draw", "band_hz", "snr_db", "view", "correlation"])
    series.to_csv(out / "audio_draws.csv", index=False, float_format="%.12g")
    summary = series.groupby(["call", "band_hz", "snr_db", "view"]).correlation.agg(
        median="median", p10=lambda x: x.quantile(.1), p90=lambda x: x.quantile(.9))
    summary.to_csv(out / "audio_summary.csv", float_format="%.12g")
    pd.DataFrame(inventory, columns=["call", "sample_rate_hz", "samples", "band_hz",
                                     "original_power_fraction"]).to_csv(
        out / "audio_sources.csv", index=False, float_format="%.12g")


if __name__ == "__main__":
    run()
