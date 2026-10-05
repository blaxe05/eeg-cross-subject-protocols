from __future__ import annotations

import math

import numpy as np
from scipy.signal import welch


DE_BANDS: dict[str, tuple[float, float]] = {
    "delta": (1.0, 4.0),
    "theta": (4.0, 8.0),
    "alpha": (8.0, 14.0),
    "beta": (14.0, 31.0),
    "gamma": (31.0, 50.0),
}


class DifferentialEntropyExtractor:
    """Stateless per-window Gaussian DE from Welch band power."""

    def __init__(self, sampling_rate: float, window_seconds: float = 1.0, epsilon: float = 1e-12):
        self.sampling_rate = float(sampling_rate)
        self.window_seconds = float(window_seconds)
        self.epsilon = float(epsilon)
        self.window_samples = int(round(self.sampling_rate * self.window_seconds))
        if self.window_samples < 2:
            raise ValueError("DE window must contain at least two samples")
        if max(high for _, high in DE_BANDS.values()) > self.sampling_rate / 2:
            raise ValueError("Sampling rate is too low for the configured gamma band")

    def transform_trial(self, eeg: np.ndarray) -> np.ndarray:
        eeg = np.asarray(eeg, dtype=np.float64)
        if eeg.ndim != 2:
            raise ValueError("EEG trial must have shape (channels, time)")
        if not np.isfinite(eeg).all():
            raise ValueError("EEG trial contains NaN or infinite values")
        n_windows = eeg.shape[1] // self.window_samples
        if n_windows == 0:
            return np.empty((0, eeg.shape[0], len(DE_BANDS)), dtype=np.float64)
        windows = eeg[:, : n_windows * self.window_samples]
        windows = windows.reshape(eeg.shape[0], n_windows, self.window_samples).transpose(1, 0, 2)
        frequencies, psd = welch(
            windows,
            fs=self.sampling_rate,
            nperseg=self.window_samples,
            axis=-1,
            detrend="constant",
            scaling="density",
        )
        features: list[np.ndarray] = []
        for low, high in DE_BANDS.values():
            mask = (frequencies >= low) & (frequencies < high)
            if not np.any(mask):
                raise ValueError(f"No FFT bins available for band [{low}, {high})")
            band_power = np.trapz(psd[..., mask], frequencies[mask], axis=-1)
            de = 0.5 * np.log(2.0 * math.pi * math.e * np.maximum(band_power, self.epsilon))
            features.append(de)
        return np.stack(features, axis=-1)
