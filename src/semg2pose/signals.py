"""EMG filtering and envelope helpers. Arrays are (T, channels) along axis 0."""

import numpy as np
from scipy import signal

from semg2pose.data import FS


def bandpass(x, lo, hi, fs=FS, order=4):
    """Zero-phase Butterworth band-pass between lo and hi Hz."""
    sos = signal.butter(order, [lo, hi], btype="bandpass", fs=fs, output="sos")
    return signal.sosfiltfilt(sos, x, axis=0)


def rms_envelope(x, window_s=0.2, fs=FS):
    """Moving RMS: square root of the mean of x^2 over a centred sliding window."""
    n = int(window_s * fs)
    kernel = np.ones(n) / n
    power = signal.fftconvolve(x ** 2, kernel[:, None], mode="same", axes=0)
    return np.sqrt(np.clip(power, 0, None))  # clip: the convolution can give -1e-12 from rounding


def log_envelope(emg, band=(20, 450), window_s=0.05, fs=FS):
    """Band-pass, moving RMS, then log(1 + x): a standard sEMG amplitude feature."""
    return np.log(rms_envelope(bandpass(emg, *band, fs=fs), window_s, fs) + 1)
