import numpy as np
from scipy.signal import butter, filtfilt


def normalize_signal(signal):

    mean = np.mean(signal)
    std = np.std(signal)

    if std == 0:
        return signal

    return (signal - mean) / std


def bandpass_filter(signal, fps, lowcut=0.7, highcut=4.0):

    nyquist = 0.5 * fps

    low = lowcut / nyquist
    high = highcut / nyquist

    b, a = butter(
        3,
        [low, high],
        btype="band"
    )

    filtered = filtfilt(b, a, signal)

    return filtered