import numpy as np
from scipy.signal import find_peaks


def calculate_heart_rate(signal, fps):

    if len(signal) < fps * 5:
        return None

    peaks, _ = find_peaks(
        signal,
        distance=int(fps * 0.4)
    )

    if len(peaks) < 2:
        return None

    intervals = np.diff(peaks) / fps

    heart_rate = 60 / np.mean(intervals)

    return heart_rate


def extract_features(rppg_signal, fps):

    heart_rate = calculate_heart_rate(
        rppg_signal,
        fps
    )

    features = {
        "heart_rate": heart_rate,
        "signal_mean": np.mean(rppg_signal),
        "signal_std": np.std(rppg_signal),
        "signal_min": np.min(rppg_signal),
        "signal_max": np.max(rppg_signal)
    }

    return features