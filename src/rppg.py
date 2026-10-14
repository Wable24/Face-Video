import numpy as np


def extract_rppg(rgb_signal):

    if len(rgb_signal) == 0:
        return np.array([])

    rgb = rgb_signal.astype(float)

    # Normalize RGB channels
    rgb_mean = np.mean(rgb, axis=1, keepdims=True)

    normalized = rgb / (rgb_mean + 1e-8)

    r = normalized[:, 0]
    g = normalized[:, 1]
    b = normalized[:, 2]

    # Chrominance projection
    x = 3 * r - 2 * g
    y = 1.5 * r + g - 1.5 * b

    alpha = np.std(x) / (np.std(y) + 1e-8)

    pulse = x - alpha * y

    pulse = pulse - np.mean(pulse)

    return pulse