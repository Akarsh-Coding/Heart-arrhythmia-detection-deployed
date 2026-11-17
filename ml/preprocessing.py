"""Shared training and serving preprocessing."""
import numpy as np
from ml import config


def normalize_beat(signal):
    arr = np.asarray(signal, dtype=np.float64)
    if arr.shape != (config.WINDOW_SIZE,):
        raise ValueError(f'Expected {config.WINDOW_SIZE} samples.')
    if not np.isfinite(arr).all():
        raise ValueError('Signal samples must be finite.')
    centered = arr - arr.mean()
    std = arr.std()
    return (centered / (std if std > 1e-6 else 1.0)).astype(np.float32)
