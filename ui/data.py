"""Strict input parsing and real annotated demo beats, without Streamlit state."""
import numpy as np
import pandas as pd
from ml import config
from ml.preprocessing import normalize_beat


def read_csv_frame(source, has_header=False):
    frame = pd.read_csv(source, header=0 if has_header else None, skip_blank_lines=False)
    if frame.empty:
        raise ValueError('CSV contains no heartbeat data.')
    return frame


def numeric_values(frame):
    try:
        values = frame.apply(pd.to_numeric, errors='raise').to_numpy(dtype=np.float64)
    except (ValueError, TypeError) as exc:
        raise ValueError('Selected signal cells must all be numeric; check headers and columns.') from exc
    if not np.isfinite(values).all():
        raise ValueError('Signal contains missing, NaN, or infinite values. Correct these cells before uploading.')
    return values


def single_beat(frame, column=None):
    if column is not None:
        frame = frame[[column]]
    if frame.shape not in {(1, config.WINDOW_SIZE), (config.WINDOW_SIZE, 1)}:
        raise ValueError('Select exactly one 360-sample column or upload one row of 360 samples.')
    return normalize_beat(numeric_values(frame).reshape(config.WINDOW_SIZE))


def batch_beats(frame, label_column=None):
    if label_column is not None:
        frame = frame.drop(columns=[label_column])
    if frame.empty or frame.shape[1] != config.WINDOW_SIZE:
        raise ValueError('Each heartbeat row must have exactly 360 signal columns. Select a label column explicitly if present.')
    values = numeric_values(frame)
    return np.stack([normalize_beat(row) for row in values])


def continuous_beats(frame, column, sampling_rate, sample_column=None):
    """Detect on a standardized copy; segment original amplitudes like training."""
    from wfdb.processing import xqrs_detect
    if sampling_rate != config.SAMPLING_RATE:
        raise ValueError('This model currently supports 360 Hz recordings only. Do not relabel a different sampling rate as 360 Hz.')
    signal = numeric_values(frame[[column]]).ravel()
    if len(signal) < 5 * sampling_rate:
        raise ValueError('Provide at least 5 seconds of continuous ECG for peak detection.')
    if sample_column is not None:
        indices = numeric_values(frame[[sample_column]]).ravel()
        if not np.all(np.diff(indices) == 1):
            raise ValueError('Sample indices must increase by one without gaps or duplicates.')
    if signal.std() <= 1e-6:
        raise ValueError('The selected signal is flat; no heartbeats can be detected.')
    detection_signal = (signal - signal.mean()) / signal.std()
    peaks = np.asarray(xqrs_detect(detection_signal, fs=sampling_rate, verbose=False), dtype=int)
    half = config.WINDOW_SIZE // 2
    valid = peaks[(peaks >= half) & (peaks + half <= len(signal))]
    if not len(valid):
        raise ValueError('No complete heartbeats detected. Check the lead, sampling rate, and signal quality.')
    beats = np.stack([normalize_beat(signal[p-half:p+half]) for p in valid])
    return signal, valid, beats, int(len(peaks) - len(valid))


def annotated_example(class_code):
    from ml.dataset import load_record_beats
    if class_code not in config.CLASS_TO_INDEX:
        raise ValueError('Unknown class.')
    for record in config.TRAIN_RECORDS:
        beats, labels = load_record_beats(record)
        matches = np.flatnonzero(labels == config.CLASS_TO_INDEX[class_code])
        if len(matches):
            return beats[matches[0]], record
    raise ValueError(f'No annotated example available for {class_code}.')
