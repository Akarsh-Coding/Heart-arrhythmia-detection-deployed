"""
ml/dataset.py
Segmentation, normalization, and balanced loading.    
"""
import numpy as np
import wfdb
from ml import config
from ml.preprocessing import normalize_beat

def load_record_beats(record_id, data_dir=config.DATA_DIR):
    record_path = f"{data_dir}/{record_id}"
    record = wfdb.rdrecord(record_path)
    annotation = wfdb.rdann(record_path, 'atr')
    
    # Prioritize Modified Limb Lead II (MLII) 
    channel = 0
    if 'MLII' in record.sig_name:
        channel = record.sig_name.index('MLII')
    signal = record.p_signal[:, channel]
    
    half_window = config.WINDOW_SIZE // 2
    beats,labels = [], []
    
    for r_peak, symbol in zip(annotation.sample, annotation.symbol):
        if symbol in config.CLASS_TO_INDEX:
            start = r_peak - half_window
            end = r_peak + half_window
            
            if start >= 0 and end <= len(signal):
                beat_segment = normalize_beat(signal[start:end])
                beats.append(beat_segment)
                labels.append(config.CLASS_TO_INDEX[symbol])
    if len(beats) == 0: 
        return np.empty((0, config.WINDOW_SIZE), dtype=np.float32), np.empty((0,), dtype=np.int32)
    return np.array(beats, dtype=np.float32), np.array(labels, dtype=np.int32)

def load_split(record_list):
    all_beats, all_labels = [], []
    
    for record_id in record_list:
        beats, labels = load_record_beats(record_id)
        all_beats.append(beats)
        all_labels.append(labels)
    if not all_beats or not any(len(beats) for beats in all_beats):
        raise ValueError(f"No valid heartbeat segments found for records: {record_list}")
    X = np.concatenate(all_beats, axis=0)
    y = np.concatenate(all_labels, axis=0)
    return np.expand_dims(X, axis=-1), y  # Add channel dimension for CNN input

def balance_train_split(X_train, y_train , seed=config.RANDOM_SEED):
    """
    Preserve all original beats and add minority copies to the majority count.
    Apply exclusively to the training partition after validation assignment.
    """
    X_train, y_train = np.asarray(X_train), np.asarray(y_train)
    if y_train.ndim != 1 or len(X_train) != len(y_train):
        raise ValueError('Training inputs and one-dimensional labels must have equal lengths.')
    classes, counts = np.unique(y_train, return_counts=True)
    if not np.array_equal(classes, np.arange(config.NUM_CLASSES)):
        raise ValueError('Training partition must contain all configured classes.')
    max_count = np.max(counts)
    rng = np.random.default_rng(seed)
    indices = [np.arange(len(y_train))]
    for cls, count in zip(classes, counts):
        if count < max_count:
            indices.append(rng.choice(np.flatnonzero(y_train == cls),
                                      size=int(max_count - count), replace=True))
    selected = np.concatenate(indices)
    rng.shuffle(selected)
    return X_train[selected], y_train[selected]
