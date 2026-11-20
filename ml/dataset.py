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

def load_split(record_list, return_groups=False):
    all_beats, all_labels, all_groups = [], [], []
    
    for record_id in record_list:
        beats, labels = load_record_beats(record_id)
        all_beats.append(beats)
        all_labels.append(labels)
        all_groups.extend([patient_id(record_id)] * len(labels))
    if not all_beats or not any(len(beats) for beats in all_beats):
        raise ValueError(f"No valid heartbeat segments found for records: {record_list}")
    X = np.concatenate(all_beats, axis=0)
    y = np.concatenate(all_labels, axis=0)
    result = (np.expand_dims(X, axis=-1), y)
    return (*result, np.asarray(all_groups)) if return_groups else result


def patient_id(record_id):
    return '201_202' if str(record_id) in {'201', '202'} else str(record_id)


def stratified_sample_indices(labels, size, seed=config.RANDOM_SEED):
    """Sample every class without replacement for explainability, not evaluation."""
    labels = np.asarray(labels)
    if size < config.NUM_CLASSES or size > len(labels):
        raise ValueError('Sample size must cover all classes and not exceed available beats.')
    rng = np.random.default_rng(seed)
    pools = [rng.permutation(np.flatnonzero(labels == i)).tolist() for i in range(config.NUM_CLASSES)]
    if any(not pool for pool in pools):
        raise ValueError('Explainability sampling requires every configured class.')
    selected = []
    while len(selected) < size:
        for pool in pools:
            if pool and len(selected) < size:
                selected.append(pool.pop())
    return np.asarray(selected)

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
