"""Bind checkpoint semantics to a specific model file."""
import hashlib
import json
from pathlib import Path
from ml import config


def model_metadata(path):
    return {'classes': config.CLASSES, 'preprocessing': config.PREPROCESSING,
            'window_size': config.WINDOW_SIZE,
            'sha256': hashlib.sha256(Path(path).read_bytes()).hexdigest()}


def write_metadata(path):
    Path(path).with_suffix('.json').write_text(json.dumps(model_metadata(path), indent=2))


def validate_metadata(path):
    path = Path(path)
    metadata_path = path.with_suffix('.json')
    if not path.exists() or not metadata_path.exists():
        raise ValueError('Compatible trained model missing. Run python -m ml.train first.')
    if json.loads(metadata_path.read_text()) != model_metadata(path):
        raise ValueError('Checkpoint labels, preprocessing or checksum do not match. Retrain the model.')
