import asyncio
import json
from types import SimpleNamespace
import numpy as np
import pytest
from ml import config
from ml.preprocessing import normalize_beat
from ml.dataset import load_record_beats, balance_train_split
from ml.train import training_folds
from ml.evaluate import compute_metrics, plot_roc_curves
from ml.artifacts import write_metadata, validate_metadata
from app.schemas import HeartbeatInput
from app.main import preprocess_signal, lifespan, app, health_check


def test_segmentation_labels_and_serving_preprocessing(monkeypatch):
    signal = np.sin(np.arange(2400) / 9) + np.arange(2400) / 100
    monkeypatch.setattr('ml.dataset.wfdb.rdrecord', lambda _: SimpleNamespace(sig_name=['MLII'], p_signal=signal[:, None]))
    monkeypatch.setattr('ml.dataset.wfdb.rdann', lambda *args: SimpleNamespace(sample=[200,600,1000,1400,1800,2200], symbol=['N','L','R','A','V','F']))
    beats, labels = load_record_beats('fake')
    assert labels.tolist() == [0,1,2,3,4]
    np.testing.assert_allclose(beats.mean(axis=1), 0, atol=1e-6)
    np.testing.assert_allclose(beats.std(axis=1), 1, atol=1e-6)
    np.testing.assert_array_equal(beats[0], preprocess_signal(signal[20:380]).ravel())


def test_validation_never_contains_oversampled_training_beats():
    y = np.repeat(np.arange(5), [10,15,20,25,30])
    X = np.arange(len(y))[:, None]
    for train, val in training_folds(y):
        balanced, labels = balance_train_split(X[train], y[train])
        assert not set(balanced.ravel()) & set(X[val].ravel())
        assert len(set(np.bincount(labels))) == 1
        assert set(balanced.ravel()) == set(X[train].ravel())
        np.testing.assert_array_equal(labels, y[balanced.ravel()])
        majority = np.argmax(np.bincount(y[train], minlength=5))
        assert len(np.unique(balanced[labels == majority])) == sum(labels == majority)
        again, _ = balance_train_split(X[train], y[train])
        np.testing.assert_array_equal(balanced, again)


def test_balancing_equal_classes_preserves_each_beat_once():
    y = np.repeat(np.arange(5), 3)
    X = np.arange(len(y))[:, None]
    balanced, labels = balance_train_split(X, y)
    np.testing.assert_array_equal(np.sort(balanced.ravel()), X.ravel())
    np.testing.assert_array_equal(labels, y[balanced.ravel()])


def test_balancing_rejects_missing_or_invalid_labels():
    for y in [np.array([0, 1, 2, 3]), np.array([1, 2, 3, 4, 5])]:
        with pytest.raises(ValueError, match='all configured classes'):
            balance_train_split(np.arange(len(y))[:, None], y)
    with pytest.raises(ValueError, match='equal lengths'):
        balance_train_split(np.zeros((6, 360, 1)), np.arange(5))


def test_constant_and_invalid_signals():
    np.testing.assert_array_equal(normalize_beat(np.ones(360)), np.zeros(360))
    for value in [float('nan'),float('inf')]:
        with pytest.raises(ValueError):
            HeartbeatInput(signal=[value]*360)
    with pytest.raises(ValueError):
        HeartbeatInput(signal=[0]*359)


def test_missing_classes_metrics():
    metrics = compute_metrics(np.array([0,0]), np.array([0,0]))
    assert len(metrics['Per_Class']) == 5
    assert metrics['Per_Class']['L']['Support'] == 0
    json.dumps(metrics, allow_nan=False)
    with pytest.raises(ValueError, match='every configured class'):
        plot_roc_curves(np.array([0,0]), np.ones((2,5))/5)


def test_checkpoint_contract(tmp_path):
    path = tmp_path / 'model.keras'
    path.write_bytes(b'checkpoint')
    with pytest.raises(ValueError):
        validate_metadata(path)
    write_metadata(path)
    validate_metadata(path)
    path.write_bytes(b'other checkpoint')
    with pytest.raises(ValueError):
        validate_metadata(path)


def test_missing_model_is_degraded(monkeypatch, tmp_path):
    monkeypatch.setattr(config, 'MODEL_DIR', tmp_path)
    async def check():
        async with lifespan(app):
            assert health_check().status == 'degraded'
    asyncio.run(check())

def test_training_step_and_inference_contract():
    from ml.train import compile_model
    from app.main import ml_models, predict_heartbeat, predict_batch_heartbeats, explain_heartbeat
    from app.schemas import BatchHeartbeatInput
    model = compile_model()
    X = np.random.default_rng(42).normal(size=(5,360,1)).astype('float32')
    loss = model.train_on_batch(X, np.eye(5, dtype='float32'))
    assert np.isfinite(loss).all()
    ml_models['mshan'] = model
    try:
        beat = HeartbeatInput(signal=X[0,:,0].tolist())
        single = predict_heartbeat(beat)
        batch = predict_batch_heartbeats(BatchHeartbeatInput(heartbeats=[beat]))
        explanation = explain_heartbeat(beat)
        assert list(single.probabilities) == config.CLASSES
        assert batch.results[0].predicted_class == single.predicted_class
        assert explanation.attribution_method == 'input_x_gradient'
        assert len(explanation.attribution_values) == 360
        assert np.isfinite(explanation.attribution_values).all()
    finally:
        ml_models.clear()
