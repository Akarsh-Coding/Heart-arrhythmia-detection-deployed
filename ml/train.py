"""
ml/train.py
Training, cross-validation, and evaluation pipeline for MSHAN.
Uses patient-grouped DS1 validation and excludes 201 to separate subject 201/202.
"""
import os
import hashlib
from types import SimpleNamespace
from pathlib import Path
from sklearn.model_selection import StratifiedGroupKFold
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Training writes reports without a desktop GUI.
import matplotlib.pyplot as plt
import tensorflow as tf

from ml import config
from ml.model import build_mshan
from ml.artifacts import write_metadata
from ml.dataset import balance_train_split, load_split, patient_id

def plot_and_save_history(history, save_dir=config.REPORTS_DIR):
    """Plots and saves training & validation loss and accuracy curves."""
    save_dir = Path(save_dir)
    os.makedirs(save_dir, exist_ok=True)
    epochs_range = range(1, len(history.history['loss']) + 1)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Loss plot
    ax1.plot(epochs_range, history.history['loss'], label='Training loss', color='royalblue', lw=2)
    ax1.plot(epochs_range, history.history['val_loss'], label='validation loss', color='crimson', lw=2)
    ax1.set_title('Categorical Cross-entropy Loss over Epochs', fontsize=12)
    ax1.set_xlabel('Epoch', fontsize=10)
    ax1.set_ylabel('Loss', fontsize=10)
    ax1.legend(loc='upper right')
    ax1.grid(True, linestyle='--', alpha=0.6)
    
    # Accuracy plot
    ax1_acc = history.history.get('accuracy', history.history.get('acc'))
    val_accuracy = history.history.get('val_accuracy', history.history.get('val_acc'))
    ax2.plot(epochs_range, ax1_acc, label='Training Accuracy', color='royalblue', lw=2)
    ax2.plot(epochs_range, val_accuracy, label='Validation Accuracy', color='crimson', lw=2)
    ax2.set_title('Accuracy over Epochs', fontsize=12)
    ax2.set_xlabel('Epoch', fontsize=10)
    ax2.set_ylabel('Accuracy', fontsize=10)
    ax2.legend(loc='lower right')
    ax2.grid(True, linestyle='--', alpha=0.6)
    
    plt.tight_layout()
    plot_path = save_dir / "training_history.png"
    plt.savefig(plot_path, dpi=300)
    plt.close()
    print(f"[+] Saved training & validation history curves to: {plot_path}")

def training_folds(y, groups):
    """Keep patients together and require every class on both sides of each fold."""
    y, groups = np.asarray(y), np.asarray(groups)
    if len(y) != len(groups):
        raise ValueError("Each beat needs a patient group.")
    for cls in range(config.NUM_CLASSES):
        if len(np.unique(groups[y == cls])) < config.CV_FOLDS:
            raise ValueError(f"Class {config.CLASSES[cls]} occurs in too few patients for {config.CV_FOLDS} folds.")
    # Select only on class coverage, never on model scores or DS2 outcomes.
    for seed in range(config.RANDOM_SEED, config.RANDOM_SEED + 100):
        splitter = StratifiedGroupKFold(config.CV_FOLDS, shuffle=True, random_state=seed)
        folds = list(splitter.split(np.zeros(len(y)), y, groups))
        if all(len(np.unique(y[t])) == config.NUM_CLASSES and
               len(np.unique(y[v])) == config.NUM_CLASSES for t, v in folds):
            return folds
    raise ValueError("Could not construct grouped folds with all classes; review patient coverage.")


def compile_model():
    model = build_mshan()
    model.compile(optimizer=tf.keras.optimizers.Adam(config.INITIAL_LR),
                  loss="categorical_crossentropy", metrics=["accuracy"])
    return model


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False))
    os.replace(temporary, path)


def final_schedule(folds):
    """Median DS1-selected duration and per-epoch rates; carry stopped folds forward."""
    epochs = max(1, int(np.median([f['best_epoch'] for f in folds])))
    return [float(np.median([f['history'][min(i, len(f['history']) - 1)]['learning_rate']
                            for f in folds])) for i in range(epochs)]


def fit_stage(directory, X, y, epochs, *, validation=None, schedule=None,
              seed=config.RANDOM_SEED, model_factory=compile_model, batch_size=config.BATCH_SIZE):
    """Commit model+optimizer first, then atomically commit epoch/controller history.

    An interrupted epoch is repeated. Completed epochs are never repeated.
    Resumption preserves optimizer/controller state, not bitwise random streams.
    """
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    state_path = directory / 'state.json'
    tf.keras.utils.set_random_seed(seed)
    if state_path.exists():
        state = json.loads(state_path.read_text())
        model = tf.keras.models.load_model(directory / state['checkpoint'])
    else:
        model = model_factory()
        state = {'history': [], 'best_loss': None, 'best_epoch': 0,
                 'early_wait': 0, 'lr_wait': 0, 'lr_best': None, 'stopped': False}
    for epoch in range(len(state['history']), epochs):
        if state['stopped']:
            break
        tf.keras.utils.set_random_seed(seed + epoch)
        if schedule is not None:
            model.optimizer.learning_rate.assign(schedule[epoch])
        rate = float(model.optimizer.learning_rate.numpy())
        history = model.fit(X, y, validation_data=validation, initial_epoch=epoch,
                            epochs=epoch + 1, batch_size=batch_size, shuffle=True, verbose=2)
        row = {key: float(values[-1]) for key, values in history.history.items()}
        row['learning_rate'] = rate
        if not all(np.isfinite(value) for value in row.values()):
            raise ValueError('Non-finite training metrics; last valid checkpoint retained.')
        state['history'].append(row)
        checkpoint = f'epoch_{epoch + 1:03d}.keras'
        if validation is not None:
            loss = row['val_loss']
            if state['best_loss'] is None or loss < state['best_loss']:
                state.update(best_loss=loss, best_epoch=epoch + 1, early_wait=0,
                             best_checkpoint=checkpoint)
            else:
                state['early_wait'] += 1
            if state['lr_best'] is None or loss < state['lr_best'] - 1e-4:
                state.update(lr_best=loss, lr_wait=0)
            else:
                state['lr_wait'] += 1
                if state['lr_wait'] >= 2:
                    model.optimizer.learning_rate.assign(max(rate * 0.5, 1e-5))
                    state['lr_wait'] = 0
            state['stopped'] = state['early_wait'] >= 6
        temporary = directory / 'pending.keras'
        model.save(temporary)
        os.replace(temporary, directory / checkpoint)
        state['checkpoint'] = checkpoint
        write_json(state_path, state)
    if validation is not None:
        model = tf.keras.models.load_model(directory / state['best_checkpoint'])
    return model, state


def run_training():
    from ml.evaluate import evaluate_model
    if {patient_id(r) for r in config.TRAIN_RECORDS} & {patient_id(r) for r in config.DS2_TEST_RECORDS}:
        raise ValueError("Training and test patient identities overlap.")
    X, y, groups = load_split(config.TRAIN_RECORDS, return_groups=True)
    split_indices = training_folds(y, groups)
    from ml import evaluate as evaluate_module
    evaluate_module.REPORTS_DIR = config.REPORTS_DIR
    config.REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    directory = config.MODEL_DIR / 'training'
    source_hash = hashlib.sha256()
    for filename in ['model.py', 'train.py', 'dataset.py', 'preprocessing.py', 'config.py']:
        source_hash.update(Path(__file__).with_name(filename).read_bytes())
    settings = {'classes': config.CLASSES, 'preprocessing': config.PREPROCESSING,
                'seed': config.RANDOM_SEED, 'batch_size': config.BATCH_SIZE,
                'epochs': config.EPOCHS, 'folds': config.CV_FOLDS,
                'initial_lr': config.INITIAL_LR, 'dropout': config.DROPOUT_RATE,
                'l2': config.L2_REG, 'window_size': config.WINDOW_SIZE,
                'ds1': config.TRAIN_RECORDS, 'ds2': config.DS2_TEST_RECORDS,
                'validation': 'patient-grouped DS1 folds; 201 excluded to separate subject 201/202',
                'tensorflow': tf.__version__, 'numpy': np.__version__,
                'source_sha256': source_hash.hexdigest(),
                'data_sha256': hashlib.sha256(X.tobytes() + y.tobytes() + groups.tobytes()).hexdigest()}
    manifest = directory / 'settings.json'
    if manifest.exists() and json.loads(manifest.read_text()) != settings:
        raise ValueError('Training settings/data/code changed. Use a new MODEL_DIR for a new run.')
    write_json(manifest, settings)
    folds = []
    for fold, (train, val) in enumerate(split_indices, 1):
        tf.keras.backend.clear_session()
        X_train, y_train = balance_train_split(X[train], y[train])
        model, state = fit_stage(directory / f'fold_{fold}', X_train,
                                 tf.keras.utils.to_categorical(y_train, config.NUM_CLASSES),
                                 config.EPOCHS,
                                 validation=(X[val], tf.keras.utils.to_categorical(y[val], config.NUM_CLASSES)),
                                 seed=config.RANDOM_SEED + fold)
        predictions = np.argmax(model.predict(X[val], batch_size=config.BATCH_SIZE, verbose=0), axis=1)
        from ml.evaluate import compute_metrics
        fold_report = {'metrics': compute_metrics(y[val], predictions),
                       'training_patients': np.unique(groups[train]).tolist(),
                       'validation_patients': np.unique(groups[val]).tolist()}
        write_json(config.REPORTS_DIR / f'fold_{fold}' / 'validation_metrics.json', fold_report)
        folds.append(state)
        plot_and_save_history(SimpleNamespace(history={
            key: [row[key] for row in state['history']] for key in state['history'][0]}),
            config.REPORTS_DIR / f'fold_{fold}')
        del model, X_train, y_train
    schedule = final_schedule(folds)
    write_json(config.REPORTS_DIR / 'cross_validation.json',
               {'folds': folds, 'final_epochs': len(schedule), 'final_learning_rates': schedule,
                'selection_rule': 'median best epoch; median per-epoch DS1 fold rates'})
    tf.keras.backend.clear_session()
    X_train, y_train = balance_train_split(X, y)
    model, state = fit_stage(directory / 'final', X_train,
                             tf.keras.utils.to_categorical(y_train, config.NUM_CLASSES),
                             len(schedule), schedule=schedule)
    write_json(config.REPORTS_DIR / 'final_history.json', state['history'])
    model_path = config.MODEL_DIR / 'best_mshan_model.keras'
    temporary = config.MODEL_DIR / 'pending.keras'
    model.save(temporary)
    os.replace(temporary, model_path)
    write_metadata(model_path)
    if not (directory / 'evaluation_complete.json').exists():
        evaluate_model(model)
        write_json(directory / 'evaluation_complete.json', {'complete': True})


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description="Patient-grouped MSHAN training")
    parser.add_argument('--run-name', default='nlrav_grouped_v2')
    args = parser.parse_args()
    if Path(args.run_name).name != args.run_name or args.run_name in {'.', '..'}:
        parser.error('run-name must be a single directory name')
    config.MODEL_DIR = config.ROOT_DIR / 'models' / args.run_name
    config.REPORTS_DIR = config.ROOT_DIR / 'reports' / args.run_name
    run_training()
