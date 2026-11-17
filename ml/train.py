"""
ml/train.py
Training, cross-validation, and evaluation pipeline for MSHAN.
Adheres strictly to the inter-patient DS1/DS2 evaluation protocol.
"""
import os
from pathlib import Path
from sklearn.model_selection import StratifiedKFold
import json
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

from ml import config
from ml.dataset import load_split, balance_train_split
from ml.model import build_mshan

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

def training_folds(y):
    """Assign original beats to folds before any duplication."""
    counts = np.bincount(y, minlength=config.NUM_CLASSES)
    if np.any(counts < config.CV_FOLDS):
        raise ValueError("Each class needs at least five original beats for cross-validation.")
    return StratifiedKFold(config.CV_FOLDS, shuffle=True,
                          random_state=config.RANDOM_SEED).split(np.zeros(len(y)), y)


def compile_model():
    model = build_mshan()
    model.compile(optimizer=tf.keras.optimizers.Adam(config.INITIAL_LR),
                  loss="categorical_crossentropy", metrics=["accuracy"])
    return model


def run_training():
    from ml.artifacts import write_metadata
    from ml.evaluate import evaluate_model
    config.MODEL_DIR.mkdir(parents=True, exist_ok=True)
    config.REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    tf.keras.utils.set_random_seed(config.RANDOM_SEED)
    X, y = load_split(config.DS1_TRAIN_RECORDS)
    print("Original DS1 class counts:", dict(zip(config.CLASSES, np.bincount(y).tolist())))
    fold_results = []
    for fold, (train_idx, val_idx) in enumerate(training_folds(y), 1):
        tf.keras.backend.clear_session()
        tf.keras.utils.set_random_seed(config.RANDOM_SEED + fold)
        X_train, y_train = balance_train_split(X[train_idx], y[train_idx])
        model = compile_model()
        history = model.fit(
            X_train, tf.keras.utils.to_categorical(y_train, config.NUM_CLASSES),
            validation_data=(X[val_idx], tf.keras.utils.to_categorical(y[val_idx], config.NUM_CLASSES)),
            epochs=config.EPOCHS, batch_size=config.BATCH_SIZE, shuffle=True,
            callbacks=[EarlyStopping(monitor="val_loss", patience=6, restore_best_weights=True),
                       ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=2, min_lr=1e-5)],
            verbose=2)
        best_epoch = int(np.argmin(history.history["val_loss"])) + 1
        fold_results.append({"fold": fold, "best_epoch": best_epoch,
                             "val_loss": float(min(history.history["val_loss"]))})
        plot_and_save_history(history, config.REPORTS_DIR / f"fold_{fold}")
        del model, X_train, y_train
    # Select training duration using DS1 only; refit on all DS1, then evaluate DS2 once.
    final_epochs = max(1, int(np.median([r["best_epoch"] for r in fold_results])))
    with open(config.REPORTS_DIR / "cross_validation.json", "w") as f:
        json.dump({"folds": fold_results, "final_epochs": final_epochs,
                   "split": "stratified original beats within DS1"}, f, indent=2, allow_nan=False)
    tf.keras.backend.clear_session()
    tf.keras.utils.set_random_seed(config.RANDOM_SEED)
    model = compile_model()
    X_train, y_train = balance_train_split(X, y)
    model.fit(X_train, tf.keras.utils.to_categorical(y_train, config.NUM_CLASSES),
              epochs=final_epochs, batch_size=config.BATCH_SIZE, shuffle=True, verbose=2)
    model_path = config.MODEL_DIR / "best_mshan_model.keras"
    model.save(model_path)
    write_metadata(model_path)
    evaluate_model(model)


if __name__ == '__main__':
    run_training()
