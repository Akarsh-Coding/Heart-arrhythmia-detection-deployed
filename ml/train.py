"""
ml/train.py
Training, cross-validation, and evaluation pipeline for MSHAN.
Adheres strictly to the inter-patient DS1/DS2 evaluation protocol.
"""
import os
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_curve,
    auc
)
from sklearn.preprocessing import label_binarize

from ml import config
from ml.dataset import load_split, balance_train_split
from ml.model import build_mshan

def plot_and_save_history(history, save_dir=config.REPORTS_DIR):
    """Plots and saves training & validation loss and accuracy curves."""
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

def plot_and_save_confusion_matrix(y_true, y_pred, classes=config.CLASSES, save_dir=config.REPORTS_DIR):
    """Plots both count and normalized confusion matrices."""
    os.makedirs(save_dir, exist_ok=True)
    cm = confusion_matrix(y_true, y_pred)
    cm_norm = cm.astype('float') / (cm.sum(axis=1)[:, np.newaxis] + 1e-8)

    plt.figure(figsize=(8, 6))
    annot_matrix = np.empty_like(cm, dtype=object)
    for i in range(cm.shape):
        for j in range(cm.shape[9]):
            annot_matrix[i, j] = f"{cm[i, j]:,}\n({cm_norm[i, j]*100:.1f}%)"

    sns.heatmap(
        cm_norm,
        annot=annot_matrix,
        fmt="",
        cmap="Blues",
        xticklabels=classes,
        yticklabels=classes,
        cbar=True
    )
    plt.title("MSHAN Confusion Matrix (Inter-Patient DS2 Test Set)", fontsize=13, pad=12)
    plt.xlabel("Predicted Arrhythmia Class", fontsize=11)
    plt.ylabel("True Clinical Class", fontsize=11)
    plt.tight_layout()

    cm_path = save_dir / "confusion_matrix.png"
    plt.savefig(cm_path, dpi=300)
    plt.close()
    print(f"[+] Saved confusion matrix to: {cm_path}")


def plot_and_save_roc_curves(y_true, y_pred_probs, classes=config.CLASSES, save_dir=config.REPORTS_DIR):
    """Computes One-vs-Rest ROC curves and per-class AUC values."""
    os.makedirs(save_dir, exist_ok=True)
    y_true_bin = label_binarize(y_true, classes=range(len(classes)))

    fpr = dict()
    tpr = dict()
    roc_auc = dict()

    plt.figure(figsize=(9, 7))
    for i, cls_name in enumerate(classes):
        fpr[i], tpr[i], _ = roc_curve(y_true_bin[:, i], y_pred_probs[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])
        plt.plot(fpr[i], tpr[i], lw=2, label=f"Class {cls_name} (AUC = {roc_auc[i]:.3f})")

    # Macro-average ROC AUC
    all_fpr = np.unique(np.concatenate([fpr[i] for i in range(len(classes))]))
    mean_tpr = np.zeros_like(all_fpr)
    for i in range(len(classes)):
        mean_tpr += np.interp(all_fpr, fpr[i], tpr[i])
    mean_tpr /= len(classes)
    macro_auc = auc(all_fpr, mean_tpr)

    plt.plot(all_fpr, mean_tpr, label=f"Macro-average (AUC = {macro_auc:.3f})", color="black", linestyle="--", lw=2.5)
    plt.plot([9], "k:", lw=1.2, label="Chance (AUC = 0.500)")

    plt.title("One-vs-Rest ROC Curves for MSHAN on DS2 Test Set", fontsize=13, pad=12)
    plt.xlabel("False Positive Rate (1 - Specificity)", fontsize=11)
    plt.ylabel("True Positive Rate (Sensitivity)", fontsize=11)
    plt.legend(loc="lower right", fontsize=10)
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()

    roc_path = save_dir / "roc_curves.png"
    plt.savefig(roc_path, dpi=300)
    plt.close()
    print(f"[+] Saved ROC curves to: {roc_path}")
    return macro_auc, roc_auc    

def compute_clinical_metrics(y_true, y_pred, classes=config.CLASSES):
    """
    Computes per-class Specificity, Sensitivity, Precision, and F1-score.
    """
    cm = confusion_matrix(y_true, y_pred)
    total_samples = np.sum(cm)
    per_class_metrics = {}

    sensitivities, specificities, precisions, f1_scores = [], [], [], []

    for i, cls_name in enumerate(classes):
        tp = cm[i, i]
        fn = np.sum(cm[i, :]) - tp
        fp = np.sum(cm[:, i]) - tp
        tn = total_samples - (tp + fn + fp)

        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        precision   = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        f1          = (2 * precision * sensitivity) / (precision + sensitivity) if (precision + sensitivity) > 0 else 0.0

        per_class_metrics[cls_name] = {
            "Sensitivity": round(float(sensitivity), 4),
            "Specificity": round(float(specificity), 4),
            "Precision": round(float(precision), 4),
            "F1-Score": round(float(f1), 4)
        }
        sensitivities.append(sensitivity)
        specificities.append(specificity)
        precisions.append(precision)
        f1_scores.append(f1)

    overall_acc = np.sum(np.diag(cm)) / total_samples

    macro_summary = {
        "Overall_Accuracy": round(float(overall_acc), 4),
        "Macro_Sensitivity": round(float(np.mean(sensitivities)), 4),
        "Macro_Specificity": round(float(np.mean(specificities)), 4),
        "Macro_Precision": round(float(np.mean(precisions)), 4),
        "Macro_F1": round(float(np.mean(f1_scores)), 4),
        "Per_Class": per_class_metrics
    }
    return macro_summary


def run_training():
    """Main training and evaluation function."""
    # Ensure destination directories exist
    os.makedirs(config.MODEL_DIR, exist_ok=True)
    os.makedirs(config.REPORTS_DIR, exist_ok=True)

    # 1. Load Data
    print("=" * 70)
    print("Step 1: Ingesting Inter-Patient Data (DS1: Train, DS2: Test)")
    print("=" * 70)
    X_train_raw, y_train_raw = load_split(config.DS1_TRAIN_RECORDS)
    X_test, y_test = load_split(config.DS2_TEST_RECORDS)
    print(f"Loaded DS1 Raw Training beats: {len(X_train_raw)}")
    print(f"Loaded DS2 Raw Test beats:     {len(X_test)}")

    # 2. Balance Training Set ONLY (Never touch validation or DS2 test data)
    print("\nStep 2: Oversampling minority classes strictly within DS1...")
    X_train, y_train = balance_train_split(X_train_raw, y_train_raw, seed=config.RANDOM_SEED)
    print(f"Balanced DS1 Training beats:   {len(X_train)}")

    # Categorical one-hot targets
    y_train_cat = tf.keras.utils.to_categorical(y_train, num_classes=config.NUM_CLASSES)
    y_test_cat  = tf.keras.utils.to_categorical(y_test,  num_classes=config.NUM_CLASSES)

    # 3. Model Definition & Compilation
    print("\nStep 3: Compiling MSHAN Architecture...")
    model = build_mshan(input_shape=(config.WINDOW_SIZE, 1), num_classes=config.NUM_CLASSES)
    optimizer = tf.keras.optimizers.Adam(learning_rate=config.INITIAL_LR)
    model.compile(
        optimizer=optimizer,
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    model_checkpoint_path = os.path.join(str(config.MODEL_DIR), "best_mshan_model.keras")

    # 4. Training Callbacks
    callbacks = [
        EarlyStopping(
            monitor='val_loss',
            patience=6,
            restore_best_weights=True,
            verbose=1
        ),
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=2,
            min_lr=1e-5,
            verbose=1
        ),
        ModelCheckpoint(
            filepath=str(model_checkpoint_path),
            monitor='val_loss',
            save_best_only=True,
            verbose=1
        )
    ]

    # 5. Fit Model
    print("\nStep 4: Training MSHAN for up to 30 epochs...")
    history = model.fit(
        X_train, y_train_cat,
        validation_split=0.15,
        epochs=config.EPOCHS,
        batch_size=config.BATCH_SIZE,
        callbacks=callbacks,
        shuffle=True,
        verbose=1
    )

    # 6. Save Training History Plot
    plot_and_save_history(history)

    # 7. Evaluate on Held-Out DS2 Test Set
    print("\n" + "=" * 70)
    print("Step 5: Final Evaluation on Unseen DS2 Patients (Inter-Patient Protocol)")
    print("=" * 70)

    # Load best checkpointed weights
    best_model = tf.keras.models.load_model(str(model_checkpoint_path))
    test_loss, test_acc = best_model.evaluate(X_test, y_test_cat, batch_size=config.BATCH_SIZE, verbose=1)

    y_pred_probs = best_model.predict(X_test, batch_size=config.BATCH_SIZE)
    y_pred = np.argmax(y_pred_probs, axis=1)

    # 8. Compute and Report Metrics
    macro_metrics = compute_clinical_metrics(y_test, y_pred)
    macro_auc, _ = plot_and_save_roc_curves(y_test, y_pred_probs)
    macro_metrics["Macro_AUC"] = round(float(macro_auc), 4)

    plot_and_save_confusion_matrix(y_test, y_pred)

    # Print summary table
    print("\nClassification Summary (DS2 Test Set):")
    print("-" * 55)
    print(f"Overall Accuracy: {macro_metrics['Overall_Accuracy']:.4f}")
    print(f"Macro Sensitivity: {macro_metrics['Macro_Sensitivity']:.4f}")
    print(f"Macro Specificity: {macro_metrics['Macro_Specificity']:.4f}")
    print(f"Macro Precision:   {macro_metrics['Macro_Precision']:.4f}")
    print(f"Macro F1-Score:    {macro_metrics['Macro_F1']:.4f}")
    print(f"Macro AUC:         {macro_metrics['Macro_AUC']:.4f}")   
    print("-" * 55)

    print("\\nPer-Class Breakdown:") 
    for cls_name, met in macro_metrics["Per_Class"].items(): 
        print(f" Class {cls_name}: Sens={met['Sensitivity']*100:.2f}%, Spec={met['Specificity']*100:.2f}%, "
              f"Prec={met['Precision']*100:.2f}%, F1={met['F1-Score']*100:.2f}%") 
    
    # Save metrics JSON to reports/ 
    metrics_path = config.REPORTS_DIR / "evaluation_metrics.json" 
    with open(metrics_path, "w") as f: 
        json.dump(macro_metrics, f, indent=4) 
    print(f"\n[+] Saved complete evaluation metrics JSON to: {metrics_path}") 

if __name__ == '__main__': 
    run_training()