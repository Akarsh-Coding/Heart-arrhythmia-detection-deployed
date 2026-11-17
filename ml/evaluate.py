"""
ml/evaluate.py
Standalone evaluation script to generate all performance reports from a trained model.
Runs strictly on the held-out inter-patient DS2 test set.
"""
import os
import json
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from sklearn.metrics import confusion_matrix, roc_curve, auc
from sklearn.preprocessing import label_binarize

from ml import config
from ml.dataset import load_split
from ml.artifacts import validate_metadata

# Ensure reports directory exists
REPORTS_DIR = Path(config.REPORTS_DIR)
REPORTS_DIR.mkdir(parents=True, exist_ok=True)


def plot_confusion_matrix(y_true, y_pred, classes=config.CLASSES):
    cm = confusion_matrix(y_true, y_pred, labels=range(len(classes)))
    cm_norm = cm.astype('float') / (cm.sum(axis=1)[:, np.newaxis] + 1e-8)

    plt.figure(figsize=(8, 6))
    annot_matrix = np.empty_like(cm, dtype=object)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
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
    plt.title("MSHAN Confusion Matrix (Held-Out DS2 Test Set)", fontsize=13, pad=12)
    plt.xlabel("Predicted Arrhythmia Class", fontsize=11)
    plt.ylabel("True Clinical Class", fontsize=11)
    plt.tight_layout()

    out_path = REPORTS_DIR / "confusion_matrix.png"
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"[+] Saved confusion matrix to: {out_path}")


def plot_roc_curves(y_true, y_pred_probs, classes=config.CLASSES):
    if set(np.unique(y_true)) != set(range(len(classes))):
        raise ValueError("ROC-AUC requires test examples from every configured class.")
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
    plt.plot([0, 1], [0, 1], "k:", lw=1.2, label="Chance (AUC = 0.500)")

    plt.title("One-vs-Rest ROC Curves (MSHAN DS2 Evaluation)", fontsize=13, pad=12)
    plt.xlabel("False Positive Rate (1 - Specificity)", fontsize=11)
    plt.ylabel("True Positive Rate (Sensitivity)", fontsize=11)
    plt.legend(loc="lower right", fontsize=10)
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()

    out_path = REPORTS_DIR / "roc_curves.png"
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"[+] Saved ROC curves to: {out_path}")
    return macro_auc


def compute_metrics(y_true, y_pred, classes=config.CLASSES):
    cm = confusion_matrix(y_true, y_pred, labels=range(len(classes)))
    total = np.sum(cm)
    if total == 0:
        raise ValueError("Cannot evaluate an empty test set.")
    per_class = {}
    sens_list, spec_list, prec_list, f1_list = [], [], [], []

    for i, cls_name in enumerate(classes):
        tp = cm[i, i]
        fn = np.sum(cm[i, :]) - tp
        fp = np.sum(cm[:, i]) - tp
        tn = total - (tp + fn + fp)

        sens = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        f1 = (2 * prec * sens) / (prec + sens) if (prec + sens) > 0 else 0.0

        per_class[cls_name] = {
            "Support": int(cm[i, :].sum()),
            "Sensitivity": round(float(sens), 4),
            "Specificity": round(float(spec), 4),
            "Precision": round(float(prec), 4),
            "F1-Score": round(float(f1), 4)
        }
        sens_list.append(sens)
        spec_list.append(spec)
        prec_list.append(prec)
        f1_list.append(f1)

    return {
        "Overall_Accuracy": round(float(np.sum(np.diag(cm)) / total), 4),
        "Macro_Sensitivity": round(float(np.mean(sens_list)), 4),
        "Macro_Specificity": round(float(np.mean(spec_list)), 4),
        "Macro_Precision": round(float(np.mean(prec_list)), 4),
        "Macro_F1": round(float(np.mean(f1_list)), 4),
        "Per_Class": per_class
    }


def main():
    model_path = Path(config.MODEL_DIR) / "best_mshan_model.keras"
    if not model_path.exists():
        raise FileNotFoundError(f"Model checkpoint not found at: {model_path}")

    print(f"[*] Loading trained model from: {model_path}")
    validate_metadata(model_path)
    model = tf.keras.models.load_model(str(model_path))
    evaluate_model(model)


def evaluate_model(model):

    print(f"[*] Loading DS2 held-out test records ({len(config.DS2_TEST_RECORDS)} patients)...")
    X_test, y_test = load_split(config.DS2_TEST_RECORDS)
    print(f"[+] Loaded {len(X_test)} test beats.")

    print("[*] Running inference on DS2 test set...")
    y_pred_probs = model.predict(X_test, batch_size=config.BATCH_SIZE, verbose=1)
    y_pred = np.argmax(y_pred_probs, axis=1)

    print("[*] Generating confusion matrix...")
    plot_confusion_matrix(y_test, y_pred)

    print("[*] Generating ROC curves...")
    macro_auc = plot_roc_curves(y_test, y_pred_probs)

    print("[*] Calculating clinical diagnostic metrics...")
    metrics = compute_metrics(y_test, y_pred)
    metrics["classes"] = config.CLASSES
    metrics["preprocessing"] = config.PREPROCESSING
    metrics["Macro_AUC"] = round(float(macro_auc), 4)

    # Save JSON summary
    metrics_path = REPORTS_DIR / "evaluation_metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=4, allow_nan=False)
    print(f"[+] Saved evaluation metrics JSON to: {metrics_path}")

    # Print summary to console
    print("\n" + "=" * 60)
    print("DS2 Inter-Patient Evaluation Summary:")
    print("=" * 60)
    print(f"Overall Accuracy:  {metrics['Overall_Accuracy'] * 100:.2f}%")
    print(f"Macro Sensitivity: {metrics['Macro_Sensitivity'] * 100:.2f}%")
    print(f"Macro Specificity: {metrics['Macro_Specificity'] * 100:.2f}%")
    print(f"Macro Precision:   {metrics['Macro_Precision'] * 100:.2f}%")
    print(f"Macro F1-Score:    {metrics['Macro_F1'] * 100:.2f}%")
    print(f"Macro ROC-AUC:     {metrics['Macro_AUC']:.4f}")
    print("=" * 60)


if __name__ == "__main__":
    main()