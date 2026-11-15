"""
ml/explain.py
Generates SHAP interpretability reports from the trained MSHAN model.
Uses shap.GradientExplainer for full TensorFlow 2.x and GRU compatibility.
"""
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import shap
import tensorflow as tf

from ml import config
from ml.dataset import load_split

REPORTS_DIR = Path(config.REPORTS_DIR)
REPORTS_DIR.mkdir(parents=True, exist_ok=True)


def run_explainability(n_background=50, n_explain=30):
    model_path = Path(config.MODEL_DIR) / "best_mshan_model.keras"
    if not model_path.exists():
        raise FileNotFoundError(f"Model checkpoint not found at: {model_path}")

    print(f"[*] Loading trained MSHAN model from: {model_path}")
    model = tf.keras.models.load_model(str(model_path))

    print("[*] Loading sample test beats from DS2...")
    X_test, y_test = load_split(config.DS2_TEST_RECORDS[:3])

    # Sample background distribution & evaluation batch
    np.random.seed(config.RANDOM_SEED)
    bg_idx = np.random.choice(len(X_test), n_background, replace=False)
    exp_idx = np.random.choice(len(X_test), n_explain, replace=False)

    background = X_test[bg_idx]
    samples_to_explain = X_test[exp_idx]

    print("[*] Computing SHAP values using GradientExplainer (TF2 / GRU compatible)...")
    # Use GradientExplainer to bypass TensorListStack GRU gradient overrides
    explainer = shap.GradientExplainer(model, background)
    shap_values = explainer.shap_values(samples_to_explain)

    # Handle shape format across SHAP versions: list of (N, 360, 1) or ndarray (N, 360, 1, 5)
    if isinstance(shap_values, list):
        # shap_values is a list of length 5 (one array per class)
        shap_by_class = [np.squeeze(s) for s in shap_values]
    else:
        # Array shape: (n_explain, 360, 1, 5) -> transpose to list per class
        shap_by_class = [np.squeeze(shap_values[..., i]) for i in range(config.NUM_CLASSES)]

    time_steps = np.arange(config.WINDOW_SIZE)

    # 1. Global Feature Attribution Plot
    plt.figure(figsize=(11, 5))
    for i, cls_name in enumerate(config.CLASSES):
        mean_abs = np.mean(np.abs(shap_by_class[i]), axis=0)
        plt.plot(time_steps, mean_abs, label=f"Class {cls_name}", lw=1.8)

    plt.axvline(x=180, color='red', linestyle='--', label='R-Peak Center (Index 180)')
    plt.title("MSHAN Global Temporal Feature Attribution (SHAP)", fontsize=13)
    plt.xlabel("Sample Index (-180 to +180 relative to R-peak)", fontsize=11)
    plt.ylabel("Mean |SHAP Value|", fontsize=11)
    plt.legend(loc="upper right")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()

    global_path = REPORTS_DIR / "shap_global_summary.png"
    plt.savefig(global_path, dpi=300)
    plt.close()
    print(f"[+] Saved global SHAP summary to: {global_path}")

    # 2. Local Beat Waveform Attribution Overlay
    sample_id = 0
    raw_signal = samples_to_explain[sample_id].squeeze()
    pred_probs = model.predict(samples_to_explain[sample_id:sample_id+1], verbose=0)
    pred_idx = int(np.argmax(pred_probs))
    true_cls = config.CLASSES[y_test[exp_idx[sample_id]]]
    pred_cls = config.CLASSES[pred_idx]

    local_shap = shap_by_class[pred_idx][sample_id]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 6), sharex=True)
    ax1.plot(time_steps, raw_signal, color='black', lw=1.5, label='Normalized ECG')
    ax1.set_title(f"ECG Heartbeat (True: {true_cls} | Predicted: {pred_cls})", fontsize=12)
    ax1.set_ylabel("Amplitude (Z-score)")
    ax1.grid(True, linestyle="--", alpha=0.5)
    ax1.legend(loc="upper right")

    colors = ['crimson' if val > 0 else 'royalblue' for val in local_shap]
    ax2.bar(time_steps, local_shap, color=colors, width=1.0)
    ax2.set_title(f"Local SHAP Attributions for Predicted Class ({pred_cls})", fontsize=12)
    ax2.set_xlabel("Sample Index (R-Peak at 180)")
    ax2.set_ylabel("SHAP Attribution")
    ax2.grid(True, linestyle="--", alpha=0.5)

    plt.tight_layout()
    local_path = REPORTS_DIR / "shap_local_overlay.png"
    plt.savefig(local_path, dpi=300)
    plt.close()
    print(f"[+] Saved local SHAP overlay to: {local_path}")


if __name__ == "__main__":
    run_explainability()