"""
app/main.py
FastAPI application serving the MSHAN ECG Arrhythmia Classifier.
"""
import os
from contextlib import asynccontextmanager
import numpy as np
import tensorflow as tf
from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from pathlib import Path

from ml import config
from app.schemas import (
    HeartbeatInput,
    BatchHeartbeatInput,
    PredictionResult,
    BatchPredictionResponse,
    ExplainResponse,
    HealthResponse
)

# Clinical mapping for human-readable diagnostic labels
CLASS_NAMES = {
    'N': 'Normal Sinus Rhythm',
    'L': 'Left Bundle Branch Block Beat (LBBB)',
    'R': 'Right Bundle Branch Block Beat (RBBB)',
    'A': 'Atrial Premature Contraction (APC)',
    'V': 'Premature Ventricular Contraction (PVC)'
}

# Global runtime state
ml_models = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    model_path = Path(config.MODEL_DIR) / "best_mshan_model.keras"
    if not model_path.exists():
        print(f"[!] Warning: Model not found at '{model_path}'")
        ml_models["mshan"] = None
    else:
        print(f"[*] Loading MSHAN model from '{model_path}'...")
        ml_models["mshan"] = tf.keras.models.load_model(str(model_path))
        
        # --- Warmup Pass ---
        print("[*] Warming up TensorFlow engine...")
        dummy_input = np.zeros((1, config.WINDOW_SIZE, 1), dtype=np.float32)
        _ = ml_models["mshan"].predict(dummy_input, verbose=0)
        print("[+] MSHAN model loaded & warmed up successfully.")

    yield
    ml_models.clear()

app = FastAPI(
    title="MSHAN ECG Arrhythmia Detection API",
    description=(
        "Production REST API for classifying ECG heartbeats and diagnosing arrhythmias "
        "using the Interpretable Multi-Scale Hybrid Attention Network (MSHAN)."
    ),
    version="1.0.0",
    lifespan=lifespan
)

# Allow Cross-Origin Resource Sharing for Streamlit or web frontends
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def preprocess_signal(signal_list: list) -> np.ndarray:
    """Standardizes signal to zero-mean and unit variance, reshaping for 1D CNN."""
    arr = np.array(signal_list, dtype=np.float32)
    std = np.std(arr)
    if std > 1e-6:
        arr = (arr - np.mean(arr)) / std
    return arr.reshape(1, config.WINDOW_SIZE, 1)


@app.get("/health", response_model=HealthResponse, tags=["Monitoring"])
def health_check():
    """Health check endpoint to monitor service readiness."""
    return HealthResponse(
        status="healthy" if ml_models.get("mshan") is not None else "degraded",
        model_loaded=ml_models.get("mshan") is not None,
        num_classes=config.NUM_CLASSES,
        classes=config.CLASSES
    )


@app.post("/predict", response_model=PredictionResult, tags=["Inference"])
def predict_heartbeat(input_data: HeartbeatInput):
    """
    Classify a single 360-sample heartbeat segment.
    Returns the predicted class, clinical name, and probability distribution.
    """
    model = ml_models.get("mshan")
    if model is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="MSHAN model checkpoint not loaded. Please train or provide model weights first."
        )

    # Preprocess and infer
    x = preprocess_signal(input_data.signal)
    probs = model.predict(x, verbose=0)[0]
    pred_idx = int(np.argmax(probs))
    pred_class = config.CLASSES[pred_idx]

    prob_dict = {
        config.CLASSES[i]: float(probs[i]) for i in range(config.NUM_CLASSES)
    }

    return PredictionResult(
        predicted_class=pred_class,
        class_name=CLASS_NAMES.get(pred_class, "Unknown Arrhythmia"),
        confidence=float(probs[pred_idx]),
        probabilities=prob_dict
    )


@app.post("/predict/batch", response_model=BatchPredictionResponse, tags=["Inference"])
def predict_batch_heartbeats(batch: BatchHeartbeatInput):
    """
    Classify a batch of 360-sample heartbeat segments simultaneously.
    """
    model = ml_models.get("mshan")
    if model is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="MSHAN model checkpoint not loaded."
        )

    if not batch.heartbeats:
        return BatchPredictionResponse(results=[])

    # Batch preprocessing
    batch_tensors = np.vstack([
        preprocess_signal(beat.signal) for beat in batch.heartbeats
    ])

    batch_probs = model.predict(batch_tensors, batch_size=config.BATCH_SIZE, verbose=0)
    results = []

    for probs in batch_probs:
        pred_idx = int(np.argmax(probs))
        pred_class = config.CLASSES[pred_idx]
        prob_dict = {
            config.CLASSES[i]: float(probs[i]) for i in range(config.NUM_CLASSES)
        }
        results.append(
            PredictionResult(
                predicted_class=pred_class,
                class_name=CLASS_NAMES.get(pred_class, "Unknown Arrhythmia"),
                confidence=float(probs[pred_idx]),
                probabilities=prob_dict
            )
        )

    return BatchPredictionResponse(results=results)


@app.post("/explain", response_model=ExplainResponse, tags=["Interpretability"])
def explain_heartbeat(input_data: HeartbeatInput):
    """
    Classifies a heartbeat and returns sample-level attribution weights
    highlighting the specific segments (P-wave, QRS, T-wave) guiding the prediction.
    """
    model = ml_models.get("mshan")
    if model is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="MSHAN model checkpoint not loaded."
        )

    x = preprocess_signal(input_data.signal)
    probs = model.predict(x, verbose=0)[0]
    pred_idx = int(np.argmax(probs))
    pred_class = config.CLASSES[pred_idx]

    # Compute gradient-based saliency attribution for real-time responsiveness
    x_tensor = tf.convert_to_tensor(x)
    with tf.GradientTape() as tape:
        tape.watch(x_tensor)
        preds = model(x_tensor, training=False)
        target_score = preds[:, pred_idx]

    grads = tape.gradient(target_score, x_tensor)
    # Saliency / Input x Gradient attribution over the 360 time samples
    attribution = tf.squeeze(grads * x_tensor).numpy().tolist()

    prob_dict = {
        config.CLASSES[i]: float(probs[i]) for i in range(config.NUM_CLASSES)
    }

    return ExplainResponse(
        predicted_class=pred_class,
        class_name=CLASS_NAMES.get(pred_class, "Unknown Arrhythmia"),
        confidence=float(probs[pred_idx]),
        probabilities=prob_dict,
        shap_values=attribution,
        baseline_time_index=list(range(config.WINDOW_SIZE))
    )