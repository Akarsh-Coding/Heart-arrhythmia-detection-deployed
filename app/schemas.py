"""
app/schemas.py
Pydantic data models for FastAPI request and response validation.
"""
from typing import List, Optional, Dict
from pydantic import BaseModel, Field, validator


class HeartbeatInput(BaseModel):
    """
    A single segmented heartbeat window.
    Must consist of exactly 360 normalized samples (-180 to +180 relative to R-peak).
    """
    signal: List[float] = Field(
        ...,
        description="Normalized 1D ECG amplitude values (360 samples at 360 Hz).",
        example=[0.05 * i for i in range(360)]
    )

    @validator("signal")
    def validate_signal_length(cls, v):
        if len(v) != 360:
            raise ValueError(f"Signal must contain exactly 360 samples, received {len(v)} samples.")
        return v


class BatchHeartbeatInput(BaseModel):
    """Batch of heartbeat windows for multi-beat prediction."""
    heartbeats: List[HeartbeatInput]


class PredictionResult(BaseModel):
    predicted_class: str = Field(..., description="Arrhythmia class code (N, L, R, A, V)")
    class_name: str = Field(..., description="Full clinical diagnostic name")
    confidence: float = Field(..., description="Prediction probability for the predicted class")
    probabilities: Dict[str, float] = Field(..., description="Probability distribution across all 5 classes")


class BatchPredictionResponse(BaseModel):
    results: List[PredictionResult]


class ExplainResponse(BaseModel):
    predicted_class: str
    class_name: str
    confidence: float
    probabilities: Dict[str, float]
    shap_values: List[float] = Field(
        ...,
        description="Temporal SHAP attribution values across each of the 360 time samples."
    )
    baseline_time_index: List[int] = Field(
        ...,
        description="Sample indices from 0 to 359 (R-peak centered at 180)."
    )


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    num_classes: int
    classes: List[str]