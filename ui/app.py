"""
ui/app.py
Streamlit Clinical Dashboard for MSHAN ECG Arrhythmia Classification & Interpretability.
Integrates with the FastAPI backend service for inference and explainability.
Updated with Streamlit 2026 container width syntax and robust CSV parsing.
"""
import os
import json
from pathlib import Path
import requests
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st

# Streamlit Page Setup
st.set_page_config(
    page_title="MSHAN ECG Diagnostic Dashboard",
    page_icon="🫀",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Backend API Configuration
API_URL = os.getenv("API_URL", "http://localhost:8000")

# Diagnostic Class Metadata
CLASS_INFO = {
    'N': {
        'name': 'Normal Sinus Rhythm',
        'color': '#2ecc71',
        'desc': 'Normal cardiac rhythm with typical P-wave, QRS, and T-wave morphology.'
    },
    'L': {
        'name': 'Left Bundle Branch Block (LBBB)',
        'color': '#e67e22',
        'desc': 'Delayed activation of the left ventricle causing widened QRS complexes.'
    },
    'R': {
        'name': 'Right Bundle Branch Block (RBBB)',
        'color': '#f39c12',
        'desc': 'Delayed conduction through right bundle branch.'
    },
    'A': {
        'name': 'Atrial Premature Contraction (APC)',
        'color': '#9b59b6',
        'desc': 'Ectopic beat originating in the atria prior to normal sinus discharge.'
    },
    'V': {
        'name': 'Premature Ventricular Contraction (PVC)',
        'color': '#e74c3c',
        'desc': 'Ectopic beat originating in the ventricles, exhibiting wide, abnormal QRS.'
    }
}


def check_api_health():
    """Queries the FastAPI /health endpoint to check system readiness."""
    try:
        res = requests.get(f"{API_URL}/health", timeout=3)
        if res.status_code == 200:
            return res.json()
    except Exception:
        pass
    return {"status": "offline", "model_loaded": False}


def generate_synthetic_beat(beat_type='N'):
    """Generates synthetic 360-sample heartbeat waveforms for quick UI demonstration."""
    t = np.linspace(-1, 1, 360)
    p_wave = 0.15 * np.exp(-((t + 0.5) ** 2) / 0.01)
    q_wave = -0.15 * np.exp(-((t + 0.08) ** 2) / 0.001)
    r_wave = 1.2 * np.exp(-(t ** 2) / 0.002)
    s_wave = -0.25 * np.exp(-((t - 0.08) ** 2) / 0.001)
    t_wave = 0.35 * np.exp(-((t - 0.4) ** 2) / 0.02)

    base_signal = p_wave + q_wave + r_wave + s_wave + t_wave

    if beat_type == 'L':
        base_signal += 0.4 * np.exp(-((t - 0.05) ** 2) / 0.004) - 0.4 * np.exp(-((t - 0.4) ** 2) / 0.02)
    elif beat_type == 'V':
        r_wave = 1.5 * np.exp(-(t ** 2) / 0.015)
        t_wave = -0.6 * np.exp(-((t - 0.45) ** 2) / 0.03)
        base_signal = r_wave + t_wave
    elif beat_type == 'A':
        p_wave = 0.35 * np.exp(-((t + 0.35) ** 2) / 0.008)
        base_signal = p_wave + q_wave + r_wave + s_wave + t_wave

    noise = np.random.normal(0, 0.02, 360)
    signal = base_signal + noise
    return (signal - np.mean(signal)) / (np.std(signal) + 1e-8)


# Sidebar Navigation & Backend Status
st.sidebar.title("🫀 MSHAN ECG AI")
health = check_api_health()

if health.get("status") == "healthy":
    st.sidebar.success("✅ Backend API: Connected")
elif health.get("status") == "degraded":
    st.sidebar.warning("⚠️ Backend API: Ready (Model Unloaded)")
else:
    st.sidebar.error("❌ Backend API: Offline (Run FastAPI on port 8000)")

st.sidebar.markdown("---")
nav = st.sidebar.radio(
    "Navigation Menu",
    [
        "Single Heartbeat Diagnosis",
        "Batch ECG Screening",
        "Model Performance & Reports",
        "Architecture & Clinical Protocols"
    ]
)

# SECTION 1: Single Heartbeat Diagnosis
if nav == "Single Heartbeat Diagnosis":
    st.title("Single Heartbeat Diagnostic Analysis")
    st.markdown(
        "Upload a 360-sample 1D ECG segment or select a preset waveform to evaluate "
        "arrhythmia probabilities and view sample-level temporal feature attributions."
    )

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Input Selection")
        input_mode = st.radio("Source", ["Preset Heartbeats", "Upload Custom CSV/JSON"])

        if input_mode == "Preset Heartbeats":
            preset_choice = st.selectbox(
                "Select Preset Rhythm",
                options=['N', 'L', 'R', 'A', 'V'],
                format_func=lambda x: f"{x} - {CLASS_INFO[x]['name']}"
            )
            signal_data = generate_synthetic_beat(preset_choice)
        else:
            uploaded_file = st.file_uploader("Upload 1D CSV/JSON (360 samples)", type=["csv", "json"])
            if uploaded_file is not None:
                try:
                    if uploaded_file.name.endswith('.csv'):
                        df = pd.read_csv(uploaded_file, header=None, low_memory=False)
                        # Coerce values to float, handling any string headers
                        numeric_vals = pd.to_numeric(df.values.flatten(), errors='coerce')
                        signal_data = numeric_vals[~np.isnan(numeric_vals)]
                    else:
                        data = json.load(uploaded_file)
                        signal_data = np.array(data['signal'], dtype=float)

                    if len(signal_data) != 360:
                        st.error(f"Input contains {len(signal_data)} samples. Expected exactly 360.")
                        st.stop()
                except Exception as e:
                    st.error(f"Error parsing file: {e}")
                    st.stop()
            else:
                st.info("Awaiting file upload. Using default Normal (N) sinus rhythm sample.")
                signal_data = generate_synthetic_beat('N')

        # Updated width syntax
        analyze_btn = st.button("🔬 Run Diagnostic Inference", type="primary", width="stretch")

    with col2:
        st.subheader("Raw 1D ECG Waveform (360 Hz)")
        fig, ax = plt.subplots(figsize=(10, 3.5))
        time_axis = np.arange(360)
        ax.plot(time_axis, signal_data, color='#2c3e50', lw=1.8, label="Lead II Signal")
        ax.axvline(x=180, color='#e74c3c', linestyle='--', label="Annotated R-Peak (Sample 180)")
        ax.set_xlabel("Sample Index (R-peak at 180; 360 Hz)")
        ax.set_ylabel("Normalized Amplitude (Z-score)")
        ax.grid(True, linestyle="--", alpha=0.5)
        ax.legend(loc="upper right")
        st.pyplot(fig)

    st.markdown("---")

    # Diagnostic Results & Explainability
    if analyze_btn:
        st.session_state['last_signal'] = signal_data

        try:
            with st.spinner("Processing through MSHAN Network & computing attributions..."):
                response = requests.post(
                    f"{API_URL}/explain",
                    json={"signal": signal_data.tolist()},
                    timeout=30
                )

            if response.status_code == 200:
                res = response.json()
                pred_cls = res["predicted_class"]
                info = CLASS_INFO[pred_cls]

                res_col1, res_col2 = st.columns(2)

                with res_col1:
                    st.subheader("Diagnostic Verdict")
                    st.markdown(
                        f"""
                        <div style="background-color: {info['color']}22; border-left: 6px solid {info['color']}; padding: 16px; border-radius: 4px;">
                            <h2 style="margin: 0; color: {info['color']};">{pred_cls} — {info['name']}</h2>
                            <p style="font-size: 1.1em; margin-top: 8px;"><b>Diagnostic Confidence:</b> {res['confidence'] * 100:.2f}%</p>
                            <p style="color: #555; margin-bottom: 0;">{info['desc']}</p>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )

                    st.markdown("#### Probability Distribution")
                    probs = res["probabilities"]
                    prob_df = pd.DataFrame({
                        'Class': list(probs.keys()),
                        'Probability': [val * 100 for val in probs.values()]
                    })
                    st.bar_chart(prob_df.set_index('Class'), height=220)

                with res_col2:
                    st.subheader("Sample-Level Attributions (Input × Gradient)")
                    st.markdown(
                        "Highlights specific morphological features (P-wave, QRS, T-wave) "
                        "driving the prediction (**red**: positive weight, **blue**: negative weight)."
                    )

                    shap_vals = np.array(res["attribution_values"])
                    fig_attr, ax_attr = plt.subplots(figsize=(10, 4.2))
                    colors = ['#e74c3c' if v > 0 else '#3498db' for v in shap_vals]
                    ax_attr.bar(time_axis, shap_vals, color=colors, width=1.0)
                    ax_attr.axvline(x=180, color='black', linestyle=':', label='R-Peak Center')
                    ax_attr.set_xlabel("Sample Index")
                    ax_attr.set_ylabel("Attribution Weight")
                    ax_attr.grid(True, linestyle="--", alpha=0.5)
                    st.pyplot(fig_attr)
            else:
                st.error(f"API Error ({response.status_code}): {response.text}")

        except Exception as e:
            st.error(f"Could not connect to FastAPI server at `{API_URL}`: {e}")


# SECTION 2: Batch ECG Screening
elif nav == "Batch ECG Screening":
    st.title("Batch ECG Arrhythmia Screening")
    st.markdown("Upload a multi-row CSV file (where each row represents a 360-sample heartbeat) for high-throughput screening.")

    batch_file = st.file_uploader("Upload Batch CSV Dataset", type=["csv"])

    if batch_file is not None:
        try:
            # Fix DtypeWarning by specifying low_memory=False
            raw_df = pd.read_csv(batch_file, header=None, low_memory=False)

            # Convert all cells to numeric, turning string headers into NaN
            clean_matrix = raw_df.apply(pd.to_numeric, errors='coerce')
            
            # Drop rows that are entirely NaN (e.g. text header row)
            clean_matrix = clean_matrix.dropna(how='all')

            # Handle 361-column files (360 samples + 1 target class column)
            if clean_matrix.shape[1] == 361:
                st.warning("⚠️ Detected 361 columns. Automatically stripping the last column (assumed target label).")
                clean_matrix = clean_matrix.iloc[:, :360]

            if clean_matrix.shape[1] != 360:
                st.error(f"❌ Incorrect format: Expected 360 columns per row, but found {clean_matrix.shape[1]}.")
            else:
                st.info(f"Loaded batch shape: {clean_matrix.shape[0]} heartbeats × {clean_matrix.shape[1]} samples")

                # Updated width syntax
                if st.button("Run Batch Classification", type="primary", width="stretch"):
                    payload = {
                        "heartbeats": [{"signal": row.dropna().tolist()} for _, row in clean_matrix.iterrows()]
                    }
                    with st.spinner("Classifying batch..."):
                        res = requests.post(f"{API_URL}/predict/batch", json=payload, timeout=30)

                    if res.status_code == 200:
                        results = res.json()["results"]
                        res_df = pd.DataFrame([
                            {
                                'Beat_ID': i + 1,
                                'Predicted_Class': r['predicted_class'],
                                'Diagnostic_Name': r['class_name'],
                                'Confidence': f"{r['confidence'] * 100:.2f}%"
                            }
                            for i, r in enumerate(results)
                        ])

                        c1, c2 = st.columns(2)
                        with c1:
                            st.subheader("Class Summary")
                            summary = res_df['Predicted_Class'].value_counts()
                            st.dataframe(summary, width="stretch")

                        with c2:
                            st.subheader("Detailed Predictions")
                            st.dataframe(res_df, width="stretch", height=300)
                    else:
                        st.error(f"Batch prediction error ({res.status_code}): {res.text}")
        except Exception as e:
            st.error(f"Failed to process batch CSV: {e}")


# SECTION 3: Model Performance & Reports
elif nav == "Model Performance & Reports":
    st.title("Model Performance & Reports")
    st.markdown("Evaluations generated on the held-out **DS2 Inter-Patient Test Set**.")

    reports_dir = Path(__file__).resolve().parent.parent / "reports" / "nlrav_v1"
    metrics_file = reports_dir / "evaluation_metrics.json"

    if metrics_file.exists():
        with open(metrics_file, "r") as f:
            metrics = json.load(f)

        m1, m2, m3, m4, m5 = st.columns(5)
        m1.metric("Overall Accuracy", f"{metrics['Overall_Accuracy'] * 100:.2f}%")
        m2.metric("Macro Sensitivity", f"{metrics['Macro_Sensitivity'] * 100:.2f}%")
        m3.metric("Macro Specificity", f"{metrics['Macro_Specificity'] * 100:.2f}%")
        m4.metric("Macro Precision", f"{metrics['Macro_Precision'] * 100:.2f}%")
        m5.metric("Macro F1-Score", f"{metrics['Macro_F1'] * 100:.2f}%")

    st.markdown("### Generated Diagnostic Figures")
    r_col1, r_col2 = st.columns(2)

    with r_col1:
        st.subheader("Confusion Matrix")
        cm_img = reports_dir / "confusion_matrix.png"
        if cm_img.exists():
            st.image(str(cm_img), width="stretch")
        else:
            st.info("Run `python -m ml.evaluate` to generate figures.")

        st.subheader("SHAP Global Summary")
        shap_img = reports_dir / "shap_global_summary.png"
        if shap_img.exists():
            st.image(str(shap_img), width="stretch")

    with r_col2:
        st.subheader("ROC Curves")
        roc_img = reports_dir / "roc_curves.png"
        if roc_img.exists():
            st.image(str(roc_img), width="stretch")

        st.subheader("Training History")
        hist_img = reports_dir / "training_history.png"
        if hist_img.exists():
            st.image(str(hist_img), width="stretch")


# SECTION 4: Architecture & Clinical Protocols
elif nav == "Architecture & Clinical Protocols":
    st.title("Architecture & Clinical Protocols")
    st.markdown(
        """
        ### Multi-Scale Hybrid Attention Network (MSHAN)
        
        The MSHAN pipeline categorizes heartbeats into 5 clinical classes (**N, L, R, A, V**) 
        using raw 360-sample lead II ECG inputs without manual feature engineering.
        
        #### Key Technical Innovations
        1. **Multi-Scale Convolutional Blocks:**
           - **Stage 1:** Parallel Conv1D kernels (**7, 13, 19** with 32 filters each) capturing short and medium wave features.
           - **Stage 2:** Parallel Conv1D kernels (**11, 17, 23** with 64 filters each) capturing broader morphological context.
        2. **Hierarchical Attention:**
           - **Temporal Attention:** $1 \\times 1$ Conv1D with sigmoid activation generating time-step weighting.
           - **Channel Attention:** Global Average Pooling + MLP recalibrating channel feature maps.
        3. **Residual Learning & Sequence Modeling:**
           - $1 \\times 1$ Conv shortcut matching channels ($96 \\to 192$), stabilized by Layer Normalization.
           - Stacked GRUs (**128** and **64** units) capturing sequential temporal dynamics.
        
        #### Inter-Patient Evaluation Protocol
        To prevent intra-patient data leakage, dataset records are strictly divided using the **de Chazal et al. (2004)** benchmark:
        - **DS1 Training/Validation (22 records):** 101, 106, 108, 109, 112, 114, 115, 116, 118, 119, 122, 124, 201, 203, 205, 207, 208, 209, 215, 220, 223, 230.
        - **DS2 Held-Out Test (22 records):** 100, 103, 105, 111, 113, 117, 121, 123, 200, 202, 210, 212, 213, 214, 219, 221, 222, 228, 231, 232, 233, 234.
        """
    )
