"""Streamlit ECG research dashboard with explicit input validation."""
import json
import os
import sys
from pathlib import Path

# Also support `streamlit run ui/app.py` from outside the repository root.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import pandas as pd
import requests
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import streamlit as st
from ml import config
from ml.preprocessing import normalize_beat
from ui.data import read_csv_frame, single_beat, annotated_example

st.set_page_config(page_title="MSHAN ECG Research Dashboard", page_icon="🫀", layout="wide")
API_URL = os.getenv('API_URL', 'http://localhost:8000').rstrip('/')
CLASS_NAMES = {'N': 'Normal beat', 'L': 'Left bundle branch block beat',
               'R': 'Right bundle branch block beat', 'A': 'Atrial premature beat',
               'V': 'Ventricular premature beat'}


def api_request(path, payload=None):
    try:
        response = (requests.get(f'{API_URL}/{path}', timeout=3) if payload is None else
                    requests.post(f'{API_URL}/{path}', json=payload, timeout=120))
        response.raise_for_status()
        return response.json()
    except requests.RequestException as exc:
        raise ValueError(f'API request failed: {exc}') from exc


@st.cache_data
def example(code):
    return annotated_example(code)


st.sidebar.title('MSHAN ECG')
try:
    health = api_request('health')
    if health.get('model_loaded'):
        st.sidebar.success('Model loaded')
    else:
        st.sidebar.warning('API available; model unavailable')
except ValueError:
    st.sidebar.warning('API offline')
nav = st.sidebar.radio('Navigation', ['Single heartbeat', 'Model performance', 'About the project'])
st.caption('Research classifier. Review held-out per-class performance before interpreting predictions.')

if nav == 'Single heartbeat':
    st.title('Single heartbeat analysis')
    st.info('Input must be an MLII heartbeat sampled at 360 Hz, with its R-peak at index 180. Continuous ECG requires segmentation before upload.')
    source = st.radio('Source', ['Annotated PhysioNet example', 'Upload CSV/JSON'])
    signal = None
    if source == 'Annotated PhysioNet example':
        code = st.selectbox('Annotated class', config.CLASSES, format_func=lambda c: f'{c}: {CLASS_NAMES[c]}')
        try:
            signal, record = example(code)
            st.caption(f'Real annotated training example from record {record}; this is a demonstration, not a held-out test.')
        except (OSError, ValueError) as exc:
            st.warning(f'Example unavailable: {exc}. Upload a segmented heartbeat instead.')
    else:
        uploaded = st.file_uploader('Segmented heartbeat', type=['csv', 'json'])
        if uploaded is not None:
            try:
                if uploaded.name.lower().endswith('.json'):
                    signal = normalize_beat(json.load(uploaded)['signal'])
                else:
                    header = st.checkbox('CSV has a header row')
                    frame = read_csv_frame(uploaded, header)
                    column = None
                    if frame.shape[1] > 1 and frame.shape[0] != 1:
                        column = st.selectbox('Select the ECG amplitude column (exclude time/ID columns)', list(frame.columns), index=None)
                        if column is None:
                            raise ValueError('Choose the ECG amplitude column before running inference.')
                    signal = single_beat(frame, column)
            except (ValueError, KeyError, TypeError) as exc:
                st.error(str(exc))
    if signal is not None:
        fig, ax = plt.subplots(figsize=(10, 3))
        ax.plot(signal)
        ax.axvline(180, linestyle='--', color='red')
        ax.set(xlabel='Sample index (R-peak at 180)', ylabel='Per-beat normalized amplitude')
        st.pyplot(fig)
        plt.close(fig)
        if st.button('Run inference', type='primary'):
            try:
                with st.spinner('Computing prediction and attribution...'):
                    result = api_request('explain', {'signal': signal.tolist()})
                st.subheader(f"{result['predicted_class']}: {CLASS_NAMES[result['predicted_class']]}")
                st.metric('Model probability (not calibrated diagnostic confidence)', f"{result['confidence']:.1%}")
                st.bar_chart(pd.Series(result['probabilities'], name='Probability'))
                st.subheader('Input × gradient attribution')
                st.line_chart(pd.Series(result['attribution_values'], name='Attribution'))
            except ValueError as exc:
                st.error(str(exc))

elif nav == 'Model performance':
    st.title('Held-out model performance')
    reports = config.REPORTS_DIR
    metrics_file = reports / 'evaluation_metrics.json'
    if metrics_file.exists():
        metrics = json.loads(metrics_file.read_text())
        cols = st.columns(3)
        cols[0].metric('Accuracy', f"{metrics['Overall_Accuracy']:.2%}")
        cols[1].metric('Macro F1', f"{metrics['Macro_F1']:.2%}")
        cols[2].metric('Macro recall', f"{metrics['Macro_Sensitivity']:.2%}")
        st.dataframe(pd.DataFrame(metrics['Per_Class']).T, width='stretch')
        weak = [c for c, values in metrics['Per_Class'].items() if values['Sensitivity'] < 0.5]
        if weak:
            st.warning('This evaluation missed more than half of the beats in: ' + ', '.join(weak))
        st.caption(f"Evaluation protocol: {metrics.get('protocol', 'Historical DS1/DS2 record split; not a verified patient-independent evaluation.')}")
    else:
        st.info('No evaluation metrics for this run.')
    for filename, title in [('confusion_matrix.png', 'Confusion matrix'), ('roc_curves.png', 'ROC curves'),
                            ('shap_global_summary.png', 'SHAP class-balanced sample summary'),
                            ('shap_local_overlay.png', 'SHAP local explanation')]:
        path = reports / filename
        if path.exists():
            st.subheader(title)
            st.image(str(path), width='stretch')
    if not (reports / 'shap_global_summary.png').exists():
        st.info('SHAP reports have not been generated. Run python -m ml.explain for the selected model.')
    for path in sorted(reports.glob('fold_*/training_history.png')):
        with st.expander(f'Training history: {path.parent.name}'):
            st.image(str(path), width='stretch')
    final_history = reports / 'final_history.json'
    if final_history.exists():
        st.subheader('Final training history')
        history = pd.DataFrame(json.loads(final_history.read_text()))
        history.index = np.arange(1, len(history) + 1)
        st.line_chart(history[['loss', 'accuracy']])

else:
    st.title('Heart arrhythmia detection from ECG signals')
    st.markdown("""
    This project uses deep learning to classify individual heartbeats from ECG signals.
    Trained on the MIT-BIH Arrhythmia Database, the model combines a 1D convolutional
    neural network with attention and GRU layers to recognize normal beats and four
    types of abnormal heartbeats.

    Upload a segmented heartbeat to explore its predicted class, view the ECG waveform,
    and see which parts of the signal influenced the prediction. The model performance
    page presents evaluation results and visual explanations.
    """)
