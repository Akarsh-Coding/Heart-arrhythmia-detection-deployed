# MSHAN ECG arrhythmia classification

Implementation following https://www.frontiersin.org/journals/artificial-intelligence/articles/10.3389/frai.2026.1867173/full

## Corrected pipeline

- Literal MIT-BIH classes, in output order: N, L, R, A, V (not AAMI groups).
- MLII preferred; 360 samples around annotated R-peaks; per-beat Z-score normalization shared with the API.
- The paper's DS1/DS2 record lists; five stratified folds of original DS1 beats. Oversampling occurs only inside each training fold. These are beat-level validation folds, not patient-grouped folds.
- Median best validation epoch selects the duration for a fresh fit on all DS1. DS2 is evaluated only after this fit. This final refit rule is an explicit implementation choice.
- Batch size 64, Adam learning rate 0.001, at most 30 epochs per fold.

## Run (Python 3.12)

```powershell
python -m pip install -r requirements.txt
python -m scripts.download_data
python -m pytest -q
python -m ml.train
python -m ml.evaluate
python -m ml.explain
python -m uvicorn app.main:app --port 8000
# In another terminal:
python -m streamlit run ui/app.py
```

Training performs five validation fits plus a final fit and can take a long time on CPU. It writes compatible checkpoints and a checksum-bound metadata file into `models/nlrav_v1/`, and reports into `reports/nlrav_v1/`. The old checkpoint and reports are retained in their original directories but are not used by the corrected application. Retraining is required before inference is available. Do not copy or relabel the old checkpoint as a corrected model.

The API exposes `/health`, `/predict`, `/predict/batch`, and `/explain`. Inputs are 360 finite samples centered on an R-peak. `/explain` returns `attribution_values` and `attribution_method="input_x_gradient"`; offline `ml.explain` uses SHAP GradientExplainer with DS1 background beats. This replaces the misleading API field `shap_values`.

Saved historical metrics do not describe the corrected model. Regenerate results after training; paper-level performance is not guaranteed. ROC evaluation fails explicitly if a configured class is absent instead of writing NaN JSON.

## Local dataset audit

With the corrected loader, the supplied local files contain these usable windows:

| Split | N | L | R | A | V | Total |
|---|---:|---:|---:|---:|---:|---:|
| DS1 | 38,082 | 3,947 | 3,780 | 810 | 3,683 | 50,302 |
| DS2 | 36,423 | 4,124 | 3,475 | 1,736 | 3,219 | 48,977 |

These differ from the paper's reported counts. The implementation keeps the stated record split and actual annotations; it does not duplicate or move test beats to reproduce table counts. Resolve this discrepancy with the authors' data/code before claiming an exact reproduction.

## Phase 2: balancing and validation coverage

Balancing retains every original training beat and adds copies only to minority classes. The majority class is preserved exactly once. Fold assignment precedes balancing; validation and DS2 retain their natural distributions.

The local DS1 audit found N in 18 records, L in 2 (109, 207), R in 3 (118, 124, 207), A in 14, and V in 17. Consequently, five record-grouped validation folds cannot each contain all five classes. Record coverage is not a verified patient identity mapping. The existing five stratified beat-level folds remain in place; they do not establish patient-independent validation. A separately reviewed grouped strategy could use two folds with the L-containing records separated, after verifying patient identities and all-class coverage on both sides. This would change the validation protocol and should be documented explicitly.
