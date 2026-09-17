# Semantic Workload Awareness in Serverless Computing

ML pipeline that classifies serverless function traffic into four archetypes — SPIKE, PERIODIC, RAMP, STATIONARY — using statistical, spectral (FFT), and trend features, Snorkel weak supervision, and a LightGBM classifier.

## Status
Work in progress. Core pipeline (data acquisition → feature engineering → weak supervision → classification) is functional end-to-end. Labeling functions are being expanded for fuller alignment with the proposed methodology.

## Pipeline
1. `phase1_preprocess.py` — downloads and preprocesses the Azure Functions 2019 invocation trace (14 days, 10,000 sampled functions)
2. Feature engineering — statistical, spectral (FFT), and trend features
3. Snorkel weak supervision — labeling functions + LabelModel
4. LightGBM classifier with GridSearch + Stratified K-Fold

## Results
See `final_confusion_matrix.png` for current evaluation results.

Raw dataset files are not included (regenerate via `phase1_preprocess.py`; source: [Azure Public Dataset](https://github.com/Azure/AzurePublicDataset)).
