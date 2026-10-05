# Semantic Workload Awareness in Serverless Computing

An ML-based pipeline for classifying serverless function workloads into four
behavioral archetypes:

- SPIKE
- PERIODIC
- RAMP
- STATIONARY

The project uses statistical, spectral (FFT), and trend-based features,
Snorkel for weak supervision, and LightGBM for workload classification.
## Requirements

- Python 3.10+
- Jupyter Notebook
- Required Python packages are listed in `requirements.txt`

## Problem Statement

Serverless workloads can exhibit different traffic patterns over time.
Identifying these patterns can help systems better understand workload
behavior and make informed resource management and scaling decisions.

This project aims to automatically classify serverless function invocation
traces into different workload archetypes using machine learning.

## Methodology

The pipeline consists of the following stages:

```text
Azure Functions Invocation Trace
              ↓
       Data Preprocessing
              ↓
       Feature Engineering
              ↓
  Statistical / FFT / Trend Features
              ↓
       Snorkel Weak Supervision
              ↓
          Label Model
              ↓
      LightGBM Classifier
              ↓



       Workload Classification
