# Precog NLP Task - Gyandeep

This repository contains a complete evaluation and ensembling pipeline for AI-text vs Human-text detection using three pretrained models (XGBoost, Semantic FFNN, DistilBERT) and a meta-ensemble stacker. It also includes analysis notebooks, saliency mapping, and a genetic text mutation experiment.

## Directory Structure

```
.
├── Meta Model/
│   ├── ensemble_stacking.py
│   ├── evaluate_ensemble.py
│   ├── linguistic_features.py
│   ├── detailed_results.csv
│   ├── test_set1.csv
│   ├── Distilbert model/
│   │   ├── config.json
│   │   └── model.safetensors
│   ├── semantic_nnmodel/
│   │   ├── config.json
│   │   ├── metadata.json
│   │   └── model.weights.h5
│   └── Xgboost model/
│       └── xgboost_ai_detector.json
├── NLP models/
│   ├── evaluate_models.py
│   ├── evaluate_models.ipynb
│   ├── saliency_analysis.py
│   ├── genetic_mutation.py
│   ├── final_dataset_gemma_complete.csv
│   ├── test_set1.csv
│   ├── glove.6B.100d.txt
│   └── Distilbert model/
│       ├── config.json
│       └── model.safetensors
├── ModelTraining.ipynb
├── README.md
├── Report.pdf
└── .gitignore
```

## Project Approach

1. **Base Models (Frozen)**
	- **DistilBERT**: Transformer classifier for AI/Human detection.
	- **Semantic FFNN**: 3-layer feedforward NN using GloVe embeddings.
	- **XGBoost**: Gradient boosting model using 14 linguistic features.

2. **Evaluation & Error Analysis**
	- Unified evaluation scripts in [NLP models/evaluate_models.py](NLP%20models/evaluate_models.py) and notebook.
	- Confusion matrix breakdown with TP/TN/FP/FN.
	- False positive extraction for manual inspection.

3. **Interpretability**
	- Captum-based saliency mapping for DistilBERT word attributions in [NLP models/saliency_analysis.py](NLP%20models/saliency_analysis.py).

4. **Meta-Ensemble (Stacking)**
	- Stacking model combines p_xgb, p_ffnn, p_bert + disagreement features.
	- Code in [Meta Model/ensemble_stacking.py](Meta%20Model/ensemble_stacking.py).

5. **Text Mutation Experiment**
	- Genetic algorithm that mutates AI text via synonym swaps to reduce AI score.
	- Script: [NLP models/genetic_mutation.py](NLP%20models/genetic_mutation.py).

6. **Model Training Notebook**
	- [ModelTraining.ipynb](ModelTraining.ipynb) documents the end-to-end training workflow used to produce the base models, including preprocessing, feature extraction, and export of trained artifacts used in this repo.

## Dependencies

Core Python libraries used:

- `torch`, `transformers`
- `tensorflow` / `keras`
- `xgboost`
- `scikit-learn`
- `spacy`
- `nltk`
- `textstat`
- `captum`
- `pandas`, `numpy`, `matplotlib`, `seaborn`

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# NLP assets
python -m spacy download en_core_web_sm
python -m nltk.downloader wordnet omw-1.4 punkt_tab averaged_perceptron_tagger_eng
```

> Note: `requirements.txt` is not included. Install dependencies manually or generate with `pip freeze > requirements.txt` after your environment is ready.

## Running the Project

### 1) Evaluate Base Models

```bash
python "NLP models/evaluate_models.py"
```

### 2) Run Ensemble Stacking

```bash
python "Meta Model/ensemble_stacking.py"
python "Meta Model/evaluate_ensemble.py"
```

### 3) Saliency Mapping (DistilBERT)

```bash
python "NLP models/saliency_analysis.py"
```

### 4) Genetic Text Mutation

```bash
python "NLP models/genetic_mutation.py"
```

### 5) Model Training Notebook

Open and run [ModelTraining.ipynb](ModelTraining.ipynb) to review the training pipeline and reproducibility steps.

## Author

Gyandeep