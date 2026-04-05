# Multimodal Crop Health Monitoring and Yield Prediction

**Author:** Khang Phan

An AI-powered system that combines computer vision and tabular data to detect crop diseases from images and predict crop yield from environmental data.

## Overview

Farmers currently rely on manual observation or delayed indicators, leading to late disease detection and inefficient resource usage. This project addresses that gap with a modular, budget-friendly ML pipeline:

- **Image model** — CNN (ResNet-based) trained on PlantVillage to classify plant diseases
- **Yield model** — Random Forest / XGBoost trained on weather and historical yield data

## Project Structure

```
crop-health-monitoring/
├── notebooks/
│   └── setup.ipynb          # Environment verification and data download
├── src/
│   ├── disease_classifier.py  # CNN model for crop disease detection
│   └── yield_predictor.py     # RF/XGBoost model for yield prediction
├── ui/
│   └── app.py               # Optional Gradio dashboard
├── data/
│   ├── raw/                 # Downloaded datasets (not tracked)
│   └── processed/           # Cleaned and preprocessed data
├── results/                 # Saved models, plots, metrics
└── docs/                    # Project documentation
```

## Datasets

| Dataset | Source | Purpose |
|---|---|---|
| PlantVillage | Kaggle / TensorFlow Datasets | Disease classification (50k+ images) |
| Crop Yield | Kaggle | Yield prediction (temperature, rainfall, region) |

## Models

### A. Image-Based Disease Detection
- Architecture: ResNet-18 with transfer learning (TensorFlow/Keras)
- Input: Crop leaf images (224x224)
- Output: Disease class (healthy vs. disease types)

### B. Yield Prediction
- Models: Random Forest (baseline), XGBoost
- Input: Temperature, rainfall, historical yield by region/year
- Output: Predicted crop yield (regression)

## Setup

```bash
pip install -r requirements.txt
```

Then open `notebooks/setup.ipynb` to verify your environment and download datasets.

## Challenges

- **Data modality mismatch** — models trained separately; multimodal fusion treated as optional
- **Class imbalance** — handled via data augmentation and transfer learning
