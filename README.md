# Multimodal Crop Health Monitoring and Yield Prediction

**Author:** Khang Phan | itsmekhang  
**Course Project** | Spring 2026

An AI-powered system that combines computer vision and tabular data to detect crop diseases from leaf images and predict crop yield from environmental data.

---

## Overview

Farmers currently rely on manual observation or delayed indicators, leading to late disease detection and inefficient use of water, pesticides, and fertilizers. This project addresses that gap with a modular, two-model ML pipeline:

- **Image model** — ResNet-18 (PyTorch, transfer learning) trained on PlantVillage to classify plant diseases
- **Yield model** — XGBoost / Random Forest trained on weather and historical yield data
- **UI** — Streamlit dashboard for interactive inference

---

## Project Structure

```
crop-health-monitoring/
├── notebooks/
│   └── setup.ipynb            # Environment check, data loading, EDA plots
├── src/
│   ├── disease_classifier.py  # CNN model training and inference
│   └── yield_predictor.py     # XGBoost/RF model training and inference
├── ui/
│   └── app.py                 # Gradio dashboard
├── data/
│   ├── raw/                   # Downloaded datasets (not tracked by git)
│   └── processed/             # Cleaned data outputs
├── results/                   # Saved models, metrics, plots
└── docs/                      # Architecture diagrams, blueprint report
```

---

## Installation

```bash
git clone https://github.com/itsmekhang/crop-health-monitoring.git
cd crop-health-monitoring
pip install -r requirements.txt
```

**Requirements:** Python 3.10+

---

## How to Run

### 1. Setup and Data Exploration
Open and run the setup notebook:
```bash
jupyter notebook notebooks/setup.ipynb
```
This will:
- Verify all dependencies are installed
- Download datasets via Kaggle API (see Dataset section below)
- Run exploratory data analysis with summary statistics and plots

### 2. Train Disease Classifier
```bash
python src/disease_classifier.py
```
Trains a ResNet50V2 model on PlantVillage. Model saved to `results/disease_model.keras`.

### 3. Train Yield Predictor
```bash
python src/yield_predictor.py
```
Trains an XGBoost regressor on crop yield data. Model saved to `results/yield_model.pkl`.

### 4. Launch the UI
```bash
streamlit run ui/app.py
```
Opens a Streamlit dashboard in your browser at `http://localhost:8501`.

---

## Datasets

| Dataset | Source | Type | Size |
|---|---|---|---|
| PlantVillage | [Kaggle](https://www.kaggle.com/datasets/emmarex/plantdisease) | Labeled leaf images | ~54,000 images, 38 classes |
| Crop Yield | [Kaggle](https://www.kaggle.com/datasets/patelris/crop-yield-prediction-dataset) | Tabular (weather + yield) | ~28,000 rows |

### Kaggle API Setup
1. Go to https://www.kaggle.com/settings → API → **Create New Token**
2. Place the downloaded `kaggle.json` at `~/.kaggle/kaggle.json`
3. Run the download cells in `notebooks/setup.ipynb`

---

## Architecture

```
[PlantVillage Images] ──► ResNet-18 ──► Disease Classification
                                               │
                                               ▼
                                        Streamlit Dashboard
                                               ▲
[Crop Yield CSV] ──► XGBoost/RF ──► Yield Prediction (hg/ha)
```

---

## Author

**Khang Phan**  
GitHub: [@itsmekhang](https://github.com/itsmekhang)
