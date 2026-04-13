
# Multimodal Crop Health Monitor

**Author:** Khang Phan | itsmekhang
**Course Project** | Spring 2026

A multimodal AI system that detects crop diseases from leaf images and assesses environmental risk from live weather data — combining computer vision and machine learning for real-time field diagnosis.

---

## How It Works

```
Leaf Image  →  ResNet-18          →  Disease class + confidence
Location    →  Open-Meteo API     →  Temperature, humidity, days since rain
                                           ↓
                                       XGBoost
                                           ↓
                              Risk score + treatment plan + revenue loss estimate
```

Two separate models handle two separate modalities:

- **ResNet-18** — image-only disease classifier trained on PlantVillage (38 classes, ~54k images)
- **XGBoost** — environmental risk scorer trained on disease type + weather conditions

Weather is intentionally kept out of the image classifier — a diseased leaf looks the same regardless of humidity. Weather only influences how urgently to act, not what disease is present.

---

## Project Structure

```
crop-health-monitoring/
├── notebooks/
│   └── setup.ipynb            # Dataset download, EDA, model training
├── src/
│   ├── disease_classifier.py  # ResNet-18 training script
│   ├── multimodal_model.py    # (unused in final version)
│   ├── fusion.py              # XGBoost risk scoring
│   ├── recommendations.py     # Treatment plans per disease
│   └── weather.py             # Open-Meteo API integration
├── ui/
│   └── app.py                 # Streamlit dashboard
├── data/
│   └── raw/                   # Downloaded datasets (not tracked)
├── results/                   # Saved models (not tracked)
└── docs/                      # Blueprint report
```

---

## Installation

```bash
git clone https://github.com/itsmekhang/crop-health-monitoring.git
cd crop-health-monitoring
pip install -r requirements.txt
```

---

## Usage

### 1. Train the models
Open `notebooks/setup.ipynb` and run all cells. This will:
- Clone PlantVillage dataset from GitHub (no Kaggle needed)
- Train ResNet-18 disease classifier → `results/disease_model.pth`
- Train XGBoost risk model → `results/risk_model.pkl`

### 2. Launch the dashboard
```bash
streamlit run ui/app.py
```

---

## Dataset

| Dataset | Source | Use |
|---|---|---|
| PlantVillage | [gabrieldgf4/PlantVillage-Dataset](https://github.com/gabrieldgf4/PlantVillage-Dataset) | ResNet-18 training |
| Open-Meteo | [open-meteo.com](https://open-meteo.com) | Live weather at inference (free, no key) |

---

## Models

| Model | Architecture | Input | Output |
|---|---|---|---|
| Disease classifier | ResNet-18 (frozen backbone, fine-tuned FC) | Leaf image | Disease class + confidence |
| Risk scorer | XGBoost regressor | Disease class + temp + humidity + rainfall | Risk score 0–1 |
<img width="1189" height="396" alt="download" src="https://github.com/user-attachments/assets/ef73153b-3059-4a57-a8b1-54f9a519f2f7" />
---

## Author

**Khang Phan**
GitHub: [@itsmekhang](https://github.com/itsmekhang)
