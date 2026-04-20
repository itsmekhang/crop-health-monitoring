# Crop Health Monitor

**Author:** Khang Phan | [@itsmekhang](https://github.com/itsmekhang)  
**Course:** EGN6217 — Applied Deep Learning | Spring 2026

A modular crop health system with two components: a ResNet-18 disease classifier trained on PlantVillage leaf images, and a rule-based environmental risk layer that combines the classifier output with live weather conditions to produce actionable field guidance.

---

## How It Works

```
Leaf Image  →  ResNet-18 (frozen backbone, fine-tuned FC)
                    ↓
              Disease class + confidence
                    ↓
Location    →  Open-Meteo API  →  Temp, humidity, days since rain
                    ↓
                XGBoost (rule approximator)
                    ↓
       Risk score + treatment plan + revenue loss estimate
```

**Two independent models, one interface:**
- **ResNet-18** — image-only disease classifier. 38 disease/healthy classes across 14 crops, ~54k lab images. Val accuracy: **0.934** (frozen backbone, per-class eval) / **0.996** (layer4 unfrozen, Section 8B).
- **XGBoost** — decision-support risk layer. Approximates agronomic rules using disease type + weather. **Note:** training target is derived from coded rules, not observed field data — Val R² = 0.966 reflects formula reproduction only, not real-world validity.

---

## Project Structure

```
crop-health-monitoring/
├── notebooks/
│   └── setup.ipynb            # Dataset download, EDA, training, per-class eval, frozen vs unfrozen comparison
├── src/
│   ├── disease_classifier.py  # ResNet-18 training (frozen or layer4-unfrozen backbone)
│   ├── fusion.py              # Environmental risk scoring with agronomic weight documentation
│   ├── recommendations.py     # Treatment plans per disease class
│   └── weather.py             # Open-Meteo API integration
├── ui/
│   └── app.py                 # Streamlit dashboard
├── data/
│   └── raw/PlantVillage/      # Downloaded dataset (not tracked in git)
├── results/                   # Saved models and output plots (not tracked in git)
└── docs/                      # Report PDF and interface screenshots
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

Open `notebooks/setup.ipynb` and run all cells in order. This will:

1. Check your environment
2. Clone PlantVillage from GitHub into `data/raw/` — no Kaggle account needed
3. Run EDA and plot class distribution
4. Train ResNet-18 (frozen backbone) → `results/disease_model.pth`
5. Evaluate per-class precision, recall, F1, and generate confusion matrix → `results/confusion_matrix.png`
6. Train XGBoost risk model → `results/risk_model.pkl`
7. **(Section 8B)** Train unfrozen variant (`layer4` + FC) and plot frozen vs. unfrozen curves → `results/disease_model_unfrozen.pth`

### 2. Launch the dashboard

```bash
streamlit run ui/app.py
```

Then open `http://localhost:8501` in your browser. Upload a leaf image, enter a city name for live weather, and optionally set field size and crop price for revenue loss estimates.

---

## Dataset

| Dataset | Source | Use |
|---|---|---|
| PlantVillage | [gabrieldgf4/PlantVillage-Dataset](https://github.com/gabrieldgf4/PlantVillage-Dataset) | ResNet-18 training |
| Open-Meteo | [open-meteo.com](https://open-meteo.com) | Live weather at inference (free, no API key) |

**Scope:** 38 disease/healthy classes across 14 crops, ~54k lab images (one artifact class filtered). Class imbalance present — Tomato dominates with 10 of the 38 classes.

---

## Models

| Model | Architecture | Input | Output |
|---|---|---|---|
| Disease classifier | ResNet-18 (frozen backbone, fine-tuned FC) | Leaf image (lab) | Disease class + confidence |
| Risk layer | XGBoost (rule approximator) | Disease class + temp + humidity + rainfall | Derived risk score 0–1 |

<img width="1312" height="495" alt="download" src="https://github.com/user-attachments/assets/1e883cc4-6a03-477c-a27c-4707ad6b794a" />
<img width="1490" height="396" alt="download" src="https://github.com/user-attachments/assets/fc059617-bb1b-4bb7-975e-2d48bde639e4" />
<img width="1124" height="990" alt="download" src="https://github.com/user-attachments/assets/1a329477-5e31-4292-8dff-b8211465f13d" />
<img width="1189" height="396" alt="download" src="https://github.com/user-attachments/assets/f81685e6-f69a-4b1a-a790-fd345e39cd06" />

---

## Results

| Metric | Frozen baseline | layer4 unfrozen (Section 8B) |
|---|---|---|
| Val accuracy (per-class eval) | **0.934** | **0.996** |
| Val accuracy (in-loop) | 0.955 | 0.996 |
| Train accuracy (epoch 10) | 0.951 | 0.999 |
| Train loss (epoch 10) | 0.155 | 0.0047 |
| Trainable params | 20,007 (FC only) | ~2.1M (layer4 + FC) |

| Metric | Value |
|---|---|
| Lowest class accuracy (frozen, per-class eval) | Tomato mosaic virus 0.488, Tomato Early blight 0.620, Potato healthy 0.630 |
| XGBoost Val R² | 0.966 (formula reproduction, not field validity) |
| XGBoost Val MAE | 0.018 |

Three classes fall below 70% accuracy on the frozen baseline. The per-class evaluation reports 0.934 vs the in-loop 0.955 — the gap reflects the augmentation bleed fix (D2 leaked augmented images into validation) and the artifact class (`x_Removed_from_Healthy_leaves`, 0/2 correct) being included in the evaluation set. Per-class precision/recall/F1 breakdown is in the notebook.

## Known Issues and Limitations

| Issue | Detail |
|---|---|
| **Lab-to-field gap** | All PlantVillage images are staged on uniform backgrounds. Field photos with soil, shadows, and partial occlusion will produce lower accuracy |
| **Class imbalance** | Tomato dominates with 10 of 38 classes; small classes like Tomato mosaic virus (82 val samples) underperform |
| **XGBoost target is derived** | Val R² = 0.966 measures how well XGBoost reproduces the hand-coded formula — not real-world predictive validity |
| **Random split inflation** | PlantVillage near-duplicates in both train/val inflate reported accuracy relative to a leaf-stratified split |
| **Location–image coupling** | The risk score assumes the uploaded photo was taken at the city the user enters — this is unverified |

---

## Reproducing Results

After running `notebooks/setup.ipynb` in full, you will find in `results/`:
- `disease_model.pth` — trained ResNet-18 weights and class list (frozen baseline)
- `disease_model_unfrozen.pth` — trained ResNet-18 with `layer4` unfrozen (Section 8B)
- `risk_model.pkl` — trained XGBoost model, label encoder, feature list
- `disease_training_curves.png` — loss/accuracy per epoch
- `frozen_vs_unfrozen_comparison.png` — side-by-side training curves (Section 8B)
- `confusion_matrix.png` — per-class confusion matrix
- `eda_plantvillage.png` — class distribution bar chart
- `xgb_risk_evaluation.png` — XGBoost predicted vs. derived target

---

## Author

**Khang Phan**  
GitHub: [@itsmekhang](https://github.com/itsmekhang)  
Course: EGN6217 — Applied Deep Learning, Spring 2026
