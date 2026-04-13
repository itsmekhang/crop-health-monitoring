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
- **ResNet-18** — image-only disease classifier. 15 classes, 3 crops (Tomato, Potato, Pepper), ~41k lab images.
- **XGBoost** — decision-support risk layer. Approximates agronomic rules using disease type + weather. **Note:** training target is derived from coded rules, not observed field data — the high R² reflects formula reproduction, not real-world validity.

---

## Project Structure

```
crop-health-monitoring/
├── notebooks/
│   └── setup.ipynb            # Dataset download, EDA, training, per-class evaluation
├── src/
│   ├── disease_classifier.py  # ResNet-18 training script
│   ├── fusion.py              # Environmental risk scoring (rule-based)
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
4. Train ResNet-18 disease classifier → `results/disease_model.pth`
5. Evaluate per-class accuracy and generate confusion matrix → `results/confusion_matrix.png`
6. Train XGBoost risk model → `results/risk_model.pkl`

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

**Scope:** 15 classes, 3 crops, ~41,274 images. Class imbalance: 21× (Tomato\_YellowLeaf\_Curl\_Virus: 6,416 images vs Potato\_healthy: 304 images).

---

## Models

| Model | Architecture | Input | Output |
|---|---|---|---|
| Disease classifier | ResNet-18 (frozen backbone, fine-tuned FC) | Leaf image (lab) | Disease class + confidence |
| Risk layer | XGBoost (rule approximator) | Disease class + temp + humidity + rainfall | Derived risk score 0–1 |

<img width="1189" height="396" alt="download" src="https://github.com/user-attachments/assets/ef73153b-3059-4a57-a8b1-54f9a519f2f7" />

---

## Current Results and Known Issues

**What works:**
- ResNet-18 classifies the 15 PlantVillage classes with high top-line accuracy on the lab validation set
- XGBoost risk layer produces consistent scores; treatment recommendations are agronomically reasonable
- Streamlit interface runs end-to-end: image upload → live weather → risk output → revenue estimate

**Known issues and limitations:**

| Issue | Detail |
|---|---|
| **Lab-to-field gap** | All PlantVillage images are staged on uniform backgrounds. Field photos with soil, shadows, and partial leaves will produce lower accuracy — this has not been formally quantified yet |
| **Class imbalance (21×)** | Top-line accuracy is dominated by Tomato classes. Per-class accuracy (especially Potato\_healthy, ~304 images) is expected to be much lower |
| **XGBoost target is derived** | The risk score is computed from a deterministic formula, not field observations. Val R² ≈ 0.998 reflects formula reproduction only — not real-world predictive validity |
| **Random split inflation** | PlantVillage near-duplicates in both train/val likely inflate reported accuracy. A leaf-level split would be more conservative |
| **Model/data scope mismatch** | The pre-trained `disease_model.pth` was originally fit on a 39-class dataset; retraining on the 15-class local subset is needed for a fully consistent evaluation |

---

## Reproducing Results

After running `notebooks/setup.ipynb` in full, you will find in `results/`:
- `disease_model.pth` — trained ResNet-18 weights and class list
- `risk_model.pkl` — trained XGBoost model, label encoder, feature list
- `disease_training_curves.png` — loss/accuracy per epoch
- `confusion_matrix.png` — per-class confusion matrix
- `eda_plantvillage.png` — class distribution bar chart
- `xgb_risk_evaluation.png` — XGBoost predicted vs. derived target

---

## Author

**Khang Phan**  
GitHub: [@itsmekhang](https://github.com/itsmekhang)  
Course: EGN6217 — Applied Deep Learning, Spring 2026
