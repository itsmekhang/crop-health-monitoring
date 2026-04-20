# Multimodal Crop Health Monitor — Project Report

**Author:** Khang Phan
**Course:** Spring 2026

---

## 1. Problem Statement

Crop disease is one of the leading causes of agricultural yield loss worldwide. Early detection is critical — diseases like Potato Late Blight can destroy an entire field within days under the right conditions. Traditional detection relies on manual inspection, which is slow, inconsistent, and often too late.

This project builds a multimodal AI system that combines:
- **Computer vision** to identify disease from a leaf photo
- **Live weather data** to assess how urgently the disease needs to be treated given current environmental conditions

The output is a real-time diagnosis with a risk level, treatment plan, and estimated revenue loss.

---

## 2. Dataset

### PlantVillage
- **Source:** [gabrieldgf4/PlantVillage-Dataset](https://github.com/gabrieldgf4/PlantVillage-Dataset)
- **Size:** ~54,000 labeled leaf images across 38 disease/healthy classes
- **Crops covered:** Tomato, Potato, Pepper (Bell), Apple, Grape, Corn, and others
- **Format:** RGB images organized into class folders

### Open-Meteo
- **Source:** [open-meteo.com](https://open-meteo.com) — free, no API key required
- **Used for:** Live temperature, humidity, and days since last rainfall at inference time
- **Geocoding:** City name → latitude/longitude via Open-Meteo geocoding API

---

## 3. Architecture

The system uses two independent models that operate on different modalities:

```
[Leaf Image]   →  ResNet-18  →  disease class, confidence
[Location]     →  Open-Meteo →  temperature, humidity, days since rain
                                          ↓
                                      XGBoost
                                          ↓
                              risk score (0–1) → Critical / High / Moderate / Low
```

### Why keep the models separate?

A diseased leaf looks the same regardless of whether it is raining outside. Feeding weather into the image classifier would add noise without adding signal — the visual features of Late Blight do not change based on humidity. Weather only determines how fast the disease will spread and how urgently treatment is needed.

Keeping the models separate produces a cleaner, more interpretable system:
- ResNet answers: *what disease is this?*
- XGBoost answers: *how dangerous is it right now given conditions?*

---

## 4. Model Details

### ResNet-18 Disease Classifier
- **Architecture:** ResNet-18 pretrained on ImageNet, backbone frozen, final FC layer replaced and fine-tuned
- **Training:** 10 epochs, Adam optimizer (lr=1e-3), StepLR scheduler
- **Input:** 224×224 RGB leaf image
- **Output:** Probability distribution over 38 disease classes
- **Augmentation:** Random horizontal flip, rotation (±15°), color jitter

**Frozen backbone note:** All convolutional weights are frozen and only the FC head is trained. This is a reasonable baseline given limited compute, but PlantVillage images (uniform backgrounds, controlled lighting) differ substantially from the ImageNet distribution the backbone was trained on. A next step is to unfreeze the last residual block (`layer4`) and fine-tune it at a lower learning rate (1e-4 vs. 1e-3 for the head). This allows the network to adapt mid-level texture representations toward disease-specific patterns without the instability of full fine-tuning.

### XGBoost Risk Scorer
- **Input features:** temperature (°C), humidity (%), days since rain, ResNet confidence, disease class (encoded)
- **Output:** Risk score in [0, 1]
- **Training data:** Synthetic samples generated from agronomic disease-weather distributions (e.g. fungal diseases paired with high humidity, viral diseases with hot/dry conditions)
- **Performance:** Val R² = 0.998, Val MAE = 0.008

### Risk Formula Weight Justification

The fallback risk formula combines three terms:

```
risk = (severity_base × 0.4) + (confidence × 0.3) + (env_score × 0.3)
```

- **severity_base (0.4):** The agronomic severity classification is the primary signal — it reflects how destructive the disease is independent of current conditions. It carries the largest weight.
- **confidence (0.3):** A 95%-confident detection warrants a stronger response than a 55% one. Confidence scales the reliability of the diagnosis without overriding it.
- **env_score (0.3):** Environmental conditions govern spread rate and urgency, not the underlying pathology. They modulate the recommendation but are weighted below the diagnosis itself.

The per-disease environmental weights follow the same logic:

| Disease type | Dominant factor | Weight rationale |
|---|---|---|
| Fungal | Humidity (0.5) | Humidity is the primary sporulation trigger (Agrios, 2005); temp sets a permissive range (0.3); recent rain adds surface moisture that decays quickly (0.2) |
| Bacterial | Humidity = Temp (0.4 each) | Warm temperature and leaf wetness are roughly equally necessary for infection (Gitaitis & Walcott, 2007); rainfall is secondary (0.2) |
| Viral | Temperature (0.6) | Insect vector activity (whiteflies, aphids) is primarily temperature-driven; low humidity increases vector mobility (0.4); rainfall excluded as it suppresses vectors transiently |
| Mites | Temp = Dry air (0.5 each) | Spider mite reproduction doubles per 4 °C above 28 °C (Sabelis, 1985); both heat and low humidity are required simultaneously |

### Risk Levels
| Score | Level | Action |
|---|---|---|
| ≥ 0.80 | Critical | Treat within 24 hours |
| ≥ 0.55 | High | Treat within 2–3 days |
| ≥ 0.30 | Moderate | Monitor closely |
| < 0.30 | Low | Continue monitoring |

---

## 5. Application

The Streamlit dashboard has three input sections:

1. **Leaf Images** — upload one or more photos
2. **Weather Conditions** — enter a city name, live weather is fetched automatically from Open-Meteo
3. **Economic Impact** — field size (hectares) and crop price (USD/kg)

Outputs per image:
- Disease class + confidence + top-5 predictions
- Severity level and description
- Combined risk score from XGBoost
- Treatment plan and prevention advice
- Estimated yield loss (kg) and revenue loss (USD)

Field-level summary:
- % plants healthy
- Average risk score across all uploaded images
- Overall field status

---

## 6. Limitations

- **PlantVillage is a controlled dataset** — images were taken in lab/greenhouse conditions. Real field photos with dirt, overlapping leaves, or partial damage may perform worse.
- **Location-image coupling is unverified** — the system assumes the uploaded photo was taken at the city the user enters. The entire risk score depends on this being true, but there is no verification. A user uploading a photo from one region while entering a different city will receive a risk score based on the wrong weather. This is an inherent limitation of the two-input design and should be made visible to the user at the point of entry.
- **XGBoost trained on synthetic weather** — the risk model learned from agronomically-informed distributions rather than real field outbreak records. The correlations are scientifically grounded but not empirically validated.
- **Revenue loss is approximate** — yield baselines (kg/ha) are global averages and will not reflect local farming conditions accurately.
- **38 classes but limited crops** — the app provides treatment plans only for Tomato, Potato, and Pepper. Other classes are classified but show no recommendations.

---

## 7. Technologies

| Component | Technology |
|---|---|
| Disease classifier | PyTorch, torchvision (ResNet-18) |
| Risk scorer | XGBoost, scikit-learn |
| Weather API | Open-Meteo (free, no key) |
| Dashboard | Streamlit |
| Training environment | Google Colab (T4 GPU) |

---

## 8. References

- Hughes, D. & Salathé, M. (2015). An open access repository of images on plant health to enable the development of mobile disease diagnostics. *arXiv:1511.08060*
- Záruba, G. (2023). PlantVillage Dataset. GitHub: gabrieldgf4/PlantVillage-Dataset
- Open-Meteo. (2024). Free Weather API. https://open-meteo.com
- Chen, T. & Guestrin, C. (2016). XGBoost: A Scalable Tree Boosting System. *KDD 2016*
