"""
Streamlit dashboard — Multimodal Crop Health Monitor
Inputs: leaf images + field conditions sliders + farmer CSV
Output: disease diagnosis + yield impact + combined risk report
"""

import sys
import pickle
import streamlit as st
import torch
import torch.nn as nn
from torchvision import transforms, models
from pathlib import Path
from PIL import Image
import pandas as pd
import numpy as np

sys.path.append(str(Path(__file__).parent.parent))
from src.recommendations import RECOMMENDATIONS, SEVERITY_COLOR
from src.fusion import assess_risk

DISEASE_MODEL = Path("results/disease_model.pth")
YIELD_MODEL   = Path("results/yield_model.pkl")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

REQUIRED_CSV_COLS = {"Crop_Type", "Soil_Type", "pH", "N", "P", "K",
                     "Irrigation_Frequency", "Fertilizer_Type", "Pesticide_Usage"}

# Yield penalty per severity (how much disease reduces expected yield)
YIELD_PENALTY = {"None": 0.0, "Moderate": 0.20, "High": 0.35, "Critical": 0.60}


@st.cache_resource
def load_disease_model():
    if not DISEASE_MODEL.exists():
        return None, None
    checkpoint = torch.load(DISEASE_MODEL, map_location=DEVICE, weights_only=False)
    classes = checkpoint["classes"]
    model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, len(classes))
    model.load_state_dict(checkpoint["model_state"])
    model.to(DEVICE)
    model.eval()
    return model, classes


@st.cache_resource
def load_yield_model():
    if not YIELD_MODEL.exists():
        return None
    with open(YIELD_MODEL, "rb") as f:
        return pickle.load(f)


def predict_disease(model, classes, img: Image.Image):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    tensor = transform(img).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        probs = torch.softmax(model(tensor), dim=1)[0].cpu().numpy()
    top_idx  = probs.argsort()[::-1][:5]
    return [classes[i] for i in top_idx], [float(probs[i]) for i in top_idx]


def predict_yield(bundle, row, temp, humidity, rainfall):
    model    = bundle["model"]
    encoders = bundle["encoders"]
    features = bundle["features"]

    record = {
        "Temperature": temp,
        "Humidity": humidity,
        "Rainfall": rainfall,
        "pH": row.get("pH", 6.5),
        "N": row.get("N", 100.0),
        "P": row.get("P", 50.0),
        "K": row.get("K", 150.0),
        "Irrigation_Frequency": row.get("Irrigation_Frequency", 5),
    }
    for col in ["Crop_Type", "Soil_Type", "Fertilizer_Type", "Pesticide_Usage"]:
        le = encoders.get(col)
        val = str(row.get(col, ""))
        try:
            record[f"{col}_enc"] = le.transform([val])[0]
        except (ValueError, AttributeError):
            record[f"{col}_enc"] = 0

    X = pd.DataFrame([record])[features]
    return float(model.predict(X)[0])


# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(page_title="Crop Health Monitor", page_icon="🌿", layout="wide")
st.title("Multimodal Crop Health Monitor")
st.caption("Leaf images + field conditions + farm CSV → disease diagnosis, yield impact, and risk report.")

st.divider()

# ── Inputs row ───────────────────────────────────────────────────────────────
col1, col2, col3 = st.columns(3)

with col1:
    st.subheader("1. Leaf Images")
    uploaded_files = st.file_uploader(
        "Upload one or more leaf photos",
        type=["jpg", "jpeg", "png"],
        accept_multiple_files=True,
    )

with col2:
    st.subheader("2. Weather Conditions")
    st.caption("Used to assess how fast disease will spread.")
    temp            = st.slider("Temperature (°C)", -10, 50, 24)
    humidity        = st.slider("Humidity (%)", 0, 100, 70)
    days_since_rain = st.slider("Days since last rainfall", 0, 30, 2)

    if temp >= 28 and humidity >= 75:
        st.warning("Hot & humid — high fungal risk")
    elif temp >= 28 and humidity < 50:
        st.warning("Hot & dry — watch for mites/pests")
    else:
        st.success("Conditions are moderate")

with col3:
    st.subheader("3. Farm Data CSV")
    st.caption("Your field records — used for yield prediction.")

    sample_csv = pd.DataFrame([{
        "Crop_Type": "Maize",
        "Soil_Type": "Loamy",
        "pH": 6.5,
        "N": 120.0,
        "P": 60.0,
        "K": 180.0,
        "Irrigation_Frequency": 5,
        "Fertilizer_Type": "Chemical",
        "Pesticide_Usage": "Low",
    }])
    st.download_button("Download CSV Template", sample_csv.to_csv(index=False),
                       "field_data_template.csv", "text/csv")

    uploaded_csv = st.file_uploader("Upload your field data (.csv)", type=["csv"])
    field_df = None
    if uploaded_csv:
        field_df = pd.read_csv(uploaded_csv)
        missing = REQUIRED_CSV_COLS - set(field_df.columns)
        if missing:
            st.error(f"Missing columns: {missing}")
            field_df = None
        else:
            st.dataframe(field_df, use_container_width=True)

st.divider()

if not uploaded_files:
    st.info("Upload at least one leaf image to get started.")
    st.stop()

disease_model, classes = load_disease_model()
yield_bundle = load_yield_model()

if disease_model is None:
    st.error("Disease model not found at `results/disease_model.pth`.")
    st.stop()

# ── Run predictions ──────────────────────────────────────────────────────────
results = []
for i, f in enumerate(uploaded_files):
    img = Image.open(f).convert("RGB")
    top_cls, top_conf = predict_disease(disease_model, classes, img)
    label, confidence = top_cls[0], top_conf[0]
    rec      = RECOMMENDATIONS.get(label, {})
    severity = rec.get("severity", "Moderate")
    risk     = assess_risk(label, confidence, severity, temp, humidity, days_since_rain)

    # Yield prediction — use matching CSV row if available, else first row
    baseline_yield, adjusted_yield = None, None
    csv_row = None
    if field_df is not None:
        csv_row = field_df.iloc[i] if i < len(field_df) else field_df.iloc[0]
        if yield_bundle:
            # days_since_rain → approximate mm by scaling (0d=200mm, 30d=0mm)
            approx_rainfall = max(0, 200 - days_since_rain * 6.5)
            baseline_yield  = predict_yield(yield_bundle, csv_row, temp, humidity, approx_rainfall)
            penalty         = YIELD_PENALTY.get(severity, 0.2)
            adjusted_yield  = baseline_yield * (1 - penalty)

    results.append({
        "filename": f.name, "img": img,
        "label": label, "confidence": confidence,
        "top_cls": top_cls, "top_conf": top_conf,
        "rec": rec, "severity": severity, "risk": risk,
        "baseline_yield": baseline_yield,
        "adjusted_yield": adjusted_yield,
        "csv_row": csv_row,
    })

# ── Field Health Overview ────────────────────────────────────────────────────
n_healthy  = sum(1 for r in results if "healthy" in r["label"].lower())
n_total    = len(results)
health_pct = n_healthy / n_total * 100
avg_risk   = np.mean([r["risk"]["risk_score"] for r in results])

st.subheader("Field Health Overview")
c1, c2, c3, c4 = st.columns(4)
color = "green" if health_pct >= 70 else "orange" if health_pct >= 40 else "red"
c1.markdown(f"<h2 style='color:{color}'>{health_pct:.0f}%</h2>**Plants Healthy**", unsafe_allow_html=True)
c2.metric("Plants Scanned", n_total)
c3.metric("Diseased", n_total - n_healthy)
c4.metric("Avg Risk Score", f"{avg_risk:.2f} / 1.0")
st.progress(int(health_pct))

if health_pct >= 70:
    st.success("Field is in good shape under current conditions.")
elif health_pct >= 40:
    st.warning("Moderate disease pressure — current weather may accelerate spread.")
else:
    st.error("Severe outbreak — treat immediately. Current conditions increase spread risk.")

st.divider()

# ── Per-plant results ────────────────────────────────────────────────────────
st.subheader("Plant-by-Plant Report")

for r in results:
    risk    = r["risk"]
    label   = r["label"].replace("_", " ")
    r_level = risk["risk_level"]
    r_color = risk["color"]

    with st.expander(f"{r['filename']} — {label} | Risk: {r_level}", expanded=True):
        col_img, col_diag, col_fuse, col_treat = st.columns([1, 1, 1, 2])

        with col_img:
            st.image(r["img"], use_container_width=True)
            if r["csv_row"] is not None:
                st.caption(f"Crop: {r['csv_row']['Item']} | Region: {r['csv_row']['Area']}")

        with col_diag:
            st.markdown("**Diagnosis**")
            if "healthy" in r["label"].lower():
                st.success(f"**{label}**")
            elif r["severity"] == "Critical":
                st.error(f"**{label}**")
            else:
                st.warning(f"**{label}**")

            st.markdown(f"Severity: **:{SEVERITY_COLOR.get(r['severity'],'gray')}[{r['severity']}]**")
            st.markdown(f"Confidence: **{r['confidence']:.1%}**")
            st.caption(r["rec"].get("description", ""))

            st.markdown("**Top Predictions**")
            for cls, conf in zip(r["top_cls"], r["top_conf"]):
                st.progress(conf, text=f"{cls.replace('_',' ')} ({conf:.1%})")

        with col_fuse:
            st.markdown("**Combined Risk Assessment**")
            st.markdown(f"<h2 style='color:{r_color}'>{r_level}</h2>", unsafe_allow_html=True)
            st.progress(risk["risk_score"], text=f"Risk score: {risk['risk_score']:.2f}")
            st.markdown(f"**Action:** {risk['action']}")
            if risk["env_warning"]:
                st.caption(risk["env_warning"])
            st.caption(f"Temp {temp}°C · Humidity {humidity}% · {days_since_rain}d dry")

            # Yield impact
            if r["baseline_yield"] is not None:
                st.markdown("---")
                st.markdown("**Yield Impact**")
                penalty_pct = YIELD_PENALTY.get(r["severity"], 0) * 100
                st.metric(
                    "Baseline Yield",
                    f"{r['baseline_yield']:.2f} t/ha",
                )
                st.metric(
                    "Est. Yield with Disease",
                    f"{r['adjusted_yield']:.2f} t/ha",
                    f"-{penalty_pct:.0f}% from disease",
                    delta_color="inverse",
                )
            elif field_df is None:
                st.caption("Upload a farm CSV to see yield impact.")

        with col_treat:
            st.markdown("**Treatment Plan**")
            for step in r["rec"].get("treatment", []):
                st.markdown(f"- {step}")
            st.markdown("**Prevention**")
            st.info(r["rec"].get("prevention", ""))
