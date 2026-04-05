"""
Streamlit dashboard — Multimodal Crop Health Monitor
Single unified input: leaf image + environmental data → combined report
"""

import streamlit as st
import torch
import pickle
import pandas as pd
import numpy as np
from pathlib import Path
from PIL import Image

DISEASE_MODEL = Path("results/disease_model.pth")
YIELD_MODEL   = Path("results/yield_model.pkl")

@st.cache_resource
def load_disease_model():
    if not DISEASE_MODEL.exists():
        return None, None
    from src.disease_classifier import build_model, DEVICE
    checkpoint = torch.load(DISEASE_MODEL, map_location=DEVICE)
    classes = checkpoint["classes"]
    model = build_model(num_classes=len(classes))
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    return model, classes

@st.cache_resource
def load_yield_model():
    if not YIELD_MODEL.exists():
        return None
    with open(YIELD_MODEL, "rb") as f:
        return pickle.load(f)

# ── Page config ────────────────────────────────────────────────────────────
st.set_page_config(page_title="Crop Health Monitor", page_icon="🌿", layout="wide")
st.title("Multimodal Crop Health Monitor")
st.caption("Upload a leaf image and enter field conditions to get a combined crop health report.")

st.divider()

# ── Inputs ─────────────────────────────────────────────────────────────────
col_img, col_env = st.columns(2)

with col_img:
    st.subheader("Leaf Image")
    uploaded = st.file_uploader("Upload a leaf photo (.jpg / .png)", type=["jpg", "jpeg", "png"])
    if uploaded:
        img = Image.open(uploaded).convert("RGB")
        st.image(img, use_container_width=True)

with col_env:
    st.subheader("Field Conditions")
    region   = st.selectbox("Region / Country", ["India", "USA", "Brazil", "China", "Australia", "France", "Other"])
    crop     = st.selectbox("Crop Type", ["Wheat", "Rice", "Maize", "Potato", "Soybean", "Cotton"])
    year     = st.slider("Year", 1990, 2025, 2023)
    rainfall = st.number_input("Avg Rainfall (mm/year)", min_value=0.0, max_value=5000.0, value=1100.0, step=10.0)
    pesticides = st.number_input("Pesticides (tonnes)", min_value=0.0, max_value=500000.0, value=40000.0, step=1000.0)
    temp     = st.number_input("Avg Temperature (°C)", min_value=-10.0, max_value=50.0, value=20.0, step=0.5)

st.divider()

# ── Run button ─────────────────────────────────────────────────────────────
run = st.button("Generate Crop Health Report", type="primary", use_container_width=True)

if run:
    if not uploaded:
        st.warning("Please upload a leaf image before running.")
        st.stop()

    st.subheader("Crop Health Report")
    res_disease, res_yield = st.columns(2)

    # ── Disease result ──────────────────────────────────────────────────────
    with res_disease:
        st.markdown("#### Disease Detection")
        disease_model, classes = load_disease_model()

        if disease_model is None:
            st.warning("Disease model not trained yet. Run `src/disease_classifier.py` first.")
        else:
            from torchvision import transforms
            transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ])
            tensor = transform(img).unsqueeze(0)
            with torch.no_grad():
                probs = torch.softmax(disease_model(tensor), dim=1)[0].numpy()

            top5_idx  = probs.argsort()[::-1][:5]
            top5_cls  = [classes[i].replace("_", " ") for i in top5_idx]
            top5_conf = [float(probs[i]) for i in top5_idx]
            label, conf = top5_cls[0], top5_conf[0]

            if "healthy" in label.lower():
                st.success(f"**{label}**  \n{conf:.1%} confidence")
            else:
                st.error(f"**{label}**  \n{conf:.1%} confidence")

            chart_df = pd.DataFrame({"Confidence": top5_conf}, index=top5_cls)
            st.bar_chart(chart_df)

    # ── Yield result ────────────────────────────────────────────────────────
    with res_yield:
        st.markdown("#### Yield Prediction")
        yield_model = load_yield_model()

        if yield_model is None:
            st.warning("Yield model not trained yet. Run `src/yield_predictor.py` first.")
        else:
            row = pd.DataFrame([{
                "Area": hash(region) % 100,
                "Item": hash(crop) % 100,
                "Year": year,
                "average_rain_fall_mm_per_year": rainfall,
                "pesticides_tonnes": pesticides,
                "avg_temp": temp,
            }])
            pred = float(yield_model.predict(row)[0])
            st.metric("Predicted Yield", f"{pred:,.0f} hg/ha", f"≈ {pred/10000:.2f} t/ha")

            # Simple context based on yield
            if pred >= 50000:
                st.success("Yield outlook is strong given these conditions.")
            elif pred >= 25000:
                st.info("Yield outlook is moderate.")
            else:
                st.warning("Yield outlook is low — consider adjusting inputs.")
