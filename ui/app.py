"""
Streamlit dashboard for Multimodal Crop Health Monitoring.
Tabs: Disease Detection (image) | Yield Prediction (tabular)
"""

import streamlit as st
import torch
import pickle
import pandas as pd
import numpy as np
from pathlib import Path
from PIL import Image

# ── paths ──────────────────────────────────────────────────────────────────
DISEASE_MODEL = Path("results/disease_model.pth")
YIELD_MODEL   = Path("results/yield_model.pkl")

# ── model loaders (cached) ─────────────────────────────────────────────────
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


# ── page config ────────────────────────────────────────────────────────────
st.set_page_config(page_title="Crop Health Monitor", page_icon="🌿", layout="wide")
st.title("🌿 Multimodal Crop Health Monitoring")
st.caption("Disease detection from leaf images · Yield prediction from environmental data")

tab1, tab2 = st.tabs(["🔬 Disease Detection", "📈 Yield Prediction"])

# ── Tab 1: Disease Detection ───────────────────────────────────────────────
with tab1:
    st.subheader("Upload a Leaf Image")
    st.write("The model classifies the leaf as healthy or identifies the disease type.")

    uploaded = st.file_uploader("Choose a .jpg or .png image", type=["jpg", "jpeg", "png"])

    if uploaded:
        img = Image.open(uploaded).convert("RGB")
        col1, col2 = st.columns([1, 2])

        with col1:
            st.image(img, caption="Uploaded image", use_container_width=True)

        with col2:
            model, classes = load_disease_model()
            if model is None:
                st.warning("Disease model not found. Run `python src/disease_classifier.py` first.")
            else:
                from torchvision import transforms
                transform = transforms.Compose([
                    transforms.Resize((224, 224)),
                    transforms.ToTensor(),
                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
                ])
                tensor = transform(img).unsqueeze(0)
                with torch.no_grad():
                    probs = torch.softmax(model(tensor), dim=1)[0].numpy()

                top5_idx  = probs.argsort()[::-1][:5]
                top5_cls  = [classes[i].replace("_", " ") for i in top5_idx]
                top5_conf = [float(probs[i]) for i in top5_idx]

                pred_label = top5_cls[0]
                pred_conf  = top5_conf[0]

                if "healthy" in pred_label.lower():
                    st.success(f"Result: **{pred_label}** ({pred_conf:.1%} confidence)")
                else:
                    st.error(f"Result: **{pred_label}** ({pred_conf:.1%} confidence)")

                st.write("**Top 5 predictions:**")
                chart_df = pd.DataFrame({"Class": top5_cls, "Confidence": top5_conf})
                st.bar_chart(chart_df.set_index("Class"))

# ── Tab 2: Yield Prediction ────────────────────────────────────────────────
with tab2:
    st.subheader("Predict Crop Yield")
    st.write("Enter environmental conditions to estimate yield in hg/ha.")

    col1, col2 = st.columns(2)
    with col1:
        region = st.selectbox("Region / Country", ["India", "USA", "Brazil", "China", "Australia", "France", "Other"])
        crop   = st.selectbox("Crop Type", ["Wheat", "Rice", "Maize", "Potato", "Soybean", "Cotton"])
        year   = st.slider("Year", 1990, 2025, 2020)
    with col2:
        rainfall    = st.number_input("Avg Rainfall (mm/year)", min_value=0.0, max_value=5000.0, value=1100.0, step=10.0)
        pesticides  = st.number_input("Pesticides (tonnes)", min_value=0.0, max_value=500000.0, value=40000.0, step=1000.0)
        temperature = st.number_input("Avg Temperature (°C)", min_value=-10.0, max_value=50.0, value=20.0, step=0.5)

    if st.button("Predict Yield", type="primary"):
        model = load_yield_model()
        if model is None:
            st.warning("Yield model not found. Run `python src/yield_predictor.py` first.")
        else:
            row = pd.DataFrame([{
                "Area": hash(region) % 100,
                "Item": hash(crop) % 100,
                "Year": year,
                "average_rain_fall_mm_per_year": rainfall,
                "pesticides_tonnes": pesticides,
                "avg_temp": temperature,
            }])
            pred = float(model.predict(row)[0])
            st.metric("Predicted Yield", f"{pred:,.0f} hg/ha", f"≈ {pred/10000:.2f} t/ha")
            st.info("Higher pesticide use, moderate rainfall, and optimal temperature generally improve yield.")
