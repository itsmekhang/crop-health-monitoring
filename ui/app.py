"""
Streamlit dashboard — Multimodal Crop Health Monitor
Inputs: leaf image + farmer CSV → combined disease + yield report
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

# Expected CSV columns
REQUIRED_COLS = {"average_rain_fall_mm_per_year", "pesticides_tonnes", "avg_temp", "Item", "Area", "Year"}

@st.cache_resource
def load_disease_model():
    if not DISEASE_MODEL.exists():
        return None, None
    from src.disease_classifier import build_model, DEVICE
    checkpoint = torch.load(DISEASE_MODEL, map_location=DEVICE, weights_only=False)
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
st.caption("Upload a leaf image and your field data CSV to get a combined crop health report.")

# CSV template download
sample_csv = pd.DataFrame([{
    "Area": "India",
    "Item": "Maize",
    "Year": 2024,
    "average_rain_fall_mm_per_year": 1200.0,
    "pesticides_tonnes": 500.0,
    "avg_temp": 22.5,
}])
st.download_button(
    "Download CSV Template",
    data=sample_csv.to_csv(index=False),
    file_name="field_data_template.csv",
    mime="text/csv",
)

st.divider()

# ── Inputs ─────────────────────────────────────────────────────────────────
col_img, col_csv = st.columns(2)

with col_img:
    st.subheader("1. Leaf Image")
    uploaded_img = st.file_uploader("Upload a leaf photo (.jpg / .png)", type=["jpg", "jpeg", "png"])
    if uploaded_img:
        img = Image.open(uploaded_img).convert("RGB")
        st.image(img, use_container_width=True)

with col_csv:
    st.subheader("2. Field Data CSV")
    uploaded_csv = st.file_uploader("Upload your field data (.csv)", type=["csv"])

    if uploaded_csv:
        field_df = pd.read_csv(uploaded_csv)
        st.dataframe(field_df, use_container_width=True)

        missing = REQUIRED_COLS - set(field_df.columns)
        if missing:
            st.error(f"Missing columns: {missing}. Download the template above.")
            uploaded_csv = None
    else:
        st.info("No CSV uploaded — download the template above, fill it in, and upload it here.")

st.divider()

# ── Run ────────────────────────────────────────────────────────────────────
run = st.button("Generate Crop Health Report", type="primary", use_container_width=True)

if run:
    if not uploaded_img:
        st.warning("Please upload a leaf image.")
        st.stop()
    if not uploaded_csv:
        st.warning("Please upload a field data CSV.")
        st.stop()

    st.subheader("Crop Health Report")

    for i, row_data in field_df.iterrows():
        if len(field_df) > 1:
            st.markdown(f"**Row {i+1} — {row_data.get('Item','Crop')} | {row_data.get('Area','Region')} | {row_data.get('Year','')}**")

        res_disease, res_yield = st.columns(2)

        # ── Disease result ──────────────────────────────────────────────────
        with res_disease:
            st.markdown("#### Disease Detection")
            disease_model, classes = load_disease_model()

            if disease_model is None:
                st.warning("Disease model not found. Run `src/disease_classifier.py` first.")
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

        # ── Yield result ────────────────────────────────────────────────────
        with res_yield:
            st.markdown("#### Yield Prediction")
            yield_bundle = load_yield_model()

            if yield_bundle is None:
                st.warning("Yield model not found. Run `src/yield_predictor.py` first.")
            else:
                model   = yield_bundle["model"]
                le_area = yield_bundle["le_area"]
                le_item = yield_bundle["le_item"]

                # Encode — handle unseen labels gracefully
                try:
                    area_enc = le_area.transform([row_data["Area"]])[0]
                except ValueError:
                    area_enc = 0
                try:
                    item_enc = le_item.transform([row_data["Item"]])[0]
                except ValueError:
                    item_enc = 0

                X = pd.DataFrame([{
                    "Area_enc": area_enc,
                    "Item_enc": item_enc,
                    "Year": row_data["Year"],
                    "average_rain_fall_mm_per_year": row_data["average_rain_fall_mm_per_year"],
                    "pesticides_tonnes": row_data["pesticides_tonnes"],
                    "avg_temp": row_data["avg_temp"],
                }])

                pred = float(model.predict(X)[0])
                st.metric("Predicted Yield", f"{pred:,.0f} hg/ha", f"≈ {pred/10000:.2f} t/ha")

                if pred >= 50000:
                    st.success("Yield outlook is strong.")
                elif pred >= 25000:
                    st.info("Yield outlook is moderate.")
                else:
                    st.warning("Yield outlook is low — review field conditions.")

        if len(field_df) > 1 and i < len(field_df) - 1:
            st.divider()
