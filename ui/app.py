"""
Streamlit dashboard — Multimodal Crop Health Monitor
Image (leaf photo) + Tabular (field conditions) → unified risk report
"""

import sys
import streamlit as st
import torch
import torch.nn as nn
from torchvision import transforms, models
from pathlib import Path
from PIL import Image
import numpy as np

sys.path.append(str(Path(__file__).parent.parent))
from src.recommendations import RECOMMENDATIONS, SEVERITY_COLOR
from src.fusion import assess_risk

DISEASE_MODEL = Path("results/disease_model.pth")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


@st.cache_resource
def load_model():
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


def predict(model, classes, img: Image.Image):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    tensor = transform(img).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        probs = torch.softmax(model(tensor), dim=1)[0].cpu().numpy()
    top_idx  = probs.argsort()[::-1][:5]
    top_cls  = [classes[i] for i in top_idx]
    top_conf = [float(probs[i]) for i in top_idx]
    return top_cls, top_conf


# ── Page config ─────────────────────────────────────────────────────────────
st.set_page_config(page_title="Crop Health Monitor", page_icon="🌿", layout="wide")
st.title("Multimodal Crop Health Monitor")
st.caption("Combine leaf images with field conditions for a risk-adjusted disease report.")

st.divider()

# ── Inputs ───────────────────────────────────────────────────────────────────
col_left, col_right = st.columns(2)

with col_left:
    st.subheader("1. Leaf Images")
    uploaded_files = st.file_uploader(
        "Upload one or more leaf photos",
        type=["jpg", "jpeg", "png"],
        accept_multiple_files=True,
    )

with col_right:
    st.subheader("2. Field Conditions")
    st.caption("These are combined with the image results to assess spread risk.")
    temp           = st.slider("Temperature (°C)", -10, 50, 24)
    humidity       = st.slider("Humidity (%)", 0, 100, 70)
    days_since_rain = st.slider("Days since last rainfall", 0, 30, 2)

    st.markdown("")
    col_a, col_b = st.columns(2)
    with col_a:
        if temp >= 28 and humidity >= 75:
            st.warning("Hot & humid — high fungal risk")
        elif temp >= 28 and humidity < 50:
            st.warning("Hot & dry — watch for mites/pests")
        else:
            st.success("Conditions are moderate")
    with col_b:
        st.metric("Temp", f"{temp}°C")
        st.metric("Humidity", f"{humidity}%")
        st.metric("Days dry", days_since_rain)

st.divider()

if not uploaded_files:
    st.info("Upload at least one leaf image to get started.")
    st.stop()

model, classes = load_model()
if model is None:
    st.error("Disease model not found at `results/disease_model.pth`.")
    st.stop()

# ── Run predictions ───────────────────────────────────────────────────────────
results = []
for f in uploaded_files:
    img = Image.open(f).convert("RGB")
    top_cls, top_conf = predict(model, classes, img)
    label, confidence = top_cls[0], top_conf[0]
    rec = RECOMMENDATIONS.get(label, {})
    severity = rec.get("severity", "Moderate")
    risk = assess_risk(label, confidence, severity, temp, humidity, days_since_rain)
    results.append({
        "filename": f.name,
        "img": img,
        "label": label,
        "confidence": confidence,
        "top_cls": top_cls,
        "top_conf": top_conf,
        "rec": rec,
        "severity": severity,
        "risk": risk,
    })

# ── Field Health Score ────────────────────────────────────────────────────────
n_healthy   = sum(1 for r in results if "healthy" in r["label"].lower())
n_total     = len(results)
health_pct  = n_healthy / n_total * 100
avg_risk    = np.mean([r["risk"]["risk_score"] for r in results])

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
    st.error("Severe disease outbreak — treat immediately. Current conditions increase spread risk.")

st.divider()

# ── Per-plant results ─────────────────────────────────────────────────────────
st.subheader("Plant-by-Plant Report")

for r in results:
    risk     = r["risk"]
    label    = r["label"].replace("_", " ")
    severity = r["severity"]
    r_level  = risk["risk_level"]
    r_color  = risk["color"]

    header = f"{r['filename']} — {label} | Risk: {r_level}"
    with st.expander(header, expanded=True):
        col_img, col_diag, col_fuse, col_treat = st.columns([1, 1, 1, 2])

        # Image
        with col_img:
            st.image(r["img"], use_container_width=True)

        # Disease diagnosis
        with col_diag:
            st.markdown("**Diagnosis**")
            if "healthy" in r["label"].lower():
                st.success(f"**{label}**")
            elif severity == "Critical":
                st.error(f"**{label}**")
            else:
                st.warning(f"**{label}**")

            st.markdown(f"Severity: **:{SEVERITY_COLOR.get(severity,'gray')}[{severity}]**")
            st.markdown(f"Confidence: **{r['confidence']:.1%}**")
            st.caption(r["rec"].get("description", ""))

            st.markdown("**Top Predictions**")
            for cls, conf in zip(r["top_cls"], r["top_conf"]):
                st.progress(conf, text=f"{cls.replace('_',' ')} ({conf:.1%})")

        # Multimodal fusion result
        with col_fuse:
            st.markdown("**Combined Risk Assessment**")
            st.markdown(
                f"<h2 style='color:{r_color}'>{r_level}</h2>",
                unsafe_allow_html=True,
            )
            st.progress(risk["risk_score"], text=f"Risk score: {risk['risk_score']:.2f}")
            st.markdown(f"**Action:** {risk['action']}")

            if risk["env_warning"]:
                st.caption(f"**Current conditions:** {risk['env_warning']}")

            st.markdown("---")
            st.caption(f"Environmental factor: {risk['env_multiplier']:.0%} of max")
            st.caption(f"Temp {temp}°C · Humidity {humidity}% · {days_since_rain}d dry")

        # Treatment
        with col_treat:
            st.markdown("**Treatment Plan**")
            for step in r["rec"].get("treatment", []):
                st.markdown(f"- {step}")
            st.markdown("**Prevention**")
            st.info(r["rec"].get("prevention", ""))
