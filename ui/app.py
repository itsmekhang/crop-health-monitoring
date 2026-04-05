"""
Streamlit dashboard — Multimodal Crop Health Monitor
Inputs: leaf images + field conditions sliders + field size / price
Output: disease diagnosis + environmental risk + estimated revenue loss
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

# Typical yields (kg/ha) based on global averages for crops the model covers
TYPICAL_YIELD_KG_HA = {"Tomato": 40_000, "Potato": 20_000, "Pepper": 15_000}
DEFAULT_PRICE_PER_KG = {"Tomato": 0.50, "Potato": 0.25, "Pepper": 1.20}

# Disease severity → fraction of yield lost
YIELD_PENALTY = {"None": 0.0, "Moderate": 0.20, "High": 0.35, "Critical": 0.60}


def _crop_family(label: str) -> str:
    if label.lower().startswith("tomato"):
        return "Tomato"
    if label.lower().startswith("potato"):
        return "Potato"
    if label.lower().startswith("pepper"):
        return "Pepper"
    return "Tomato"  # fallback


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


def predict_disease(model, classes, img: Image.Image):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    tensor = transform(img).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        probs = torch.softmax(model(tensor), dim=1)[0].cpu().numpy()
    top_idx = probs.argsort()[::-1][:5]
    return [classes[i] for i in top_idx], [float(probs[i]) for i in top_idx]


def estimate_loss(crop: str, severity: str, field_ha: float, price_per_kg: float):
    typical  = TYPICAL_YIELD_KG_HA.get(crop, 20_000)
    penalty  = YIELD_PENALTY.get(severity, 0.0)
    lost_kg  = typical * penalty * field_ha
    lost_usd = lost_kg * price_per_kg
    return typical, penalty, lost_kg, lost_usd


# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(page_title="Crop Health Monitor", page_icon="🌿", layout="wide")
st.title("Multimodal Crop Health Monitor")
st.caption("Leaf images + field conditions → disease diagnosis, environmental risk, and estimated revenue loss.")

st.divider()

# ── Inputs row ────────────────────────────────────────────────────────────────
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
    st.subheader("3. Economic Impact")
    st.caption("Used to estimate revenue loss from disease.")
    field_ha     = st.number_input("Field size (hectares)", min_value=0.1, max_value=10_000.0,
                                   value=1.0, step=0.5)
    price_per_kg = st.number_input("Crop price (USD per kg)", min_value=0.01, max_value=100.0,
                                   value=0.50, step=0.05,
                                   help="Tomato ≈ $0.50 · Potato ≈ $0.25 · Pepper ≈ $1.20")

st.divider()

if not uploaded_files:
    st.info("Upload at least one leaf image to get started.")
    st.stop()

disease_model, classes = load_disease_model()

if disease_model is None:
    st.error("Disease model not found at `results/disease_model.pth`.")
    st.stop()

# ── Run predictions ───────────────────────────────────────────────────────────
results = []
for f in uploaded_files:
    img = Image.open(f).convert("RGB")
    top_cls, top_conf = predict_disease(disease_model, classes, img)
    label, confidence = top_cls[0], top_conf[0]
    rec      = RECOMMENDATIONS.get(label, {})
    severity = rec.get("severity", "Moderate")
    risk     = assess_risk(label, confidence, severity, temp, humidity, days_since_rain)
    crop     = _crop_family(label)
    typical, penalty, lost_kg, lost_usd = estimate_loss(crop, severity, field_ha, price_per_kg)

    results.append({
        "filename": f.name, "img": img,
        "label": label, "confidence": confidence,
        "top_cls": top_cls, "top_conf": top_conf,
        "rec": rec, "severity": severity, "risk": risk,
        "crop": crop,
        "typical_kg_ha": typical,
        "penalty": penalty,
        "lost_kg": lost_kg,
        "lost_usd": lost_usd,
    })

# ── Field Health Overview ─────────────────────────────────────────────────────
n_healthy  = sum(1 for r in results if "healthy" in r["label"].lower())
n_total    = len(results)
health_pct = n_healthy / n_total * 100
avg_risk   = np.mean([r["risk"]["risk_score"] for r in results])
total_loss = sum(r["lost_usd"] for r in results if "healthy" not in r["label"].lower())

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

# ── Per-plant results ─────────────────────────────────────────────────────────
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
            st.caption(f"Detected crop: **{r['crop']}**")

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

            # Revenue loss estimate
            if "healthy" not in r["label"].lower():
                st.markdown("---")
                st.markdown("**Estimated Revenue Loss**")
                penalty_pct = r["penalty"] * 100
                st.metric("Yield Loss", f"{r['lost_kg']:,.0f} kg",
                          f"{penalty_pct:.0f}% of {r['typical_kg_ha']:,} kg/ha baseline")
                st.metric("Revenue Loss", f"${r['lost_usd']:,.0f}",
                          f"over {field_ha:.1f} ha at ${price_per_kg:.2f}/kg",
                          delta_color="inverse")
            else:
                st.success("No yield loss expected.")

        with col_treat:
            st.markdown("**Treatment Plan**")
            for step in r["rec"].get("treatment", []):
                st.markdown(f"- {step}")
            st.markdown("**Prevention**")
            st.info(r["rec"].get("prevention", ""))
