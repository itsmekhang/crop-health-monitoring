"""
Streamlit dashboard — Crop Disease Detection & Treatment Advisor
Upload one or more leaf images → get disease diagnosis, treatment plan, and field health score.
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


# ── Page config ────────────────────────────────────────────────────────────
st.set_page_config(page_title="Crop Disease Advisor", page_icon="🌿", layout="wide")
st.title("Crop Disease Detection & Treatment Advisor")
st.caption("Upload leaf images from your field — get an instant diagnosis and treatment plan.")

st.divider()

# ── Upload ─────────────────────────────────────────────────────────────────
uploaded_files = st.file_uploader(
    "Upload leaf images (one per plant sample — upload multiple for a field health score)",
    type=["jpg", "jpeg", "png"],
    accept_multiple_files=True,
)

if not uploaded_files:
    st.info("Upload one or more leaf images to get started.")
    st.stop()

model, classes = load_model()
if model is None:
    st.error("Disease model not found at `results/disease_model.pth`.")
    st.stop()

st.divider()

# ── Run predictions ────────────────────────────────────────────────────────
results = []
for f in uploaded_files:
    img = Image.open(f).convert("RGB")
    top_cls, top_conf = predict(model, classes, img)
    results.append({
        "filename": f.name,
        "img": img,
        "label": top_cls[0],
        "confidence": top_conf[0],
        "top_cls": top_cls,
        "top_conf": top_conf,
        "rec": RECOMMENDATIONS.get(top_cls[0], {}),
    })

# ── Field Health Score ─────────────────────────────────────────────────────
n_healthy  = sum(1 for r in results if "healthy" in r["label"].lower())
n_total    = len(results)
health_pct = n_healthy / n_total * 100

st.subheader("Field Health Score")
col_score, col_bar = st.columns([1, 3])
with col_score:
    color = "green" if health_pct >= 70 else "orange" if health_pct >= 40 else "red"
    st.markdown(f"<h1 style='color:{color}'>{health_pct:.0f}%</h1>", unsafe_allow_html=True)
    st.caption(f"{n_healthy} of {n_total} plants healthy")
with col_bar:
    st.progress(int(health_pct))
    if health_pct >= 70:
        st.success("Field is in good health.")
    elif health_pct >= 40:
        st.warning("Field has moderate disease pressure — action recommended.")
    else:
        st.error("Field is severely affected — immediate treatment required.")

# Severity summary
severities = [r["rec"].get("severity", "Unknown") for r in results if "healthy" not in r["label"].lower()]
if severities:
    critical = severities.count("Critical")
    high     = severities.count("High")
    moderate = severities.count("Moderate")
    st.caption(f"Diseased plants — Critical: {critical} | High: {high} | Moderate: {moderate}")

st.divider()

# ── Per-image results ──────────────────────────────────────────────────────
st.subheader("Plant-by-Plant Results")

for r in results:
    with st.expander(f"{r['filename']} — {r['label'].replace('_', ' ')} ({r['confidence']:.1%})", expanded=True):
        col_img, col_diag, col_treat = st.columns([1, 1, 2])

        with col_img:
            st.image(r["img"], use_container_width=True)

        with col_diag:
            st.markdown("**Diagnosis**")
            severity = r["rec"].get("severity", "Unknown")
            color    = SEVERITY_COLOR.get(severity, "gray")

            if "healthy" in r["label"].lower():
                st.success(f"**{r['label'].replace('_', ' ')}**")
            elif severity == "Critical":
                st.error(f"**{r['label'].replace('_', ' ')}**")
            else:
                st.warning(f"**{r['label'].replace('_', ' ')}**")

            st.markdown(f"Severity: **:{color}[{severity}]**")
            st.markdown(f"Confidence: **{r['confidence']:.1%}**")
            st.caption(r["rec"].get("description", ""))

            st.markdown("**Top 5 Predictions**")
            for cls, conf in zip(r["top_cls"], r["top_conf"]):
                st.progress(conf, text=f"{cls.replace('_', ' ')} ({conf:.1%})")

        with col_treat:
            st.markdown("**Treatment Plan**")
            for step in r["rec"].get("treatment", []):
                st.markdown(f"- {step}")

            st.markdown("**Prevention**")
            st.info(r["rec"].get("prevention", ""))
