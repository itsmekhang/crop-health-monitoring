"""
Streamlit dashboard — Multimodal Crop Health Monitor
Inputs: leaf images + live weather (Open-Meteo) + field size / price
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
from src.multimodal_model import MultimodalCropNet
from src.weather import geocode, fetch_weather

DISEASE_MODEL = Path("results/disease_model.pth")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Typical yields in lbs/acre (US averages)
TYPICAL_YIELD_LBS_ACRE = {"Tomato": 35_700, "Potato": 41_000, "Pepper": 13_400}
YIELD_PENALTY          = {"None": 0.0, "Moderate": 0.20, "High": 0.35, "Critical": 0.60}

# Maps new dataset class names → keys used in RECOMMENDATIONS / fusion.py
CLASS_NAME_MAP = {
    "Tomato___Bacterial_spot":                          "Tomato_Bacterial_spot",
    "Tomato___Early_blight":                            "Tomato_Early_blight",
    "Tomato___Late_blight":                             "Tomato_Late_blight",
    "Tomato___Leaf_Mold":                               "Tomato_Leaf_Mold",
    "Tomato___Septoria_leaf_spot":                      "Tomato_Septoria_leaf_spot",
    "Tomato___Spider_mites Two-spotted_spider_mite":    "Tomato_Spider_mites_Two_spotted_spider_mite",
    "Tomato___Target_Spot":                             "Tomato__Target_Spot",
    "Tomato___Tomato_Yellow_Leaf_Curl_Virus":           "Tomato__Tomato_YellowLeaf__Curl_Virus",
    "Tomato___Tomato_mosaic_virus":                     "Tomato__Tomato_mosaic_virus",
    "Tomato___healthy":                                 "Tomato_healthy",
    "Pepper,_bell___Bacterial_spot":                    "Pepper__bell___Bacterial_spot",
    "Pepper,_bell___healthy":                           "Pepper__bell___healthy",
}

def _normalize_class(label: str) -> str:
    return CLASS_NAME_MAP.get(label, label)


_CROP_PREFIXES = [
    "Tomato", "Potato", "Pepper,_bell", "Pepper__bell",
    "Apple", "Grape", "Corn_(maize)", "Cherry_(including_sour)",
    "Strawberry", "Peach", "Orange", "Raspberry", "Soybean",
    "Squash", "Blueberry",
]

def _strip_crop_prefix(label: str) -> str:
    for prefix in sorted(_CROP_PREFIXES, key=len, reverse=True):
        if label.lower().startswith(prefix.lower()):
            rest = label[len(prefix):].lstrip("_,( ")
            # Also clean up remaining underscores/separators
            return rest.replace("_", " ").replace("  ", " ").strip() or label
    return label.replace("_", " ").strip()


def _crop_family(label: str) -> str:
    if label.lower().startswith("tomato"): return "Tomato"
    if label.lower().startswith("potato"): return "Potato"
    if label.lower().startswith("pepper"): return "Pepper"
    return "Tomato"


@st.cache_resource
def load_disease_model():
    if not DISEASE_MODEL.exists():
        return None, None, None
    checkpoint = torch.load(DISEASE_MODEL, map_location=DEVICE, weights_only=False)
    classes    = checkpoint["classes"]
    model_type = checkpoint.get("model_type", "resnet18")

    if model_type == "multimodal":
        model = MultimodalCropNet(num_classes=len(classes), freeze_backbone=False)
    else:
        model = models.resnet18(weights=None)
        model.fc = nn.Linear(model.fc.in_features, len(classes))

    model.load_state_dict(checkpoint["model_state"])
    model.to(DEVICE)
    model.eval()
    return model, classes, model_type


def predict_disease(model, classes, model_type, img: Image.Image,
                    temp: float, humidity: float, days_since_rain: float):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    tensor = transform(img).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        if model_type == "multimodal":
            weather = torch.tensor([[temp, humidity, days_since_rain]],
                                   dtype=torch.float32).to(DEVICE)
            logits = model(tensor, weather)
        else:
            logits = model(tensor)
        probs = torch.softmax(logits, dim=1)[0].cpu().numpy()

    # Filter out garbage classes before ranking
    IGNORE = {"x_Removed_from_Healthy_leaves"}
    valid_idx = [i for i, c in enumerate(classes) if c not in IGNORE]
    valid_probs = [(probs[i], i) for i in valid_idx]
    top = sorted(valid_probs, reverse=True)[:5]
    return [classes[i] for _, i in top], [float(p) for p, _ in top]


def estimate_loss(crop: str, severity: str, field_acres: float, price_per_lb: float):
    typical   = TYPICAL_YIELD_LBS_ACRE.get(crop, 18_000)
    penalty   = YIELD_PENALTY.get(severity, 0.0)
    lost_lbs  = typical * penalty * field_acres
    lost_usd  = lost_lbs * price_per_lb
    return typical, penalty, lost_lbs, lost_usd


# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(page_title="Crop Health Monitor", page_icon="🌿", layout="wide")
st.title("Multimodal Crop Health Monitor")
st.caption("Leaf images + live weather → disease diagnosis, environmental risk, and estimated revenue loss.")

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
    location_input = st.text_input("Location (city or region)", placeholder="e.g. Hanoi, Vietnam")

    temp, humidity, days_since_rain = 24, 70, 2  # silent defaults

    if location_input:
        geo = geocode(location_input)
        if geo is None:
            st.error("Location not found.")
        else:
            lat, lon, display_name = geo
            wx = fetch_weather(lat, lon)
            if wx is None:
                st.error("Weather fetch failed.")
            else:
                temp            = wx["temperature_c"]
                humidity        = wx["humidity_pct"]
                days_since_rain = wx["days_since_rain"]
                st.caption(f"Live weather · **{display_name}** · via Open-Meteo")
                c_t, c_h, c_r = st.columns(3)
                temp_f = round(temp * 9/5 + 32, 1)
                c_t.metric("Temperature", f"{temp_f}°F")
                c_h.metric("Humidity", f"{humidity}%")
                c_r.metric("Days since rain", days_since_rain)
    else:
        st.info("Enter a location above to fetch live weather automatically.")

    if temp >= 28 and humidity >= 75:
        st.warning("Hot & humid — high fungal risk")
    elif temp >= 28 and humidity < 50:
        st.warning("Hot & dry — watch for mites/pests")
    else:
        st.success("Conditions are moderate")

with col3:
    st.subheader("3. Economic Impact")
    st.caption("Used to estimate revenue loss from disease.")
    field_acres  = st.number_input("Field size (acres)", min_value=0.1, max_value=25_000.0,
                                   value=2.5, step=1.0)
    price_per_lb = st.number_input("Crop price (USD per lb)", min_value=0.001, max_value=50.0,
                                   value=0.23, step=0.01,
                                   help="Tomato ≈ $0.23 · Potato ≈ $0.11 · Pepper ≈ $0.55")

st.divider()

if not uploaded_files:
    st.info("Upload at least one leaf image to get started.")
    st.stop()

disease_model, classes, model_type = load_disease_model()

if disease_model is None:
    st.error("Disease model not found at `results/disease_model.pth`.")
    st.stop()

# ── Run predictions ───────────────────────────────────────────────────────────
results = []
for f in uploaded_files:
    img = Image.open(f).convert("RGB")
    top_cls, top_conf = predict_disease(disease_model, classes, model_type, img,
                                        temp, humidity, days_since_rain)
    label, confidence = _normalize_class(top_cls[0]), top_conf[0]
    top_cls = [_normalize_class(c) for c in top_cls]
    rec      = RECOMMENDATIONS.get(label, {})
    severity = rec.get("severity", "Moderate")
    risk     = assess_risk(label, confidence, severity, temp, humidity, days_since_rain)
    crop     = _crop_family(label)
    typical, penalty, lost_lbs, lost_usd = estimate_loss(crop, severity, field_acres, price_per_lb)

    results.append({
        "filename": f.name, "img": img,
        "label": label, "confidence": confidence,
        "top_cls": top_cls, "top_conf": top_conf,
        "rec": rec, "severity": severity, "risk": risk,
        "crop": crop,
        "typical_lbs_acre": typical,
        "penalty": penalty,
        "lost_lbs": lost_lbs,
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
    label = _strip_crop_prefix(r["label"])
    r_level = risk["risk_level"]
    r_color = risk["color"]

    with st.expander(f"{r['filename']} — {label} | Risk: {r_level}", expanded=True):
        col_img, col_diag, col_fuse, col_treat = st.columns([1, 1, 1,2])

        with col_img:
            st.image(r["img"], use_container_width=True)

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
                st.progress(conf, text=f"{_strip_crop_prefix(cls)} ({conf:.1%})")

        with col_fuse:
            st.markdown("**Combined Risk Assessment**")
            st.markdown(f"<h2 style='color:{r_color}'>{r_level}</h2>", unsafe_allow_html=True)
            st.progress(risk["risk_score"], text=f"Risk score: {risk['risk_score']:.2f}")
            st.markdown(f"**Action:** {risk['action']}")
            if risk["env_warning"]:
                st.caption(risk["env_warning"])
            temp_f = round(temp * 9/5 + 32, 1)
            st.caption(f"Temp {temp_f}°F · Humidity {humidity}% · {days_since_rain}d dry")

            if "healthy" not in r["label"].lower():
                st.markdown("---")
                st.markdown("**Estimated Revenue Loss**")
                penalty_pct = r["penalty"] * 100
                st.metric("Yield Loss", f"{r['lost_lbs']:,.0f} lbs",
                          f"{penalty_pct:.0f}% of {r['typical_lbs_acre']:,} lbs/acre baseline")
                st.metric("Revenue Loss", f"${r['lost_usd']:,.0f}",
                          f"over {field_acres:.1f} acres at ${price_per_lb:.2f}/lb",
                          delta_color="inverse")
            else:
                st.success("No yield loss expected.")

        with col_treat:
            st.markdown("**Treatment Plan**")
            treatment = r["rec"].get("treatment", [])
            if treatment:
                for step in treatment:
                    st.markdown(f"- {step}")
            else:
                st.caption("No specific treatment data available for this crop.")
            prevention = r["rec"].get("prevention", "")
            if prevention:
                st.markdown("**Prevention**")
                st.info(prevention)
