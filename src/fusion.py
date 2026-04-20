"""
Environmental risk scoring: combines disease classification output with
field conditions (temperature, humidity, days since rain) to produce a
decision-support risk score.

The risk score is derived from agronomic rules (see DISEASE_CONDITIONS and
SEVERITY_BASE below), not learned from observed field data. XGBoost is used
to approximate this formula at inference — not as a deeply trained multimodal model.
"""

# Environmental conditions that amplify each disease
# Keys map to disease class names
DISEASE_CONDITIONS = {
    # Fungal diseases — thrive in warm + humid + wet
    # Weight rationale: humidity is the primary sporulation trigger for most
    # fungal pathogens (0.5); temperature sets a permissive range but is
    # secondary (0.3); recent rainfall adds surface moisture but effect decays
    # quickly once canopy dries (0.2). Consistent with Agrios (2005) and
    # APS disease management guidelines for Late Blight and Early Blight.
    "fungal": {
        "classes": [
            # Tomato (old naming via CLASS_NAME_MAP)
            "Tomato_Early_blight", "Tomato_Late_blight", "Tomato_Leaf_Mold",
            "Tomato_Septoria_leaf_spot", "Tomato__Target_Spot",
            # Tomato (new triple-underscore naming)
            "Tomato___Early_blight", "Tomato___Late_blight", "Tomato___Leaf_Mold",
            "Tomato___Septoria_leaf_spot", "Tomato___Target_Spot",
            # Potato
            "Potato___Early_blight", "Potato___Late_blight",
            # Apple
            "Apple___Apple_scab", "Apple___Black_rot", "Apple___Cedar_apple_rust",
            # Cherry
            "Cherry_(including_sour)___Powdery_mildew",
            # Corn
            "Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot",
            "Corn_(maize)___Common_rust_",
            "Corn_(maize)___Northern_Leaf_Blight",
            # Grape
            "Grape___Black_rot", "Grape___Esca_(Black_Measles)",
            "Grape___Leaf_blight_(Isariopsis_Leaf_Spot)",
            # Squash / Strawberry
            "Squash___Powdery_mildew", "Strawberry___Leaf_scorch",
        ],
        "risk_fn": lambda temp, humidity, days_since_rain: (
            _scale(humidity, 60, 90) * 0.5 +
            _scale(temp, 15, 28) * 0.3 +
            _scale(7 - days_since_rain, 0, 7) * 0.2
        ),
        "warning": "Fungal diseases spread rapidly in humid, wet conditions.",
    },
    # Bacterial diseases — warm + wet + physical spread
    # Weight rationale: temperature and humidity contribute roughly equally
    # for bacterial pathogens (0.4 each) — warm temperatures accelerate
    # cell division and humidity promotes leaf wetness needed for infection
    # (Gitaitis & Walcott, 2007). Recent rainfall contributes less because
    # bacteria can persist on dry surfaces between rain events (0.2).
    "bacterial": {
        "classes": [
            # Tomato / Pepper (old naming)
            "Tomato_Bacterial_spot", "Pepper__bell___Bacterial_spot",
            # Tomato / Pepper (new naming)
            "Tomato___Bacterial_spot", "Pepper,_bell___Bacterial_spot",
            # Peach / Orange
            "Peach___Bacterial_spot",
            "Orange___Haunglongbing_(Citrus_greening)",
        ],
        "risk_fn": lambda temp, humidity, days_since_rain: (
            _scale(humidity, 65, 95) * 0.4 +
            _scale(temp, 20, 32) * 0.4 +
            _scale(7 - days_since_rain, 0, 7) * 0.2
        ),
        "warning": "Bacteria spread through water splash and humid conditions.",
    },
    # Viral diseases — spread by insect vectors (whiteflies, aphids)
    # Weight rationale: insect vector activity is primarily temperature-driven
    # (0.6) — whitefly and aphid populations peak above 25 °C. Low humidity
    # (dry_air term) further increases vector mobility and flight activity (0.4).
    # Rainfall is excluded because it temporarily suppresses vector movement
    # but does not affect the underlying viral spread mechanism.
    "viral": {
        "classes": [
            # Tomato (old naming)
            "Tomato__Tomato_YellowLeaf__Curl_Virus",
            "Tomato__Tomato_mosaic_virus",
            # Tomato (new naming)
            "Tomato___Tomato_Yellow_Leaf_Curl_Virus",
            "Tomato___Tomato_mosaic_virus",
        ],
        "risk_fn": lambda temp, humidity, days_since_rain: (
            _scale(temp, 25, 38) * 0.6 +
            _scale(100 - humidity, 20, 60) * 0.4
        ),
        "warning": "Viral spread driven by insect vectors — more active in warm, dry weather.",
    },
    # Spider mites — hot and dry
    # Weight rationale: spider mite reproduction rate roughly doubles for
    # every 4 °C rise above 28 °C (Sabelis, 1985), and low humidity speeds
    # desiccation of natural enemies. Both factors carry equal weight (0.5 each)
    # because mite outbreaks require both conditions simultaneously.
    "mites": {
        "classes": [
            "Tomato_Spider_mites_Two_spotted_spider_mite",
            "Tomato___Spider_mites Two-spotted_spider_mite",
        ],
        "risk_fn": lambda temp, humidity, days_since_rain: (
            _scale(temp, 28, 40) * 0.5 +
            _scale(100 - humidity, 30, 70) * 0.5
        ),
        "warning": "Spider mites thrive in hot, dry conditions.",
    },
}

SEVERITY_BASE = {"None": 0.0, "Moderate": 0.4, "High": 0.7, "Critical": 1.0}

RISK_LEVELS = [
    (0.80, "Critical", "red",    "Immediate action required — treat within 24 hours."),
    (0.55, "High",     "red",    "Treat within 2-3 days before conditions worsen."),
    (0.30, "Moderate", "orange", "Monitor closely and prepare treatment."),
    (0.00, "Low",      "green",  "Low risk — continue monitoring."),
]


import pickle
from pathlib import Path

_RISK_MODEL = None  # lazy-loaded on first call

def _load_risk_model():
    global _RISK_MODEL
    if _RISK_MODEL is not None:
        return _RISK_MODEL
    model_path = Path(__file__).parent.parent / "results" / "risk_model.pkl"
    if model_path.exists():
        with open(model_path, "rb") as f:
            _RISK_MODEL = pickle.load(f)
    return _RISK_MODEL


def _scale(value, low, high):
    """Normalize value to [0, 1] clamped between low and high."""
    return max(0.0, min(1.0, (value - low) / (high - low)))


def _get_disease_type(disease_class):
    for dtype, info in DISEASE_CONDITIONS.items():
        if disease_class in info["classes"]:
            return dtype, info
    return None, None


def assess_risk(disease_class, confidence, severity, temp, humidity, days_since_rain):
    """
    Combine disease classification + field conditions into a risk score.
    Uses XGBoost risk model when available, falls back to weighted formula.

    Returns:
        dict with risk_score, risk_level, color, action, env_warning, env_multiplier
    """
    if "healthy" in disease_class.lower():
        return {
            "risk_score": 0.0,
            "risk_level": "None",
            "color": "green",
            "action": "No action needed. Continue monitoring.",
            "env_warning": None,
            "env_multiplier": 1.0,
        }

    dtype, info = _get_disease_type(disease_class)
    env_warning = info["warning"] if info else None

    # Try XGBoost model first
    risk_data = _load_risk_model()
    if risk_data is not None:
        try:
            le  = risk_data["label_encoder"]
            xgb = risk_data["model"]
            cls_enc = le.transform([disease_class])[0]
            risk_score = float(xgb.predict([[temp, humidity, days_since_rain,
                                             confidence, cls_enc]])[0])
            risk_score = max(0.0, min(1.0, risk_score))
            env_multiplier = round(float(info["risk_fn"](temp, humidity, days_since_rain)), 2) if info else 0.5
        except Exception:
            risk_data = None  # fall through to formula

    # Fallback: hand-coded weighted formula
    # Weight rationale:
    #   base severity (0.4) — primary signal; reflects the agronomic classification
    #     of how destructive the disease is regardless of conditions.
    #   model confidence (0.3) — scales the diagnosis reliability; a 95%-confident
    #     detection warrants a stronger response than a 55% one.
    #   env_score (0.3) — modulates urgency based on spread conditions; environment
    #     affects how fast the disease progresses, not the underlying pathology,
    #     so it carries less weight than the diagnosis itself.
    if risk_data is None:
        base = SEVERITY_BASE.get(severity, 0.4)
        env_score = info["risk_fn"](temp, humidity, days_since_rain) if info else 0.5
        risk_score = (base * 0.4) + (confidence * 0.3) + (env_score * 0.3)
        env_multiplier = round(env_score, 2)

    for threshold, level, color, action in RISK_LEVELS:
        if risk_score >= threshold:
            return {
                "risk_score": round(risk_score, 3),
                "risk_level": level,
                "color": color,
                "action": action,
                "env_warning": env_warning,
                "env_multiplier": env_multiplier,
            }
