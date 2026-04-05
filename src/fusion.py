"""
Multimodal fusion: combines disease classification (image) +
field conditions (tabular) into a unified risk score and recommendation.
"""

# Environmental conditions that amplify each disease
# Keys map to disease class names
DISEASE_CONDITIONS = {
    # Fungal diseases — thrive in warm + humid + wet
    "fungal": {
        "classes": [
            "Tomato_Early_blight", "Tomato_Late_blight", "Tomato_Leaf_Mold",
            "Tomato_Septoria_leaf_spot", "Tomato__Target_Spot",
            "Potato___Early_blight", "Potato___Late_blight",
        ],
        "risk_fn": lambda temp, humidity, days_since_rain: (
            _scale(humidity, 60, 90) * 0.5 +
            _scale(temp, 15, 28) * 0.3 +
            _scale(7 - days_since_rain, 0, 7) * 0.2
        ),
        "warning": "Fungal diseases spread rapidly in humid, wet conditions.",
    },
    # Bacterial diseases — warm + wet + physical spread
    "bacterial": {
        "classes": [
            "Tomato_Bacterial_spot", "Pepper__bell___Bacterial_spot",
        ],
        "risk_fn": lambda temp, humidity, days_since_rain: (
            _scale(humidity, 65, 95) * 0.4 +
            _scale(temp, 20, 32) * 0.4 +
            _scale(7 - days_since_rain, 0, 7) * 0.2
        ),
        "warning": "Bacteria spread through water splash and humid conditions.",
    },
    # Viral diseases — spread by insect vectors (whiteflies, aphids)
    # Insects are more active in warm, dry conditions
    "viral": {
        "classes": [
            "Tomato__Tomato_YellowLeaf__Curl_Virus",
            "Tomato__Tomato_mosaic_virus",
        ],
        "risk_fn": lambda temp, humidity, days_since_rain: (
            _scale(temp, 25, 38) * 0.6 +
            _scale(100 - humidity, 20, 60) * 0.4
        ),
        "warning": "Viral spread driven by insect vectors — more active in warm, dry weather.",
    },
    # Spider mites — hot and dry
    "mites": {
        "classes": [
            "Tomato_Spider_mites_Two_spotted_spider_mite",
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
