"""
Optional Gradio dashboard for crop health monitoring.
Provides a simple UI to run disease classification and yield prediction.
"""

import gradio as gr
import tensorflow as tf
import numpy as np
import pickle
import pandas as pd
from PIL import Image


# Load models (update paths as needed)
DISEASE_MODEL_PATH = "../results/disease_model.keras"
YIELD_MODEL_PATH = "../results/yield_model.pkl"

disease_model = None
yield_model = None


def load_models():
    global disease_model, yield_model
    try:
        disease_model = tf.keras.models.load_model(DISEASE_MODEL_PATH)
    except Exception:
        pass
    try:
        with open(YIELD_MODEL_PATH, "rb") as f:
            yield_model = pickle.load(f)
    except Exception:
        pass


def predict_disease(image: Image.Image) -> str:
    if disease_model is None:
        return "Disease model not loaded. Run src/disease_classifier.py first."
    img = image.resize((224, 224))
    arr = np.array(img) / 255.0
    arr = np.expand_dims(arr, axis=0)
    preds = disease_model.predict(arr)[0]
    top_idx = int(np.argmax(preds))
    return f"Predicted class index: {top_idx} (confidence: {preds[top_idx]:.2%})"


def predict_yield(area: str, crop: str, year: int, rainfall: float, pesticides: float, temp: float) -> str:
    if yield_model is None:
        return "Yield model not loaded. Run src/yield_predictor.py first."
    # Simple label encoding — in production use the fitted encoder
    row = pd.DataFrame([{
        "Area": hash(area) % 100,
        "Item": hash(crop) % 100,
        "Year": year,
        "average_rain_fall_mm_per_year": rainfall,
        "pesticides_tonnes": pesticides,
        "avg_temp": temp,
    }])
    pred = yield_model.predict(row)[0]
    return f"Predicted yield: {pred:,.0f} hg/ha"


load_models()

with gr.Blocks(title="Crop Health Monitor") as demo:
    gr.Markdown("# Multimodal Crop Health Monitoring")

    with gr.Tab("Disease Detection"):
        img_input = gr.Image(type="pil", label="Upload crop leaf image")
        disease_output = gr.Textbox(label="Result")
        gr.Button("Classify").click(predict_disease, inputs=img_input, outputs=disease_output)

    with gr.Tab("Yield Prediction"):
        with gr.Row():
            area = gr.Textbox(label="Region/Country")
            crop = gr.Textbox(label="Crop type")
            year = gr.Number(label="Year", value=2023)
        with gr.Row():
            rainfall = gr.Number(label="Avg rainfall (mm/year)")
            pesticides = gr.Number(label="Pesticides (tonnes)")
            temp = gr.Number(label="Avg temperature (°C)")
        yield_output = gr.Textbox(label="Predicted Yield")
        gr.Button("Predict").click(
            predict_yield,
            inputs=[area, crop, year, rainfall, pesticides, temp],
            outputs=yield_output,
        )

if __name__ == "__main__":
    demo.launch()
