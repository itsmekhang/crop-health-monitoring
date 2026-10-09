"""
Field-photo evaluation on PlantDoc — the honest "unseen samples" benchmark.

PlantVillage validation accuracy (0.93–0.996) only measures lab photos the model
has effectively already seen (same backgrounds, same sessions, near-duplicates).
PlantDoc (Singh et al., 2020) has 2,500+ real-world photos — field, greenhouse
and web images with clutter, hands and multiple leaves — for 27 of the same
crop/disease classes. Nothing from it is used for training here.

Setup (images are CC BY 4.0; not tracked in git):
    git clone --depth 1 https://github.com/pratikkayal/PlantDoc-Dataset data/raw/PlantDoc

Usage:
    python -m src.field_eval results/disease_model.pth data/raw/PlantDoc
"""

import argparse
from collections import defaultdict
from pathlib import Path

import torch
from PIL import Image

from src.disease_classifier import load_model, prepare_image

# PlantDoc folder → PlantVillage class
PLANTDOC_TO_PLANTVILLAGE = {
    "Apple Scab Leaf":                      "Apple___Apple_scab",
    "Apple leaf":                           "Apple___healthy",
    "Apple rust leaf":                      "Apple___Cedar_apple_rust",
    "Bell_pepper leaf":                     "Pepper,_bell___healthy",
    "Bell_pepper leaf spot":                "Pepper,_bell___Bacterial_spot",
    "Blueberry leaf":                       "Blueberry___healthy",
    "Cherry leaf":                          "Cherry_(including_sour)___healthy",
    "Corn Gray leaf spot":                  "Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot",
    "Corn leaf blight":                     "Corn_(maize)___Northern_Leaf_Blight",
    "Corn rust leaf":                       "Corn_(maize)___Common_rust_",
    "Peach leaf":                           "Peach___healthy",
    "Potato leaf early blight":             "Potato___Early_blight",
    "Potato leaf late blight":              "Potato___Late_blight",
    "Raspberry leaf":                       "Raspberry___healthy",
    "Soyabean leaf":                        "Soybean___healthy",
    "Squash Powdery mildew leaf":           "Squash___Powdery_mildew",
    "Strawberry leaf":                      "Strawberry___healthy",
    "Tomato Early blight leaf":             "Tomato___Early_blight",
    "Tomato Septoria leaf spot":            "Tomato___Septoria_leaf_spot",
    "Tomato leaf":                          "Tomato___healthy",
    "Tomato leaf bacterial spot":           "Tomato___Bacterial_spot",
    "Tomato leaf late blight":              "Tomato___Late_blight",
    "Tomato leaf mosaic virus":             "Tomato___Tomato_mosaic_virus",
    "Tomato leaf yellow virus":             "Tomato___Tomato_Yellow_Leaf_Curl_Virus",
    "Tomato mold leaf":                     "Tomato___Leaf_Mold",
    "Tomato two spotted spider mites leaf": "Tomato___Spider_mites Two-spotted_spider_mite",
    "grape leaf":                           "Grape___healthy",
    "grape leaf black rot":                 "Grape___Black_rot",
}
IMG_EXTS = {".jpg", ".jpeg", ".png"}


def crop_of(cls: str) -> str:
    return cls.split("___")[0]


def plantdoc_images(plantdoc_dir: str, splits=("train", "test")):
    """Yields (path, PlantVillage class) for every mapped PlantDoc image."""
    for split in splits:
        for folder in sorted((Path(plantdoc_dir) / split).iterdir()):
            target = PLANTDOC_TO_PLANTVILLAGE.get(folder.name)
            if target is None:
                continue
            for p in sorted(folder.iterdir()):
                if p.suffix.lower() in IMG_EXTS:
                    yield p, target


def evaluate_field(model_path: str, plantdoc_dir: str, splits=("train", "test"),
                   preprocess: str | None = None, verbose: bool = True) -> dict:
    """
    Top-1 / top-3 accuracy on PlantDoc photos, predicting over all classes.
    "given crop" restricts the prediction to the true crop's classes — what you
    get if the user tells the app which crop they photographed.
    """
    model, classes, ckpt_preprocess = load_model(model_path)
    preprocess = preprocess or ckpt_preprocess
    idx = {c: i for i, c in enumerate(classes)}
    crop_masks = {}
    for c in classes:
        crop_masks.setdefault(crop_of(c), torch.tensor([crop_of(k) == crop_of(c) for k in classes]))

    hits = defaultdict(lambda: [0, 0])          # class → [correct, total]
    top1 = top3 = given_crop = n = 0
    with torch.no_grad():
        for path, target in plantdoc_images(plantdoc_dir, splits):
            if target not in idx:
                continue
            try:
                im = Image.open(path)
                im.draft("RGB", (1024, 1024))   # fast reduced-size JPEG decode; model input is 224px
                tensor, _ = prepare_image(im, preprocess)
            except OSError:
                continue
            probs = torch.softmax(model(tensor), 1)[0].cpu()
            t = idx[target]
            rank = probs.argsort(descending=True)
            top1 += int(rank[0] == t)
            top3 += int(t in rank[:3])
            given_crop += int((probs * crop_masks[crop_of(target)]).argmax() == t)
            hits[target][0] += int(rank[0] == t)
            hits[target][1] += 1
            n += 1

    result = {"n": n, "preprocess": preprocess, "top1": top1 / n, "top3": top3 / n,
              "top1_given_crop": given_crop / n,
              "per_class": {c: h[0] / h[1] for c, h in sorted(hits.items())}}
    if verbose:
        print(f"PlantDoc field photos: {n} images, preprocess={preprocess}")
        print(f"  top-1 {result['top1']:.3f} | top-3 {result['top3']:.3f} | "
              f"top-1 given crop {result['top1_given_crop']:.3f}")
        for c, acc in sorted(result["per_class"].items(), key=lambda kv: kv[1]):
            print(f"    {acc:.2f}  {c}  (n={hits[c][1]})")
    return result


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Evaluate a disease model on PlantDoc field photos.")
    ap.add_argument("model_path")
    ap.add_argument("plantdoc_dir")
    ap.add_argument("--splits", nargs="+", default=["train", "test"])
    ap.add_argument("--preprocess", choices=["resize", "leaf_cutout"], default=None,
                    help="override the checkpoint's preprocessing")
    a = ap.parse_args()
    evaluate_field(a.model_path, a.plantdoc_dir, a.splits, a.preprocess)
