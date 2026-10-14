"""
Field-photo evaluation on PlantDoc — the honest "unseen samples" benchmark.

PlantVillage validation accuracy (0.93–0.996) only measures lab photos the model
has effectively already seen (same backgrounds, same sessions, near-duplicates).
PlantDoc (Singh et al., 2020) has 2,500+ real-world photos — field, greenhouse
and web images with clutter, hands and multiple leaves — for 27 of the same
crop/disease classes. Nothing from it is used for training here.

Two benchmarks (images are CC BY 4.0; not tracked in git):
  - Full photos — PlantDoc-Dataset (classification repo), what the dashboard sees:
        git clone --depth 1 https://github.com/pratikkayal/PlantDoc-Dataset data/raw/PlantDoc
  - Cropped leaves (C-PD) — single-leaf crops from the bounding boxes of
    PlantDoc-Object-Detection-Dataset; the setting of the paper's Table 2 and
    the training data for field fine-tuning (see build_cropped_plantdoc):
        git clone --depth 1 https://github.com/pratikkayal/PlantDoc-Object-Detection-Dataset data/raw/PlantDoc-OD

Usage:
    python -m src.field_eval results/disease_model.pth data/raw/PlantDoc --splits test
    python -m src.field_eval results/disease_model.pth data/processed/PlantDoc_cropped --crops
"""

import argparse
import csv
import hashlib
import random
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
    "Potato leaf":                          "Potato___healthy",   # detection dataset only
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


def cropped_images(cropped_dir: str, split: str = "test"):
    """Yields (path, PlantVillage class) from a build_cropped_plantdoc() split."""
    for folder in sorted((Path(cropped_dir) / split).iterdir()):
        for p in sorted(folder.iterdir()):
            if p.suffix.lower() in IMG_EXTS:
                yield p, folder.name


def _md5(path: Path) -> str:
    return hashlib.md5(path.read_bytes()).hexdigest()


def build_cropped_plantdoc(detection_dir: str, out_dir: str, classification_dir: str | None = None,
                           val_frac: float = 0.1, min_side: int = 32, seed: int = 0) -> dict:
    """
    Crop every annotated leaf in PlantDoc-Object-Detection-Dataset into
    out_dir/{train,val,test}/<PlantVillage class>/<photo>_<n>.jpg — the paper's
    Cropped-PlantDoc (C-PD), with two deliberate differences:
      - split by photo, not by crop: leaves of one photo never span train and test
        (the paper's random 80/20 crop split lets near-identical leaves leak);
      - training photos that also appear in a test split (PlantDoc ships a few
        duplicates) are dropped, so test accuracy stays honest. Pass the
        classification repo as classification_dir to also guard its test split.
    "val" is val_frac of the training photos, for choosing the best epoch.
    """
    detection_dir, out_dir = Path(detection_dir), Path(out_dir)
    test_md5 = {_md5(p) for p in (detection_dir / "TEST").iterdir() if p.suffix.lower() in IMG_EXTS}
    if classification_dir:
        test_md5 |= {_md5(p) for p, _ in plantdoc_images(classification_dir, splits=("test",))}

    counts = defaultdict(int)
    rng = random.Random(seed)
    for split, folder in (("train", "TRAIN"), ("test", "TEST")):
        boxes = defaultdict(list)
        for r in csv.DictReader(open(detection_dir / f"{split}_labels.csv")):
            boxes[r["filename"]].append(r)
        for fname in sorted(boxes):
            src = detection_dir / folder / fname
            if not src.exists():
                counts["missing_photo"] += 1
                continue
            out_split = split
            if split == "train":
                if _md5(src) in test_md5:
                    counts["dropped_duplicate_of_test"] += 1
                    continue
                out_split = "val" if rng.random() < val_frac else "train"
            try:
                im = Image.open(src).convert("RGB")
            except OSError:
                counts["unreadable"] += 1
                continue
            for k, r in enumerate(boxes[fname]):
                target = PLANTDOC_TO_PLANTVILLAGE.get(r["class"])
                x0, y0 = max(0, int(r["xmin"])), max(0, int(r["ymin"]))
                x1, y1 = min(im.width, int(r["xmax"])), min(im.height, int(r["ymax"]))
                if target is None or min(x1 - x0, y1 - y0) < min_side:
                    counts["skipped_box"] += 1
                    continue
                dst = out_dir / out_split / target / f"{Path(fname).stem}_{k}.jpg"
                dst.parent.mkdir(parents=True, exist_ok=True)
                crop = im.crop((x0, y0, x1, y1))
                crop.thumbnail((512, 512))          # plenty for a 224px model; keeps the set small
                crop.save(dst, quality=95)
                counts[out_split] += 1
    print("Cropped-PlantDoc:", dict(counts))
    return dict(counts)


def _score(model_path: str, items, preprocess: str | None, segmenter: str) -> dict:
    """Top-1/top-3 over all classes, plus restricted variants, for (path, class) items."""
    model, classes, ckpt_preprocess = load_model(model_path)
    preprocess = preprocess or ckpt_preprocess
    idx = {c: i for i, c in enumerate(classes)}
    crop_masks = {}
    for c in classes:
        crop_masks.setdefault(crop_of(c), torch.tensor([crop_of(k) == crop_of(c) for k in classes]))
    # The paper's setting: the model only chooses among PlantDoc's classes
    plantdoc_mask = torch.tensor([c in set(PLANTDOC_TO_PLANTVILLAGE.values()) for c in classes])

    hits = defaultdict(lambda: [0, 0])          # class → [correct, total]
    top1 = top3 = given_crop = among_plantdoc = n = 0
    with torch.no_grad():
        for path, target in items:
            if target not in idx:
                continue
            try:
                im = Image.open(path)
                im.draft("RGB", (1024, 1024))   # fast reduced-size JPEG decode; model input is 224px
                tensor, _ = prepare_image(im, preprocess, segmenter)
            except OSError:
                continue
            probs = torch.softmax(model(tensor), 1)[0].cpu()
            t = idx[target]
            rank = probs.argsort(descending=True)
            top1 += int(rank[0] == t)
            top3 += int(t in rank[:3])
            given_crop += int((probs * crop_masks[crop_of(target)]).argmax() == t)
            among_plantdoc += int((probs * plantdoc_mask).argmax() == t)
            hits[target][0] += int(rank[0] == t)
            hits[target][1] += 1
            n += 1

    return {"n": n, "preprocess": preprocess, "top1": top1 / n, "top3": top3 / n,
            "top1_given_crop": given_crop / n, "top1_plantdoc_classes": among_plantdoc / n,
            "per_class": {c: h[0] / h[1] for c, h in sorted(hits.items())},
            "support": {c: h[1] for c, h in sorted(hits.items())}}


def _report(title: str, r: dict) -> None:
    print(f"{title}: {r['n']} images, preprocess={r['preprocess']}")
    print(f"  top-1 {r['top1']:.3f} | top-3 {r['top3']:.3f} | top-1 among PlantDoc classes "
          f"{r['top1_plantdoc_classes']:.3f} | top-1 given crop {r['top1_given_crop']:.3f}")
    for c, acc in sorted(r["per_class"].items(), key=lambda kv: kv[1]):
        print(f"    {acc:.2f}  {c}  (n={r['support'][c]})")


def evaluate_field(model_path: str, plantdoc_dir: str, splits=("train", "test"),
                   preprocess: str | None = None, verbose: bool = True,
                   segmenter: str = "hybrid") -> dict:
    """
    Accuracy on full PlantDoc photos (what the dashboard sees), predicting over all
    classes. "given crop" restricts the prediction to the true crop's classes — what
    you get if the user tells the app which crop they photographed. Use
    splits=("test",) for any model fine-tuned on PlantDoc's training photos.
    """
    r = _score(model_path, plantdoc_images(plantdoc_dir, splits), preprocess, segmenter)
    if verbose:
        _report("PlantDoc field photos", r)
    return r


def evaluate_crops(model_path: str, cropped_dir: str, split: str = "test",
                   preprocess: str | None = None, verbose: bool = True,
                   segmenter: str = "hybrid") -> dict:
    """Accuracy on Cropped-PlantDoc single leaves — comparable to the paper's Table 2."""
    r = _score(model_path, cropped_images(cropped_dir, split), preprocess, segmenter)
    if verbose:
        _report(f"Cropped-PlantDoc {split} leaves", r)
    return r


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Evaluate a disease model on PlantDoc field photos.")
    ap.add_argument("model_path")
    ap.add_argument("plantdoc_dir", help="PlantDoc-Dataset clone, or a Cropped-PlantDoc dir with --crops")
    ap.add_argument("--crops", action="store_true", help="evaluate Cropped-PlantDoc test leaves")
    ap.add_argument("--splits", nargs="+", default=["train", "test"])
    ap.add_argument("--preprocess", choices=["resize", "leaf_cutout"], default=None,
                    help="override the checkpoint's preprocessing")
    ap.add_argument("--segmenter", choices=["hybrid", "grabcut"], default="hybrid",
                    help="leaf segmentation for leaf_cutout models (hybrid = MobileSAM + GrabCut)")
    a = ap.parse_args()
    if a.crops:
        evaluate_crops(a.model_path, a.plantdoc_dir, "test", a.preprocess, segmenter=a.segmenter)
    else:
        evaluate_field(a.model_path, a.plantdoc_dir, a.splits, a.preprocess, segmenter=a.segmenter)
