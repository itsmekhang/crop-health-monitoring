"""
Classical machine-learning baselines — SVM, KNN and Random Forest — for the comparison
table in Jha et al. (ICACRS 2025, Table 1), but measured on PlantDoc field data instead
of PlantVillage, where every method scores high and the comparison says little.

Each baseline sees the same input as the CNN: the segmented leaf cutout, described by
frozen ImageNet ResNet-18 features (the 512 values before its final layer). They train
on the same data as the field fine-tuned CNN (a per-class sample of PlantVillage plus
the Cropped-PlantDoc training leaves), so the table compares classifiers, not data.

Usage (after notebook Section 8D has built Cropped-PlantDoc and its masks):
    from src.baselines import run_baselines
    table = run_baselines(pv_dir, pv_mask_dir, cpd_dir, cpd_mask_dir, plantdoc_dir)
"""

import random
from collections import defaultdict

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from PIL import Image
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_recall_fscore_support
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import Normalizer, StandardScaler
from sklearn.svm import SVC
from torch.utils.data import ConcatDataset, DataLoader, Subset

from src.disease_classifier import (BATCH_SIZE, DEVICE, FieldFolder, PlantFolder, _image_loader,
                                    _pv_datasets, build_model, get_transforms, prepare_image)
from src.field_eval import cropped_images, plantdoc_images

CLASSIFIERS = {
    "SVM (RBF)":     lambda: make_pipeline(StandardScaler(), SVC(kernel="rbf", C=10)),
    "KNN (k=5)":     lambda: make_pipeline(Normalizer(), KNeighborsClassifier(n_neighbors=5)),
    "Random Forest": lambda: RandomForestClassifier(n_estimators=300, n_jobs=-1, random_state=0),
}


def _feature_extractor() -> nn.Module:
    model = build_model(num_classes=1, finetune="fc", pretrained=True, arch="resnet18")
    model.fc = nn.Identity()                     # 512-d ImageNet features
    return model.eval()


@torch.no_grad()
def _features_from_loader(model, loader) -> tuple[np.ndarray, np.ndarray]:
    feats, labels = [], []
    for images, y in loader:
        feats.append(model(images.to(DEVICE)).cpu().numpy())
        labels.append(y.numpy())
    return np.concatenate(feats), np.concatenate(labels)


@torch.no_grad()
def _features_from_photos(model, items, idx, segmenter) -> tuple[np.ndarray, np.ndarray]:
    """Test photos/crops go through prepare_image — exactly what the CNN is scored on."""
    feats, labels = [], []
    for path, target in items:
        if target not in idx:
            continue
        try:
            im = Image.open(path)
            im.draft("RGB", (1024, 1024))
            tensor, _ = prepare_image(im, "leaf_cutout", segmenter)
        except OSError:
            continue
        feats.append(model(tensor).cpu().numpy()[0])
        labels.append(idx[target])
    return np.stack(feats), np.array(labels)


def _scores(y_true, y_pred) -> dict:
    present = sorted(set(y_true))
    p, r, f1, _ = precision_recall_fscore_support(y_true, y_pred, labels=present,
                                                  average="macro", zero_division=0)
    return {"top1": float((y_true == y_pred).mean()), "macro_precision": float(p),
            "macro_recall": float(r), "macro_f1": float(f1)}


def run_baselines(pv_dir: str, pv_mask_dir: str, field_dir: str, field_mask_dir: str,
                  plantdoc_dir: str, per_class: int = 200, segmenter: str = "hybrid",
                  num_workers: int = 2, seed: int = 0) -> pd.DataFrame:
    """
    Train SVM / KNN / Random Forest on ImageNet features of leaf cutouts and score them on
    the Cropped-PlantDoc test leaves and the full PlantDoc test photos.
    per_class: PlantVillage training images sampled per class (keeps the SVM tractable).
    """
    model = _feature_extractor()

    # Training set: a per-class sample of the PlantVillage *train* split + C-PD train crops
    pv_train, _, _, classes = _pv_datasets(pv_dir, "leaf_cutout", pv_mask_dir, None)
    eval_tf = get_transforms(False, "leaf_cutout")
    pv_eval = PlantFolder(pv_dir, transform=eval_tf, loader=_image_loader(pv_dir, pv_mask_dir, "leaf_cutout"))
    by_class = defaultdict(list)
    for i in pv_train.indices:
        by_class[pv_eval.samples[i][1]].append(i)
    rng = random.Random(seed)
    chosen = [i for idxs in by_class.values() for i in rng.sample(idxs, min(per_class, len(idxs)))]
    field_train = FieldFolder(f"{field_dir}/train", classes, eval_tf,
                              _image_loader(field_dir, field_mask_dir, "leaf_cutout"))
    train_loader = DataLoader(ConcatDataset([Subset(pv_eval, chosen), field_train]),
                              batch_size=BATCH_SIZE, num_workers=num_workers)
    x_train, y_train = _features_from_loader(model, train_loader)
    print(f"Baseline training features: {len(chosen)} PlantVillage + {len(field_train)} field crops")

    idx = {c: i for i, c in enumerate(classes)}
    tests = {
        "C-PD test leaves": _features_from_photos(model, cropped_images(field_dir, "test"), idx, segmenter),
        "Full test photos": _features_from_photos(model, plantdoc_images(plantdoc_dir, ("test",)), idx, segmenter),
    }

    rows = []
    for name, make in CLASSIFIERS.items():
        clf = make().fit(x_train, y_train)
        row = {"model": name}
        for test_name, (x, y) in tests.items():
            s = _scores(y, clf.predict(x))
            row[f"{test_name}: top-1"] = s["top1"]
            row[f"{test_name}: macro F1"] = s["macro_f1"]
        rows.append(row)
        print(f"  {name}: done")
    return pd.DataFrame(rows).set_index("model")
