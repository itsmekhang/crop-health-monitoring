"""
Image-based crop disease classification using ResNet-18 (PyTorch + torchvision).
Dataset: PlantVillage — 14 crops, 38 disease/healthy classes, ~54,000 images.
All images are lab-controlled (uniform backgrounds). Field performance will differ.

Preprocessing modes (stored in the checkpoint so inference always matches training):
  "resize"      — legacy: the whole photo, background included, resized to 224x224.
  "leaf_cutout" — the leaf is segmented out (src/segmentation.py) and cropped to its
                  bounding box. Training pastes it onto random backgrounds; inference
                  pastes it onto black. This removes PlantVillage's background shortcut,
                  which is the main reason the classifier fails on unseen field photos.
"""

import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms, models
from pathlib import Path
from PIL import Image, ImageOps

from src.segmentation import RandomBackground, apply_mask, composite, cut_out_leaf, mask_path_for


IMG_SIZE = 224
BATCH_SIZE = 32
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MEAN, STD = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
EXCLUDED_CLASSES = {"x_Removed_from_Healthy_leaves"}   # 5-image artifact folder


class CompositeOnBlack:
    """RGBA cutout → RGB on black (inference-time background)."""
    def __call__(self, rgba: Image.Image) -> Image.Image:
        return composite(rgba, (0, 0, 0))


def get_transforms(augment: bool = False, preprocess: str = "leaf_cutout",
                   backgrounds_dir: str | None = None):
    tail = [transforms.ToTensor(), transforms.Normalize(MEAN, STD)]

    if preprocess == "resize":
        base = [transforms.Resize((IMG_SIZE, IMG_SIZE))] + tail
        if augment:
            base = [
                transforms.RandomHorizontalFlip(),
                transforms.RandomRotation(15),
                transforms.ColorJitter(brightness=0.2, contrast=0.2),
            ] + base
        return transforms.Compose(base)

    # leaf_cutout: input is a square RGBA cutout from segmentation.cut_out_leaf
    if not augment:
        return transforms.Compose([transforms.Resize((IMG_SIZE, IMG_SIZE)),
                                   CompositeOnBlack()] + tail)
    return transforms.Compose([
        # Geometric augmentation on the RGBA cutout — exposed corners stay transparent
        transforms.RandomResizedCrop(IMG_SIZE, scale=(0.6, 1.0), ratio=(0.85, 1.15)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),        # leaves have no canonical "up"
        transforms.RandomRotation(30),
        RandomBackground(backgrounds_dir),       # RGBA → RGB on a random background
        # Photometric augmentation: field lighting, white balance, focus
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2, hue=0.02),
        transforms.RandomApply([transforms.GaussianBlur(5, sigma=(0.1, 2.0))], p=0.3),
        *tail,
        transforms.RandomErasing(p=0.25, scale=(0.02, 0.1)),   # partial occlusion
    ])


class LeafCutoutLoader:
    """
    ImageFolder loader returning an RGBA leaf cutout. Uses the precomputed mask
    from `python -m src.segmentation` when available, otherwise segments on the fly.
    """
    def __init__(self, image_root: str, mask_root: str | None = None):
        self.image_root = Path(image_root)
        self.mask_root = Path(mask_root) if mask_root else None

    def __call__(self, path: str) -> Image.Image:
        img = Image.open(path).convert("RGB")
        if self.mask_root:
            mp = mask_path_for(Path(path), self.image_root, self.mask_root)
            if mp.exists():
                return apply_mask(img, np.asarray(Image.open(mp)))
        return cut_out_leaf(img)[0]


class PlantFolder(datasets.ImageFolder):
    """ImageFolder that skips hidden folders (.git) and known artifact classes."""
    def find_classes(self, directory):
        classes = sorted(e.name for e in Path(directory).iterdir()
                         if e.is_dir() and not e.name.startswith(".")
                         and e.name not in EXCLUDED_CLASSES)
        return classes, {c: i for i, c in enumerate(classes)}


def load_datasets(data_dir: str, preprocess: str = "leaf_cutout", mask_dir: str | None = None,
                  backgrounds_dir: str | None = None, val_frac: float = 0.2, seed: int = 42,
                  num_workers: int = 2):
    loader = LeafCutoutLoader(data_dir, mask_dir) if preprocess == "leaf_cutout" else None
    kw = {"loader": loader} if loader else {}
    # Two dataset instances so train and val each keep their own transform
    # (re-assigning .transform on a shared dataset silently changes both splits).
    train_full = PlantFolder(data_dir, transform=get_transforms(True, preprocess, backgrounds_dir), **kw)
    val_full   = PlantFolder(data_dir, transform=get_transforms(False, preprocess), **kw)

    perm  = torch.randperm(len(train_full), generator=torch.Generator().manual_seed(seed)).tolist()
    n_val = int(val_frac * len(train_full))
    train_ds, val_ds = Subset(train_full, perm[n_val:]), Subset(val_full, perm[:n_val])

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=num_workers, pin_memory=DEVICE == "cuda")
    val_loader   = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False,
                              num_workers=num_workers, pin_memory=DEVICE == "cuda")
    return train_loader, val_loader, train_full.classes


def build_model(num_classes: int, finetune: str = "all", pretrained: bool = True) -> nn.Module:
    """
    finetune: "fc"     — frozen ImageNet backbone, train the head only (D2 baseline)
              "layer4" — also adapt the last residual block (D3, Section 8B)
              "all"    — fine-tune every layer with a low backbone learning rate
    """
    weights = models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
    model = models.resnet18(weights=weights)
    for name, param in model.named_parameters():
        param.requires_grad = finetune == "all" or (finetune == "layer4" and name.startswith("layer4"))
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model.to(DEVICE)


def train(data_dir: str, epochs: int = 10, save_path: str = "results/disease_model.pth",
          finetune: str = "all", preprocess: str = "leaf_cutout", mask_dir: str | None = None,
          backgrounds_dir: str | None = None, backbone_lr: float = 1e-4, head_lr: float = 1e-3,
          num_workers: int = 2):
    train_loader, val_loader, classes = load_datasets(
        data_dir, preprocess, mask_dir, backgrounds_dir, num_workers=num_workers)
    print(f"Classes ({len(classes)}): {classes[:5]} ...  preprocess={preprocess}  finetune={finetune}")

    model = build_model(num_classes=len(classes), finetune=finetune)
    backbone = [p for n, p in model.named_parameters() if p.requires_grad and not n.startswith("fc.")]
    groups = [{"params": model.fc.parameters(), "lr": head_lr}]
    if backbone:
        # Differential learning rates: small steps for pretrained weights
        groups.append({"params": backbone, "lr": backbone_lr})
    optimizer = torch.optim.AdamW(groups, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    # Label smoothing keeps confidences honest on images unlike the training set
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

    history = {"train_loss": [], "train_acc": [], "val_acc": []}
    best_acc, best_state = -1.0, None
    for epoch in range(1, epochs + 1):
        model.train()
        total_loss, correct, total = 0.0, 0, 0
        for images, labels in train_loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            correct += (outputs.argmax(1) == labels).sum().item()
            total += labels.size(0)
        scheduler.step()

        val_acc = evaluate(model, val_loader)
        history["train_loss"].append(total_loss / len(train_loader))
        history["train_acc"].append(correct / total)
        history["val_acc"].append(val_acc)
        if val_acc > best_acc:
            best_acc = val_acc
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        print(f"Epoch {epoch}/{epochs}  loss={history['train_loss'][-1]:.4f}  "
              f"train_acc={correct/total:.3f}  val_acc={val_acc:.3f}")

    model.load_state_dict(best_state)
    torch.save({"model_state": best_state, "classes": classes, "model_type": "resnet18",
                "preprocess": preprocess, "finetune": finetune}, save_path)
    print(f"Best val_acc={best_acc:.3f} — model saved to {save_path}")
    return model, classes, history


def evaluate(model: nn.Module, loader: DataLoader, classes=None):
    """
    Evaluate model on loader.

    Args:
        classes: if provided, returns a dict with overall accuracy and
                 per-class precision, recall, F1, and support.
                 If None, returns overall accuracy as a float (for use
                 inside the training loop).
    """
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            preds = model(images).argmax(1)
            all_preds.extend(preds.cpu().tolist())
            all_labels.extend(labels.cpu().tolist())

    accuracy = sum(p == l for p, l in zip(all_preds, all_labels)) / len(all_labels)

    if classes is None:
        return accuracy

    from sklearn.metrics import precision_recall_fscore_support
    precision, recall, f1, support = precision_recall_fscore_support(
        all_labels, all_preds,
        labels=list(range(len(classes))),
        zero_division=0,
    )
    return {
        "accuracy": round(accuracy, 4),
        "per_class": {
            cls: {
                "precision": round(float(p), 3),
                "recall":    round(float(r), 3),
                "f1":        round(float(f), 3),
                "support":   int(s),
            }
            for cls, p, r, f, s in zip(classes, precision, recall, f1, support)
        },
    }


def load_model(model_path: str = "results/disease_model.pth"):
    """Returns (model, classes, preprocess). Old checkpoints default to "resize"."""
    checkpoint = torch.load(model_path, map_location=DEVICE, weights_only=False)
    classes = checkpoint["classes"]
    model = build_model(num_classes=len(classes), finetune="fc", pretrained=False)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    return model, classes, checkpoint.get("preprocess", "resize")


def prepare_image(img: Image.Image, preprocess: str = "leaf_cutout"):
    """
    Turn a user photo into the model's input tensor, exactly as at validation time.
    Returns (tensor of shape (1, 3, H, W), RGB image the model actually sees).
    """
    img = ImageOps.exif_transpose(img).convert("RGB")   # phone photos store rotation in EXIF
    resize = transforms.Resize((IMG_SIZE, IMG_SIZE))
    if preprocess == "leaf_cutout":
        view = composite(resize(cut_out_leaf(img)[0]), (0, 0, 0))
    else:
        view = resize(img)
    tensor = transforms.Compose([transforms.ToTensor(), transforms.Normalize(MEAN, STD)])(view)
    return tensor.unsqueeze(0).to(DEVICE), view


def predict(image_path: str, model_path: str = "results/disease_model.pth") -> tuple[str, float]:
    model, classes, preprocess = load_model(model_path)
    tensor, _ = prepare_image(Image.open(image_path), preprocess)
    with torch.no_grad():
        probs = torch.softmax(model(tensor), dim=1)[0]
    idx = probs.argmax().item()
    return classes[idx], float(probs[idx])


if __name__ == "__main__":
    # 1. Precompute leaf masks once (resumable):
    #      python -m src.segmentation data/raw/PlantVillage data/processed/PlantVillage_masks
    # 2. Train on leaf cutouts with random backgrounds, full fine-tuning:
    train(data_dir="data/raw/PlantVillage", epochs=10,
          mask_dir="data/processed/PlantVillage_masks", finetune="all", preprocess="leaf_cutout")
