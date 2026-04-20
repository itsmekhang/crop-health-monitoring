"""
Image-based crop disease classification using ResNet-18 (PyTorch + torchvision).
Dataset: PlantVillage — 14 crops, 39 disease/healthy classes, ~54,000 images.
All images are lab-controlled (uniform backgrounds). Field performance will differ.
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms, models
from pathlib import Path


IMG_SIZE = 224
BATCH_SIZE = 32
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def get_transforms(augment: bool = False):
    base = [
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ]
    if augment:
        base = [
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(15),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
        ] + base
    return transforms.Compose(base)


def build_model(num_classes: int, unfreeze_last_block: bool = False) -> nn.Module:
    model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    # Freeze all layers — backbone acts as a fixed ImageNet feature extractor
    for param in model.parameters():
        param.requires_grad = False
    # Optionally unfreeze the last residual block (layer4) for domain adaptation.
    # PlantVillage images differ substantially from ImageNet — unfreezing layer4
    # lets the network adapt mid-level texture features toward disease-specific
    # patterns without the cost or instability of full fine-tuning.
    if unfreeze_last_block:
        for param in model.layer4.parameters():
            param.requires_grad = True
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model.to(DEVICE)


def load_datasets(data_dir: str):
    full_ds = datasets.ImageFolder(data_dir, transform=get_transforms(augment=True))
    n_val = int(0.2 * len(full_ds))
    train_ds, val_ds = random_split(full_ds, [len(full_ds) - n_val, n_val],
                                    generator=torch.Generator().manual_seed(42))
    val_ds.dataset.transform = get_transforms(augment=False)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False, num_workers=2)
    return train_loader, val_loader, full_ds.classes


def train(data_dir: str, epochs: int = 10, save_path: str = "results/disease_model.pth",
          unfreeze_last_block: bool = False):
    train_loader, val_loader, classes = load_datasets(data_dir)
    print(f"Classes ({len(classes)}): {classes[:5]} ...")

    model = build_model(num_classes=len(classes), unfreeze_last_block=unfreeze_last_block)
    criterion = nn.CrossEntropyLoss()
    # When layer4 is unfrozen, use a lower lr for backbone params to avoid
    # destroying pretrained features (differential learning rates).
    if unfreeze_last_block:
        optimizer = torch.optim.Adam([
            {"params": model.layer4.parameters(), "lr": 1e-4},
            {"params": model.fc.parameters(),     "lr": 1e-3},
        ])
    else:
        optimizer = torch.optim.Adam(model.fc.parameters(), lr=1e-3)

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

        val_acc = evaluate(model, val_loader)
        print(f"Epoch {epoch}/{epochs}  loss={total_loss/len(train_loader):.4f}  "
              f"train_acc={correct/total:.3f}  val_acc={val_acc:.3f}")

    torch.save({"model_state": model.state_dict(), "classes": classes}, save_path)
    print(f"Model saved to {save_path}")
    return model, classes


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


def predict(image_path: str, model_path: str = "results/disease_model.pth") -> tuple[str, float]:
    from PIL import Image
    checkpoint = torch.load(model_path, map_location=DEVICE)
    classes = checkpoint["classes"]
    model = build_model(num_classes=len(classes))
    model.load_state_dict(checkpoint["model_state"])
    model.eval()

    img = Image.open(image_path).convert("RGB")
    tensor = get_transforms()(img).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        probs = torch.softmax(model(tensor), dim=1)[0]
    idx = probs.argmax().item()
    return classes[idx], float(probs[idx])


if __name__ == "__main__":
    # Set unfreeze_last_block=True to fine-tune layer4 in addition to the FC head.
    # Use this to test whether partially thawed backbone weights improve accuracy
    # on PlantVillage vs. the frozen-backbone baseline.
    train(data_dir="data/raw/PlantVillage", epochs=10, unfreeze_last_block=False)
