"""
Image-based crop disease classification using ResNet-18 (PyTorch + torchvision).
Dataset: PlantVillage — 3 crops (Tomato, Potato, Pepper), 15 classes, ~41,000 images.
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


def build_model(num_classes: int) -> nn.Module:
    model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    # Freeze all layers except the final classifier
    for param in model.parameters():
        param.requires_grad = False
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


def train(data_dir: str, epochs: int = 10, save_path: str = "results/disease_model.pth"):
    train_loader, val_loader, classes = load_datasets(data_dir)
    print(f"Classes ({len(classes)}): {classes[:5]} ...")

    model = build_model(num_classes=len(classes))
    criterion = nn.CrossEntropyLoss()
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


def evaluate(model: nn.Module, loader: DataLoader) -> float:
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            preds = model(images).argmax(1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
    return correct / total


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
    train(data_dir="data/raw/PlantVillage", epochs=10)
