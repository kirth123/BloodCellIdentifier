import argparse
import random
from pathlib import Path

import medmnist
import numpy as np
import torch
from medmnist import INFO
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from torch import nn
from torch.utils.data import DataLoader


# source of data: https://medmnist.com/

class SmallCNN(nn.Module):
    def __init__(self, num_classes: int) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes),
        )

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.features(images))


def image_to_tensor(image: object) -> torch.Tensor:
    array = np.asarray(image, dtype=np.float32) / 255.0
    return torch.from_numpy(array).permute(2, 0, 1)


def make_loader(
    dataset_class: type,
    split: str,
    data_dir: Path,
    batch_size: int,
    shuffle: bool,
    seed: int,
) -> DataLoader:
    data_dir.mkdir(parents=True, exist_ok=True)
    dataset = dataset_class(
        split=split,
        root=str(data_dir),
        download=True,
        transform=image_to_tensor,
    )
    generator = torch.Generator().manual_seed(seed)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        generator=generator if shuffle else None,
        num_workers=0,
    )


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None = None,
) -> tuple[float, float]:
    is_training = optimizer is not None
    model.train(is_training)
    total_loss = 0.0
    predictions: list[int] = []
    targets: list[int] = []

    for images, labels in loader:
        images = images.to(device)
        labels = labels.squeeze(1).long().to(device)

        if is_training:
            optimizer.zero_grad()

        with torch.set_grad_enabled(is_training):
            logits = model(images)
            loss = criterion(logits, labels)
            if is_training:
                loss.backward()
                optimizer.step()

        total_loss += loss.item() * labels.size(0)
        predictions.extend(logits.argmax(dim=1).detach().cpu().tolist())
        targets.extend(labels.detach().cpu().tolist())

    return total_loss / len(loader.dataset), accuracy_score(targets, predictions)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a small CNN on BloodMNIST.")
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--output", type=Path, default=Path("models/best_model.pt"))
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if args.epochs < 1 or args.batch_size < 1 or args.learning_rate <= 0:
        parser.error("epochs, batch-size, and learning-rate must be positive")

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset_info = INFO["bloodmnist"]
    dataset_class = getattr(medmnist, dataset_info["python_class"])
    class_names = [dataset_info["label"][str(i)] for i in range(len(dataset_info["label"]))]
    num_classes = len(class_names)

    train_loader = make_loader(
        dataset_class, "train", args.data_dir, args.batch_size, True, args.seed
    )
    val_loader = make_loader(
        dataset_class, "val", args.data_dir, args.batch_size, False, args.seed
    )
    test_loader = make_loader(
        dataset_class, "test", args.data_dir, args.batch_size, False, args.seed
    )

    model = SmallCNN(num_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
    best_val_accuracy = -1.0
    args.output.parent.mkdir(parents=True, exist_ok=True)

    print(f"Device: {device}")
    print(f"Classes ({num_classes}): {', '.join(class_names)}")
    for epoch in range(1, args.epochs + 1):
        train_loss, train_accuracy = run_epoch(
            model, train_loader, criterion, device, optimizer
        )
        val_loss, val_accuracy = run_epoch(model, val_loader, criterion, device)
        print(
            f"Epoch {epoch:02d}/{args.epochs} "
            f"train_loss={train_loss:.4f} train_acc={train_accuracy:.4f} "
            f"val_loss={val_loss:.4f} val_acc={val_accuracy:.4f}"
        )

        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "class_names": class_names,
                    "val_accuracy": best_val_accuracy,
                },
                args.output,
            )

    checkpoint = torch.load(args.output, map_location=device, weights_only=True)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    test_predictions: list[int] = []
    test_targets: list[int] = []
    with torch.inference_mode():
        for images, labels in test_loader:
            logits = model(images.to(device))
            test_predictions.extend(logits.argmax(dim=1).cpu().tolist())
            test_targets.extend(labels.squeeze(1).tolist())

    print(f"\nBest validation accuracy: {best_val_accuracy:.4f}")
    print(f"Test accuracy: {accuracy_score(test_targets, test_predictions):.4f}")
    print("\nPer-class test metrics:")
    print(
        classification_report(
            test_targets,
            test_predictions,
            labels=list(range(num_classes)),
            target_names=class_names,
            zero_division=0,
        )
    )
    print("Test confusion matrix (rows=true, columns=predicted):")
    print(confusion_matrix(test_targets, test_predictions, labels=list(range(num_classes))))
    print(f"\nSaved best checkpoint to {args.output}")


if __name__ == "__main__":
    main()
