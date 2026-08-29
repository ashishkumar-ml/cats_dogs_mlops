
#!/usr/bin/env python3
"""
train.py — End-to-end training script for Cats vs Dogs classification.

Tracks experiments with MLflow and saves the best model checkpoint.

Usage:
    python train.py --data_dir data/processed --epochs 15
"""

# ---------------------------------------------------------------------------
# Python / environment diagnostics
# ---------------------------------------------------------------------------

import sys

print("Python executable:")
print(sys.executable)

print("\nPython version:")
print(sys.version)

# ---------------------------------------------------------------------------
# Imports
# ---------------------------------------------------------------------------

import argparse
import logging
import os

import matplotlib

# Use a non-GUI backend so plotting works in VS Code / terminal
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import mlflow
import mlflow.pytorch
import seaborn as sns
import torch
import torch.nn as nn
import torch.optim as optim

from sklearn.metrics import classification_report, confusion_matrix
from torch.utils.data import DataLoader
from torchvision import datasets

from src.model import SimpleCNN
from src.preprocess import get_transforms


# ---------------------------------------------------------------------------
# PyTorch diagnostics
# ---------------------------------------------------------------------------

print("\nPyTorch:")
print(torch.__version__)

print("Torch location:")
print(torch.__file__)

try:
    import torchvision

    print("\nTorchVision:")
    print(torchvision.__version__)

    print("TorchVision location:")
    print(torchvision.__file__)

except Exception as e:
    print("\nTorchVision import failed:")
    print(e)
    raise


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s"
)

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Training / evaluation helpers
# ---------------------------------------------------------------------------

def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()

    total_loss = 0.0
    correct = 0
    total = 0

    for imgs, labels in loader:
        imgs = imgs.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        outputs = model(imgs)
        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

        total_loss += loss.item()

        predictions = outputs.argmax(dim=1)
        correct += predictions.eq(labels).sum().item()
        total += labels.size(0)

    avg_loss = total_loss / len(loader)
    accuracy = 100.0 * correct / total

    return avg_loss, accuracy


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()

    total_loss = 0.0
    correct = 0
    total = 0

    all_preds = []
    all_labels = []

    for imgs, labels in loader:
        imgs = imgs.to(device)
        labels = labels.to(device)

        outputs = model(imgs)
        loss = criterion(outputs, labels)

        total_loss += loss.item()

        predictions = outputs.argmax(dim=1)

        correct += predictions.eq(labels).sum().item()
        total += labels.size(0)

        all_preds.extend(predictions.cpu().tolist())
        all_labels.extend(labels.cpu().tolist())

    avg_loss = total_loss / len(loader)
    accuracy = 100.0 * correct / total

    return avg_loss, accuracy, all_preds, all_labels


# ---------------------------------------------------------------------------
# Plotting helpers
# ---------------------------------------------------------------------------

def save_loss_curves(
    train_losses,
    val_losses,
    path="artifacts/loss_curves.png"
):
    os.makedirs(os.path.dirname(path), exist_ok=True)

    plt.figure(figsize=(8, 4))

    plt.plot(train_losses, label="Train Loss")
    plt.plot(val_losses, label="Val Loss")

    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training & Validation Loss")
    plt.legend()

    plt.tight_layout()
    plt.savefig(path)
    plt.close()

    return path


def save_confusion_matrix(
    labels,
    preds,
    class_names,
    path="artifacts/confusion_matrix.png"
):
    os.makedirs(os.path.dirname(path), exist_ok=True)

    cm = confusion_matrix(labels, preds)

    plt.figure(figsize=(6, 5))

    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names
    )

    plt.title("Confusion Matrix")
    plt.ylabel("True")
    plt.xlabel("Predicted")

    plt.tight_layout()
    plt.savefig(path)
    plt.close()

    return path


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(args):

    # ---------------------------------------------------------------
    # Device
    # ---------------------------------------------------------------

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    log.info("Device: %s", device)

    if torch.cuda.is_available():
        log.info("CUDA device: %s", torch.cuda.get_device_name(0))
    else:
        log.info("CUDA not available. Using CPU.")


    # ---------------------------------------------------------------
    # Transforms
    # ---------------------------------------------------------------

    train_tf, val_tf = get_transforms(args.img_size)


    # ---------------------------------------------------------------
    # Dataset / DataLoader
    # ---------------------------------------------------------------

    def make_loader(split, transform, shuffle):

        split_path = os.path.join(
            args.data_dir,
            split
        )

        if not os.path.exists(split_path):
            raise FileNotFoundError(
                f"Dataset directory not found: {split_path}"
            )

        dataset = datasets.ImageFolder(
            split_path,
            transform=transform
        )

        loader = DataLoader(
            dataset,
            batch_size=args.batch_size,
            shuffle=shuffle,

            # Windows / VS Code safe setting
            num_workers=args.num_workers,

            pin_memory=torch.cuda.is_available()
        )

        return loader, dataset


    train_loader, train_ds = make_loader(
        "train",
        train_tf,
        shuffle=True
    )

    val_loader, val_ds = make_loader(
        "val",
        val_tf,
        shuffle=False
    )

    test_loader, test_ds = make_loader(
        "test",
        val_tf,
        shuffle=False
    )


    # ---------------------------------------------------------------
    # Classes
    # ---------------------------------------------------------------

    class_names = train_ds.classes

    log.info(
        "Classes: %s | train=%d | val=%d | test=%d",
        class_names,
        len(train_ds),
        len(val_ds),
        len(test_ds)
    )

    if len(class_names) != 2:
        raise ValueError(
            f"Expected 2 classes, but found {len(class_names)}: "
            f"{class_names}"
        )


    # ---------------------------------------------------------------
    # Model
    # ---------------------------------------------------------------

    model = SimpleCNN(
        num_classes=2
    ).to(device)


    # ---------------------------------------------------------------
    # Loss / optimizer / scheduler
    # ---------------------------------------------------------------

    criterion = nn.CrossEntropyLoss()

    optimizer = optim.Adam(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.wd
    )

    scheduler = optim.lr_scheduler.StepLR(
        optimizer,
        step_size=5,
        gamma=0.5
    )


    # ---------------------------------------------------------------
    # Model directory
    # ---------------------------------------------------------------

    model_dir = os.path.dirname(args.model_path)

    if model_dir:
        os.makedirs(
            model_dir,
            exist_ok=True
        )


    # ---------------------------------------------------------------
    # MLflow
    # ---------------------------------------------------------------

    mlflow.set_tracking_uri(
        args.mlflow_uri
    )

    mlflow.set_experiment(
        args.experiment
    )


    # ---------------------------------------------------------------
    # Training
    # ---------------------------------------------------------------

    with mlflow.start_run(
        run_name=args.run_name
    ):

        mlflow.log_params({
            "architecture": "SimpleCNN",
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "lr": args.lr,
            "weight_decay": args.wd,
            "img_size": args.img_size,
            "optimizer": "Adam",
            "scheduler": "StepLR(step=5,gamma=0.5)",
            "device": str(device),
        })


        best_val_acc = 0.0

        train_losses = []
        val_losses = []


        # -----------------------------------------------------------
        # Epoch loop
        # -----------------------------------------------------------

        for epoch in range(
            1,
            args.epochs + 1
        ):

            train_loss, train_acc = train_one_epoch(
                model,
                train_loader,
                criterion,
                optimizer,
                device
            )

            val_loss, val_acc, _, _ = evaluate(
                model,
                val_loader,
                criterion,
                device
            )

            scheduler.step()


            train_losses.append(train_loss)
            val_losses.append(val_loss)


            # -------------------------------------------------------
            # MLflow metrics
            # -------------------------------------------------------

            mlflow.log_metrics(
                {
                    "train_loss": train_loss,
                    "train_acc": train_acc,
                    "val_loss": val_loss,
                    "val_acc": val_acc,
                },
                step=epoch
            )


            log.info(
                "Epoch %2d/%d | "
                "tr_loss=%.4f tr_acc=%.2f%% | "
                "vl_loss=%.4f vl_acc=%.2f%%",
                epoch,
                args.epochs,
                train_loss,
                train_acc,
                val_loss,
                val_acc
            )


            # -------------------------------------------------------
            # Save best model
            # -------------------------------------------------------

            if val_acc > best_val_acc:

                best_val_acc = val_acc

                torch.save(
                    model.state_dict(),
                    args.model_path
                )

                log.info(
                    "  -> Saved best model "
                    "(val_acc=%.2f%%)",
                    val_acc
                )


        # -----------------------------------------------------------
        # Verify checkpoint exists
        # -----------------------------------------------------------

        if not os.path.exists(args.model_path):
            raise FileNotFoundError(
                f"Best model checkpoint was not created: "
                f"{args.model_path}"
            )


        # -----------------------------------------------------------
        # Final test evaluation
        # -----------------------------------------------------------

        model.load_state_dict(
            torch.load(
                args.model_path,
                map_location=device
            )
        )


        test_loss, test_acc, test_preds, test_labels = evaluate(
            model,
            test_loader,
            criterion,
            device
        )


        mlflow.log_metrics({
            "test_loss": test_loss,
            "test_acc": test_acc,
            "best_val_acc": best_val_acc
        })


        log.info(
            "Test | loss=%.4f | acc=%.2f%%",
            test_loss,
            test_acc
        )


        # -----------------------------------------------------------
        # Loss curve
        # -----------------------------------------------------------

        loss_curve_path = save_loss_curves(
            train_losses,
            val_losses
        )


        # -----------------------------------------------------------
        # Confusion matrix
        # -----------------------------------------------------------

        confusion_matrix_path = save_confusion_matrix(
            test_labels,
            test_preds,
            class_names
        )


        # -----------------------------------------------------------
        # MLflow artifacts
        # -----------------------------------------------------------

        mlflow.log_artifact(
            loss_curve_path
        )

        mlflow.log_artifact(
            confusion_matrix_path
        )

        mlflow.log_artifact(
            args.model_path,
            "model_weights"
        )


        # -----------------------------------------------------------
        # Classification report
        # -----------------------------------------------------------

        report = classification_report(
            test_labels,
            test_preds,
            target_names=class_names,
            zero_division=0
        )

        log.info(
            "\n%s",
            report
        )


        report_path = os.path.join(
            "artifacts",
            "classification_report.txt"
        )

        os.makedirs(
            "artifacts",
            exist_ok=True
        )

        with open(
            report_path,
            "w",
            encoding="utf-8"
        ) as f:
            f.write(report)


        mlflow.log_artifact(
            report_path
        )


        # -----------------------------------------------------------
        # MLflow model
        # -----------------------------------------------------------

        try:

            mlflow.pytorch.log_model(
                model,
                "model",
                registered_model_name="cats_dogs_classifier"
            )

        except Exception as e:

            log.warning(
                "MLflow model registration failed: %s",
                e
            )

            # Still save the model to MLflow
            mlflow.pytorch.log_model(
                model,
                "model"
            )


    # ---------------------------------------------------------------
    # Finished
    # ---------------------------------------------------------------

    log.info(
        "Run complete. "
        "Best val_acc=%.2f%% | test_acc=%.2f%%",
        best_val_acc,
        test_acc
    )


# ---------------------------------------------------------------------------
# Command-line arguments
# ---------------------------------------------------------------------------

if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Train Cats vs Dogs classifier"
    )

    parser.add_argument(
        "--data_dir",
        default="data/processed"
    )

    parser.add_argument(
        "--model_path",
        default="models/best_model.pt"
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=15
    )

    parser.add_argument(
        "--batch_size",
        type=int,
        default=32
    )

    parser.add_argument(
        "--lr",
        type=float,
        default=1e-3
    )

    parser.add_argument(
        "--wd",
        type=float,
        default=1e-4
    )

    parser.add_argument(
        "--img_size",
        type=int,
        default=224
    )

    parser.add_argument(
        "--experiment",
        default="cats-dogs-classification"
    )

    parser.add_argument(
        "--run_name",
        default="simple-cnn-baseline"
    )

    parser.add_argument(
        "--mlflow_uri",
        default="mlruns"
    )

    # Important for Windows / VS Code
    parser.add_argument(
        "--num_workers",
        type=int,
        default=0
    )

    args = parser.parse_args()

    main(args)

