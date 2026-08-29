#!/usr/bin/env python3
"""
train.py — End-to-end training script for Cats vs Dogs classification.
Tracks experiments with MLflow and saves the best model checkpoint.
import sys

print("Python executable:")
print(sys.executable)

print("Python version:")
print(sys.version)

import torch

print("PyTorch:")
print(torch.__version__)

print("Torch location:")
print(torch.__file__)

import torchvision

print("TorchVision:")
print(torchvision.__version__)

print("TorchVision location:")
print(torchvision.__file__)
Usage:
    python train.py --data_dir data/processed --epochs 15
"""
import argparse
import logging
import os

import matplotlib
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

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Training / evaluation helpers
# ---------------------------------------------------------------------------

def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss, correct, total = 0.0, 0, 0
    for imgs, labels in loader:
        imgs, labels = imgs.to(device), labels.to(device)
        optimizer.zero_grad()
        out  = model(imgs)
        loss = criterion(out, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        correct    += out.argmax(1).eq(labels).sum().item()
        total      += labels.size(0)
    return total_loss / len(loader), 100.0 * correct / total


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    all_preds, all_labels = [], []
    for imgs, labels in loader:
        imgs, labels = imgs.to(device), labels.to(device)
        out  = model(imgs)
        loss = criterion(out, labels)
        total_loss += loss.item()
        preds       = out.argmax(1)
        correct    += preds.eq(labels).sum().item()
        total      += labels.size(0)
        all_preds.extend(preds.cpu().tolist())
        all_labels.extend(labels.cpu().tolist())
    return total_loss / len(loader), 100.0 * correct / total, all_preds, all_labels


# ---------------------------------------------------------------------------
# Plotting helpers
# ---------------------------------------------------------------------------

def save_loss_curves(train_losses, val_losses, path="/tmp/loss_curves.png"):
    plt.figure(figsize=(8, 4))
    plt.plot(train_losses, label="Train Loss")
    plt.plot(val_losses,   label="Val Loss")
    plt.xlabel("Epoch"); plt.ylabel("Loss")
    plt.title("Training & Validation Loss"); plt.legend()
    plt.tight_layout(); plt.savefig(path); plt.close()
    return path


def save_confusion_matrix(labels, preds, class_names, path="/tmp/confusion_matrix.png"):
    cm = confusion_matrix(labels, preds)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=class_names, yticklabels=class_names)
    plt.title("Confusion Matrix")
    plt.ylabel("True"); plt.xlabel("Predicted")
    plt.tight_layout(); plt.savefig(path); plt.close()
    return path


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info("Device: %s", device)

    train_tf, val_tf = get_transforms(args.img_size)

    def make_loader(split, tf, shuffle):
        ds = datasets.ImageFolder(os.path.join(args.data_dir, split), transform=tf)
        return DataLoader(ds, batch_size=args.batch_size, shuffle=shuffle,
                          num_workers=2, pin_memory=True), ds

    train_loader, train_ds = make_loader("train", train_tf, shuffle=True)
    val_loader,   _        = make_loader("val",   val_tf,   shuffle=False)
    test_loader,  _        = make_loader("test",  val_tf,   shuffle=False)
    class_names = train_ds.classes
    log.info("Classes: %s  |  train=%d  val=%d  test=%d",
             class_names, len(train_ds), len(val_loader.dataset), len(test_loader.dataset))

    model     = SimpleCNN(num_classes=2).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.wd)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)

    os.makedirs(os.path.dirname(args.model_path), exist_ok=True)

    mlflow.set_tracking_uri(args.mlflow_uri)
    mlflow.set_experiment(args.experiment)

    with mlflow.start_run(run_name=args.run_name):
        mlflow.log_params({
            "architecture": "SimpleCNN",
            "epochs":       args.epochs,
            "batch_size":   args.batch_size,
            "lr":           args.lr,
            "weight_decay": args.wd,
            "img_size":     args.img_size,
            "optimizer":    "Adam",
            "scheduler":    "StepLR(step=5,gamma=0.5)",
        })

        best_val_acc      = 0.0
        train_losses, val_losses = [], []

        for epoch in range(1, args.epochs + 1):
            tr_loss, tr_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
            vl_loss, vl_acc, _, _ = evaluate(model, val_loader, criterion, device)
            scheduler.step()

            train_losses.append(tr_loss)
            val_losses.append(vl_loss)

            mlflow.log_metrics(
                {"train_loss": tr_loss, "train_acc": tr_acc,
                 "val_loss":   vl_loss, "val_acc":   vl_acc},
                step=epoch,
            )
            log.info("Epoch %2d/%d | tr_loss=%.4f tr_acc=%.2f%% | vl_loss=%.4f vl_acc=%.2f%%",
                     epoch, args.epochs, tr_loss, tr_acc, vl_loss, vl_acc)

            if vl_acc > best_val_acc:
                best_val_acc = vl_acc
                torch.save(model.state_dict(), args.model_path)
                log.info("  -> Saved best model (val_acc=%.2f%%)", vl_acc)

        # ---- Final evaluation on test set ----
        model.load_state_dict(torch.load(args.model_path, map_location=device))
        te_loss, te_acc, te_preds, te_labels = evaluate(model, test_loader, criterion, device)
        mlflow.log_metrics({"test_loss": te_loss, "test_acc": te_acc,
                             "best_val_acc": best_val_acc})
        log.info("Test  | loss=%.4f  acc=%.2f%%", te_loss, te_acc)

        # ---- Artifacts ----
        lc_path = save_loss_curves(train_losses, val_losses)
        cm_path = save_confusion_matrix(te_labels, te_preds, class_names)
        mlflow.log_artifact(lc_path)
        mlflow.log_artifact(cm_path)
        mlflow.log_artifact(args.model_path, "model_weights")

        report = classification_report(te_labels, te_preds, target_names=class_names)
        log.info("\n%s", report)
        rpt_path = "/tmp/classification_report.txt"
        with open(rpt_path, "w") as f:
            f.write(report)
        mlflow.log_artifact(rpt_path)

        mlflow.pytorch.log_model(model, "model",
                                 registered_model_name="cats_dogs_classifier")

    log.info("Run complete. Best val_acc=%.2f%%  test_acc=%.2f%%", best_val_acc, te_acc)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--data_dir",   default="data/processed")
    p.add_argument("--model_path", default="models/best_model.pt")
    p.add_argument("--epochs",     type=int,   default=15)
    p.add_argument("--batch_size", type=int,   default=32)
    p.add_argument("--lr",         type=float, default=1e-3)
    p.add_argument("--wd",         type=float, default=1e-4)
    p.add_argument("--img_size",   type=int,   default=224)
    p.add_argument("--experiment", default="cats-dogs-classification")
    p.add_argument("--run_name",   default="simple-cnn-baseline")
    p.add_argument("--mlflow_uri", default="mlruns")
    main(p.parse_args())
