import os
import random
import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def split_zone_chronologically(zone_df, train_ratio=0.7, val_ratio=0.15):
    zone_df = zone_df.sort_values("time").reset_index(drop=True)
    n = len(zone_df)

    train_end = int(n * train_ratio)
    val_end = int(n * (train_ratio + val_ratio))

    train_df = zone_df.iloc[:train_end].copy()
    val_df = zone_df.iloc[train_end:val_end].copy()
    test_df = zone_df.iloc[val_end:].copy()

    return train_df, val_df, test_df


def plot_loss_curves(history, save_dir, prefix, y_label="Loss", title="Training Curve"):
    epochs = range(1, len(history["train_loss"]) + 1)

    plt.figure(figsize=(8, 5))
    plt.plot(epochs, history["train_loss"], label="Train")
    plt.plot(epochs, history["val_loss"], label="Val")
    plt.xlabel("Epoch")
    plt.ylabel(y_label)
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f"{prefix}_loss.png"), dpi=300)
    plt.close()


def plot_classifier_history(history, save_dir, prefix):
    epochs = range(1, len(history["train_loss"]) + 1)

    plt.figure(figsize=(8, 5))
    plt.plot(epochs, history["train_loss"], label="Train Loss")
    plt.plot(epochs, history["val_loss"], label="Val Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Cross-Entropy Loss")
    plt.title(f"{prefix} Loss")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f"{prefix}_loss.png"), dpi=300)
    plt.close()

    plt.figure(figsize=(8, 5))
    plt.plot(epochs, history["train_acc"], label="Train Accuracy")
    plt.plot(epochs, history["val_acc"], label="Val Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title(f"{prefix} Accuracy")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f"{prefix}_accuracy.png"), dpi=300)
    plt.close()

    plt.figure(figsize=(8, 5))
    plt.plot(epochs, history["train_f1"], label="Train Macro-F1")
    plt.plot(epochs, history["val_f1"], label="Val Macro-F1")
    plt.xlabel("Epoch")
    plt.ylabel("Macro-F1")
    plt.title(f"{prefix} Macro-F1")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f"{prefix}_macro_f1.png"), dpi=300)
    plt.close()


def save_test_outputs(test_loss, test_acc, test_f1, y_true, y_pred, reports_dir, stage_name):
    cm = confusion_matrix(y_true, y_pred)
    cls_report = classification_report(y_true, y_pred, digits=4)

    with open(os.path.join(reports_dir, f"{stage_name}_test_metrics.txt"), "w") as f:
        f.write("===== TEST RESULTS =====\n")
        f.write(f"Test Loss     : {test_loss:.6f}\n")
        f.write(f"Test Accuracy : {test_acc:.6f}\n")
        f.write(f"Test Macro-F1 : {test_f1:.6f}\n")

    with open(os.path.join(reports_dir, f"{stage_name}_confusion_matrix.txt"), "w") as f:
        f.write("Confusion Matrix\n")
        f.write(np.array2string(cm))

    with open(os.path.join(reports_dir, f"{stage_name}_classification_report.txt"), "w") as f:
        f.write(cls_report)

    print(f"\n===== {stage_name.upper()} TEST RESULTS =====")
    print(f"Test Loss     : {test_loss:.4f}")
    print(f"Test Accuracy : {test_acc:.4f}")
    print(f"Test Macro-F1 : {test_f1:.4f}")
    print("\nConfusion Matrix:")
    print(cm)
    print("\nClassification Report:")
    print(cls_report)



# Classifier helpers
def freeze_encoder(model):
    for p in model.encoder.parameters():
        p.requires_grad = False


def unfreeze_encoder(model):
    for p in model.encoder.parameters():
        p.requires_grad = True