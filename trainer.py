import copy

import numpy as np
from sklearn.metrics import accuracy_score, f1_score

import torch

# SSL training
def train_one_epoch_ssl(model, loader, optimizer, device):
    model.train()
    total_loss = 0.0
    total_samples = 0

    for batch in loader:
        x = batch["masked_input"].to(device)
        y = batch["target"].to(device)
        m = batch["mask"].to(device)

        optimizer.zero_grad()
        pred = model(x)
        loss = masked_mse_loss(pred, y, m)
        loss.backward()
        optimizer.step()

        batch_size = x.size(0)
        total_loss += loss.item() * batch_size
        total_samples += batch_size

    return total_loss / total_samples


@torch.no_grad()
def evaluate_ssl(model, loader, device):
    model.eval()
    total_loss = 0.0
    total_samples = 0

    for batch in loader:
        x = batch["masked_input"].to(device)
        y = batch["target"].to(device)
        m = batch["mask"].to(device)

        pred = model(x)
        loss = masked_mse_loss(pred, y, m)

        batch_size = x.size(0)
        total_loss += loss.item() * batch_size
        total_samples += batch_size

    return total_loss / total_samples


def train_ssl_model(model, train_loader, val_loader, optimizer, device, num_epochs=30, save_path="best_ssl_model.pt"):
    history = {"train_loss": [], "val_loss": []}

    best_val_loss = float("inf")
    best_state = None
    best_epoch = -1

    for epoch in range(1, num_epochs + 1):
        train_loss = train_one_epoch_ssl(model, train_loader, optimizer, device)
        val_loss = evaluate_ssl(model, val_loader, device)

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)

        print(f"Epoch [{epoch}/{num_epochs}] | Train Loss: {train_loss:.6f} | Val Loss: {val_loss:.6f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
            torch.save(best_state, save_path)
            print(f"  -> Best SSL model saved at epoch {epoch}")

    print(f"\nBest validation loss: {best_val_loss:.6f} at epoch {best_epoch}")

    if best_state is not None:
        model.load_state_dict(best_state)

    return model, history

def train_one_epoch_classifier(model, loader, optimizer, criterion, device, weather_mode=False):
    model.train()
    total_loss = 0.0
    total_samples = 0
    all_y, all_pred = [], []

    for batch in loader:
        optimizer.zero_grad()

        if weather_mode:
            x_img, x_weather, y = batch
            x_img = x_img.to(device)
            x_weather = x_weather.to(device)
            y = y.to(device)
            logits = model(x_img, x_weather)
            batch_size = x_img.size(0)
        else:
            x, y = batch
            x = x.to(device)
            y = y.to(device)
            logits = model(x)
            batch_size = x.size(0)

        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()

        preds = logits.argmax(dim=1)
        total_loss += loss.item() * batch_size
        total_samples += batch_size
        all_y.extend(y.detach().cpu().numpy())
        all_pred.extend(preds.detach().cpu().numpy())

    return (
        total_loss / total_samples,
        accuracy_score(all_y, all_pred),
        f1_score(all_y, all_pred, average="macro")
    )


@torch.no_grad()
def evaluate_classifier(model, loader, criterion, device, weather_mode=False):
    model.eval()
    total_loss = 0.0
    total_samples = 0
    all_y, all_pred = [], []

    for batch in loader:
        if weather_mode:
            x_img, x_weather, y = batch
            x_img = x_img.to(device)
            x_weather = x_weather.to(device)
            y = y.to(device)
            logits = model(x_img, x_weather)
            batch_size = x_img.size(0)
        else:
            x, y = batch
            x = x.to(device)
            y = y.to(device)
            logits = model(x)
            batch_size = x.size(0)

        loss = criterion(logits, y)
        preds = logits.argmax(dim=1)

        total_loss += loss.item() * batch_size
        total_samples += batch_size
        all_y.extend(y.detach().cpu().numpy())
        all_pred.extend(preds.detach().cpu().numpy())

    return (
        total_loss / total_samples,
        accuracy_score(all_y, all_pred),
        f1_score(all_y, all_pred, average="macro"),
        np.array(all_y),
        np.array(all_pred)
    )


def train_classifier_model(model, train_loader, val_loader, optimizer, criterion, device, num_epochs, save_path, weather_mode=False):
    history = {
        "train_loss": [], "val_loss": [],
        "train_acc": [], "val_acc": [],
        "train_f1": [], "val_f1": []
    }

    best_val_f1 = -1.0
    best_epoch = -1
    best_state = None

    for epoch in range(1, num_epochs + 1):
        train_loss, train_acc, train_f1 = train_one_epoch_classifier(
            model, train_loader, optimizer, criterion, device, weather_mode=weather_mode
        )
        val_loss, val_acc, val_f1, _, _ = evaluate_classifier(
            model, val_loader, criterion, device, weather_mode=weather_mode
        )

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_acc"].append(train_acc)
        history["val_acc"].append(val_acc)
        history["train_f1"].append(train_f1)
        history["val_f1"].append(val_f1)

        print(
            f"Epoch [{epoch}/{num_epochs}] | "
            f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | Train F1: {train_f1:.4f} | "
            f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f} | Val F1: {val_f1:.4f}"
        )

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
            torch.save(best_state, save_path)
            print(f"  -> Best classifier saved at epoch {epoch}")

    print(f"\nBest validation macro-F1: {best_val_f1:.4f} at epoch {best_epoch}")

    if best_state is not None:
        model.load_state_dict(best_state)

    return model, history

# Loss function
def masked_mse_loss(pred, target, mask, eps=1e-8):
    sq_error = (pred - target) ** 2
    masked_sq_error = sq_error * mask
    return masked_sq_error.sum() / (mask.sum() + eps)
