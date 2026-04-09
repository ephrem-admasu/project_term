import os
import json


import numpy as np
import pandas as pd


import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from config import (
    CSV_PATH, SEED, HISTORY, TRAIN_RATIO, VAL_RATIO, NUM_CLASSES, BATCH_SIZE, 
    NUM_WORKERS, PIN_MEMORY, SSL_EPOCHS, LINEAR_PROBE_EPOCHS, FINETUNE_EPOCHS, 
    DEVICE, DEMAND_ROOT, WEATHER_ROOT, WEATHER_COLS
)

from utilities import set_seed, plot_loss_curves, plot_classifier_history, save_test_outputs, freeze_encoder, unfreeze_encoder

from models import (
    GAFMaskedAutoencoder, GAFDemandClassifier, GAFWeatherHistoryFusionClassifier
)

from dataset import (
    MaskedGAFDataset, GAFClassificationDataset, GAFWeatherClassificationDataset
)

from preprocessing import (
    load_prepared_final_df, save_demand_only_arrays, save_weather_fusion_arrays
)

from trainer import train_ssl_model, evaluate_classifier, train_classifier_model

DEMAND_DATA_DIR = os.path.join(DEMAND_ROOT, "gaf_zone_data_ssl")
DEMAND_OUTPUT_DIR = os.path.join(DEMAND_ROOT, "outputs")
DEMAND_PLOTS_DIR = os.path.join(DEMAND_OUTPUT_DIR, "plots")
DEMAND_CHECKPOINTS_DIR = os.path.join(DEMAND_OUTPUT_DIR, "checkpoints")
DEMAND_REPORTS_DIR = os.path.join(DEMAND_OUTPUT_DIR, "reports")
DEMAND_SUMMARY_DIR = os.path.join(DEMAND_OUTPUT_DIR, "data_summary")

# demand+weather experiment folders

WEATHER_DATA_DIR = os.path.join(WEATHER_ROOT, "gaf_zone_data_weatherhist")
WEATHER_OUTPUT_DIR = os.path.join(WEATHER_ROOT, "outputs")
WEATHER_PLOTS_DIR = os.path.join(WEATHER_OUTPUT_DIR, "plots")
WEATHER_CHECKPOINTS_DIR = os.path.join(WEATHER_OUTPUT_DIR, "checkpoints")
WEATHER_REPORTS_DIR = os.path.join(WEATHER_OUTPUT_DIR, "reports")
WEATHER_SUMMARY_DIR = os.path.join(WEATHER_OUTPUT_DIR, "data_summary")

for d in [
    DEMAND_ROOT, DEMAND_DATA_DIR, DEMAND_OUTPUT_DIR, DEMAND_PLOTS_DIR,
    DEMAND_CHECKPOINTS_DIR, DEMAND_REPORTS_DIR, DEMAND_SUMMARY_DIR,
    WEATHER_ROOT, WEATHER_DATA_DIR, WEATHER_OUTPUT_DIR, WEATHER_PLOTS_DIR,
    WEATHER_CHECKPOINTS_DIR, WEATHER_REPORTS_DIR, WEATHER_SUMMARY_DIR
]:
    os.makedirs(d, exist_ok=True)


set_seed(SEED)


# Array Loaders
def load_split_from_npy(root_dir, split="train"):
    x_files = sorted([
        os.path.join(root_dir, f)
        for f in os.listdir(root_dir)
        if f.endswith(f"_X_{split}.npy")
    ])
    y_files = sorted([
        os.path.join(root_dir, f)
        for f in os.listdir(root_dir)
        if f.endswith(f"_y_{split}.npy")
    ])

    assert len(x_files) > 0, f"No X files found for split={split}"

    X = np.concatenate([np.load(f) for f in x_files], axis=0).astype(np.float32)
    y = None
    if len(y_files) > 0:
        y = np.concatenate([np.load(f) for f in y_files], axis=0).astype(np.int64)

    return X, y


def load_xy_split_from_npy(root_dir, split="train"):
    x_files = sorted([
        os.path.join(root_dir, f)
        for f in os.listdir(root_dir)
        if f.endswith(f"_X_{split}.npy")
    ])
    y_files = sorted([
        os.path.join(root_dir, f)
        for f in os.listdir(root_dir)
        if f.endswith(f"_y_{split}.npy")
    ])

    assert len(x_files) > 0, f"No X files found for split={split}"
    assert len(y_files) > 0, f"No y files found for split={split}"
    assert len(x_files) == len(y_files), "Mismatch in X/y file counts"

    X = np.concatenate([np.load(f) for f in x_files], axis=0).astype(np.float32)
    y = np.concatenate([np.load(f) for f in y_files], axis=0).astype(np.int64)
    return X, y


def load_xwy_split_from_npy(root_dir, split="train"):
    x_files = sorted([
        os.path.join(root_dir, f)
        for f in os.listdir(root_dir)
        if f.endswith(f"_X_{split}.npy")
    ])
    w_files = sorted([
        os.path.join(root_dir, f)
        for f in os.listdir(root_dir)
        if f.endswith(f"_W_{split}.npy")
    ])
    y_files = sorted([
        os.path.join(root_dir, f)
        for f in os.listdir(root_dir)
        if f.endswith(f"_y_{split}.npy")
    ])

    assert len(x_files) > 0, f"No X files found for split={split}"
    assert len(w_files) > 0, f"No W files found for split={split}"
    assert len(y_files) > 0, f"No y files found for split={split}"
    assert len(x_files) == len(w_files) == len(y_files), "Mismatch in X/W/y file counts"

    X = np.concatenate([np.load(f) for f in x_files], axis=0).astype(np.float32)
    W = np.concatenate([np.load(f) for f in w_files], axis=0).astype(np.float32)
    y = np.concatenate([np.load(f) for f in y_files], axis=0).astype(np.int64)

    return X, W, y




def run_ssl_pretraining(data_dir, checkpoints_dir, plots_dir, reports_dir, exp_name):
    print(f"\n================ SSL PRETRAINING: {exp_name} ================")

    X_train, _ = load_split_from_npy(data_dir, split="train")
    X_val, _ = load_split_from_npy(data_dir, split="val")

    train_dataset = MaskedGAFDataset(X_train, patch_size=4, mask_ratio=0.4, mask_value=0.0)
    val_dataset = MaskedGAFDataset(X_val, patch_size=4, mask_ratio=0.4, mask_value=0.0)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False,
                            num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY)

    model = GAFMaskedAutoencoder().to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)

    ssl_ckpt_path = os.path.join(checkpoints_dir, "best_ssl_model.pt")
    model, history = train_ssl_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        device=DEVICE,
        num_epochs=SSL_EPOCHS,
        save_path=ssl_ckpt_path
    )

    plot_loss_curves(history, plots_dir, prefix=f"{exp_name}_ssl", y_label="Masked MSE Loss", title=f"{exp_name} SSL")
    pd.DataFrame(history).to_csv(os.path.join(reports_dir, f"{exp_name}_ssl_history.csv"), index=False)

    return ssl_ckpt_path



def build_demand_classifier_from_ssl(ssl_ckpt_path, num_classes=4, device="cpu"):
    ssl_model = GAFMaskedAutoencoder()
    ssl_model.load_state_dict(torch.load(ssl_ckpt_path, map_location=device))
    return GAFDemandClassifier(
        encoder=ssl_model.encoder,
        num_classes=num_classes,
        dropout=0.2
    )



def build_weather_fusion_classifier_from_ssl(ssl_ckpt_path, num_classes=4, history=24, num_weather_features=3, device="cpu"):
    ssl_model = GAFMaskedAutoencoder()
    ssl_model.load_state_dict(torch.load(ssl_ckpt_path, map_location=device))

    return GAFWeatherHistoryFusionClassifier(
        encoder=ssl_model.encoder,
        history=history,
        num_weather_features=num_weather_features,
        num_classes=num_classes,
        dropout=0.2
    )


# Run Demand-Only Classification
def run_demand_only_pipeline(data_dir, checkpoints_dir, plots_dir, reports_dir, ssl_ckpt_path):
    print("\n================ DEMAND-ONLY CLASSIFICATION ================")

    X_train, y_train = load_xy_split_from_npy(data_dir, split="train")
    X_val, y_val = load_xy_split_from_npy(data_dir, split="val")
    X_test, y_test = load_xy_split_from_npy(data_dir, split="test")

    train_dataset = GAFClassificationDataset(X_train, y_train)
    val_dataset = GAFClassificationDataset(X_val, y_val)
    test_dataset = GAFClassificationDataset(X_test, y_test)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False,
                            num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False,
                             num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY)

    class_counts = np.bincount(y_train, minlength=NUM_CLASSES)
    class_weights = class_counts.sum() / (len(class_counts) * np.maximum(class_counts, 1))
    class_weights = torch.tensor(class_weights, dtype=torch.float32, device=DEVICE)
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    # stage 1
    model = build_demand_classifier_from_ssl(ssl_ckpt_path, num_classes=NUM_CLASSES, device=DEVICE).to(DEVICE)
    freeze_encoder(model)

    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=1e-3,
        weight_decay=1e-4
    )

    linear_probe_ckpt = os.path.join(checkpoints_dir, "best_classifier_linear_probe.pt")

    print("\n===== DEMAND-ONLY STAGE 1: LINEAR PROBE =====")
    model, history_probe = train_classifier_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        criterion=criterion,
        device=DEVICE,
        num_epochs=LINEAR_PROBE_EPOCHS,
        save_path=linear_probe_ckpt,
        weather_mode=False
    )

    plot_classifier_history(history_probe, plots_dir, prefix="demand_only_linear_probe")
    pd.DataFrame(history_probe).to_csv(os.path.join(reports_dir, "demand_only_linear_probe_history.csv"), index=False)

    test_loss, test_acc, test_f1, y_true, y_pred = evaluate_classifier(
        model, test_loader, criterion, DEVICE, weather_mode=False
    )
    save_test_outputs(test_loss, test_acc, test_f1, y_true, y_pred, reports_dir, "demand_only_linear_probe")

    # stage 2
    model = build_demand_classifier_from_ssl(ssl_ckpt_path, num_classes=NUM_CLASSES, device=DEVICE).to(DEVICE)
    model.load_state_dict(torch.load(linear_probe_ckpt, map_location=DEVICE))
    unfreeze_encoder(model)

    optimizer = torch.optim.AdamW(
        [
            {"params": model.encoder.parameters(), "lr": 1e-4},
            {"params": model.classifier.parameters(), "lr": 1e-3},
        ],
        weight_decay=1e-4
    )

    finetune_ckpt = os.path.join(checkpoints_dir, "best_classifier_finetuned.pt")

    print("\n===== DEMAND-ONLY STAGE 2: FINE-TUNING =====")
    model, history_ft = train_classifier_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        criterion=criterion,
        device=DEVICE,
        num_epochs=FINETUNE_EPOCHS,
        save_path=finetune_ckpt,
        weather_mode=False
    )

    plot_classifier_history(history_ft, plots_dir, prefix="demand_only_finetuned")
    pd.DataFrame(history_ft).to_csv(os.path.join(reports_dir, "demand_only_finetuned_history.csv"), index=False)

    model.load_state_dict(torch.load(finetune_ckpt, map_location=DEVICE))
    test_loss, test_acc, test_f1, y_true, y_pred = evaluate_classifier(
        model, test_loader, criterion, DEVICE, weather_mode=False
    )
    save_test_outputs(test_loss, test_acc, test_f1, y_true, y_pred, reports_dir, "demand_only_finetuned")



# Run Weather-Fusion Classification
def run_weather_fusion_pipeline(data_dir, checkpoints_dir, plots_dir, reports_dir, ssl_ckpt_path):
    print("\n================ WEATHER-HISTORY FUSION CLASSIFICATION ================")

    X_train, W_train, y_train = load_xwy_split_from_npy(data_dir, split="train")
    X_val, W_val, y_val = load_xwy_split_from_npy(data_dir, split="val")
    X_test, W_test, y_test = load_xwy_split_from_npy(data_dir, split="test")

    train_dataset = GAFWeatherClassificationDataset(X_train, W_train, y_train)
    val_dataset = GAFWeatherClassificationDataset(X_val, W_val, y_val)
    test_dataset = GAFWeatherClassificationDataset(X_test, W_test, y_test)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False,
                            num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False,
                             num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY)

    class_counts = np.bincount(y_train, minlength=NUM_CLASSES)
    class_weights = class_counts.sum() / (len(class_counts) * np.maximum(class_counts, 1))
    class_weights = torch.tensor(class_weights, dtype=torch.float32, device=DEVICE)
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    # stage 1
    model = build_weather_fusion_classifier_from_ssl(
        ssl_ckpt_path=ssl_ckpt_path,
        num_classes=NUM_CLASSES,
        history=HISTORY,
        num_weather_features=3,
        device=DEVICE
    ).to(DEVICE)

    freeze_encoder(model)

    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=1e-3,
        weight_decay=1e-4
    )

    linear_probe_ckpt = os.path.join(checkpoints_dir, "best_weatherhist_linear_probe.pt")

    print("\n===== WEATHER-FUSION STAGE 1: LINEAR PROBE =====")
    model, history_probe = train_classifier_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        criterion=criterion,
        device=DEVICE,
        num_epochs=LINEAR_PROBE_EPOCHS,
        save_path=linear_probe_ckpt,
        weather_mode=True
    )

    plot_classifier_history(history_probe, plots_dir, prefix="weatherhist_linear_probe")
    pd.DataFrame(history_probe).to_csv(os.path.join(reports_dir, "weatherhist_linear_probe_history.csv"), index=False)

    test_loss, test_acc, test_f1, y_true, y_pred = evaluate_classifier(
        model, test_loader, criterion, DEVICE, weather_mode=True
    )
    save_test_outputs(test_loss, test_acc, test_f1, y_true, y_pred, reports_dir, "weatherhist_linear_probe")

    # stage 2
    model = build_weather_fusion_classifier_from_ssl(
        ssl_ckpt_path=ssl_ckpt_path,
        num_classes=NUM_CLASSES,
        history=HISTORY,
        num_weather_features=3,
        device=DEVICE
    ).to(DEVICE)

    model.load_state_dict(torch.load(linear_probe_ckpt, map_location=DEVICE))
    unfreeze_encoder(model)

    optimizer = torch.optim.AdamW(
        [
            {"params": model.encoder.parameters(), "lr": 1e-4},
            {"params": model.weather_mlp.parameters(), "lr": 1e-3},
            {"params": model.classifier.parameters(), "lr": 1e-3},
        ],
        weight_decay=1e-4
    )

    finetune_ckpt = os.path.join(checkpoints_dir, "best_weatherhist_finetuned.pt")

    print("\n===== WEATHER-FUSION STAGE 2: FINE-TUNING =====")
    model, history_ft = train_classifier_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        criterion=criterion,
        device=DEVICE,
        num_epochs=FINETUNE_EPOCHS,
        save_path=finetune_ckpt,
        weather_mode=True
    )

    plot_classifier_history(history_ft, plots_dir, prefix="weatherhist_finetuned")
    pd.DataFrame(history_ft).to_csv(os.path.join(reports_dir, "weatherhist_finetuned_history.csv"), index=False)

    model.load_state_dict(torch.load(finetune_ckpt, map_location=DEVICE))
    test_loss, test_acc, test_f1, y_true, y_pred = evaluate_classifier(
        model, test_loader, criterion, DEVICE, weather_mode=True
    )
    save_test_outputs(test_loss, test_acc, test_f1, y_true, y_pred, reports_dir, "weatherhist_finetuned")



def main():
    # use same cleaned dataframe for both branches
    df = load_prepared_final_df(CSV_PATH, DEMAND_SUMMARY_DIR)
    df.to_csv(os.path.join(WEATHER_SUMMARY_DIR, "prepared_final_df_checked.csv"), index=False)

    # -----------------------------
    # 1) DEMAND-ONLY BRANCH
    # -----------------------------
    print("\n############################################################")
    print("# 1) DEMAND-ONLY DATA PREP")
    print("############################################################")
    save_demand_only_arrays(df, DEMAND_DATA_DIR, DEMAND_SUMMARY_DIR)

    print("\n############################################################")
    print("# 2) DEMAND-ONLY SSL PRETRAINING")
    print("############################################################")
    demand_ssl_ckpt = run_ssl_pretraining(
        data_dir=DEMAND_DATA_DIR,
        checkpoints_dir=DEMAND_CHECKPOINTS_DIR,
        plots_dir=DEMAND_PLOTS_DIR,
        reports_dir=DEMAND_REPORTS_DIR,
        exp_name="demand_only"
    )

    print("\n############################################################")
    print("# 3) DEMAND-ONLY CLASSIFICATION")
    print("############################################################")
    run_demand_only_pipeline(
        data_dir=DEMAND_DATA_DIR,
        checkpoints_dir=DEMAND_CHECKPOINTS_DIR,
        plots_dir=DEMAND_PLOTS_DIR,
        reports_dir=DEMAND_REPORTS_DIR,
        ssl_ckpt_path=demand_ssl_ckpt
    )

    # -----------------------------
    # 2) WEATHER-FUSION BRANCH
    # -----------------------------
    print("\n############################################################")
    print("# 4) WEATHER-FUSION DATA PREP")
    print("############################################################")
    save_weather_fusion_arrays(df, WEATHER_DATA_DIR, WEATHER_SUMMARY_DIR)

    print("\n############################################################")
    print("# 5) WEATHER-FUSION CLASSIFICATION")
    print("############################################################")
    # uses the same demand-only SSL encoder checkpoint, now fused with weather
    run_weather_fusion_pipeline(
        data_dir=WEATHER_DATA_DIR,
        checkpoints_dir=WEATHER_CHECKPOINTS_DIR,
        plots_dir=WEATHER_PLOTS_DIR,
        reports_dir=WEATHER_REPORTS_DIR,
        ssl_ckpt_path=demand_ssl_ckpt
    )

    # save run config
    config = {
        "CSV_PATH": CSV_PATH,
        "SEED": SEED,
        "HISTORY": HISTORY,
        "TRAIN_RATIO": TRAIN_RATIO,
        "VAL_RATIO": VAL_RATIO,
        "NUM_CLASSES": NUM_CLASSES,
        "BATCH_SIZE": BATCH_SIZE,
        "SSL_EPOCHS": SSL_EPOCHS,
        "LINEAR_PROBE_EPOCHS": LINEAR_PROBE_EPOCHS,
        "FINETUNE_EPOCHS": FINETUNE_EPOCHS,
        "DEVICE": str(DEVICE),
        "WEATHER_COLS": WEATHER_COLS,
        "DEMAND_SSL_CKPT_USED_FOR_WEATHER_FUSION": demand_ssl_ckpt
    }

    with open(os.path.join(DEMAND_REPORTS_DIR, "run_config.json"), "w") as f:
        json.dump(config, f, indent=2)

    with open(os.path.join(WEATHER_REPORTS_DIR, "run_config.json"), "w") as f:
        json.dump(config, f, indent=2)

    print("\nDone.")


if __name__ == "__main__":
    main()
