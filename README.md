# project_term

Self-Supervised Masked Autoencoder for Curb Demand Prediction with Gramian Angual Fields

## Overview

This project studies curb demand prediction by converting historical demand sequences into GAF images and learning representations with a masked autoencoder. The learned encoder is then used for downstream demand classification under two settings:
- demand only
- demand + weather history fusion

## Model Architecture

See the full diagram here:

[Download model diagram](proj_method.pdf)

## Main idea

The workflow is:

1. Load and clean curb demand data
2. Split each zone chronologically into train/validation/test
3. Build historical windows of length 24
4. Convert windows into GAF images
5. Pretrain a masked autoencoder with self-supervision
6. Train downstream classifiers using:
   - linear probing
   - fine-tuning
7. Compare performance across demand-only and weather-fusion settings

## Files

- `config.py` – configuration values and paths
- `dataset.py` – PyTorch dataset classes
- `models.py` – masked autoencoder and classifier models
- `preprocessing.py` – data cleaning, window generation, GAF conversion, and saved arrays
- `trainer.py` – training and evaluation functions
- `utilities.py` – helper functions for plotting, reporting, and reproducibility
- `train_curb_demand.py` – main script to run the full pipeline

## Data format

The input CSV is not shared here due to large but can be shared upon request. It is expected to contain:

- `zone`
- `time`
- `demand`
- `observed`
- `temperature_2m`
- `precipitation`
- `wind_speed_10m`

## Demand classes

Demand is converted into 4 classes:

- `0` = idle
- `1` = low
- `2` = medium
- `3` = high

## Outputs

The pipeline saves:

- processed arrays
- checkpoints
- training curves
- confusion matrices
- classification reports
- run configuration files
