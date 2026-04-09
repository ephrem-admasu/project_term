import os
import gc

import numpy as np
import pandas as pd

from sklearn.preprocessing import MinMaxScaler
from pyts.image import GramianAngularField

from config import HISTORY, TRAIN_RATIO, VAL_RATIO, WEATHER_COLS
from utilities import split_zone_chronologically

# Load and prepare find_df 
def load_prepared_final_df(csv_path, summary_dir):
    df = pd.read_csv(csv_path)
    print("Raw shape:", df.shape)

    required_cols = [
        "zone", "time", "demand", "observed",
        "temperature_2m", "precipitation", "wind_speed_10m"
    ]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns in final_df.csv: {missing}")

    df = df[required_cols].copy()

    df["time"] = pd.to_datetime(df["time"], errors="coerce")
    df["zone"] = df["zone"].astype(str)

    for c in ["demand", "observed"] + WEATHER_COLS:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    df = df.dropna(subset=["zone", "time"]).copy()
    df["observed"] = df["observed"].fillna(0)
    df["observed"] = (df["observed"] > 0).astype(int)
    df["demand"] = df["demand"].fillna(0)

    df = df.sort_values(["zone", "time"]).reset_index(drop=True)

    # weather interpolation within zone
    def fill_weather(g):
        g = g.sort_values("time").copy()
        for c in WEATHER_COLS:
            g[c] = g[c].interpolate(method="linear", limit_direction="both")
            g[c] = g[c].ffill().bfill()
        return g

    df = df.groupby("zone", group_keys=False).apply(fill_weather).reset_index(drop=True)
    df = df.dropna(subset=WEATHER_COLS).reset_index(drop=True)

    # demand classes
    non_zero = df.loc[df["demand"] > 0, "demand"]
    if len(non_zero) == 0:
        raise ValueError("All demand values are zero. Cannot build classes.")

    q1 = non_zero.quantile(0.33)
    q2 = non_zero.quantile(0.66)

    def classify_demand(x):
        if x == 0:
            return 0
        elif x <= q1:
            return 1
        elif x <= q2:
            return 2
        else:
            return 3

    df["demand_class"] = df["demand"].apply(classify_demand).astype(np.int64)

    with open(os.path.join(summary_dir, "demand_quantiles.txt"), "w") as f:
        f.write(f"q1={q1}\nq2={q2}\n")

    df.to_csv(os.path.join(summary_dir, "prepared_final_df_checked.csv"), index=False)

    print("\nDemand class distribution:")
    print(df["demand_class"].value_counts(normalize=True).sort_index())

    return df

# Window builder
def make_demand_windows(series_values, class_values=None, history=24):
    X_seq, y = [], []

    for t in range(history, len(series_values)):
        X_seq.append(series_values[t-history:t])
        if class_values is not None:
            y.append(class_values[t])

    if len(X_seq) == 0:
        X_empty = np.empty((0, history), dtype=np.float32)
        y_empty = np.empty((0,), dtype=np.int64) if class_values is not None else None
        return X_empty, y_empty

    X_seq = np.asarray(X_seq, dtype=np.float32)

    if class_values is not None:
        y = np.asarray(y, dtype=np.int64)
        return X_seq, y

    return X_seq, None

def make_windows_with_weather_history(
    demand_values,
    weather_values,
    class_values,
    time_values,
    zone_name,
    history=24
):
    X_seq, W_seq, y, meta_rows = [], [], [], []

    for t in range(history, len(demand_values)):
        X_seq.append(demand_values[t-history:t])
        W_seq.append(weather_values[t-history:t])
        y.append(class_values[t])

        meta_rows.append({
            "zone": zone_name,
            "target_time": pd.Timestamp(time_values[t]).isoformat()
        })

    if len(X_seq) == 0:
        return (
            np.empty((0, history), dtype=np.float32),
            np.empty((0, history, weather_values.shape[1]), dtype=np.float32),
            np.empty((0,), dtype=np.int64),
            pd.DataFrame(columns=["zone", "target_time"])
        )

    return (
        np.asarray(X_seq, dtype=np.float32),
        np.asarray(W_seq, dtype=np.float32),
        np.asarray(y, dtype=np.int64),
        pd.DataFrame(meta_rows)
    )

# Save processed arrays
def save_demand_only_arrays(df, save_dir, summary_dir):
    gaf = GramianAngularField(method="summation")
    summary_rows = []

    for zone, zone_df in df.groupby("zone"):
        zone_df = zone_df.sort_values("time").reset_index(drop=True)

        if len(zone_df) < HISTORY + 5:
            continue

        train_df, val_df, test_df = split_zone_chronologically(zone_df, TRAIN_RATIO, VAL_RATIO)

        if len(train_df) <= HISTORY or len(val_df) <= HISTORY or len(test_df) <= HISTORY:
            continue

        scaler = MinMaxScaler(feature_range=(-1, 1))
        train_scaled = scaler.fit_transform(train_df[["demand"]]).astype(np.float32).ravel()
        val_scaled = scaler.transform(val_df[["demand"]]).astype(np.float32).ravel()
        test_scaled = scaler.transform(test_df[["demand"]]).astype(np.float32).ravel()

        train_classes = train_df["demand_class"].to_numpy(dtype=np.int64)
        val_classes = val_df["demand_class"].to_numpy(dtype=np.int64)
        test_classes = test_df["demand_class"].to_numpy(dtype=np.int64)

        X_train_seq, y_train = make_demand_windows(train_scaled, train_classes, history=HISTORY)
        X_val_seq, y_val = make_demand_windows(val_scaled, val_classes, history=HISTORY)
        X_test_seq, y_test = make_demand_windows(test_scaled, test_classes, history=HISTORY)

        if len(X_train_seq) == 0 or len(X_val_seq) == 0 or len(X_test_seq) == 0:
            continue

        X_train_gaf = gaf.transform(X_train_seq).astype(np.float32)[:, None, :, :]
        X_val_gaf = gaf.transform(X_val_seq).astype(np.float32)[:, None, :, :]
        X_test_gaf = gaf.transform(X_test_seq).astype(np.float32)[:, None, :, :]

        safe_zone = str(zone).replace("/", "_").replace(" ", "_")

        np.save(os.path.join(save_dir, f"{safe_zone}_X_train.npy"), X_train_gaf)
        np.save(os.path.join(save_dir, f"{safe_zone}_X_val.npy"), X_val_gaf)
        np.save(os.path.join(save_dir, f"{safe_zone}_X_test.npy"), X_test_gaf)

        np.save(os.path.join(save_dir, f"{safe_zone}_y_train.npy"), y_train)
        np.save(os.path.join(save_dir, f"{safe_zone}_y_val.npy"), y_val)
        np.save(os.path.join(save_dir, f"{safe_zone}_y_test.npy"), y_test)

        summary_rows.append({
            "zone": zone,
            "train_samples": len(y_train),
            "val_samples": len(y_val),
            "test_samples": len(y_test),
            "image_shape": str(X_train_gaf.shape[1:])
        })

        del train_scaled, val_scaled, test_scaled
        del X_train_seq, X_val_seq, X_test_seq
        del X_train_gaf, X_val_gaf, X_test_gaf
        del y_train, y_val, y_test
        gc.collect()

    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(os.path.join(summary_dir, "zone_split_summary.csv"), index=False)

    with open(os.path.join(summary_dir, "zone_split_totals.txt"), "w") as f:
        f.write("Total sample counts across zones\n")
        f.write(str(summary_df[["train_samples", "val_samples", "test_samples"]].sum()))

    print("\nDemand-only summary:")
    print(summary_df.head())
    if len(summary_df) > 0:
        print(summary_df[["train_samples", "val_samples", "test_samples"]].sum())


def save_weather_fusion_arrays(df, save_dir, summary_dir):
    gaf = GramianAngularField(method="summation")
    summary_rows = []

    for zone, zone_df in df.groupby("zone"):
        zone_df = zone_df.sort_values("time").reset_index(drop=True)

        if len(zone_df) < HISTORY + 5:
            continue

        train_df, val_df, test_df = split_zone_chronologically(zone_df, TRAIN_RATIO, VAL_RATIO)

        if len(train_df) <= HISTORY or len(val_df) <= HISTORY or len(test_df) <= HISTORY:
            continue

        demand_scaler = MinMaxScaler(feature_range=(-1, 1))
        train_demand = demand_scaler.fit_transform(train_df[["demand"]]).astype(np.float32).ravel()
        val_demand = demand_scaler.transform(val_df[["demand"]]).astype(np.float32).ravel()
        test_demand = demand_scaler.transform(test_df[["demand"]]).astype(np.float32).ravel()

        weather_scaler = MinMaxScaler()
        train_weather = weather_scaler.fit_transform(train_df[WEATHER_COLS]).astype(np.float32)
        val_weather = weather_scaler.transform(val_df[WEATHER_COLS]).astype(np.float32)
        test_weather = weather_scaler.transform(test_df[WEATHER_COLS]).astype(np.float32)

        train_classes = train_df["demand_class"].to_numpy(dtype=np.int64)
        val_classes = val_df["demand_class"].to_numpy(dtype=np.int64)
        test_classes = test_df["demand_class"].to_numpy(dtype=np.int64)

        X_train_seq, W_train, y_train, meta_train = make_windows_with_weather_history(
            train_demand, train_weather, train_classes, train_df["time"].to_numpy(), zone, history=HISTORY
        )
        X_val_seq, W_val, y_val, meta_val = make_windows_with_weather_history(
            val_demand, val_weather, val_classes, val_df["time"].to_numpy(), zone, history=HISTORY
        )
        X_test_seq, W_test, y_test, meta_test = make_windows_with_weather_history(
            test_demand, test_weather, test_classes, test_df["time"].to_numpy(), zone, history=HISTORY
        )

        if len(X_train_seq) == 0 or len(X_val_seq) == 0 or len(X_test_seq) == 0:
            continue

        X_train_gaf = gaf.transform(X_train_seq).astype(np.float32)[:, None, :, :]
        X_val_gaf = gaf.transform(X_val_seq).astype(np.float32)[:, None, :, :]
        X_test_gaf = gaf.transform(X_test_seq).astype(np.float32)[:, None, :, :]

        safe_zone = str(zone).replace("/", "_").replace(" ", "_")

        np.save(os.path.join(save_dir, f"{safe_zone}_X_train.npy"), X_train_gaf)
        np.save(os.path.join(save_dir, f"{safe_zone}_X_val.npy"), X_val_gaf)
        np.save(os.path.join(save_dir, f"{safe_zone}_X_test.npy"), X_test_gaf)

        np.save(os.path.join(save_dir, f"{safe_zone}_W_train.npy"), W_train)
        np.save(os.path.join(save_dir, f"{safe_zone}_W_val.npy"), W_val)
        np.save(os.path.join(save_dir, f"{safe_zone}_W_test.npy"), W_test)

        np.save(os.path.join(save_dir, f"{safe_zone}_y_train.npy"), y_train)
        np.save(os.path.join(save_dir, f"{safe_zone}_y_val.npy"), y_val)
        np.save(os.path.join(save_dir, f"{safe_zone}_y_test.npy"), y_test)

        meta_train.to_csv(os.path.join(save_dir, f"{safe_zone}_meta_train.csv"), index=False)
        meta_val.to_csv(os.path.join(save_dir, f"{safe_zone}_meta_val.csv"), index=False)
        meta_test.to_csv(os.path.join(save_dir, f"{safe_zone}_meta_test.csv"), index=False)

        summary_rows.append({
            "zone": zone,
            "train_samples": len(y_train),
            "val_samples": len(y_val),
            "test_samples": len(y_test),
            "image_shape": str(X_train_gaf.shape[1:]),
            "weather_shape": str(W_train.shape[1:])
        })

        del train_demand, val_demand, test_demand
        del train_weather, val_weather, test_weather
        del X_train_seq, X_val_seq, X_test_seq
        del X_train_gaf, X_val_gaf, X_test_gaf
        del W_train, W_val, W_test
        del y_train, y_val, y_test
        gc.collect()

    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(os.path.join(summary_dir, "aligned_zone_summary.csv"), index=False)

    with open(os.path.join(summary_dir, "aligned_totals.txt"), "w") as f:
        f.write("Aligned X/W/y totals\n")
        f.write(str(summary_df[["train_samples", "val_samples", "test_samples"]].sum()))

    print("\nWeather-fusion summary:")
    print(summary_df.head())
    if len(summary_df) > 0:
        print(summary_df[["train_samples", "val_samples", "test_samples"]].sum())