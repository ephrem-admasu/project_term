import torch 


CSV_PATH = "data/final_df.csv"
# zone, time, demand, observed, temperature_2m, precipitation, wind_speed_10m

SEED = 42
HISTORY = 24
TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
NUM_CLASSES = 4

BATCH_SIZE = 128
NUM_WORKERS = 4
PIN_MEMORY = True

SSL_EPOCHS = 30
LINEAR_PROBE_EPOCHS = 20
FINETUNE_EPOCHS = 30

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

WEATHER_COLS = ["temperature_2m", "precipitation", "wind_speed_10m"]

# demand-only experiment folders
DEMAND_ROOT = "exp_demand_only"

WEATHER_ROOT = "exp_weather_history_fusion"