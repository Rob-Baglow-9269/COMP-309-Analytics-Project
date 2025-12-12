import pickle
import pandas as pd
from pathlib import Path

# ===============================
# Load model + column order
# ===============================

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "models"

COLS_PATH = MODELS_DIR / "model_columns.pkl"

with open(COLS_PATH, "rb") as f:
    model_columns = pickle.load(f)

# ===============================
# Get options for dropdowns
# ===============================
data = pd.read_csv(DATA_DIR / "Bicycle_Thefts_Open_Data.csv")

class Hood:
    def __init__(self, key, value):
        self.id = key
        self.name = value

def get_options(column_name):
    return data[column_name].dropna().unique().tolist()

def get_sorted_options(column_name):
    options = get_options(column_name)
    options.sort()
    return options

def get_hoods_mapping():
    hood_data = data[["HOOD_158", "NEIGHBOURHOOD_158"]].dropna().drop_duplicates()
    hood_mapping = list(map(Hood, hood_data["HOOD_158"], hood_data["NEIGHBOURHOOD_158"]))
    return sorted(hood_mapping, key=lambda item: item.name)

premises_types = get_sorted_options("PREMISES_TYPE")
bike_types = get_sorted_options("BIKE_TYPE")
hoods = get_hoods_mapping()