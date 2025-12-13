import pandas as pd
import numpy as np
import os


def load_raw_data(path):
    """
    Load the raw bicycle theft dataset.
    """
    return pd.read_csv(path)


def clean_and_engineer(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean the dataset and engineer features for modeling.

    Assumes (or tolerates) columns like:
      - Status
      - Occurrence_Date
      - Report_Date
      - Cost_of_Bike
      - Bike_Type, Bike_Colour, Bike_Make, Bike_Model
      - Longitude, Latitude
      - Location_Type, Premises_Type
      - ObjectId, Occurrence Number, Hood_ID, Bike Serial Number
    """

    df = df.copy()

    # 1. Drop ID-like & noisy columns
    columns_to_drop = [
        "ObjectId",
        "Occurrence Number",
        "Bike Serial Number",
        "Longitude",
        "Latitude",
        "Premises_Type",   # we decided to drop this
        "Bike_Model"       # too messy, usually free text
        # NOTE: we intentionally KEEP Location_Type and Hood_ID
    ]
    df.drop(columns=[c for c in columns_to_drop if c in df.columns], inplace=True, errors="ignore")

    # 2. Handle missing values in key columns
    if "Status" in df.columns:
        df["Status"] = df["Status"].fillna("UNKNOWN")
    if "Bike_Type" in df.columns:
        df["Bike_Type"] = df["Bike_Type"].fillna("UNKNOWN")
    if "Bike_Colour" in df.columns:
        df["Bike_Colour"] = df["Bike_Colour"].fillna("UNKNOWN")
    if "Bike_Make" in df.columns:
        df["Bike_Make"] = df["Bike_Make"].fillna("UNKNOWN")

    if "Cost_of_Bike" in df.columns:
        df["Cost_of_Bike"] = df["Cost_of_Bike"].fillna(df["Cost_of_Bike"].median())

    # 3. Convert date columns and extract time-based features
    if "Occurrence_Date" in df.columns:
        df["Occurrence_Date"] = pd.to_datetime(df["Occurrence_Date"], errors="coerce")
        df["Occ_Year"] = df["Occurrence_Date"].dt.year
        df["Occ_Month"] = df["Occurrence_Date"].dt.month
        df["Occ_Day"] = df["Occurrence_Date"].dt.day
        df["Occ_DayOfWeek"] = df["Occurrence_Date"].dt.dayofweek  # 0 = Monday
    else:
        df["Occurrence_Date"] = pd.NaT

    if "Report_Date" in df.columns:
        df["Report_Date"] = pd.to_datetime(df["Report_Date"], errors="coerce")
        # REPORT_DELAY in hours
        df["ReportDelay_Hours"] = (
            df["Report_Date"] - df["Occurrence_Date"]
        ).dt.total_seconds() / 3600.0
    else:
        df["Report_Date"] = pd.NaT
        df["ReportDelay_Hours"] = np.nan

    # Fill any missing delay with median (for safety)
    df["ReportDelay_Hours"] = df["ReportDelay_Hours"].fillna(df["ReportDelay_Hours"].median())

    # 4. Target variable: returned (1 if recovered, else 0)
    if "Status" not in df.columns:
        raise ValueError("Expected 'Status' column not found in dataset.")

    df["returned"] = (
        df["Status"].astype(str).str.upper().str.contains("RECOVERED").astype(int)
    )

    # 5. Drop raw status and date columns now that we've derived features
    df.drop(columns=[c for c in ["Status", "Occurrence_Date", "Report_Date"] if c in df.columns],
            inplace=True, errors="ignore")

    # 6. One-hot encode remaining categorical variables
    categorical_cols = df.select_dtypes(include=["object"]).columns.tolist()
    if categorical_cols:
        df = pd.get_dummies(df, columns=categorical_cols, drop_first=True)

    # Ensure target is int
    df["returned"] = df["returned"].astype(int)

    return df


def save_clean_dataset(df: pd.DataFrame, path: str):
    """
    Save the cleaned dataset to CSV.
    """
    out_dir = os.path.dirname(path)
    if out_dir and not os.path.exists(out_dir):
        os.makedirs(out_dir, exist_ok=True)

    df.to_csv(path, index=False)
    print(f"Clean dataset saved to: {path}")
    print("Final shape:", df.shape)


if __name__ == "__main__":
    raw_path = "Bicycle_Thefts_Open_Data.csv"  # adjust path if needed
    out_path = os.path.join("data", "clean_bike_data.csv")

    raw_df = load_raw_data(raw_path)
    clean_df = clean_and_engineer(raw_df)
    save_clean_dataset(clean_df, out_path)
