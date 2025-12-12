import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split


def load_raw_data(path):
    """
    Loading the raw bicycle theft dataset.
    """
    return pd.read_csv(path)


def clean_and_engineer(df):
    """
    Cleaning the dataset and engineering the features for modeling.
    """

    df = df.copy()

    # 1. Drops the columns that are IDs, coordinates, or not useful for modeling
    columns_to_drop = [
        "ObjectId",
        "Occurrence Number",
        "Hood_ID",
        "Bike Serial Number",
        "Longitude",
        "Latitude",
        "Location_Type",
        "Premises_Type",
        "Report_Date",
        "Bike_Make",
        "Bike_Model"
    ]

    df.drop(columns=[c for c in columns_to_drop if c in df.columns], inplace=True)

    # 2. Handle's missing values
    df["Status"] = df["Status"].fillna("UNKNOWN")
    df["Bike_Type"] = df["Bike_Type"].fillna("UNKNOWN")
    df["Bike_Colour"] = df["Bike_Colour"].fillna("UNKNOWN")

    if "Cost_of_Bike" in df.columns:
        df["Cost_of_Bike"] = df["Cost_of_Bike"].fillna(df["Cost_of_Bike"].median())

    # 3. Converting date column and extract time-based feature
    df["Occurrence_Date"] = pd.to_datetime(df["Occurrence_Date"], errors="coerce")

    df["Year"] = df["Occurrence_Date"].dt.year
    df["Month"] = df["Occurrence_Date"].dt.month
    df["Day"] = df["Occurrence_Date"].dt.day
    df["DayOfWeek"] = df["Occurrence_Date"].dt.dayofweek  # 0 = Monday

    # 4. Target variable: Was the bike recovered?
    df["Returned"] = df["Status"].apply(
        lambda x: 1 if "Recovered" in str(x) else 0
    )

    # 5. Drops columns that are no longer needed. 

    df.drop(columns=["Status", "Occurrence_Date"], inplace=True)

    
    # 6. Categorical variables 
    categorical_cols = df.select_dtypes(include=["object"]).columns
    df = pd.get_dummies(df, columns=categorical_cols, drop_first=True)

    return df


def create_train_test(df):
    """
    Spliting the dataset into training and testing sets.
    """
    X = df.drop("Returned, axis=1")
    y = df["Returned"]
    return train_test_split(X, y, test_size=0.2, random_state=42)


# Main execution block for data preparation
if __name__ == "__main__":
    raw_path = "Bicycle_Thefts_Open_Data.csv" 

    raw_df = load_raw_data(raw_path)
    clean_df = clean_and_engineer(raw_df)

    X_train, X_test, y_train, y_test = create_train_test(clean_df)

    print("Cleaned dataset shape:", clean_df.shape)
    print("Training set shape:", X_train.shape)
    print("Testing set shape:", X_test.shape)
