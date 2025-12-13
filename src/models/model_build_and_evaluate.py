import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_curve, roc_auc_score
)
from sklearn.preprocessing import StandardScaler

from imblearn.over_sampling import SMOTE
import pickle


# ============================================================
# PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"

RAW_PATH = DATA_DIR / "Bicycle_Thefts_Open_Data.csv"
CLEANED_PATH = DATA_DIR / "clean_bike_data.csv"
MODEL_PATH = BASE_DIR / "models/final_model.pkl"
SCALER_PATH = BASE_DIR / "models/scaler.pkl"
COLUMNS_PATH = BASE_DIR / "models/model_columns.pkl"
ROC_PATH = BASE_DIR / "models/roc_curve.png"


# ============================================================
# 1. LOAD RAW DATA
# ============================================================

def load_raw_data(path: Path):
    df = pd.read_csv(path)
    print("Raw dataset loaded:", df.shape)
    return df


# ============================================================
# 2. CLEANING WITH STRONG FEATURES
# ============================================================

def clean_bike_data(df: pd.DataFrame):

    # ------------------------------------------------------------
    # 1. Keep useful columns only
    # ------------------------------------------------------------
    keep_cols = [
        "OCC_DATE", "REPORT_DATE",
        "OCC_YEAR", "OCC_MONTH", "OCC_DAY", "OCC_HOUR",
        "OCC_DOW", "OCC_DOY",
        "REPORT_YEAR", "REPORT_MONTH", "REPORT_DAY", "REPORT_HOUR",
        "REPORT_DOW",
        "LOCATION_TYPE", "PREMISES_TYPE",
        "DIVISION", "HOOD_158",
        "BIKE_TYPE", "BIKE_SPEED", "BIKE_MAKE", "BIKE_MODEL",
        "BIKE_COLOUR", "BIKE_COST",
        "STATUS"
    ]

    df = df[keep_cols].copy()

    # ------------------------------------------------------------
    # 2. Convert dates
    # ------------------------------------------------------------
    df["OCC_DATE"] = pd.to_datetime(df["OCC_DATE"], errors="coerce")
    df["REPORT_DATE"] = pd.to_datetime(df["REPORT_DATE"], errors="coerce")

    # ------------------------------------------------------------
    # 3. Compute REPORT_DELAY (strongest predictor)
    # ------------------------------------------------------------
    df["REPORT_DELAY"] = (
        df["REPORT_DATE"] - df["OCC_DATE"]
    ).dt.total_seconds() / 3600

    df["REPORT_DELAY"] = df["REPORT_DELAY"].clip(lower=0, upper=72).fillna(0)

    # ------------------------------------------------------------
    # 4. Map month names if needed
    # ------------------------------------------------------------
    month_map = {
        "January": 1, "February": 2, "March": 3, "April": 4,
        "May": 5, "June": 6, "July": 7, "August": 8,
        "September": 9, "October": 10, "November": 11, "December": 12
    }

    df["OCC_MONTH"] = pd.to_numeric(df["OCC_MONTH"].map(month_map).fillna(df["OCC_MONTH"]), errors="coerce")
    df["REPORT_MONTH"] = pd.to_numeric(df["REPORT_MONTH"].map(month_map).fillna(df["REPORT_MONTH"]), errors="coerce")

    # ------------------------------------------------------------
    # 5. Convert day of week names → integers
    # ------------------------------------------------------------
    dow_map = {
        "Monday": 0, "Tuesday": 1, "Wednesday": 2,
        "Thursday": 3, "Friday": 4, "Saturday": 5, "Sunday": 6
    }

    df["OCC_DOW"] = df["OCC_DOW"].map(dow_map).fillna(0)
    df["REPORT_DOW"] = df["REPORT_DOW"].map(dow_map).fillna(0)

    # ------------------------------------------------------------
    # 6. Cyclical encoding
    # ------------------------------------------------------------
    df["MONTH_SIN"] = np.sin(2 * np.pi * df["OCC_MONTH"] / 12)
    df["MONTH_COS"] = np.cos(2 * np.pi * df["OCC_MONTH"] / 12)

    # ------------------------------------------------------------
    # 7. Target variable
    # ------------------------------------------------------------
    df["returned"] = df["STATUS"].apply(
        lambda x: 1 if str(x).strip().upper() == "RECOVERED" else 0
    )
    df.drop(columns=["STATUS"], inplace=True)

    # ------------------------------------------------------------
    # 8. Numeric cleaning
    # ------------------------------------------------------------
    numeric_cols = [
        "OCC_YEAR", "OCC_MONTH", "OCC_DAY", "OCC_HOUR",
        "OCC_DOW", "OCC_DOY",
        "REPORT_YEAR", "REPORT_MONTH", "REPORT_DAY", "REPORT_HOUR",
        "REPORT_DOW",
        "BIKE_SPEED", "BIKE_COST",
        "REPORT_DELAY",
        "MONTH_SIN", "MONTH_COS"
    ]

    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(df[col].median())

    # ------------------------------------------------------------
    # 9. Clean categories
    # ------------------------------------------------------------
    cat_cols = [
        "LOCATION_TYPE", "PREMISES_TYPE",
        "DIVISION", "HOOD_158",
        "BIKE_TYPE", "BIKE_COLOUR", "BIKE_MAKE", "BIKE_MODEL"
    ]

    for col in cat_cols:
        df[col] = df[col].astype(str).replace(["nan", ""], "Unknown")

    # Reduce cardinality
    df["BIKE_MAKE"] = df["BIKE_MAKE"].where(
        df["BIKE_MAKE"].isin(df["BIKE_MAKE"].value_counts().nlargest(20).index),
        "Other"
    )
    df["BIKE_MODEL"] = df["BIKE_MODEL"].where(
        df["BIKE_MODEL"].isin(df["BIKE_MODEL"].value_counts().nlargest(20).index),
        "Other"
    )

    # ------------------------------------------------------------
    # 10. One-hot encoding
    # ------------------------------------------------------------
    df = pd.get_dummies(df, columns=cat_cols, drop_first=True)

    # ------------------------------------------------------------
    # 11. REMOVE ALL DATETIME COLUMNS
    # ------------------------------------------------------------
    datetime_cols = [col for col in df.columns if np.issubdtype(df[col].dtype, np.datetime64)]
    df.drop(columns=datetime_cols, inplace=True)

    # ------------------------------------------------------------
    # 12. Final cleanup
    # ------------------------------------------------------------
    df = df.fillna(0)
    df = df.sample(frac=1, random_state=42)

    print("Cleaned dataset shape:", df.shape)
    print("Remaining NaNs:", df.isna().sum().sum())

    return df




# ============================================================
# 3. SAVE CLEANED DATASET
# ============================================================

def save_cleaned_dataset(df: pd.DataFrame):
    DATA_DIR.mkdir(exist_ok=True)
    df.to_csv(CLEANED_PATH, index=False)
    print("Saved cleaned dataset →", CLEANED_PATH)


# ============================================================
# 4. SPLIT DATA
# ============================================================

def split_data(df):
    X = df.drop("returned", axis=1)
    y = df["returned"]
    return train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )


# ============================================================
# 5. TRAIN USING SMOTE + SCALING
# ============================================================

def train_balanced_model(X_train, y_train):

    print("\n=== APPLYING SMOTE (SAFE MODE) ===")
    print("Before SMOTE:\n", y_train.value_counts())

    sm = SMOTE(sampling_strategy=0.3, random_state=42)
    X_res, y_res = sm.fit_resample(X_train, y_train)

    print("After SMOTE:\n", y_res.value_counts())

    scaler = StandardScaler()
    X_res_scaled = scaler.fit_transform(X_res)

    model = LogisticRegression(
        max_iter=2000,
        class_weight="balanced",
        solver="lbfgs"
    )
    model.fit(X_res_scaled, y_res)

    return model, scaler


# ============================================================
# 6. EVALUATE MODEL
# ============================================================

def evaluate_model(name, model, scaler, X_test, y_test):

    X_scaled = scaler.transform(X_test)
    preds = model.predict(X_scaled)

    print("\n====================================")
    print(f" Evaluation: {name}")
    print("====================================")

    print("Accuracy:", accuracy_score(y_test, preds))
    print("Precision:", precision_score(y_test, preds, zero_division=0))
    print("Recall:", recall_score(y_test, preds, zero_division=0))
    print("F1 Score:", f1_score(y_test, preds, zero_division=0))

    print("\nConfusion Matrix:\n", confusion_matrix(y_test, preds))
    print("\nClassification Report:\n", classification_report(y_test, preds, zero_division=0))

    probs = model.predict_proba(X_scaled)[:, 1]
    auc = roc_auc_score(y_test, probs)
    fpr, tpr, _ = roc_curve(y_test, probs)

    print("AUC:", auc)

    return {"name": name, "auc": auc, "fpr": fpr, "tpr": tpr}


# ============================================================
# MAIN PIPELINE
# ============================================================

def main():

    print("\n=== LOADING RAW DATA ===")
    df_raw = load_raw_data(RAW_PATH)

    print("\n=== CLEANING DATA ===")
    df_clean = clean_bike_data(df_raw)
    save_cleaned_dataset(df_clean)

    with open(COLUMNS_PATH, "wb") as f:
        pickle.dump(df_clean.drop(columns=["returned"]).columns.tolist(), f)

    print("\n=== SPLITTING DATA ===")
    X_train, X_test, y_train, y_test = split_data(df_clean)

    print("\n=== TRAINING MODEL ===")
    model, scaler = train_balanced_model(X_train, y_train)

    print("\n=== SAVING MODEL & SCALER ===")
    with open(MODEL_PATH, "wb") as f:
        pickle.dump(model, f)
    with open(SCALER_PATH, "wb") as f:
        pickle.dump(scaler, f)

    print("Saved model →", MODEL_PATH)
    print("Saved scaler →", SCALER_PATH)

    print("\n=== EVALUATING MODEL ===")
    results = evaluate_model("Logistic Regression", model, scaler, X_test, y_test)

    plt.figure(figsize=(7, 5))
    plt.plot(results["fpr"], results["tpr"], label=f"AUC={results['auc']:.3f}")
    plt.plot([0, 1], [0, 1], "k--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.legend()
    plt.grid(True)
    plt.savefig(ROC_PATH)
    print("Saved ROC curve →", ROC_PATH)

    print("\n=== DONE ===")


if __name__ == "__main__":
    main()
