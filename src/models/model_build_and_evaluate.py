import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
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

MODELS_DIR = BASE_DIR / "models"
REPORTS_DIR = BASE_DIR / "reports" / "figures"

MODEL_PATH = MODELS_DIR / "final_model.pkl"
SCALER_PATH = MODELS_DIR / "scaler.pkl"
COLUMNS_PATH = MODELS_DIR / "model_columns.pkl"
ROC_PATH = REPORTS_DIR / "roc_curve.png"


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

def clean_bike_data(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # ------------------------------------------------------------
    # 0. Drop lat/long explicitly if present
    # ------------------------------------------------------------
    coord_cols = ["LAT_WGS84", "LONG_WGS84", "Latitude", "Longitude"]
    df.drop(columns=[c for c in coord_cols if c in df.columns], inplace=True, errors="ignore")

    # ------------------------------------------------------------
    # 1. Keep useful columns only (NOTE: PREMISES_TYPE DROPPED)
    # ------------------------------------------------------------
    keep_cols = [
        "OCC_DATE", "REPORT_DATE",
        "OCC_YEAR", "OCC_MONTH", "OCC_DAY", "OCC_HOUR",
        "OCC_DOW", "OCC_DOY",
        "REPORT_YEAR", "REPORT_MONTH", "REPORT_DAY", "REPORT_HOUR",
        "REPORT_DOW",
        "LOCATION_TYPE",
        "DIVISION", "HOOD_158",
        "BIKE_TYPE", "BIKE_SPEED", "BIKE_MAKE", "BIKE_MODEL",
        "BIKE_COLOUR", "BIKE_COST",
        "STATUS"
    ]

    # Only keep columns that actually exist
    available_cols = [c for c in keep_cols if c in df.columns]
    df = df[available_cols].copy()

    # ------------------------------------------------------------
    # 2. Convert dates
    # ------------------------------------------------------------
    if "OCC_DATE" in df.columns:
        df["OCC_DATE"] = pd.to_datetime(df["OCC_DATE"], errors="coerce")
    if "REPORT_DATE" in df.columns:
        df["REPORT_DATE"] = pd.to_datetime(df["REPORT_DATE"], errors="coerce")

    # ------------------------------------------------------------
    # 3. Compute REPORT_DELAY (strong predictor)
    # ------------------------------------------------------------
    if "OCC_DATE" in df.columns and "REPORT_DATE" in df.columns:
        df["REPORT_DELAY"] = (
            df["REPORT_DATE"] - df["OCC_DATE"]
        ).dt.total_seconds() / 3600
    else:
        df["REPORT_DELAY"] = np.nan

    # Clip to a reasonable window (0–72h) and fill missing
    df["REPORT_DELAY"] = df["REPORT_DELAY"].clip(lower=0, upper=72).fillna(0)

    # ------------------------------------------------------------
    # 4. Map month names → integers (defensive)
    # ------------------------------------------------------------
    month_map = {
        "January": 1, "February": 2, "March": 3, "April": 4,
        "May": 5, "June": 6, "July": 7, "August": 8,
        "September": 9, "October": 10, "November": 11, "December": 12
    }

    for col in ["OCC_MONTH", "REPORT_MONTH"]:
        if col in df.columns:
            df[col] = pd.to_numeric(
                df[col].map(month_map).fillna(df[col]),
                errors="coerce"
            )

    # ------------------------------------------------------------
    # 5. Convert day-of-week names → integers (defensive)
    # ------------------------------------------------------------
    dow_map = {
        "Monday": 0, "Tuesday": 1, "Wednesday": 2,
        "Thursday": 3, "Friday": 4, "Saturday": 5, "Sunday": 6
    }

    for col in ["OCC_DOW", "REPORT_DOW"]:
        if col in df.columns:
            df[col] = df[col].map(dow_map).fillna(df[col]).fillna(0)

    # ------------------------------------------------------------
    # 6. Cyclical encoding for month (seasonality)
    # ------------------------------------------------------------
    if "OCC_MONTH" in df.columns:
        df["MONTH_SIN"] = np.sin(2 * np.pi * df["OCC_MONTH"] / 12)
        df["MONTH_COS"] = np.cos(2 * np.pi * df["OCC_MONTH"] / 12)
    else:
        df["MONTH_SIN"] = 0
        df["MONTH_COS"] = 0

    # ------------------------------------------------------------
    # 7. Target variable: returned
    # ------------------------------------------------------------
    if "STATUS" not in df.columns:
        raise ValueError("Expected 'STATUS' column not found in dataset.")

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
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
            df[col] = df[col].fillna(df[col].median())

    # ------------------------------------------------------------
    # 9. Clean categories (PREMISES_TYPE is intentionally omitted)
    # ------------------------------------------------------------
    cat_cols = [
        "LOCATION_TYPE",
        "DIVISION", "HOOD_158",
        "BIKE_TYPE", "BIKE_COLOUR", "BIKE_MAKE", "BIKE_MODEL"
    ]

    for col in cat_cols:
        if col in df.columns:
            df[col] = df[col].astype(str).replace(["nan", ""], "Unknown")

    # Reduce cardinality for make/model if present
    if "BIKE_MAKE" in df.columns:
        top_makes = df["BIKE_MAKE"].value_counts().nlargest(20).index
        df["BIKE_MAKE"] = df["BIKE_MAKE"].where(df["BIKE_MAKE"].isin(top_makes), "Other")

    if "BIKE_MODEL" in df.columns:
        top_models = df["BIKE_MODEL"].value_counts().nlargest(20).index
        df["BIKE_MODEL"] = df["BIKE_MODEL"].where(df["BIKE_MODEL"].isin(top_models), "Other")

    # ------------------------------------------------------------
    # 10. One-hot encoding (on available categorical columns)
    # ------------------------------------------------------------
    usable_cat = [c for c in cat_cols if c in df.columns]
    if usable_cat:
        df = pd.get_dummies(df, columns=usable_cat, drop_first=True)

    # ------------------------------------------------------------
    # 11. REMOVE ALL DATETIME COLUMNS
    # ------------------------------------------------------------
    datetime_cols = [
        col for col in df.columns
        if np.issubdtype(df[col].dtype, np.datetime64)
    ]
    if datetime_cols:
        df.drop(columns=datetime_cols, inplace=True)

    # ------------------------------------------------------------
    # 12. Final cleanup
    # ------------------------------------------------------------
    df = df.fillna(0)
    df = df.sample(frac=1, random_state=42)  # shuffle

    print("Cleaned dataset shape:", df.shape)
    print("Remaining NaNs:", df.isna().sum().sum())

    return df


# ============================================================
# 3. SAVE CLEANED DATASET
# ============================================================

def save_cleaned_dataset(df: pd.DataFrame):
    DATA_DIR.mkdir(exist_ok=True, parents=True)
    df.to_csv(CLEANED_PATH, index=False)
    print("Saved cleaned dataset →", CLEANED_PATH)


# ============================================================
# 4. SPLIT DATA
# ============================================================

def split_data(df: pd.DataFrame):
    X = df.drop("returned", axis=1)
    y = df["returned"]
    print("\n[INFO] Class balance (overall):")
    print(y.value_counts(normalize=True).rename("proportion"))
    return train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )


# ============================================================
# 5. TRAIN USING SMOTE + SCALING (LOGREG + TREE)
# ============================================================

def train_balanced_models(X_train, y_train):
    print("\n=== APPLYING SMOTE ===")
    print("Before SMOTE:\n", y_train.value_counts())

    sm = SMOTE(sampling_strategy=0.3, random_state=42)
    X_res, y_res = sm.fit_resample(X_train, y_train)

    print("After SMOTE:\n", y_res.value_counts())

    scaler = StandardScaler()
    X_res_scaled = scaler.fit_transform(X_res)

    # Logistic Regression
    log_model = LogisticRegression(
        max_iter=2000,
        class_weight="balanced",
        solver="lbfgs"
    )
    log_model.fit(X_res_scaled, y_res)

    # Decision Tree
    tree_model = DecisionTreeClassifier(
        random_state=42,
        class_weight="balanced"
    )
    tree_model.fit(X_res_scaled, y_res)

    models = {
        "Logistic Regression": log_model,
        "Decision Tree": tree_model,
    }

    return models, scaler


# ============================================================
# 6. EVALUATE MODEL(S)
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

    # Ensure output directories exist
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    print("\n=== LOADING RAW DATA ===")
    df_raw = load_raw_data(RAW_PATH)

    print("\n=== CLEANING DATA ===")
    df_clean = clean_bike_data(df_raw)
    save_cleaned_dataset(df_clean)

    # Save model columns (feature names only, no target)
    feature_cols = df_clean.drop(columns=["returned"]).columns.tolist()
    with open(COLUMNS_PATH, "wb") as f:
        pickle.dump(feature_cols, f)
    print("Saved model columns →", COLUMNS_PATH)

    print("\n=== SPLITTING DATA ===")
    X_train, X_test, y_train, y_test = split_data(df_clean)

    print("\n=== TRAINING MODELS ===")
    models, scaler = train_balanced_models(X_train, y_train)

    print("\n=== EVALUATING MODELS ===")
    results = []
    for name, model in models.items():
        res = evaluate_model(name, model, scaler, X_test, y_test)
        results.append(res)

    # Choose best model by AUC
    best = max(results, key=lambda r: r["auc"])
    best_name = best["name"]
    best_model = models[best_name]

    print("\n=== BEST MODEL SELECTED ===")
    print(f"{best_name} with AUC={best['auc']:.3f}")

    print("\n=== SAVING BEST MODEL & SCALER ===")
    with open(MODEL_PATH, "wb") as f:
        pickle.dump(best_model, f)
    with open(SCALER_PATH, "wb") as f:
        pickle.dump(scaler, f)

    print("Saved best model →", MODEL_PATH)
    print("Saved scaler →", SCALER_PATH)

    # Plot combined ROC
    plt.figure(figsize=(7, 5))
    for r in results:
        plt.plot(r["fpr"], r["tpr"], label=f"{r['name']} (AUC={r['auc']:.3f})")
    plt.plot([0, 1], [0, 1], "k--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve Comparison")
    plt.legend()
    plt.grid(True)
    plt.savefig(ROC_PATH)
    print("Saved ROC curve →", ROC_PATH)

    print("\n=== DONE ===")


if __name__ == "__main__":
    main()
