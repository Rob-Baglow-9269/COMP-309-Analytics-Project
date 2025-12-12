import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
    roc_curve,
    roc_auc_score,
)

import pickle


# ============================================================
# PATH SETUP
# This script is inside: src/models/
# Data directory is:     src/data/
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent   # goes from /src/models → /src
DATA_DIR = BASE_DIR / "data"

RAW_PATH = DATA_DIR / "Bicycle_Thefts_Open_Data.csv"
CLEANED_PATH = DATA_DIR / "clean_bike_data.csv"
MODEL_PATH =  BASE_DIR / "models/final_model.pkl"
ROC_PATH = BASE_DIR / "models/roc_curve.png"


# ============================================================
# 1. LOAD RAW DATA
# ============================================================

def load_raw_data(path: Path):
    df = pd.read_csv(path)
    print("Raw dataset loaded:", df.shape)
    return df


# ============================================================
# 2. CLEANING FUNCTION
# ============================================================

def clean_bike_data(df: pd.DataFrame):

    keep_cols = [
        "OCC_YEAR",
        "OCC_MONTH",
        "OCC_DAY",
        "OCC_HOUR",
        "BIKE_TYPE",
        "BIKE_SPEED",
        "BIKE_MAKE",
        "BIKE_MODEL",
        "BIKE_COST",
        "STATUS"
    ]

    df = df[keep_cols].copy()

    df["returned"] = df["STATUS"].apply(
        lambda x: 1 if str(x).strip().upper() == "RECOVERED" else 0
    )
    df.drop(columns=["STATUS"], inplace=True)

    numeric_cols = [
        "OCC_YEAR",
        "OCC_MONTH",
        "OCC_DAY",
        "OCC_HOUR",
        "BIKE_SPEED",
        "BIKE_COST"
    ]

    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")
        df[col] = df[col].fillna(df[col].median())

    # Fix categorical columns
    df["BIKE_TYPE"] = df["BIKE_TYPE"].astype(str).replace(["nan", ""], "Unknown")
    df["BIKE_MAKE"] = df["BIKE_MAKE"].astype(str).replace(["nan", ""], "Unknown")
    df["BIKE_MODEL"] = df["BIKE_MODEL"].astype(str).replace(["nan", ""], "Unknown")

    # Reduce cardinality
    top_makes = df["BIKE_MAKE"].value_counts().nlargest(15).index
    df["BIKE_MAKE"] = df["BIKE_MAKE"].where(df["BIKE_MAKE"].isin(top_makes), "Other")

    top_models = df["BIKE_MODEL"].value_counts().nlargest(15).index
    df["BIKE_MODEL"] = df["BIKE_MODEL"].where(df["BIKE_MODEL"].isin(top_models), "Other")

    # One-hot encode
    df = pd.get_dummies(
        df,
        columns=["BIKE_TYPE", "BIKE_MAKE", "BIKE_MODEL"],
        drop_first=True
    )

    # FINAL FIX: remove any remaining NaNs
    df = df.fillna(0)

    df = df.sample(frac=1, random_state=42)

    print("Cleaned dataset shape:", df.shape)
    print("\nAny NaNs left:", df.isna().sum().sum())
    return df


# ============================================================
# 3. SAVE CLEANED DATASET
# ============================================================

def save_cleaned_dataset(df: pd.DataFrame):
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(CLEANED_PATH, index=False)
    print("Cleaned dataset saved →", CLEANED_PATH)


# ============================================================
# 4. TRAIN / TEST Split
# ============================================================

def split_data(df, target_column="returned"):

    if target_column not in df.columns:
        raise ValueError(f"Target column '{target_column}' missing in dataset.")

    X = df.drop(target_column, axis=1)
    y = df[target_column]

    return train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )


# ============================================================
# 5. TRAIN MODELS
# ============================================================

def train_logistic_regression(X_train, y_train):
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)
    return model


def train_decision_tree(X_train, y_train):
    model = DecisionTreeClassifier(random_state=42)
    model.fit(X_train, y_train)
    return model


# ============================================================
# 6. EVALUATION HELPER
# ============================================================

def evaluate_model(name, model, X_test, y_test):
    print("\n==============================================")
    print(f" Evaluation: {name}")
    print("==============================================")

    preds = model.predict(X_test)

    accuracy = accuracy_score(y_test, preds)
    precision = precision_score(y_test, preds, zero_division=0)
    recall = recall_score(y_test, preds, zero_division=0)
    f1 = f1_score(y_test, preds, zero_division=0)
    cm = confusion_matrix(y_test, preds)

    print("Accuracy:", accuracy)
    print("Precision:", precision)
    print("Recall:", recall)
    print("F1 Score:", f1)
    print("\nConfusion Matrix:\n", cm)
    print("\nClassification Report:\n", classification_report(y_test, preds, zero_division=0))

    # Predict probabilities for AUC
    if hasattr(model, "predict_proba"):
        probs = model.predict_proba(X_test)[:, 1]
    else:
        probs = model.predict(X_test)

    auc = roc_auc_score(y_test, probs)
    print("AUC Score:", auc)

    fpr, tpr, _ = roc_curve(y_test, probs)

    return {
        "name": name,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "auc": auc,
        "fpr": fpr,
        "tpr": tpr,
        "model": model
    }


# ============================================================
# 7. SAVE BEST MODEL
# ============================================================

def save_model(model):
    with open(MODEL_PATH, "wb") as f:
        pickle.dump(model, f)
    print(f"Model saved → {MODEL_PATH}")


# ============================================================
# 8. PLOT ROC
# ============================================================

def plot_roc(results):
    plt.figure(figsize=(8, 6))

    for r in results:
        plt.plot(r["fpr"], r["tpr"], label=f"{r['name']} (AUC={r['auc']:.2f})")

    plt.plot([0, 1], [0, 1], "k--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve Comparison")
    plt.grid(True)
    plt.legend()

    plt.tight_layout()
    plt.savefig(ROC_PATH)
    print("ROC curve saved →", ROC_PATH)
    plt.show()


# ============================================================
# MAIN
# ============================================================

def main():

    print("\n=== LOADING RAW DATA ===")
    df_raw = load_raw_data(RAW_PATH)
    
    print("\n=== CLEANING DATA ===")
    df_clean = clean_bike_data(df_raw)
    save_cleaned_dataset(df_clean)
    
    COLUMNS_PATH = BASE_DIR / "models/model_columns.pkl"
    with open(COLUMNS_PATH, "wb") as f:
        pickle.dump(df_clean.drop(columns=["returned"]).columns.tolist(), f)
    
    print("\n=== SPLITTING DATA ===")
    X_train, X_test, y_train, y_test = split_data(df_clean)

    print("\n=== TRAINING MODELS ===")
    log_model = train_logistic_regression(X_train, y_train)
    tree_model = train_decision_tree(X_train, y_train)

    print("\n=== EVALUATING MODELS ===")
    results = [
        evaluate_model("Logistic Regression", log_model, X_test, y_test),
        evaluate_model("Decision Tree", tree_model, X_test, y_test)
    ]

    best_model = max(results, key=lambda r: r["auc"])

    print("\n=== BEST MODEL SELECTED ===")
    print(f"{best_model['name']} (AUC = {best_model['auc']:.3f})")

    save_model(best_model["model"])
    plot_roc(results)


if __name__ == "__main__":
    main()
