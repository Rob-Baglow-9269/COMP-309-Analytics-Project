import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
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
    roc_auc_score
)
import pickle


# ============================================================
# 1. Load dataset
# ============================================================

def load_dataset(path: str):
    df = pd.read_csv(path)
    print("Dataset loaded:", df.shape)
    return df


# ============================================================
# 2. Train-test split
# ============================================================

def split_data(df, target_column="returned"):
    if target_column not in df.columns:
        raise ValueError(f"Target column '{target_column}' not found in dataset.")

    X = df.drop(target_column, axis=1)
    y = df[target_column]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    return X_train, X_test, y_train, y_test


# ============================================================
# 3. Train Models
# ============================================================

def train_logistic_regression(X_train, y_train):
    model = LogisticRegression(max_iter=1000, solver="lbfgs")
    model.fit(X_train, y_train)
    return model


def train_decision_tree(X_train, y_train):
    model = DecisionTreeClassifier(
        random_state=42
    )
    model.fit(X_train, y_train)
    return model


# ============================================================
# 4. Evaluation helper
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

    # AUC / ROC
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
        "model": model,
    }


# ============================================================
# 5. Save best model via pickle
# ============================================================

def save_model(model, filename="final_model.pkl"):
    with open(filename, "wb") as f:
        pickle.dump(model, f)
    print(f"\nModel saved → {filename}")


# ============================================================
# 6. Plot combined ROC curve
# ============================================================

def plot_roc(results):
    plt.figure(figsize=(8,6))

    for r in results:
        plt.plot(r["fpr"], r["tpr"], label=f"{r['name']} (AUC={r['auc']:.2f})")

    plt.plot([0,1],[0,1], "k--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve Comparison")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("roc_curve.png")
    print("\nROC curve saved → roc_curve.png")
    plt.show()


# ============================================================
# MAIN PIPELINE
# ============================================================

def main():
    df = load_dataset("data/processed/clean_bike_data.csv")

    X_train, X_test, y_train, y_test = split_data(df)

    # Train both models
    log_model = train_logistic_regression(X_train, y_train)
    tree_model = train_decision_tree(X_train, y_train)

    # Evaluate
    results = []
    results.append(evaluate_model("Logistic Regression", log_model, X_test, y_test))
    results.append(evaluate_model("Decision Tree", tree_model, X_test, y_test))

    # Choose best model based on AUC
    best_model = max(results, key=lambda r: r["auc"])

    print("\n\n==============================")
    print(" BEST MODEL SELECTED")
    print("==============================")
    print(f"{best_model['name']} with AUC={best_model['auc']:.3f}")

    # Save model for Flask API
    save_model(best_model["model"], "final_model.pkl")

    # Save ROC comparison
    plot_roc(results)


if __name__ == "__main__":
    main()
