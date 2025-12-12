from flask import Flask, request, jsonify
import pickle
import pandas as pd
from pathlib import Path
import opts

app = Flask(__name__)

# ===============================
# Load model + column order
# ===============================

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "models"
TEMPLATES_DIR = BASE_DIR / "templates"

MODEL_PATH = MODELS_DIR / "final_model.pkl"
COLS_PATH = MODELS_DIR / "model_columns.pkl"

with open(MODEL_PATH, "rb") as f:
    model = pickle.load(f)

with open(COLS_PATH, "rb") as f:
    model_columns = pickle.load(f)


# ===============================
# Utility: Convert user JSON → model dataframe
# ===============================

def preprocess_input(data):
    """
    Convert the user-submitted JSON to the same format as the training dataframe.
    """

    # Extract year/month/day/hour from occ_date and occ_time
    occ_date = pd.to_datetime(data["occ_date"])
    occ_time = pd.to_datetime(data["occ_time"])

    # Build the base row
    row = {
        "OCC_YEAR": occ_date.year,
        "OCC_MONTH": occ_date.month,
        "OCC_DAY": occ_date.day,
        "OCC_HOUR": occ_time.hour,
        "BIKE_COST": float(data["value"]),
        "BIKE_SPEED": float(data["bike_speed"]),
        "BIKE_TYPE": data["bike_type"],
        "BIKE_MAKE": "UserInput",
        "BIKE_MODEL": "UserInput"
    }

    # Convert into DataFrame
    df = pd.DataFrame([row])

    # Match training-time category reduction
    df["BIKE_MAKE"] = df["BIKE_MAKE"].replace("", "Other")
    df["BIKE_MODEL"] = df["BIKE_MODEL"].replace("", "Other")

    # One-hot encode the same way training did
    df = pd.get_dummies(df)

    # Add missing columns, fill with 0
    for col in model_columns:
        if col not in df.columns:
            df[col] = 0

    # Ensure correct column order
    df = df[model_columns]

    return df


# ===============================
# Prediction endpoint
# ===============================

@app.post("/predict")
def predict():
    try:
        data = request.get_json()
        processed = preprocess_input(data)

        prediction = model.predict(processed)[0]
        probability = model.predict_proba(processed)[0][1]  # probability returned = 1

        return jsonify({
            "returned_probability": float(probability),
            "stolen_probability": float(1 - probability),
            "prediction_label": "LIKELY RETURNED" if prediction == 1 else "NOT LIKELY RETURNED"
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 400

@app.route("/")
def index():
    from flask import render_template
    return render_template("index.html",
                           premises_types=opts.premises_types,
                           bike_types=opts.bike_types,
                           hoods=opts.hoods)

@app.post("/result")
def result():
    try:
        data = request.form.to_dict()
        processed = preprocess_input(data)

        prediction = model.predict(processed)[0]
        probability = model.predict_proba(processed)[0][1]  # probability returned = 1

        from flask import render_template
        return render_template("result.html", returned_probability=float(probability),
                               stolen_probability=float(1 - probability),
                               prediction_label="LIKELY RETURNED" if prediction == 1 else "NOT LIKELY RETURNED")

    except Exception as e:
        return jsonify({"error": str(e)}), 400

# ===============================
# Run server
# ===============================

if __name__ == "__main__":
    app.run(debug=True)
