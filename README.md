# COMP-309-Analytics-Project
Group 7
# Bike Theft Prediction Project

## Overview
Predict whether a stolen bicycle is likely to be recovered, using Toronto Police data.

## Tech Stack
- Python (pandas, scikit-learn, matplotlib, seaborn)
- Flask (API)
- HTML/Jinja / React (client)

## Structure
- `data/` – raw and processed datasets
- `notebooks/` – exploratory and preprocessing notebooks
- `src/` – application source code (models, API, client)
- `models/` – serialized model files
- `reports/` – figures and final report

## How to Run the Model Script
```bash
pip install -r requirements.txt
python src/models/model_build_and_evaluate.py
