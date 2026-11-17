import pickle
from pathlib import Path

import pandas as pd
from flask import Flask, jsonify, render_template, request

BASE_DIR = Path(__file__).parent

with open(BASE_DIR / "task_slr.pkl", "rb") as f:
    model = pickle.load(f)

app = Flask(__name__)


@app.get("/")
def home():
    return render_template("Height to weight predictor.html")


@app.post("/predict")
def predict():
    data = request.get_json(silent=True) or {}
    try:
        height_cm = float(data.get("height_cm"))
    except (TypeError, ValueError):
        return jsonify(error="Enter a height in centimetres."), 400

    if not 100 <= height_cm <= 250:
        return jsonify(error="Enter a height between 100 and 250 cm."), 400

    # The model was trained on a "Height" column in metres
    X = pd.DataFrame({"Height": [height_cm / 100]})
    weight = float(model.predict(X)[0])
    return jsonify(weight_kg=round(weight, 1))


if __name__ == "__main__":
    app.run(debug=True)