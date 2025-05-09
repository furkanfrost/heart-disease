# app.py
from flask import Flask, render_template, request
import joblib
import numpy as np

app = Flask(__name__)

# Load trained model and scaler
model = joblib.load("heart_model.pkl")
scaler = joblib.load("heart_scaler.pkl")

@app.route("/", methods=["GET"])
def index():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    try:
        # Collect and convert form inputs
        features = [
            float(request.form.get("Age")),
            1 if request.form.get("Sex") == "M" else 0,
            {"ATA": 0, "NAP": 1, "ASY": 2, "TA": 3}[request.form.get("ChestPainType")],
            float(request.form.get("RestingBP")),
            float(request.form.get("Cholesterol")),
            float(request.form.get("FastingBS")),
            {"Normal": 0, "ST": 1, "LVH": 2}[request.form.get("RestingECG")],
            float(request.form.get("MaxHR")),
            1 if request.form.get("ExerciseAngina") == "Y" else 0,
            float(request.form.get("Oldpeak")),
            {"Up": 0, "Flat": 1, "Down": 2}[request.form.get("ST_Slope")],
            float(request.form.get("ca")),
            float(request.form.get("thal"))
        ]

        # Scale features
        scaled_features = scaler.transform([features])

        # Get probability prediction
        probability = model.predict_proba(scaled_features)[0][1]
        risk_percentage = round(probability * 100, 2)

        # Determine risk level and color
        if risk_percentage < 30:
            risk_level = "Low"
            color = "green"
        elif risk_percentage < 70:
            risk_level = "Moderate"
            color = "orange"
        else:
            risk_level = "High"
            color = "red"

        result = {
            "percentage": risk_percentage,
            "level": risk_level,
            "color": color
        }
        
        return render_template("index.html", prediction=result)

    except Exception as e:
        return f"Error: {e}"

if __name__ == "__main__":
    app.run(debug=True)
