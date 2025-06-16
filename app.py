# app.py
from flask import Flask, render_template, request
import joblib
import numpy as np

app = Flask(__name__)

# Load trained model and scaler with error handling
try:
    model = joblib.load("heart_model.pkl")
    scaler = joblib.load("heart_scaler.pkl")
except FileNotFoundError as e:
    print(f"Error: Model files not found. Please ensure heart_model.pkl and heart_scaler.pkl exist. Error: {e}")
    raise
except Exception as e:
    print(f"Error loading model files: {e}")
    raise

@app.route("/", methods=["GET"])
def index():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    try:
        # Validate required fields
        required_fields = ["Age", "Sex", "ChestPainType", "RestingBP", "Cholesterol", 
                         "FastingBS", "RestingECG", "MaxHR", "ExerciseAngina", 
                         "Oldpeak", "ST_Slope", "ca", "thal"]
        
        for field in required_fields:
            if not request.form.get(field):
                return render_template("index.html", error=f"Missing required field: {field}")

        # Collect and convert form inputs
        try:
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
        except ValueError as e:
            return render_template("index.html", error=f"Invalid numeric value: {str(e)}")
        except KeyError as e:
            return render_template("index.html", error=f"Invalid category value: {str(e)}")

        # Scale features
        try:
            scaled_features = scaler.transform([features])
        except Exception as e:
            return render_template("index.html", error=f"Error scaling features: {str(e)}")

        # Get probability prediction
        try:
            probability = model.predict_proba(scaled_features)[0][1]
        except Exception as e:
            return render_template("index.html", error=f"Error making prediction: {str(e)}")

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
        return render_template("index.html", error=f"An unexpected error occurred: {str(e)}")

if __name__ == "__main__":
    app.run(debug=True)
