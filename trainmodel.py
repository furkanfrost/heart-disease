# app.py
from flask import Flask, render_template, request, session, send_file
import joblib
import numpy as np
import os
from io import BytesIO
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas

app = Flask(__name__)
app.secret_key = os.environ.get("SECRET_KEY", "dev-secret-key")  # PDF için session kullanacağız

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
        required_fields = [
            "Age", "Sex", "ChestPainType", "RestingBP", "Cholesterol",
            "FastingBS", "RestingECG", "MaxHR", "ExerciseAngina",
            "Oldpeak", "ST_Slope", "ca", "thal"
        ]

        for field in required_fields:
            if not request.form.get(field):
                return render_template("index.html", error=f"Missing required field: {field}")

        try:
            chest_pain_map = {"ATA": 0, "NAP": 1, "ASY": 2, "TA": 3}
            resting_ecg_map = {"Normal": 0, "ST": 1, "LVH": 2}
            st_slope_map = {"Up": 0, "Flat": 1, "Down": 2}

            features = [
                float(request.form.get("Age")),
                1 if request.form.get("Sex") == "M" else 0,
                chest_pain_map.get(request.form.get("ChestPainType"), -1),
                float(request.form.get("RestingBP")),
                float(request.form.get("Cholesterol")),
                float(request.form.get("FastingBS")),
                resting_ecg_map.get(request.form.get("RestingECG"), -1),
                float(request.form.get("MaxHR")),
                1 if request.form.get("ExerciseAngina") == "Y" else 0,
                float(request.form.get("Oldpeak")),
                st_slope_map.get(request.form.get("ST_Slope"), -1),
                float(request.form.get("ca")),
                float(request.form.get("thal"))
            ]

            if -1 in features:
                return render_template("index.html", error="Invalid categorical value entered.")

        except ValueError as e:
            return render_template("index.html", error=f"Invalid numeric value: {str(e)}")
        except Exception as e:
            return render_template("index.html", error=f"Error processing input: {str(e)}")

        try:
            scaled_features = scaler.transform([features])
        except Exception as e:
            return render_template("index.html", error=f"Error scaling features: {str(e)}")

        try:
            probability = model.predict_proba(scaled_features)[0][1]
        except Exception as e:
            return render_template("index.html", error=f"Error making prediction: {str(e)}")

        risk_percentage = round(probability * 100, 2)

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

        session['prediction_result'] = result

        return render_template("index.html", prediction=result)

    except Exception as e:
        return render_template("index.html", error=f"An unexpected error occurred: {str(e)}")

@app.route("/download-pdf", methods=["GET"])
def download_pdf():
    result = session.get('prediction_result')
    if not result:
        return render_template("index.html", error="No prediction result available to export.")

    buffer = BytesIO()
    p = canvas.Canvas(buffer, pagesize=A4)
    p.setFont("Helvetica", 12)
    p.drawString(100, 800, "Heart Attack Risk Prediction Report")
    p.drawString(100, 770, f"Risk Level: {result['level']}")
    p.drawString(100, 750, f"Risk Percentage: {result['percentage']}%")
    p.save()
    buffer.seek(0)

    return send_file(buffer, as_attachment=True, download_name="heart_risk_report.pdf", mimetype="application/pdf")

if __name__ == "__main__":
    debug_mode = os.environ.get("FLASK_DEBUG", "true").lower() == "true"
    app.run(debug=debug_mode)
