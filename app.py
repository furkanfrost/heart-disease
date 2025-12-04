from flask import Flask, render_template, request, session, send_file
import joblib
import pandas as pd
import os
from io import BytesIO
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas

app = Flask(__name__)
app.secret_key = os.environ.get("SECRET_KEY", "super-secret-key")

MODEL_PATH = "heart_pipeline.pkl"

if not os.path.exists(MODEL_PATH) and os.path.exists("heart_model.pkl"):
    MODEL_PATH = "heart_model.pkl"

try:
    model = joblib.load(MODEL_PATH)
    print(f"Model uploaded: {MODEL_PATH}")
except Exception as e:
    model = None
    print(f"Model upload error. {e}")


@app.route("/", methods=["GET"])
def index():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    if not model:
        return render_template("index.html", error="Model file couldn't find. Please train model first.")

    try:
        input_data = {
            "age": int(request.form.get("Age")),
            "sex": int(request.form.get("Sex")),
            "cp": int(request.form.get("ChestPainType")),
            "trestbps": int(request.form.get("RestingBP")),
            "chol": int(request.form.get("Cholesterol")),
            "fbs": int(request.form.get("FastingBS")),
            "restecg": int(request.form.get("RestingECG")),
            "thalach": int(request.form.get("MaxHR")),
            "exang": int(request.form.get("ExerciseAngina")),
            "oldpeak": float(request.form.get("Oldpeak")),
            "slope": int(request.form.get("ST_Slope")),
            "ca": int(request.form.get("ca")),
            "thal": int(request.form.get("thal"))
        }

        input_df = pd.DataFrame([input_data])

        if hasattr(model, "predict_proba"):
            probability = model.predict_proba(input_df)[0][1]
            risk_percentage = round(probability * 100, 2)
        else:
            prediction = model.predict(input_df)[0]
            risk_percentage = 90 if prediction == 1 else 10

        if risk_percentage < 30:
            risk_level = "Low"
            color = "success"
        elif risk_percentage < 70:
            risk_level = "Moderate"
            color = "warning"
        else:
            risk_level = "High"
            color = "danger"

        result = {
            "percentage": risk_percentage,
            "level": risk_level,
            "color": color
        }

        session['prediction_result'] = result
        return render_template("index.html", prediction=result)

    except Exception as e:
        return render_template("index.html", error=f"Bir hata oluştu: {str(e)}")


@app.route("/download-pdf", methods=["GET"])
def download_pdf():
    result = session.get('prediction_result')
    if not result:
        return render_template("index.html", error="No data report.")

    buffer = BytesIO()
    p = canvas.Canvas(buffer, pagesize=A4)
    p.setFont("Helvetica-Bold", 16)
    p.drawString(100, 800, "Heart Disease Risk Report")
    p.setFont("Helvetica", 12)
    p.drawString(100, 770, f"Risk Level: {result['level']}")
    p.drawString(100, 750, f"Probability: {result['percentage']}%")
    p.save()
    buffer.seek(0)
    return send_file(buffer, as_attachment=True, download_name="report.pdf", mimetype="application/pdf")


if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=5000)
