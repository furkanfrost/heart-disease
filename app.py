# app.py
from flask import Flask, render_template, request, send_file
import joblib
import numpy as np
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet
import csv
from datetime import datetime
import os
import io

app = Flask(__name__)

# Load trained model and scaler
model = joblib.load("heart_model.pkl")
scaler = joblib.load("heart_scaler.pkl")

# Create reports directory if it doesn't exist
if not os.path.exists('reports'):
    os.makedirs('reports')

def generate_report(prediction_data, format_type='pdf'):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    if format_type == 'pdf':
        # Create PDF report
        buffer = io.BytesIO()
        doc = SimpleDocTemplate(buffer, pagesize=letter)
        styles = getSampleStyleSheet()
        elements = []
        
        # Add title
        title = Paragraph("Heart Disease Risk Assessment Report", styles['Title'])
        elements.append(title)
        elements.append(Spacer(1, 20))
        
        # Create data table
        data = [
            ["Parameter", "Value"],
            ["Age", prediction_data['Age']],
            ["Sex", prediction_data['Sex']],
            ["Chest Pain Type", prediction_data['ChestPainType']],
            ["Resting BP", prediction_data['RestingBP']],
            ["Cholesterol", prediction_data['Cholesterol']],
            ["Fasting BS", prediction_data['FastingBS']],
            ["Resting ECG", prediction_data['RestingECG']],
            ["Max HR", prediction_data['MaxHR']],
            ["Exercise Angina", prediction_data['ExerciseAngina']],
            ["Oldpeak", prediction_data['Oldpeak']],
            ["ST Slope", prediction_data['ST_Slope']],
            ["CA", prediction_data['ca']],
            ["Thal", prediction_data['thal']],
            ["Risk Percentage", f"{prediction_data['risk_percentage']}%"],
            ["Risk Level", prediction_data['risk_level']]
        ]
        
        table = Table(data)
        table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 14),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
            ('TEXTCOLOR', (0, 1), (-1, -1), colors.black),
            ('FONTNAME', (0, 1), (-1, -1), 'Helvetica'),
            ('FONTSIZE', (0, 1), (-1, -1), 12),
            ('GRID', (0, 0), (-1, -1), 1, colors.black)
        ]))
        
        elements.append(table)
        doc.build(elements)
        buffer.seek(0)
        return buffer, f"heart_disease_report_{timestamp}.pdf"
        
    else:  # CSV format
        buffer = io.StringIO()
        writer = csv.writer(buffer)
        writer.writerow(["Parameter", "Value"])
        writer.writerow(["Age", prediction_data['Age']])
        writer.writerow(["Sex", prediction_data['Sex']])
        writer.writerow(["Chest Pain Type", prediction_data['ChestPainType']])
        writer.writerow(["Resting BP", prediction_data['RestingBP']])
        writer.writerow(["Cholesterol", prediction_data['Cholesterol']])
        writer.writerow(["Fasting BS", prediction_data['FastingBS']])
        writer.writerow(["Resting ECG", prediction_data['RestingECG']])
        writer.writerow(["Max HR", prediction_data['MaxHR']])
        writer.writerow(["Exercise Angina", prediction_data['ExerciseAngina']])
        writer.writerow(["Oldpeak", prediction_data['Oldpeak']])
        writer.writerow(["ST Slope", prediction_data['ST_Slope']])
        writer.writerow(["CA", prediction_data['ca']])
        writer.writerow(["Thal", prediction_data['thal']])
        writer.writerow(["Risk Percentage", f"{prediction_data['risk_percentage']}%"])
        writer.writerow(["Risk Level", prediction_data['risk_level']])
        
        buffer.seek(0)
        return buffer, f"heart_disease_report_{timestamp}.csv"

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

        # Store prediction data for reporting
        prediction_data = {
            "Age": request.form.get("Age"),
            "Sex": request.form.get("Sex"),
            "ChestPainType": request.form.get("ChestPainType"),
            "RestingBP": request.form.get("RestingBP"),
            "Cholesterol": request.form.get("Cholesterol"),
            "FastingBS": request.form.get("FastingBS"),
            "RestingECG": request.form.get("RestingECG"),
            "MaxHR": request.form.get("MaxHR"),
            "ExerciseAngina": request.form.get("ExerciseAngina"),
            "Oldpeak": request.form.get("Oldpeak"),
            "ST_Slope": request.form.get("ST_Slope"),
            "ca": request.form.get("ca"),
            "thal": request.form.get("thal"),
            "risk_percentage": risk_percentage,
            "risk_level": risk_level
        }

        result = {
            "percentage": risk_percentage,
            "level": risk_level,
            "color": color,
            "prediction_data": prediction_data
        }

        return render_template("index.html", prediction=result)

    except Exception as e:
        return f"Error: {e}"

@app.route("/download_report/<format_type>", methods=["POST"])
def download_report(format_type):
    try:
        prediction_data = request.json
        buffer, filename = generate_report(prediction_data, format_type)
        
        if format_type == 'pdf':
            return send_file(
                buffer,
                mimetype='application/pdf',
                as_attachment=True,
                download_name=filename
            )
        else:
            return send_file(
                buffer,
                mimetype='text/csv',
                as_attachment=True,
                download_name=filename
            )
    except Exception as e:
        return f"Error generating report: {e}"

if __name__ == "__main__":
    app.run(debug=True)
