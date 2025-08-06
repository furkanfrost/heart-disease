from flask import Flask, render_template, request, redirect, url_for, session, send_file
from io import BytesIO
from reportlab.pdfgen import canvas
import os

app = Flask(__name__)
app.secret_key = os.environ.get('SECRET_KEY', 'dev-secret-key')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Bu örnekte dummy veriler kullanılıyor, model entegrasyonu burada yapılmalı
    age = request.form.get('Age')
    risk_level = "Moderate"
    risk_percentage = 57
    color = "#ffa726"
    session['prediction_result'] = {
        "level": risk_level,
        "percentage": risk_percentage,
        "color": color
    }
    return render_template('index.html', prediction=session['prediction_result'])

@app.route('/download-pdf')
def download_pdf():
    prediction = session.get("prediction_result")
    if not prediction:
        return "No prediction result available to export.", 400

    buffer = BytesIO()
    p = canvas.Canvas(buffer)
    p.setFont("Helvetica", 14)
    p.drawString(100, 800, "Heart Attack Risk Report")
    p.drawString(100, 770, f"Risk Level: {prediction['level']}")
    p.drawString(100, 740, f"Risk Percentage: {prediction['percentage']}%")
    p.showPage()
    p.save()
    buffer.seek(0)
    return send_file(buffer, as_attachment=True, download_name="heart_risk_report.pdf", mimetype="application/pdf")

if __name__ == '__main__':
    app.run(debug=True)
