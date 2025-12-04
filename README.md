
# ❤️ Heart Attack Risk Prediction AI

Advanced AI-powered web application that predicts the probability of heart disease based on clinical parameters using Machine Learning Pipelines.

---

## 🚀 Project Overview

This project is a **full-stack AI application** designed to assist medical professionals in early heart disease detection.  
Unlike traditional scripts, this project utilizes **Scikit-Learn Pipelines** to prevent data leakage and ensure robust production deployment.

The system processes **13 clinical features** (such as Chest Pain Type, ST Slope, Cholesterol) and provides an **instant risk assessment with a downloadable PDF Medical Report**.

---

## ✨ Key Features

- 🧠 **Pipeline Architecture:** Fully integrated preprocessing (Scaling/Encoding) + Model Inference.
- ✔️ **High Accuracy:** Achieved **98.5% Accuracy** on validation.
- 🖥️ **Interactive UI:** Responsive and modern medical data entry interface.
- 📄 **PDF Reporting:** Automated patient risk reports using ReportLab.
- 🐳 **Dockerized:** Fully containerized for seamless deployment.

---

## 📊 Model Performance

Trained using a **Random Forest Classifier** within a Pipeline.

| Metric | Score |
|--------|-------|
| Accuracy | **98.54%** |
| Precision | **99%** |
| Recall | **99%** |
| F1-Score | **99%** |

### Classification Report

```
              precision    recall  f1-score   support

           0       0.97      1.00      0.99       102
           1       1.00      0.97      0.99       103

    accuracy                           0.99       205
   macro avg       0.99      0.99      0.99       205
weighted avg      0.99      0.99      0.99       205
```

---

## 🛠️ Tech Stack & Architecture

- **Core:** Python 3.9  
- **ML Framework:** Scikit-Learn (Pipeline, RandomForest, StandardScaler, OneHotEncoder)  
- **Web Framework:** Flask (Jinja2 Templates)  
- **Containerization:** Docker  
- **Libraries:** Pandas, NumPy, Joblib, ReportLab  

---

## 💻 Installation & Usage

### **Option 1: Using Docker (Recommended)**

```bash
# Build Docker Image
docker build -t heart-disease-app .

# Run Container
docker run -p 5000:5000 heart-disease-app
```

---

### **Option 2: Local Installation**

```bash
# Clone the repository
git clone https://github.com/furkanfrost/heart-disease-prediction.git
cd heart-disease-prediction

# Create Virtual Environment (Optional)
python -m venv venv
source venv/bin/activate   # On Windows: venv\Scripts\activate

# Install Dependencies
pip install -r requirements.txt

# Train the Model (Creates heart_pipeline.pkl)
python train_pipeline.py

# Run the Application
python app.py
```

➡ Access via browser: **http://localhost:5000**

---

## 📂 Project Structure

```
├── data/                # Dataset source (heart.csv)
├── templates/           # HTML templates for the UI
├── app.py               # Main Flask Application
├── train_pipeline.py    # ML Training Script (Pipeline & Preprocessing)
├── heart_pipeline.pkl   # Serialized Model Pipeline
├── Dockerfile           # Docker configuration
└── requirements.txt     # Python dependencies
```

---

## ⚠️ Disclaimer

This application is developed for **educational and research purposes**.  
It is intended to assist, **not replace** medical professionals.  
Always consult with a qualified healthcare provider.

---

## 👤 Author

**Furkan KILBOZ**


---
