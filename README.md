# Heart Disease

This project analyzes heart disease risk using a machine learning model and exposes
the prediction through a simple Flask web application.

## Setup

1. Clone the repository and optionally create a virtual environment.
2. Install the required packages:

   ```bash
   pip install -r requirements.txt
   ```

3. Set the Flask session secret key (required for PDF downloads and session
   management):

   ```bash
   export SECRET_KEY="your-secret-key"  # macOS/Linux
   set SECRET_KEY="your-secret-key"    # Windows PowerShell
   ```

## Training the Model

1. Ensure the dataset `data/heart.csv` is present.
2. Run the training script to create `heart_model.pkl` and `heart_scaler.pkl`:

   ```bash
   python save_model.py
   ```

## Running the Web Application

1. After training, start the Flask server:

   ```bash
   python trainmodel.py
   ```

2. Open `http://127.0.0.1:5000/` in your browser to use the app.
