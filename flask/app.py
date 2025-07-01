# flask/app.py
from flask import Flask, render_template, request
import pickle
import numpy as np
import xgboost as xgb
import pandas as pd

app = Flask(__name__)

# Load the XGBoost model
model = xgb.XGBClassifier()
model.load_model(r"C:\Users\SANTHOSH\Videos\clean\xgboost_model.json")

@app.route('/')
def home():
    return render_template("home.html")

@app.route('/predict')
def predict():
    return render_template("predict.html")

@app.route('/submit', methods=["POST"])
def submit():
    if request.method == "POST":
        features = [
            float(request.form["step"]),
            float(request.form["type"]),
            float(request.form["amount"]),
            float(request.form["oldbalanceOrg"]),
            float(request.form["newbalanceOrig"]),
            float(request.form["oldbalanceDest"]),
            float(request.form["newbalanceDest"])
        ]
        final_input = np.array([features])
        
        # Get probability scores
        proba = model.predict_proba(final_input)[0]
        fraud_probability = proba[1]  # Probability of fraud
        
        # Use a lower threshold for fraud detection (0.3 instead of 0.5)
        if fraud_probability > 0.3:
            prediction = f"⚠️ Fraudulent Transaction Detected (Confidence: {fraud_probability:.2%})"
        else:
            prediction = f"✅ Transaction is Legitimate (Confidence: {(1-fraud_probability):.2%})"

        return render_template("submit.html", prediction_text=prediction)

    return render_template("predict.html", prediction_text="Please submit the form.")

if __name__ == "__main__":
    app.run(debug=True)

