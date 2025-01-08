from flask import Flask, request, jsonify, render_template
import joblib
import numpy as np

app = Flask(__name__)

# Load the saved model
try:
    model = joblib.load("final_rf_model_v2.pkl")
    print("Model loaded successfully.")
except FileNotFoundError:
    print("Error: 'final_rf_model.pkl' not found.")
    exit()

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    try:
        # Fetch required form data
        year = int(request.form.get("year"))
        rainfall = float(request.form.get("rainfall"))
        irrigation_area = float(request.form.get("irrigationArea"))

        # Prepare features for prediction
        features = np.array([[year, rainfall, irrigation_area]])
        
        # Predict yield
        prediction = model.predict(features)[0]

        return render_template('index.html',prediction=f'Prediction:- {prediction} kg/ha')

    except Exception as e:
        return jsonify({"error": str(e)}), 400

if __name__ == "__main__":
    app.run(debug=True)
