# app.py
from flask import Flask, request, jsonify
import pickle
import numpy as np

app = Flask(__name__)

# Load the trained model
model_filename = "batter_prediction_model.pkl"
with open(model_filename, "rb") as file:
    model = pickle.load(file)

@app.route("/", methods=["GET"])
def home():
    return "Batter Performance Prediction API is running!"

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()
        
        # Extract features from incoming request
        # Expecting JSON format: {"features": [5, 1, 200, ...]}
        features = np.array(data["features"]).reshape(1, -1)
        
        # Make prediction
        prediction = model.predict(features)

        return jsonify({"predicted_runs": float(prediction[0])})

    except Exception as e:
        return jsonify({"error": str(e)})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
