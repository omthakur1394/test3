# Make sure all of these are imported at the top
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import pandas as pd
import numpy as np
import joblib

# Initialize the Flask app
app = Flask(__name__)
# Enable CORS
CORS(app)

# Load the model and pipeline
try:
    model = joblib.load('model.pkl')
    pipeline = joblib.load('pipeline.pkl')
except Exception as e:
    print(f"Error loading model or pipeline: {e}")
    model = None
    pipeline = None

# This route serves the main HTML page
@app.route('/')
def home():
    return render_template('index.html')

# This route handles the predictions
@app.route('/predict', methods=['POST'])
def predict():
    if model is None or pipeline is None:
        return jsonify({'error': 'Model not loaded properly'}), 500

    try:
        json_data = request.get_json()
        input_df = pd.DataFrame([json_data])

        # Data type conversion
        input_df['vehicle_age'] = input_df['vehicle_age'].astype(int)
        input_df['km_driven'] = input_df['km_driven'].astype(int)
        input_df['mileage'] = input_df['mileage'].astype(float)
        input_df['engine'] = input_df['engine'].astype(int)
        input_df['max_power'] = input_df['max_power'].astype(float)
        input_df['seats'] = input_df['seats'].astype(int)

        transformed_data = pipeline.transform(input_df)
        prediction_log = model.predict(transformed_data)
        prediction = np.expm1(prediction_log)
        
        result_text = f'₹{prediction[0]:,.0f}'
        return jsonify({'prediction_text': result_text})

    except Exception as e:
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    app.run(debug=True)