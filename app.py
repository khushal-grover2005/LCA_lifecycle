from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
import numpy as np
import sys
import os
from src.pipeline.predictpipeline import PredictPipeline
from src.exception import CustomException
from src.logger import logging

app = Flask(__name__)
CORS(app)  # Enables cross-origin requests for your VS Code / Frontend development

@app.route('/')
def index():
    return "LCA Predictor Backend is Running on Hugging Face."

@app.route('/predict', methods=['POST'])
def predict_api():
    try:
        # 1. Capture the JSON input from the user
        json_data = request.get_json()
        if not json_data:
            return jsonify({"status": "error", "message": "No input data provided"}), 400

        # 2. Track which fields were originally left blank/zero by the user
        # This list allows the frontend to highlight "AI Estimated" values
        imputed_fields = [
            k for k, v in json_data.items() 
            if v in [None, "", 0, 0.0, "unknown"]
        ]

        # 3. Initialize the Pipeline
        # The PredictPipeline now handles the conditional imputation logic
        pipeline = PredictPipeline()

        # 4. Convert input to DataFrame for processing
        # We wrap the json_data in a list to create a single-row DataFrame
        input_df = pd.DataFrame([json_data])

        # 5. Execute the Expert System Logic
        # This returns: 
        # - Primary predictions (GWP, Circularity)
        # - Sankey visualization data
        # - The full 42-parameter technical profile (fully imputed)
        output = pipeline.predict(input_df)

        # 6. Construct the final JSON response
        response = {
            "status": "success",
            "results": {
                "gwp_total": output['gwp'],
                "circularity_index": output['circularity'],
                "resource_efficiency": output['full_profile'].get('material_efficiency_score'),
                "recycled_content_est": output['full_profile'].get('recycled_content_pct'),
                "reuse_potential": output['full_profile'].get('reuse_potential_score')
            },
            "visualizations": {
                "sankey_data": output['sankey_data']
            },
            "technical_profile": output['full_profile'],
            "imputed_fields": imputed_fields,
            "metadata": {
                "model_version": "1.0.0",
                "estimation_method": "Conditional Median/Mode Imputation"
            }
        }

        logging.info(f"Successful prediction for metal: {json_data.get('metal')}")
        return jsonify(response)

    except Exception as e:
        logging.error(f"Prediction Error: {str(e)}")
        return jsonify({
            "status": "error", 
            "message": "An error occurred during processing. Ensure all core fields (metal, route) are provided.",
            "details": str(e)
        }), 500

if __name__ == "__main__":
    # Hugging Face Spaces usually require port 7860
    app.run(host="0.0.0.0", port=7860)