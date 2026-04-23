import os
import sys
from flask import Flask, request, jsonify
from flask_cors import CORS
from src.exception import CustomException
from src.logger import logging
from src.pipeline.trainpipeline import TrainingPipeline
from src.pipeline.predictpipeline import PredictPipeline, CustomData

app = Flask(__name__)
# Allows all origins for testing; when you deploy Vercel, 
# you can replace "*" with your specific Vercel URL.
CORS(app, resources={r"/*": {"origins": "*"}})

@app.route('/')
def home():
    return "LCA Predictor Backend is Running on Hugging Face."

@app.route('/predict', methods=['POST'])
def predict_api():
    try:
        json_data = request.get_json()
        
        if not json_data:
            return jsonify({"status": "error", "message": "No input provided"}), 400

        # Mapping JSON to your explicit CustomData constructor
        data = CustomData(
            metal=json_data.get('metal'),
            production_route=json_data.get('production_route'),
            region=json_data.get('region'),
            energy_mix=json_data.get('energy_mix'),
            transport_mode=json_data.get('transport_mode'),
            transport_distance_km=float(json_data.get('transport_distance_km', 0)),
            energy_mj_per_kg=float(json_data.get('energy_mj_per_kg', 0)),
            typical_lifespan_years=int(json_data.get('typical_lifespan_years', 0)),
            recycled_content_pct=float(json_data.get('recycled_content_pct', 0)),
            eol_recovery_pct=float(json_data.get('eol_recovery_pct', 0)),
            application=json_data.get('application'),
            end_of_life_scenario=json_data.get('end_of_life_scenario'),
            pathway_type=json_data.get('pathway_type'),
            circular_flow_type=json_data.get('circular_flow_type'),
            life_cycle_stage=json_data.get('life_cycle_stage'),
            co2_capture_used=json_data.get('co2_capture_used'),
            year=int(json_data.get('year', 2024)),
            facility_age_years=int(json_data.get('facility_age_years', 0)),
            batch_size_tonnes=float(json_data.get('batch_size_tonnes', 0)),
            global_recycling_rate_pct=float(json_data.get('global_recycling_rate_pct', 0)),
            reuse_potential_score=float(json_data.get('reuse_potential_score', 0)),
            material_efficiency_score=float(json_data.get('material_efficiency_score', 0)),
            human_toxicity_score=float(json_data.get('human_toxicity_score', 0)),
            eutrophication_potential=float(json_data.get('eutrophication_potential', 0))
        )

        pred_df = data.get_data_as_data_frame()
        predict_pipeline = PredictPipeline()
        results = predict_pipeline.predict(pred_df)
        
        logging.info("Prediction and Sankey data generated successfully")

        return jsonify({
            "status": "success",
            "results": {
                "gwp_total": results['gwp'],
                "circularity_index": results['circularity']
            },
            "visualizations": {
                "sankey_data": results['sankey_data']
            }
        })

    except Exception as e:
        logging.error(f"Prediction Error: {str(e)}")
        return jsonify({"status": "error", "message": str(e)}), 500

if __name__ == "__main__":
    # Hugging Face Spaces uses port 7860 by default
    port = int(os.environ.get("PORT", 7860))
    
    # Check if models exist. 
    # Note: On HF, it's better to push artifacts to Git so it doesn't train on startup.
    if not os.path.exists("artifacts/gwp_model.pkl"):
        print("Artifacts missing. Training models...")
        try:
            train_pipeline = TrainingPipeline()
            train_pipeline.start_training()
        except Exception as e:
            print(f"Initial training failed: {e}")

    print(f"\n🚀 LCA Backend Live on port {port}")
    # debug=False is important for production/hosting
    app.run(host="0.0.0.0", port=port, debug=False)