import os
import sys
from pathlib import Path

# Path setup to ensure internal modules are discoverable
project_root = str(Path(__file__).parent)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from flask import Flask, request, render_template, jsonify
from src.exception import CustomException
from src.logger import logging
from src.pipeline.trainpipeline import TrainingPipeline
from src.pipeline.predictpipeline import PredictPipeline, CustomData

app = Flask(__name__)

## --- ROUTE 1: HOME PAGE ---
@app.route('/')
def index():
    return render_template('index.html') 

## --- ROUTE 2: PREDICTION ENDPOINT ---
@app.route('/predictdata', methods=['GET', 'POST'])
def predict_datapoint():
    if request.method == 'GET':
        return render_template('home.html') # This is your input form
    else:
        try:
            # Capturing data from the form (mapping to your CustomData class)
            data = CustomData(
                metal=request.form.get('metal'),
                production_route=request.form.get('production_route'),
                region=request.form.get('region'),
                energy_mix=request.form.get('energy_mix'),
                transport_mode=request.form.get('transport_mode'),
                transport_distance_km=float(request.form.get('transport_distance_km')),
                energy_mj_per_kg=float(request.form.get('energy_mj_per_kg')),
                typical_lifespan_years=int(request.form.get('typical_lifespan_years')),
                recycled_content_pct=float(request.form.get('recycled_content_pct')),
                eol_recovery_pct=float(request.form.get('eol_recovery_pct')),
                application=request.form.get('application'),
                end_of_life_scenario=request.form.get('end_of_life_scenario'),
                pathway_type=request.form.get('pathway_type'),
                circular_flow_type=request.form.get('circular_flow_type'),
                life_cycle_stage=request.form.get('life_cycle_stage'),
                co2_capture_used=request.form.get('co2_capture_used'),
                year=int(request.form.get('year')),
                facility_age_years=int(request.form.get('facility_age_years')),
                batch_size_tonnes=float(request.form.get('batch_size_tonnes')),
                global_recycling_rate_pct=float(request.form.get('global_recycling_rate_pct')),
                reuse_potential_score=float(request.form.get('reuse_potential_score')),
                material_efficiency_score=float(request.form.get('material_efficiency_score')),
                human_toxicity_score=float(request.form.get('human_toxicity_score')),
                eutrophication_potential=float(request.form.get('eutrophication_potential'))
            )

            pred_df = data.get_data_as_data_frame()
            logging.info("Input Dataframe created for prediction")

            predict_pipeline = PredictPipeline()
            results = predict_pipeline.predict(pred_df)
            
            logging.info(f"Prediction successful: {results}")

            # Returning results to the frontend
            return render_template('home.html', 
                                   gwp_result=results['gwp'], 
                                   circularity_result=results['circularity'])

        except Exception as e:
            raise CustomException(e, sys)

## --- EXECUTION BLOCK ---
if __name__ == "__main__":
    # If you want to trigger training before starting the server, keep this block
    # Or run it once and comment it out to just run the web server
    try:
        print("\n" + "="*50)
        print("🛠️  CHECKING MODEL STATUS")
        
        # Checking if models already exist in artifacts
        if not os.path.exists("artifacts/gwp_model.pkl"):
            print("⚠️  Models not found. Starting Training Pipeline...")
            train_pipeline = TrainingPipeline()
            train_pipeline.start_training()
            print("✅ Training Completed Successfully.")
        else:
            print("🚀 Models found. Skipping training.")

        print("="*50 + "\n")
        
        # Start the Flask Server
        # host='0.0.0.0' makes it accessible on your local network
        app.run(host="0.0.0.0", port=8080, debug=True)

    except Exception as e:
        logging.info("Application execution failed")
        raise CustomException(e, sys)