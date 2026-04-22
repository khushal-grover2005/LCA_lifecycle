import sys
import os
import pandas as pd
import numpy as np
from src.exception import CustomException
from src.logger import logging
from src.utils import load_object

class PredictPipeline:
    def __init__(self):
        pass

    def predict(self, features):
        """
        Takes processed features and returns predictions for both GWP and Circularity.
        """
        try:
            # Define paths to our saved artifacts
            model_gwp_path = os.path.join("artifacts", "gwp_model.pkl")
            model_circ_path = os.path.join("artifacts", "circularity_model.pkl")
            preprocessor_path = os.path.join("artifacts", "preprocessor.pkl")

            logging.info("Loading models and preprocessor...")
            model_gwp = load_object(file_path=model_gwp_path)
            model_circ = load_object(file_path=model_circ_path)
            preprocessor = load_object(file_path=preprocessor_path)

            # Transform the user input using the saved preprocessor
            data_scaled = preprocessor.transform(features)

            # Predict GWP (Note: This is in Log scale from training)
            gwp_log_pred = model_gwp.predict(data_scaled)
            # Convert back from Log scale to actual kg CO2/kg
            gwp_final = np.expm1(gwp_log_pred)

            # Predict Circularity Index (Already in 0-1 scale)
            circularity_pred = model_circ.predict(data_scaled)

            return {
                "gwp": round(float(gwp_final[0]), 4),
                "circularity": round(float(circularity_pred[0]), 4)
            }

        except Exception as e:
            raise CustomException(e, sys)

class CustomData:
    """
    This class is responsible for mapping the HTML/API inputs to the 
    exact column names expected by the DataTransformation object.
    """
    def __init__(self, 
                 metal: str, production_route: str, region: str, 
                 energy_mix: str, transport_mode: str, transport_distance_km: float,
                 energy_mj_per_kg: float, typical_lifespan_years: int,
                 recycled_content_pct: float, eol_recovery_pct: float,
                 application: str, end_of_life_scenario: str,
                 pathway_type: str, circular_flow_type: str,
                 life_cycle_stage: str, co2_capture_used: str,
                 year: int, facility_age_years: int, batch_size_tonnes: float,
                 global_recycling_rate_pct: float, reuse_potential_score: float,
                 material_efficiency_score: float, human_toxicity_score: float,
                 eutrophication_potential: float):

        self.data_dict = {
            "metal": [metal], "production_route": [production_route], "region": [region],
            "energy_mix": [energy_mix], "transport_mode": [transport_mode],
            "transport_distance_km": [transport_distance_km], "energy_mj_per_kg": [energy_mj_per_kg],
            "typical_lifespan_years": [typical_lifespan_years], "recycled_content_pct": [recycled_content_pct],
            "eol_recovery_pct": [eol_recovery_pct], "application": [application],
            "end_of_life_scenario": [end_of_life_scenario], "pathway_type": [pathway_type],
            "circular_flow_type": [circular_flow_type], "life_cycle_stage": [life_cycle_stage],
            "co2_capture_used": [co2_capture_used], "year": [year],
            "facility_age_years": [facility_age_years], "batch_size_tonnes": [batch_size_tonnes],
            "global_recycling_rate_pct": [global_recycling_rate_pct],
            "reuse_potential_score": [reuse_potential_score],
            "material_efficiency_score": [material_efficiency_score],
            "human_toxicity_score": [human_toxicity_score],
            "eutrophication_potential": [eutrophication_potential]
        }

    def get_data_as_data_frame(self):
        try:
            return pd.DataFrame(self.data_dict)
        except Exception as e:
            raise CustomException(e, sys)