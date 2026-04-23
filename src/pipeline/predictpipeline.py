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

    def calculate_sankey_flows(self, total_gwp, input_df):
        """
        Logic to split total GWP into nodes based on input features for visualization.
        """
        try:
            # Standard baseline weights
            upstream_w = 0.35
            production_w = 0.55
            transport_w = 0.10

            # Dynamic adjustments based on specific inputs
            # 1. Check Production Route
            route = input_df['production_route'].iloc[0]
            if route == 'Primary':
                upstream_w += 0.20
                production_w -= 0.20
            elif route == 'Secondary':
                upstream_w -= 0.15
                production_w += 0.15

            # 2. Check Transport Distance
            dist = input_df['transport_distance_km'].iloc[0]
            if dist > 3000:
                shift = 0.15
                transport_w += shift
                upstream_w -= (shift / 2)
                production_w -= (shift / 2)

            # Ensure weights remain realistic (minimum 5% per node)
            upstream_w = max(0.05, upstream_w)
            production_w = max(0.05, production_w)
            transport_w = max(0.05, transport_w)

            # Calculate flow values
            upstream_val = round(total_gwp * upstream_w, 4)
            production_val = round(total_gwp * production_w, 4)
            transport_val = round(total_gwp * transport_w, 4)

            # Construct Sankey JSON structure
            sankey_data = [
                {"source": "Raw Material Extraction", "target": "Metal Production", "value": upstream_val},
                {"source": "Energy & Processing", "target": "Metal Production", "value": production_val},
                {"source": "Logistics & Transport", "target": "Metal Production", "value": transport_val},
                {"source": "Metal Production", "target": "Finished Product", "value": round(total_gwp, 4)}
            ]
            return sankey_data
        except Exception as e:
            raise CustomException(e, sys)

    def predict(self, features):
        try:
            model_gwp_path = os.path.join("artifacts", "gwp_model.pkl")
            model_circ_path = os.path.join("artifacts", "circularity_model.pkl")
            preprocessor_path = os.path.join("artifacts", "preprocessor.pkl")

            model_gwp = load_object(file_path=model_gwp_path)
            model_circ = load_object(file_path=model_circ_path)
            preprocessor = load_object(file_path=preprocessor_path)

            logging.info("Transforming input features...")
            data_scaled = preprocessor.transform(features)

            # Predict GWP and Invert Log Scale
            gwp_log_pred = model_gwp.predict(data_scaled)
            gwp_final = np.expm1(gwp_log_pred)[0]

            # Predict Circularity
            circularity_pred = model_circ.predict(data_scaled)[0]

            # Generate Sankey Visualization Logic
            sankey_data = self.calculate_sankey_flows(gwp_final, features)

            return {
                "gwp": round(float(gwp_final), 4),
                "circularity": round(float(circularity_pred), 4),
                "sankey_data": sankey_data
            }

        except Exception as e:
            raise CustomException(e, sys)

class CustomData:
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