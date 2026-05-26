import sys
import os
import pandas as pd
import numpy as np
from src.exception import CustomException
from src.logger import logging
from src.utils import load_object

class PredictPipeline:
    def __init__(self):
        # Load knowledge base for imputation
        self.df = pd.read_csv(os.path.join("data", "lca_metals_final.csv"))

    def estimate_missing_params(self, input_df):
        try:
            # 1. Identify Context
            metal = input_df['metal'].iloc[0]
            route = input_df['production_route'].iloc[0]

            # 2. Get Contextual Data from your 42-column CSV
            context_df = self.df[(self.df['metal'] == metal) & (self.df['production_route'] == route)]
            
            # Fallback if specific route doesn't exist for that metal
            if context_df.empty:
                context_df = self.df[self.df['metal'] == metal]
            
            # Global fallback if the metal itself isn't found (safety first)
            if context_df.empty:
                context_df = self.df

            # 3. Fill EVERYTHING that is NaN/Blank/Zero
            for col in input_df.columns:
                # Check if value is NaN or null
                if pd.isna(input_df[col].iloc[0]) or input_df[col].iloc[0] in [None, "", 0, 0.0, "unknown"]:
                    
                    # Fill numeric columns with Median
                    if self.df[col].dtype in [np.float64, np.int64]:
                        val = context_df[col].median()
                        # If context median is still NaN, use global median
                        input_df[col] = val if not pd.isna(val) else self.df[col].median()
                    
                    # Fill categorical columns with Mode
                    else:
                        mode_vals = context_df[col].mode()
                        if not mode_vals.empty:
                            input_df[col] = mode_vals[0]
                        else:
                            # Global mode fallback
                            input_df[col] = self.df[col].mode()[0]
            
            return input_df
        except Exception as e:
            raise CustomException(e, sys)
        
    def calculate_sankey_flows(self, total_gwp, input_df):
        """
        Logic to split total GWP into nodes based on input features for visualization.
        Formatted specifically for Nivo Sankey (nodes array and index-based links).
        """
        try:
            upstream_w, production_w, transport_w = 0.35, 0.55, 0.10

            # Dynamic adjustments
            route = input_df['production_route'].iloc[0]
            if route == 'Primary':
                upstream_w += 0.20
                production_w -= 0.20
            elif route == 'Secondary':
                upstream_w -= 0.15
                production_w += 0.15

            dist = input_df['transport_distance_km'].iloc[0]
            if dist > 3000:
                shift = 0.15
                transport_w += shift
                upstream_w -= (shift / 2)
                production_w -= (shift / 2)

            upstream_w = max(0.05, upstream_w)
            production_w = max(0.05, production_w)
            transport_w = max(0.05, transport_w)

            # 🌟 REQUIRED NIVO FORMAT 🌟
            nodes = [
                {"name": "Raw Material Extraction"}, # Index 0
                {"name": "Energy & Processing"},     # Index 1
                {"name": "Logistics & Transport"},   # Index 2
                {"name": "Metal Production"},        # Index 3
                {"name": "Finished Product"}         # Index 4
            ]

            links = [
                {"source": 0, "target": 3, "value": round(total_gwp * upstream_w, 4)},
                {"source": 1, "target": 3, "value": round(total_gwp * production_w, 4)},
                {"source": 2, "target": 3, "value": round(total_gwp * transport_w, 4)},
                {"source": 3, "target": 4, "value": round(total_gwp, 4)}
            ]

            return {
                "nodes": nodes,
                "links": links
            }
        except Exception as e:
            raise CustomException(e, sys)

    def predict(self, features):
        try:
            # 1. TEMPLATE LOGIC: Create a template of all 42 expected columns
            # This ensures even a single-value JSON doesn't crash the preprocessor
            all_columns = [col for col in self.df.columns if col not in ['gwp_total', 'circularity_index']]
            template_df = pd.DataFrame(columns=all_columns)
            
            # Align user input into the template (fills missing columns with NaN)
            full_input_df = pd.concat([template_df, features], axis=0, sort=False).tail(1).reset_index(drop=True)

            # 2. EXPERT IMPUTATION: Fill based on Metal and Production Route context
            # This fills the NaNs using medians/modes from your specific LCA dataset
            filled_features = self.estimate_missing_params(full_input_df)

            # 3. ZERO-NaN SAFETY NET: Mandatory for ElasticNet/Linear Models
            # If any value is STILL NaN (due to empty context), fill with global median/mode
            for col in filled_features.columns:
                if filled_features[col].isnull().any():
                    if self.df[col].dtype in [np.float64, np.int64]:
                        global_median = self.df[col].median()
                        filled_features[col] = filled_features[col].fillna(global_median if not pd.isna(global_median) else 0)
                    else:
                        global_mode = self.df[col].mode()
                        filled_features[col] = filled_features[col].fillna(global_mode[0] if not global_mode.empty else "unknown")

            # Final check to catch any lingering edge-case NaNs
            filled_features = filled_features.fillna(0)

            # 4. LOAD ARTIFACTS
            model_gwp = load_object(os.path.join("artifacts", "gwp_model.pkl"))
            model_circ = load_object(os.path.join("artifacts", "circularity_model.pkl"))
            preprocessor = load_object(os.path.join("artifacts", "preprocessor.pkl"))

            # 5. TRANSFORM AND PREDICT
            logging.info("Transforming 42-parameter profile for model inference...")
            data_scaled = preprocessor.transform(filled_features)
            
            # GWP Prediction (Assumes log transformation was used during training)
            gwp_pred = model_gwp.predict(data_scaled)
            gwp_final = np.expm1(gwp_pred)[0] 
            
            # Circularity Prediction
            circularity_final = model_circ.predict(data_scaled)[0]

            # 6. GENERATE VISUALIZATION AND PROFILE
            # Uses your original logic to split the GWP into Sankey nodes
            sankey_data = self.calculate_sankey_flows(gwp_final, filled_features)
            profile = filled_features.to_dict(orient='records')[0]

            return {
                "gwp": round(float(gwp_final), 4),
                "circularity": round(float(circularity_final), 4),
                "sankey_data": sankey_data,
                "full_profile": profile
            }

        except Exception as e:
            logging.error(f"Error in PredictPipeline: {str(e)}")
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