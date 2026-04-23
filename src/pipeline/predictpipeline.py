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
        """
        Objective: Use the dataset to fill blank user inputs 
        based on specific Metal and Production Route context.
        """
        try:
            metal = input_df['metal'].iloc[0]
            route = input_df['production_route'].iloc[0]

            # Filter dataset for context
            context_df = self.df[(self.df['metal'] == metal) & (self.df['production_route'] == route)]
            if context_df.empty:
                context_df = self.df[self.df['metal'] == metal]

            # Fill blanks/zeros using median/mode
            for col in input_df.columns:
                val = input_df[col].iloc[0]
                if val in [None, "", 0, 0.0, "unknown"]:
                    if self.df[col].dtype in [np.float64, np.int64]:
                        input_df[col] = context_df[col].median()
                    else:
                        input_df[col] = context_df[col].mode()[0]
            
            return input_df
        except Exception as e:
            raise CustomException(e, sys)

    def calculate_sankey_flows(self, total_gwp, input_df):
        """
        Logic to split total GWP into nodes based on input features.
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

            upstream_w, production_w, transport_w = max(0.05, upstream_w), max(0.05, production_w), max(0.05, transport_w)

            return [
                {"source": "Raw Material Extraction", "target": "Metal Production", "value": round(total_gwp * upstream_w, 4)},
                {"source": "Energy & Processing", "target": "Metal Production", "value": round(total_gwp * production_w, 4)},
                {"source": "Logistics & Transport", "target": "Metal Production", "value": round(total_gwp * transport_w, 4)},
                {"source": "Metal Production", "target": "Finished Product", "value": round(total_gwp, 4)}
            ]
        except Exception as e:
            raise CustomException(e, sys)

    def predict(self, features):
        try:
            # 1. Create a Full Template of all 42 columns
            # This ensures even if user sends 1 value, the DataFrame structure is correct
            all_columns = self.df.columns.tolist()
            
            # Remove targets if they exist in your CSV columns
            targets = ['gwp_total', 'circularity_index']
            all_columns = [col for col in all_columns if col not in targets]
            
            # Create a blank DataFrame with the correct column order
            template_df = pd.DataFrame(columns=all_columns)
            
            # 2. Re-align user input into the template
            # This places the user's 1 (or more) value into the right column 
            # and fills the rest with NaN/None
            full_input_df = pd.concat([template_df, features], axis=0, sort=False).reset_index(drop=True)
            
            # Keep only the last row (the actual user data combined with the template)
            full_input_df = full_input_df.tail(1)

            # 3. Estimate missing parameters (The Expert System)
            # This will see the NaNs and fill them with medians based on the Metal/Route
            filled_features = self.estimate_missing_params(full_input_df)

            # 4. Load Artifacts
            model_gwp = load_object(os.path.join("artifacts", "gwp_model.pkl"))
            model_circ = load_object(os.path.join("artifacts", "circularity_model.pkl"))
            preprocessor = load_object(os.path.join("artifacts", "preprocessor.pkl"))

            # 5. Transform and Predict
            # Now preprocessor is happy because it sees all 42 expected columns
            data_scaled = preprocessor.transform(filled_features)
            
            gwp_pred = model_gwp.predict(data_scaled)
            # Invert log scaling if your model was trained on log(y)
            gwp_final = np.expm1(gwp_pred)[0] 
            
            circularity_final = model_circ.predict(data_scaled)[0]

            # 6. Generate Visualization and Technical Profile
            sankey_data = self.calculate_sankey_flows(gwp_final, filled_features)
            profile = filled_features.to_dict(orient='records')[0]

            return {
                "gwp": round(float(gwp_final), 4),
                "circularity": round(float(circularity_final), 4),
                "sankey_data": sankey_data,
                "full_profile": profile
            }

        except Exception as e:
            raise CustomException(e, sys)