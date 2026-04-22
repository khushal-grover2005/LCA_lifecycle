import sys
import os
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler, MinMaxScaler, FunctionTransformer
from sklearn.pipeline import Pipeline

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object
from dataclasses import dataclass

@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path: str = os.path.join('artifacts', "preprocessor.pkl")

class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()

    def get_data_transformer_object(self):
        try:
            # 1. CATEGORICAL FEATURES
            categorical_columns = [
                "metal", "production_route", "region", "application", "energy_mix",
                "transport_mode", "end_of_life_scenario", "pathway_type",
                "circular_flow_type", "life_cycle_stage", "co2_capture_used"
            ]

            # 2. NUMERICAL: LOG TRANSFORM GROUP
            log_columns = ["energy_mj_per_kg", "transport_distance_km", "batch_size_tonnes"]

            # 3. NUMERICAL: STANDARD SCALER GROUP
            standard_scaler_columns = ["year", "typical_lifespan_years", "facility_age_years"]

            # 4. NUMERICAL: MIN-MAX SCALER GROUP
            minmax_scaler_columns = [
                "global_recycling_rate_pct", "recycled_content_pct", "eol_recovery_pct",
                "reuse_potential_score", "material_efficiency_score", "human_toxicity_score",
                "eutrophication_potential"
            ]

            # Pipelines
            cat_pipeline = Pipeline(steps=[
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("one_hot_encoder", OneHotEncoder(handle_unknown='ignore', drop='first', sparse_output=False)),
                ("scaler", StandardScaler(with_mean=False))
            ])

            log_pipeline = Pipeline(steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("log", FunctionTransformer(np.log1p)),
                ("scaler", StandardScaler())
            ])

            num_standard_pipeline = Pipeline(steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler())
            ])

            num_minmax_pipeline = Pipeline(steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", MinMaxScaler())
            ])

            preprocessor = ColumnTransformer(
                transformers=[
                    ("num_standard_pipeline", num_standard_pipeline, standard_scaler_columns),
                    ("num_minmax_pipeline", num_minmax_pipeline, minmax_scaler_columns),
                    ("log_pipeline", log_pipeline, log_columns),
                    ("cat_pipeline", cat_pipeline, categorical_columns)
                ],
                remainder="drop" # Ensures extra columns don't sneak in
            )

            return preprocessor

        except Exception as e:
            raise CustomException(e, sys)

    def initiate_data_transformation(self, train_path, test_path):
        try:
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            logging.info("Read train and test data completed")

            # --- TARGETS ---
            target_columns = ["gwp_kg_co2_per_kg", "circularity_index"]
            
            # --- STRATEGY STEP: DROP 9 COLUMNS ---
            drop_columns = [
                "recommended_action", "data_completeness_score", "benchmark_gwp_for_metal",
                "upstream_gwp_kg_co2", "downstream_gwp_kg_co2", "transport_gwp_kg_co2_per_kg",
                "emission_reduction_vs_primary_pct", "carbon_intensity_category", "route_type"
            ]

            # --- STRATEGY STEP: OUTLIER REMOVAL (IQR Method) ---
            # IMPORTANT: Outlier removal only on training data to prevent leakage
            outlier_cols = ["water_use_l_per_kg", "so2_kg_per_kg", "unit_price_usd_per_kg", 
                            "process_temperature_celsius", "waste_generated_kg_per_kg", 
                            "acidification_potential", "abiotic_depletion_score"]

            def remove_outliers_iqr(df, cols):
                for col in cols:
                    if col in df.columns:
                        Q1 = df[col].quantile(0.25)
                        Q3 = df[col].quantile(0.75)
                        IQR = Q3 - Q1
                        lower_bound = Q1 - 1.5 * IQR
                        upper_bound = Q3 + 1.5 * IQR
                        df = df[(df[col] >= lower_bound) & (df[col] <= upper_bound)]
                return df

            logging.info(f"Shape before outlier removal: {train_df.shape}")
            train_df = remove_outliers_iqr(train_df, outlier_cols)
            train_df.reset_index(drop=True, inplace=True)
            logging.info(f"Shape after outlier removal: {train_df.shape}")

            # Define features to drop (Ensure we don't try to drop columns that don't exist)
            all_cols_to_drop = [col for col in (drop_columns + target_columns) if col in train_df.columns]

            # Separate Features and Targets
            input_feature_train_df = train_df.drop(columns=all_cols_to_drop)
            target_feature_train_df = train_df[target_columns].copy()

            input_feature_test_df = test_df.drop(columns=all_cols_to_drop)
            target_feature_test_df = test_df[target_columns].copy()

            # Apply log transformation to Target GWP (as per strategy)
            target_feature_train_df["gwp_kg_co2_per_kg"] = np.log1p(target_feature_train_df["gwp_kg_co2_per_kg"])
            target_feature_test_df["gwp_kg_co2_per_kg"] = np.log1p(target_feature_test_df["gwp_kg_co2_per_kg"])

            logging.info("Applying preprocessing object")
            preprocessing_obj = self.get_data_transformer_object()

            # Transformation
            input_feature_train_arr = preprocessing_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr = preprocessing_obj.transform(input_feature_test_df)

            # Convert target DataFrames to numpy arrays
            target_train_arr = target_feature_train_df.to_numpy()
            target_test_arr = target_feature_test_df.to_numpy()

            # Final Concatenation
            # Using np.c_ is a shorthand for concatenation along axis 1
            train_arr = np.c_[input_feature_train_arr, target_train_arr]
            test_arr = np.c_[input_feature_test_arr, target_test_arr]

            save_object(
                file_path=self.data_transformation_config.preprocessor_obj_file_path,
                obj=preprocessing_obj
            )

            logging.info("Transformation and save successful")
            return train_arr, test_arr, self.data_transformation_config.preprocessor_obj_file_path

        except Exception as e:
            raise CustomException(e, sys)