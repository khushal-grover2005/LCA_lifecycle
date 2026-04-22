import sys
import os
import numpy as np
import pandas as pd

from src.components.dataingestion import DataIngestion
from src.components.datatransformation import DataTransformation
from src.components.modeltrainer import ModelTrainer
from src.exception import CustomException
from src.logger import logging

class TrainingPipeline:
    def start_training(self):
        try:
            # Step 1: Data Ingestion
            logging.info("Starting data ingestion...")
            data_ingestion = DataIngestion()
            train_data_path, test_data_path = data_ingestion.initiate_data_ingestion()
            
            logging.info("Data ingestion completed. Train and test data paths obtained.")
            print(f"\nTrain data path: {train_data_path}")
            print(f"Test data path: {test_data_path}\n")

            # Step 2: Data Transformation
            logging.info("Starting data transformation...")
            data_transformation = DataTransformation()
            train_arr, test_arr, preprocessor_path = data_transformation.initiate_data_transformation(
                train_data_path,
                test_data_path
            )
            
            logging.info("Data Transformation Successful.")
            logging.info(f"Train array shape: {train_arr.shape}")
            logging.info(f"Test array shape: {test_arr.shape}")

            # Step 3: Model Training
            logging.info("Starting model trainer for target: gwp_kg_co2_per_kg...")
            print("\n" + "="*80)
            print("TRAINING MODELS FOR TARGET: GWP_KG_CO2_PER_KG")
            print("="*80)
            
            model_trainer = ModelTrainer()
            best_model_gwp, best_model_name_gwp = model_trainer.initiate_model_trainer(
                train_arr, 
                test_arr, 
                target_column_index=0  # gwp_kg_co2_per_kg is at index 0
            )
            
            logging.info(f"Best model for GWP selected: {best_model_name_gwp}")
            
            # Optional: Train model for second target (circularity_index)
            logging.info("Starting model trainer for target: circularity_index...")
            print("\n" + "="*80)
            print("TRAINING MODELS FOR TARGET: CIRCULARITY_INDEX")
            print("="*80)
            
            best_model_circularity, best_model_name_circularity = model_trainer.initiate_model_trainer(
                train_arr,
                test_arr,
                target_column_index=1  # circularity_index is at index 1
            )
            
            logging.info(f"Best model for Circularity Index selected: {best_model_name_circularity}")
            
            print("\n" + "="*80)
            print("TRAINING PIPELINE COMPLETED SUCCESSFULLY!")
            print("="*80)
            print(f"Best Model for GWP: {best_model_name_gwp}")
            print(f"Best Model for Circularity Index: {best_model_name_circularity}")
            print("="*80 + "\n")
            
            return {
                "gwp_model": best_model_gwp,
                "gwp_model_name": best_model_name_gwp,
                "circularity_model": best_model_circularity,
                "circularity_model_name": best_model_name_circularity
            }

        except Exception as e:
            raise CustomException(e, sys)

if __name__ == "__main__":
    pipeline = TrainingPipeline()
    results = pipeline.start_training()
