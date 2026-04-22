import sys
import os
from src.components.dataingestion import DataIngestion
from src.components.datatransformation import DataTransformation
from src.components.modeltrainer import ModelTrainer
from src.exception import CustomException
from src.logger import logging

class TrainingPipeline:
    def start_training(self):
        try:
            print("\n" + "="*80)
            print("🚀 INITIALIZING LCA SUSTAINABILITY TRAINING PIPELINE")
            print("="*80)

            # Step 1: Data Ingestion
            logging.info("Starting data ingestion...")
            print(f"\n[STEP 1/4]: Ingesting data from source...")
            data_ingestion = DataIngestion()
            train_data_path, test_data_path = data_ingestion.initiate_data_ingestion()
            print(f"✅ Ingestion Complete.")
            print(f"   -> Train Data: {train_data_path}")
            print(f"   -> Test Data:  {test_data_path}")
            
            # Step 2: Data Transformation
            logging.info("Starting data transformation...")
            print(f"\n[STEP 2/4]: Applying Transformations (OHE, Log, Scaling, Outlier Removal)...")
            data_transformation = DataTransformation()
            train_arr, test_arr, preprocessor_path = data_transformation.initiate_data_transformation(
                train_data_path, test_data_path
            )
            print(f"✅ Transformation Complete.")
            print(f"   -> Preprocessor saved at: {preprocessor_path}")
            print(f"   -> Final Feature Matrix Shape: {train_arr.shape}")

            model_trainer = ModelTrainer()

            # Step 3: Model Training for GWP
            logging.info("Training GWP Model...")
            print(f"\n[STEP 3/4]: Training Environmental Engine (Target: GWP)...")
            best_model_gwp, name_gwp = model_trainer.initiate_model_trainer(
                train_arr, test_arr, target_name="gwp"
            )
            print(f"✅ GWP Model Selection: {name_gwp} is the best performer.")

            # Step 4: Model Training for Circularity
            logging.info("Training Circularity Model...")
            print(f"\n[STEP 4/4]: Training Circularity Engine (Target: Circularity Index)...")
            best_model_circ, name_circ = model_trainer.initiate_model_trainer(
                train_arr, test_arr, target_name="circularity"
            )
            print(f"✅ Circularity Model Selection: {name_circ} is the best performer.")

            print("\n" + "="*80)
            print("🏆 PIPELINE EXECUTION SUCCESSFUL")
            print("="*80)
            print(f"📍 Artifacts generated in: {os.path.abspath('artifacts')}")
            print(f"📊 Best GWP Model:         {name_gwp}")
            print(f"📊 Best Circularity Model: {name_circ}")
            print("="*80 + "\n")

            return {
                "gwp_model_name": name_gwp,
                "circ_model_name": name_circ
            }

        except Exception as e:
            print(f"\n❌ Pipeline failed. Check logs/ for details.")
            raise CustomException(e, sys)

if __name__ == "__main__":
    pipeline = TrainingPipeline()
    pipeline.start_training()