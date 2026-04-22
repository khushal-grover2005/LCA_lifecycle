import os
import sys
from pathlib import Path
project_root = str(Path(__file__).parent.parent.parent)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.exception import CustomException
from src.logger import logging
from src.pipeline.trainpipeline import TrainingPipeline

if __name__ == "__main__":
    try:
        # Execute the training pipeline
        pipeline = TrainingPipeline()
        results = pipeline.start_training()
        logging.info("Application execution completed successfully")

    except Exception as e:
        raise CustomException(e, sys)