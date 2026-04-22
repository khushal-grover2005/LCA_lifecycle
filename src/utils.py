import os
import sys
import pickle
import numpy as np
import pandas as pd
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

from src.exception import CustomException
from src.logger import logging

def save_object(file_path, obj):
    """
    Saves a python object (like a model or preprocessor) as a pickle file.
    """
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)

        with open(file_path, "wb") as file_obj:
            pickle.dump(obj, file_obj)
            
        logging.info(f"Object saved successfully at {file_path}")

    except Exception as e:
        raise CustomException(e, sys)

def load_object(file_path):
    """
    Loads a saved pickle object.
    """
    try:
        with open(file_path, "rb") as file_obj:
            return pickle.dump(file_obj)
            
    except Exception as e:
        raise CustomException(e, sys)

def evaluate_models(X_train, y_train, X_test, y_test, models, params):
    """
    Trains multiple models, performs hyperparameter tuning, and returns 
    an evaluation report based on R2 Score.
    """
    try:
        report = {}

        for i in range(len(list(models))):
            model_name = list(models.keys())[i]
            model = list(models.values())[i]
            para = params[model_name]

            # Optional: Hyperparameter tuning using GridSearchCV/RandomizedSearchCV can be added here
            # For now, let's stick to the base training for the first iteration
            
            model.fit(X_train, y_train) # Training model

            y_train_pred = model.predict(X_train)
            y_test_pred = model.predict(X_test)

            # We use R2 Score as the primary metric for our targets
            train_model_score = r2_score(y_train, y_train_pred)
            test_model_score = r2_score(y_test, y_test_pred)

            report[model_name] = test_model_score
            
            logging.info(f"Model: {model_name} | R2 Score: {test_model_score}")

        return report

    except Exception as e:
        raise CustomException(e, sys)