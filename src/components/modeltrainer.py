import sys
import os
import numpy as np
import pandas as pd
from dataclasses import dataclass

from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object

@dataclass
class ModelTrainerConfig:
    # Split into two paths so both targets are preserved
    gwp_model_file_path: str = os.path.join("artifacts", "gwp_model.pkl")
    circularity_model_file_path: str = os.path.join("artifacts", "circularity_model.pkl")

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def initiate_model_trainer(self, train_array, test_array, target_name):
        """
        Train and evaluate multiple regression models with hyperparameter tuning.
        target_name: "gwp" or "circularity"
        """
        try:
            # gwp is at index 0, circularity is at index 1 (the last two columns)
            target_column_index = 0 if target_name == "gwp" else 1
            
            logging.info(f"Splitting training and test input and target for: {target_name}")
            
            X_train, y_train, X_test, y_test = (
                train_array[:, :-2],  
                train_array[:, -2 + target_column_index],  
                test_array[:, :-2],  
                test_array[:, -2 + target_column_index]  
            )

            models = {
                "Linear Regression": LinearRegression(),
                "Ridge": Ridge(),
                "Lasso": Lasso(),
                "ElasticNet": ElasticNet(),
                "Random Forest": RandomForestRegressor(random_state=42, n_jobs=-1),
                "Gradient Boosting": GradientBoostingRegressor(random_state=42),
                "SVR": SVR()
            }

            params = {
                "Linear Regression": {},
                "Ridge": {"alpha": [0.1, 1, 10, 100]},
                "Lasso": {"alpha": [0.001, 0.01, 0.1, 1]},
                "ElasticNet": {"alpha": [0.01, 0.1, 1], "l1_ratio": [0.5, 0.8]},
                "Random Forest": {
                    "n_estimators": [100, 200],
                    "max_depth": [15, 25],
                    "min_samples_split": [5, 10]
                },
                "Gradient Boosting": {
                    "n_estimators": [100, 200],
                    "learning_rate": [0.05, 0.1],
                    "max_depth": [5, 7]
                },
                "SVR": {"kernel": ["rbf"], "C": [1, 10], "epsilon": [0.1]}
            }

            model_report = {}
            best_models = {}

            print(f"\n--- Training Models for {target_name.upper()} ---")

            for model_name, model in models.items():
                try:
                    if len(params[model_name]) == 0:
                        model.fit(X_train, y_train)
                        best_model = model
                        best_params = {}
                    else:
                        grid_search = RandomizedSearchCV(
                            estimator=model,
                            param_distributions=params[model_name],
                            n_iter=5,
                            cv=3,
                            n_jobs=-1,
                            random_state=42,
                            verbose=0
                        )
                        grid_search.fit(X_train, y_train)
                        best_model = grid_search.best_estimator_
                        best_params = grid_search.best_params_

                    y_test_pred = best_model.predict(X_test)
                    test_r2 = r2_score(y_test, y_test_pred)

                    model_report[model_name] = {
                        "test_r2": test_r2,
                        "best_params": best_params,
                        "metrics": {
                            "rmse": np.sqrt(mean_squared_error(y_test, y_test_pred)),
                            "mae": mean_absolute_error(y_test, y_test_pred)
                        }
                    }
                    best_models[model_name] = best_model
                    print(f"{model_name}: R² = {test_r2:.4f}")

                except Exception as e:
                    logging.error(f"Error training {model_name}: {str(e)}")
                    continue

            best_model_name = max(model_report, key=lambda x: model_report[x]["test_r2"])
            best_model_obj = best_models[best_model_name]

            # Set path based on target
            save_path = self.model_trainer_config.gwp_model_file_path if target_name == "gwp" else self.model_trainer_config.circularity_model_file_path

            save_object(file_path=save_path, obj=best_model_obj)
            logging.info(f"Best model for {target_name} ({best_model_name}) saved at {save_path}")

            return best_model_obj, best_model_name

        except Exception as e:
            raise CustomException(e, sys)