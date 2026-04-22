import sys
import os
import numpy as np
import pandas as pd
from dataclasses import dataclass

from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object

@dataclass
class ModelTrainerConfig:
    trained_model_file_path: str = os.path.join("artifacts", "model.pkl")

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def initiate_model_trainer(self, train_array, test_array, target_column_index=0):
        """
        Train and evaluate multiple regression models with hyperparameter tuning.
        
        Args:
            train_array: Training data (features + target)
            test_array: Test data (features + target)
            target_column_index: Index of target column (default: 0 for GWP, 1 for circularity_index)
        """
        try:
            logging.info("Splitting training and test input and target")
            X_train, y_train, X_test, y_test = (
                train_array[:, :-2],  # All columns except last 2 (targets)
                train_array[:, target_column_index],  # Select target column
                test_array[:, :-2],  # All columns except last 2 (targets)
                test_array[:, target_column_index]  # Select target column
            )

            # Dictionary of models to try
            models = {
                "Linear Regression": LinearRegression(),
                "Ridge": Ridge(),
                "Lasso": Lasso(),
                "ElasticNet": ElasticNet(),
                "Random Forest": RandomForestRegressor(random_state=42, n_jobs=-1),
                "Gradient Boosting": GradientBoostingRegressor(random_state=42),
                "SVR": SVR()
            }

            # Hyperparameter grids for GridSearchCV
            params = {
                "Linear Regression": {},  # No hyperparameters to tune
                "Ridge": {
                    "alpha": [0.1, 1, 10, 100]
                },
                "Lasso": {
                    "alpha": [0.001, 0.01, 0.1, 1]
                },
                "ElasticNet": {
                    "alpha": [0.01, 0.1, 1],
                    "l1_ratio": [0.5, 0.8]
                },
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
                "SVR": {
                    "kernel": ["rbf"],
                    "C": [1, 10],
                    "epsilon": [0.1]
                }
            }

            model_report = {}
            best_models = {}

            logging.info("Training and hyperparameter tuning all models...")
            print("\n" + "="*80)
            print("MODEL TRAINING AND HYPERPARAMETER TUNING")
            print("="*80 + "\n")

            for model_name, model in models.items():
                logging.info(f"Training {model_name}...")
                print(f"Training {model_name}...", end=" ")
                
                try:
                    # Perform hyperparameter tuning using GridSearchCV
                    if len(params[model_name]) == 0:
                        # If no parameters to tune, just fit the model
                        model.fit(X_train, y_train)
                        best_model = model
                        best_params = {}
                    else:
                        # Use RandomizedSearchCV for models with many parameters (to save time)
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

                    # Make predictions
                    y_train_pred = best_model.predict(X_train)
                    y_test_pred = best_model.predict(X_test)

                    # Calculate metrics
                    train_r2 = r2_score(y_train, y_train_pred)
                    test_r2 = r2_score(y_test, y_test_pred)
                    train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
                    test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
                    train_mae = mean_absolute_error(y_train, y_train_pred)
                    test_mae = mean_absolute_error(y_test, y_test_pred)

                    # Store results
                    model_report[model_name] = {
                        "train_r2": train_r2,
                        "test_r2": test_r2,
                        "train_rmse": train_rmse,
                        "test_rmse": test_rmse,
                        "train_mae": train_mae,
                        "test_mae": test_mae,
                        "best_params": best_params
                    }
                    best_models[model_name] = best_model

                    print(f"✓ Test R² = {test_r2:.4f}")
                    logging.info(f"{model_name} - Train R²: {train_r2:.4f}, Test R²: {test_r2:.4f}")

                except Exception as e:
                    logging.error(f"Error training {model_name}: {str(e)}")
                    print(f"✗ Error")
                    continue

            # Find the best model based on test R2 score
            if not model_report:
                raise CustomException("No models were successfully trained", sys)

            best_model_name = max(model_report, key=lambda x: model_report[x]["test_r2"])
            best_model_metrics = model_report[best_model_name]
            best_model_obj = best_models[best_model_name]

            print("\n" + "="*80)
            print("MODEL EVALUATION RESULTS")
            print("="*80 + "\n")

            # Print all models' performance
            print(f"{'Model':<25} {'Train R²':<12} {'Test R²':<12} {'Test RMSE':<12} {'Test MAE':<12}")
            print("-" * 73)
            for model_name, metrics in model_report.items():
                print(f"{model_name:<25} {metrics['train_r2']:<12.4f} {metrics['test_r2']:<12.4f} {metrics['test_rmse']:<12.4f} {metrics['test_mae']:<12.4f}")

            print("\n" + "="*80)
            print(f"BEST MODEL: {best_model_name}")
            print("="*80)
            print(f"Test R² Score (Accuracy): {best_model_metrics['test_r2']:.4f}")
            print(f"Test RMSE: {best_model_metrics['test_rmse']:.4f}")
            print(f"Test MAE: {best_model_metrics['test_mae']:.4f}")
            print(f"Best Hyperparameters: {best_model_metrics['best_params']}")
            print("="*80 + "\n")

            # Save the best model
            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model_obj
            )

            logging.info(f"Best model {best_model_name} has been saved")

            return best_model_obj, best_model_name

        except Exception as e:
            raise CustomException(e, sys)
