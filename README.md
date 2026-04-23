# LCApredictor - Life Cycle Assessment Prediction System

## Project Overview

**LCApredictor** is a machine learning-based system designed to predict environmental impact metrics for metal production processes. It focuses on two primary targets:

1. **GWP (Global Warming Potential)**: `gwp_kg_co2_per_kg` - Carbon footprint per kilogram of metal produced
2. **Circularity Index**: `circularity_index` - Measure of product circularity and recycling potential

The system is built using scikit-learn for machine learning, Flask for the REST API backend, and includes comprehensive data preprocessing pipelines.

---

## Project Structure

```
LCApredictor/
│
├── app.py                          # Flask backend application & REST API endpoint
├── README.md                        # This file
├── requirements.txt                 # Python dependencies
├── setup.py                         # Package setup configuration
│
├── artifacts/                       # Trained models and preprocessor
│   ├── data.csv                     # Raw dataset copy
│   ├── train.csv                    # Training dataset (80%)
│   ├── test.csv                     # Test dataset (20%)
│   ├── preprocessor.pkl             # Fitted data preprocessor
│   ├── gwp_model.pkl                # Trained GWP prediction model
│   └── circularity_model.pkl        # Trained Circularity Index model
│
├── data/                            # Raw data directory
│   └── lca_metals_final.csv         # Original 42-column LCA dataset
│
├── logs/                            # Application logs
│   └── {timestamp}.log              # Timestamped log files
│
├── notebook/                        # Jupyter notebooks
│   └── EDA.ipynb                    # Exploratory Data Analysis notebook
│
└── src/                             # Source code
    ├── __init__.py
    ├── exception.py                 # Custom exception handling
    ├── logger.py                    # Logging configuration
    ├── utils.py                     # Utility functions (save/load objects)
    │
    ├── components/                  # Data processing and model training
    │   ├── __init__.py
    │   ├── dataingestion.py         # Data loading & train-test split
    │   ├── datatransformation.py    # Feature engineering & preprocessing
    │   └── modeltrainer.py          # Model training with hyperparameter tuning
    │
    └── pipeline/                    # Orchestration pipelines
        ├── __init__.py
        ├── trainpipeline.py         # End-to-end training pipeline
        └── predictpipeline.py       # Inference pipeline for predictions
```

---

## Key Features

### 📊 Data Processing
- **Data Ingestion**: Loads LCA metals dataset and performs 80-20 train-test split
- **Outlier Removal**: IQR-based outlier detection and removal on training data
- **Feature Engineering**: Specialized preprocessing for different feature types:
  - **Categorical Features** (11): OneHotEncoding + StandardScaling
  - **Numerical Features**:
    - Log-transformed features (3): log1p transformation + StandardScaling
    - Standard scaling features (3): Mean centering + variance scaling
    - MinMax scaling features (7): 0-1 normalization

### 🤖 Machine Learning Models
The system evaluates 7 different regression models per target:
1. **Linear Regression** - Baseline linear model
2. **Ridge** - L2 regularized regression
3. **Lasso** - L1 regularized regression
4. **ElasticNet** - Combined L1 & L2 regularization
5. **Random Forest** - Ensemble tree-based model
6. **Gradient Boosting** - Sequential ensemble learning
7. **SVR** - Support Vector Regression

### ⚙️ Hyperparameter Tuning
- Uses **RandomizedSearchCV** for efficient hyperparameter optimization
- Cross-validation: 3-fold CV (faster training)
- Grid search iterations: 5 per model
- Metrics: R² Score, RMSE, MAE

### 🎯 Dual Targets
- **Two separate trained models** for GWP and Circularity Index
- Independent predictions for each target
- Both models saved in artifacts directory

---

## Installation

### Prerequisites
- Python 3.8+
- pip or conda package manager

### Setup Instructions

1. **Clone the repository** (if from git):
   ```bash
   git clone <repository-url>
   cd LCApredictor
   ```

2. **Create a virtual environment** (recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Verify installation**:
   ```bash
   python -c "from src.components.modeltrainer import ModelTrainer; print('Installation successful!')"
   ```

---

## Usage

### 1. Training the Models

**Run the complete training pipeline**:
```bash
python app.py
```

This will:
- Load and split the data from `data/lca_metals_final.csv`
- Perform data transformation and outlier removal
- Train all 7 models for both targets (GWP and Circularity Index)
- Perform hyperparameter tuning on each model
- Select and save the best model for each target
- Print evaluation metrics to console and logs

**Training Output** includes:
```
================================================================================
TRAINING MODELS FOR TARGET: GWP_KG_CO2_PER_KG
================================================================================
Model                Train R²      Test R²      Test RMSE    Test MAE
Linear Regression    1.0000        1.0000       0.0012       0.0008
Ridge                1.0000        1.0000       0.0015       0.0010
... (7 models total)

BEST MODEL: Linear Regression
Test R² Score (Accuracy): 1.0000
================================================================================
```

### 2. Making Predictions

**Via Flask API** (recommended for production):
```bash
python app.py  # Starts Flask server on http://localhost:5000
```

**Make a prediction request** (e.g., using curl or Postman):
```bash
curl -X POST http://localhost:5000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "metal": "Nickel",
    "production_route": "Primary - Sulphide Ore",
    "region": "Philippines",
    "energy_mix": "Renewable Grid",
    "transport_mode": "Sea Freight",
    "transport_distance_km": 4340.8,
    "energy_mj_per_kg": 181.39,
    "typical_lifespan_years": 16,
    "recycled_content_pct": 1.33,
    "eol_recovery_pct": 59.32,
    "application": "Electroplating",
    "end_of_life_scenario": "Remanufacturing",
    "pathway_type": "Hybrid",
    "circular_flow_type": "Closed Loop",
    "life_cycle_stage": "Extraction",
    "co2_capture_used": "No",
    "year": 2024,
    "facility_age_years": 6,
    "batch_size_tonnes": 363.1,
    "global_recycling_rate_pct": 61.91,
    "reuse_potential_score": 4,
    "material_efficiency_score": 0.913,
    "human_toxicity_score": 0.7654,
    "eutrophication_potential": 0.00408
  }'
```

**Response Format**:
```json
{
  "gwp_prediction": 2.1234,
  "circularity_index_prediction": 42.56,
  "status": "success"
}
```

---

## File Descriptions

### Source Code Files

#### `src/components/dataingestion.py`
- **Purpose**: Load raw data and create train-test split
- **Key Class**: `DataIngestion`
- **Output**: `artifacts/train.csv`, `artifacts/test.csv`

#### `src/components/datatransformation.py`
- **Purpose**: Feature engineering and data preprocessing
- **Key Class**: `DataTransformation`
- **Process**:
  - Reads train/test CSV files
  - Removes outliers using IQR method (training data only)
  - Applies log transformation to target (gwp_kg_co2_per_kg)
  - Creates ColumnTransformer for feature preprocessing
  - Outputs: Transformed arrays + fitted preprocessor

#### `src/components/modeltrainer.py`
- **Purpose**: Train and evaluate multiple models with hyperparameter tuning
- **Key Class**: `ModelTrainer`
- **Methods**: `initiate_model_trainer(train_array, test_array, target_name)`
- **Outputs**: Best trained model + evaluation metrics

#### `src/pipeline/trainpipeline.py`
- **Purpose**: Orchestrate the complete training workflow
- **Key Class**: `TrainingPipeline`
- **Flow**: DataIngestion → DataTransformation → ModelTrainer (GWP) → ModelTrainer (Circularity)

#### `src/pipeline/predictpipeline.py`
- **Purpose**: Load models and make predictions on new data
- **Key Classes**: `CustomData`, `PredictPipeline`
- **Input**: Individual feature values
- **Output**: GWP and Circularity Index predictions

#### `src/exception.py`
- **Purpose**: Custom exception handling with detailed error messages
- **Features**: Captures file name, line number, and error message

#### `src/logger.py`
- **Purpose**: Centralized logging configuration
- **Output**: Timestamped log files in `logs/` directory

#### `src/utils.py`
- **Purpose**: Utility functions
- **Functions**: `save_object()`, `load_object()`, `evaluate_models()`

---

## Data Features (Input Variables)

### Categorical Features (11)
- `metal` - Type of metal (Nickel, Lithium, Silver, etc.)
- `production_route` - Manufacturing method
- `region` - Geographic region of production
- `application` - End-use application
- `energy_mix` - Energy source grid mix
- `transport_mode` - Shipping/transport method
- `end_of_life_scenario` - Recycling/disposal approach
- `pathway_type` - Production pathway classification
- `circular_flow_type` - Circularity type
- `life_cycle_stage` - LCA stage (Extraction, Manufacturing, etc.)
- `co2_capture_used` - Whether CO2 capture is employed

### Numerical Features (31)
**Log-transformed** (3):
- `energy_mj_per_kg`, `transport_distance_km`, `batch_size_tonnes`

**Standard Scaled** (3):
- `year`, `typical_lifespan_years`, `facility_age_years`

**MinMax Scaled** (7):
- `global_recycling_rate_pct`, `recycled_content_pct`, `eol_recovery_pct`, `reuse_potential_score`, `material_efficiency_score`, `human_toxicity_score`, `eutrophication_potential`

### Target Variables
- **`gwp_kg_co2_per_kg`** - Carbon footprint (kg CO2-eq per kg of metal) [Log-transformed during preprocessing]
- **`circularity_index`** - Circularity score (0-100)

---

## Model Artifacts

All trained artifacts are stored in the `artifacts/` directory:

| File | Description | Size | Format |
|------|-------------|------|--------|
| `preprocessor.pkl` | Fitted ColumnTransformer for feature preprocessing | ~MB | Pickle |
| `gwp_model.pkl` | Best trained model for GWP prediction | ~MB | Pickle |
| `circularity_model.pkl` | Best trained model for Circularity Index | ~MB | Pickle |
| `train.csv` | Training dataset (80% of data) | ~MB | CSV |
| `test.csv` | Test dataset (20% of data) | ~MB | CSV |

---

## Logging

Logs are created in the `logs/` directory with timestamps:
- Format: `MM_DD_YYYY_HH_MM_SS.log/MM_DD_YYYY_HH_MM_SS.log`
- Example: `04_22_2026_22_01_49.log/04_22_2026_22_01_49.log`

### Log Levels
- **INFO**: General process information (data loaded, model trained, etc.)
- **WARNING**: Non-critical issues (unknown categories in transform, etc.)
- **ERROR**: Critical failures and exceptions

---

## Model Performance

Typical performance metrics (may vary based on hyperparameter tuning results):

### GWP Prediction Model
- **Best Model**: Linear Regression / Ridge (generally achieving R² ≈ 1.0)
- **Test R² Score**: ~1.0000 (near-perfect fit on test data)
- **Test RMSE**: Very small (< 0.01)

### Circularity Index Prediction Model
- **Best Model**: Random Forest / Gradient Boosting (generally)
- **Test R² Score**: ~0.95-0.99
- **Test RMSE**: < 5.0 index points

---

## Error Handling

The system includes comprehensive error handling:
- **CustomException**: Provides detailed error messages with file name and line number
- **Try-Except Blocks**: Wrapped around all critical operations
- **Logging**: All errors logged for debugging

Example error output:
```
error occured in python script name [src/components/modeltrainer.py] 
line number [87] error message[Division by zero in model evaluation]
```

---

## Requirements

See `requirements.txt` for complete dependencies. Main packages:
- `scikit-learn>=1.0.0` - Machine learning models
- `pandas>=1.3.0` - Data manipulation
- `numpy>=1.21.0` - Numerical computing
- `flask>=2.0.0` - Web framework
- `flask-cors>=3.0.10` - CORS support

---

## Future Enhancements

- [ ] Add cross-validation results reporting
- [ ] Implement deep learning models (Neural Networks)
- [ ] Add feature importance analysis
- [ ] Create web UI for predictions
- [ ] Add batch prediction endpoint
- [ ] Implement model versioning
- [ ] Add explainability features (SHAP values)
- [ ] Containerization with Docker

---

## Troubleshooting

### Issue: ModuleNotFoundError
**Solution**: Ensure all dependencies are installed:
```bash
pip install -r requirements.txt
```

### Issue: FileNotFoundError for data
**Solution**: Verify `data/lca_metals_final.csv` exists and has correct permissions

### Issue: Models not found during prediction
**Solution**: Run training first to generate model artifacts:
```bash
python app.py
```

### Issue: Port 5000 already in use
**Solution**: Change the port in app.py:
```python
app.run(debug=True, port=5001)
```

---

## Author & Contact

For questions or contributions, please contact the development team.

---

## License

This project is proprietary. All rights reserved.

---

**Last Updated**: April 2026  
**Version**: 1.0.0
