---
title: LCA Predictor Backend
emoji: ♻️
colorFrom: green
colorTo: emerald
sdk: docker
app_port: 7860
pinned: false
---

# 🌍 AI-Driven Life Cycle Assessment (LCA) & Circularity Predictor

This repository houses the **Inference Engine** for an AI-powered platform designed to predict the environmental impact and circularity of metallurgical processes. Developed as part of a 14-day sprint, this system utilizes machine learning to transform complex industrial data into actionable sustainability insights.

## 🚀 Key Features
* **Dual-Target Prediction:** Predicts Global Warming Potential (GWP) and Circularity Index simultaneously.
* **Sankey Diagram Logic:** Generates source-target-value JSON flows to visualize carbon footprints from extraction to finished products.
* **Automated ML Pipeline:** Includes modular components for Data Ingestion, Transformation (OHE, Log Scaling, Outlier Handling), and Model Training.
* **RESTful API:** Flask-based backend ready for integration with React/Next.js frontends.

## 🛠️ Tech Stack
* **Language:** Python 3.9
* **ML Frameworks:** Scikit-Learn, XGBoost, CatBoost
* **API:** Flask & Flask-CORS
* **DevOps:** Docker, Hugging Face Spaces
* **Data Handling:** Pandas, NumPy, Dill

## 📊 Project Structure
```text
LCAPredictor/
├── artifacts/           # Saved model pickles and preprocessor
├── data/                # Raw and processed datasets
├── src/
│   ├── components/      # Data Ingestion, Transformation, Model Trainer
│   ├── pipeline/        # Training and Prediction Pipelines
│   ├── logger.py        # Custom logging module
│   └── exception.py     # Custom exception handling
├── app.py               # Flask API entry point
├── Dockerfile           # Container configuration
└── requirements.txt     # Project dependencies