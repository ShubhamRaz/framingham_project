# Framingham Project

## Table of Contents
- [Overview](#overview)
- [Project Features](#project-features)
- [Data Pipeline](#data-pipeline)
- [Model Training & Evaluation](#model-training--evaluation)
- [Web Application](#web-application)
- [API Usage](#api-usage)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Usage](#usage)
- [Dependencies](#dependencies)
- [License](#license)
- [References & Acknowledgments](#references--acknowledgments)

## Overview
This project predicts the 10-year risk of coronary heart disease (CHD) using the Framingham Heart Study dataset. It includes data cleaning, feature engineering, model training, hyperparameter tuning, and a Flask web application for risk assessment.

## Project Features
- **Data Cleaning:** Handles missing values, removes irrelevant columns, and prepares features for modeling.
- **Exploratory Data Analysis:** Visualizes distributions and relationships using seaborn and matplotlib.
- **Model Training:** Trains multiple classifiers (KNN, Decision Tree, AdaBoost, Gradient Boosting, XGBoost) and compares their performance.
- **Hyperparameter Tuning:** Uses RandomizedSearchCV for AdaBoost and Gradient Boosting.
- **Model Persistence:** Saves the best model and scaler for deployment.
- **Web Application:** User-friendly interface for risk prediction, with input validation and recommendations.
- **REST API:** Provides a `/api/predict` endpoint for programmatic access.

## Data Pipeline
- **Raw Data:** `data/framingham.csv` (original dataset)
- **Cleaning Steps:**
  - Drop `education` column
  - Impute missing values (e.g., median for `cigsPerDay`, `BPMeds`)
  - Save cleaned data as `data/filtered.csv`
- **Feature Selection:**
  - Features: `male`, `age`, `currentSmoker`, `cigsPerDay`, `BPMeds`, `prevalentStroke`, `prevalentHyp`, `diabetes`, `totChol`, `sysBP`, `diaBP`, `BMI`, `heartRate`, `glucose`
  - Target: `TenYearCHD`

## Model Training & Evaluation
- **Notebooks:**
  - `jupyterNB/DataCleaning.ipynb`: Data cleaning and preprocessing
  - `jupyterNB/ModelTraining.ipynb`: Model training, evaluation, and tuning
- **Models Used:**
  - KNeighborsClassifier
  - DecisionTreeClassifier
  - AdaBoostClassifier
  - GradientBoostingClassifier
  - XGBClassifier
- **Metrics:** Accuracy, Precision, Recall, F1 Score, ROC-AUC
- **Best Model:** Gradient Boosting (with tuned hyperparameters)
- **Persistence:**
  - Trained model: `trained_model/model.pkl`
  - Scaler: `trained_model/scaler.pkl`

## Web Application
- **File:** `app.py`
- **Template:** `templates/index.html`
- **Features:**
  - Input form for all model features
  - Risk threshold adjustment
  - Displays risk level, probability, and recommendations
  - Error handling and input validation
  - Disclaimer for clinical use

## API Usage
- **Endpoint:** `/api/predict`
- **Method:** POST
- **Request Body:** JSON with all model features and optional `threshold`
- **Response:**
  - `probability`: Predicted probability of CHD
  - `label`: 1 (high risk) or 0 (low risk)
- **Example:**
  ```bash
  curl -X POST http://127.0.0.1:5000/api/predict \
    -H "Content-Type: application/json" \
    -d '{"male":1,"age":55,"currentSmoker":0,"cigsPerDay":0,"BPMeds":0,"prevalentStroke":0,"prevalentHyp":1,"diabetes":0,"totChol":220,"sysBP":130,"diaBP":80,"BMI":27,"heartRate":75,"glucose":90,"threshold":0.5}'
  ```

## Project Structure
```
framingham_project/
├── app.py                # Flask application for serving predictions
├── README.md             # Project documentation
├── data/                 # Dataset folder
│   ├── filtered.csv      # Preprocessed dataset
│   └── framingham.csv    # Raw dataset
├── jupyterNB/            # Jupyter notebooks for data exploration and modeling
│   ├── DataCleaning.ipynb
│   └── ModelTraining.ipynb
├── models/               # Saved machine learning models
│   └── chd_pipeline.joblib
├── templates/            # HTML templates for the web app
│   └── index.html
├── trained_model/        # Final trained model and scaler
│   └── model.pkl
│   └── scaler.pkl
```

## Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/ShubhamRaz/framingham_project.git
   ```
2. Navigate to the project directory:
   ```bash
   cd framingham_project
   ```
3. Create and activate a virtual environment:
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   ```
4. Install the required dependencies (create `requirements.txt` if missing):
   ```bash
   pip install pandas scikit-learn seaborn matplotlib flask xgboost
   ```

## Usage
1. **Data Cleaning and Model Training:**
   - Open the Jupyter notebooks in the `jupyterNB/` directory to explore the data cleaning and model training process.
2. **Run the Flask App:**
   - Start the Flask application to serve predictions:
     ```bash
     python app.py
     ```
   - Open your browser and navigate to `http://127.0.0.1:5000` to interact with the web app.
3. **API Access:**
   - Use the `/api/predict` endpoint for programmatic predictions (see example above).

## Dependencies
- Python 3.8+
- pandas
- scikit-learn
- seaborn
- matplotlib
- Flask
- xgboost

## License
This project is licensed under the MIT License. See the LICENSE file for details.

## References & Acknowledgments
- Framingham Heart Study: [NIH Resource](https://www.framinghamheartstudy.org/)
- Open-source libraries: pandas, scikit-learn, seaborn, matplotlib, Flask, xgboost
- UI icons: [Font Awesome](https://fontawesome.com/)
- UI fonts: [Google Fonts](https://fonts.google.com/)