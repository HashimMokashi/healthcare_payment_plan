import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.cluster import KMeans
from sklearn.metrics import mean_absolute_error
import os

# === 1. Preprocessing Pipeline ===
def create_preprocessing_pipeline():
    numeric_features = ['age', 'annual_income', 'credit_score', 'insurance_coverage_pct',
                        'total_bill', 'urgency_score', 'past_defaults', 'avg_days_late']
    
    categorical_features = ['gender', 'location_code', 'marital_status', 'employment_status',
                            'insurance_type', 'treatment_type', 'preferred_payment_method']
    
    numeric_transformer = Pipeline(steps=[('scaler', StandardScaler())])
    categorical_transformer = Pipeline(steps=[('onehot', OneHotEncoder(handle_unknown='ignore'))])
    
    preprocessor = ColumnTransformer(transformers=[
        ('numeric', numeric_transformer, numeric_features),
        ('categorical', categorical_transformer, categorical_features)
    ])
    
    return preprocessor

# === 2. Global Variables (to be overwritten at runtime) ===
preprocessor = create_preprocessing_pipeline()
n_clusters = 4
kmeans_model = KMeans(n_clusters=n_clusters, random_state=42)
cluster_regressors = [RandomForestRegressor(random_state=42) for _ in range(n_clusters)]
X_train_columns = None

# === 3. Load Model ===
def load_model():
    global preprocessor, kmeans_model, cluster_regressors, X_train_columns

    if X_train_columns is not None:
        return  # Already loaded

    model_path = os.path.join("model_files", "payment_plan_predictor.pkl")
    model_data = joblib.load(model_path)

    preprocessor = model_data['preprocessor']
    kmeans_model = model_data['kmeans']
    cluster_regressors = model_data['regressors']
    X_train_columns = model_data['X_columns']

# === 4. Fit the Hybrid Model ===
def fit_payment_model(X, y):
    global X_train_columns
    X_train_columns = X.columns.tolist()
    
    X_processed = preprocessor.fit_transform(X)
    kmeans_model.fit(X_processed)
    clusters = kmeans_model.predict(X_processed)
    
    for cluster_id in range(n_clusters):
        mask = (clusters == cluster_id)
        if sum(mask) > 0:
            X_cluster = X_processed[mask]
            y_cluster = y[mask]
            cluster_regressors[cluster_id].fit(X_cluster, y_cluster)

# === 5. Predict Monthly Payment ===
def predict_payment_model(X):
    load_model()  # Ensure model is loaded

    missing_cols = set(X_train_columns) - set(X.columns)
    for col in missing_cols:
        X[col] = 0
    X = X[X_train_columns]
    
    X_processed = preprocessor.transform(X)
    clusters = kmeans_model.predict(X_processed)
    
    y_pred = np.zeros(X.shape[0])
    for i in range(X.shape[0]):
        cluster_id = clusters[i]
        y_pred[i] = cluster_regressors[cluster_id].predict(X_processed[i].reshape(1, -1))[0]
    
    return y_pred

# === 6. Determine Patient Cluster ===
def get_patient_cluster(X):
    load_model()  # Ensure model is loaded

    X = X[X_train_columns]
    X_processed = preprocessor.transform(X)
    return kmeans_model.predict(X_processed)

# === 7. Success Probability (Optional Analytics) ===
def payment_success_probability(y_true, y_pred, X):
    error = np.abs(y_true - y_pred) / y_true
    income = X['annual_income'].values
    bill = X['total_bill'].values
    bill_to_income = bill / income
    failure_prob = 0.2 * error + 0.5 * np.clip(bill_to_income, 0, 0.5)
    success_prob = 1 - failure_prob
    return np.mean(success_prob)

# === 8. Main Prediction Interface ===
def generate_payment_plan(patient_data):
    load_model()  # Ensure model is loaded

    monthly_amount = predict_payment_model(patient_data)[0]
    
    total_bill = patient_data['total_bill'].values[0]
    insurance_coverage = patient_data['insurance_coverage_pct'].values[0] / 100
    adjusted_bill = total_bill * (1 - insurance_coverage)
    
    duration = np.ceil(adjusted_bill / monthly_amount)
    monthly_amount = round(monthly_amount / 5) * 5
    final_payment = adjusted_bill - (monthly_amount * (duration - 1))
    
    cluster = get_patient_cluster(patient_data)[0]
    
    plan = {
        'total_bill': total_bill,
        'insurance_coverage': insurance_coverage * 100,
        'adjusted_bill': adjusted_bill,
        'monthly_payment': monthly_amount,
        'duration_months': int(duration),
        'final_payment': final_payment,
        'patient_segment': f"Cluster {cluster}",
        'payment_schedule': []
    }
    
    for month in range(1, int(duration) + 1):
        if month < duration:
            amount = monthly_amount
        else:
            amount = final_payment
        
        plan['payment_schedule'].append({
            'month': month,
            'amount': amount,
            'cumulative_paid': min(month * monthly_amount, adjusted_bill)
        })
    
    return plan
