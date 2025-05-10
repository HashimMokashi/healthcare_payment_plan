from fastapi import FastAPI, Request, Form
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse
import pandas as pd
import joblib
import os
print(os.path.exists("app/templates/form.html"))  # should print: True

from app.predictor import generate_payment_plan, preprocessor, kmeans_model, cluster_regressors, X_train_columns

app = FastAPI()
templates = Jinja2Templates(directory="app/templates")

# Load model components
model_data = joblib.load("model_files/payment_plan_predictor.pkl")
preprocessor = model_data['preprocessor']
kmeans_model = model_data['kmeans']
cluster_regressors = model_data['regressors']
X_train_columns = model_data['X_columns']

@app.get("/", response_class=HTMLResponse)
async def form_get(request: Request):
    return templates.TemplateResponse("form.html", {"request": request})

@app.post("/predict", response_class=HTMLResponse)
async def predict(
    request: Request,
    age: int = Form(...),
    gender: str = Form(...),
    location_code: int = Form(...),
    marital_status: str = Form(...),
    annual_income: float = Form(...),
    credit_score: int = Form(...),
    employment_status: str = Form(...),
    insurance_type: str = Form(...),
    insurance_coverage_pct: float = Form(...),
    treatment_type: str = Form(...),
    total_bill: float = Form(...),
    urgency_score: int = Form(...),
    past_defaults: int = Form(...),
    avg_days_late: float = Form(...),
    preferred_payment_method: str = Form(...)
):
    data = pd.DataFrame([{
        "age": age,
        "gender": gender,
        "location_code": location_code,
        "marital_status": marital_status,
        "annual_income": annual_income,
        "credit_score": credit_score,
        "employment_status": employment_status,
        "insurance_type": insurance_type,
        "insurance_coverage_pct": insurance_coverage_pct,
        "treatment_type": treatment_type,
        "total_bill": total_bill,
        "urgency_score": urgency_score,
        "past_defaults": past_defaults,
        "avg_days_late": avg_days_late,
        "preferred_payment_method": preferred_payment_method
    }])
    
    plan = generate_payment_plan(data)
    return templates.TemplateResponse("form.html", {
        "request": request,
        "plan": plan
    })
