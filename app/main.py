
from contextlib import asynccontextmanager

import mlflow
import mlflow.sklearn
import pandas as pd

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field


MODEL_NAME = "atm-cash-prediction"
MODEL_ALIAS = "champion"

FEATURE_COLUMNS = [
    "nearby_shops",
    "days_to_salary",
    "footfall_index",
    "distance_to_market_km",
    "avg_ticket_rs",
    "other_atms_1km",
    "atm_type",
    "area"
]


@asynccontextmanager
async def lifespan(app: FastAPI):

    mlflow.set_tracking_uri("sqlite:///mlflow.db")

    model_uri = f"models:/{MODEL_NAME}@{MODEL_ALIAS}"

    app.state.model = mlflow.sklearn.load_model(model_uri)

    yield

    app.state.model = None


app = FastAPI(
    title="ATM Cash Prediction API",
    description="MLflow-powered ATM cash withdrawal prediction",
    version="1.0.0",
    lifespan=lifespan
)


class PredictionRequest(BaseModel):

    nearby_shops: int = Field(ge=0)
    days_to_salary: int
    footfall_index: float
    distance_to_market_km: float = Field(ge=0)
    avg_ticket_rs: float = Field(ge=0)
    other_atms_1km: int = Field(ge=0)

    atm_type: str
    area: str


@app.get("/")
def home():

    return {
        "message": "ATM Cash Prediction API is running",
        "model": MODEL_NAME,
        "alias": MODEL_ALIAS
    }


@app.get("/health")
def health():

    return {
        "status": "healthy"
    }


@app.post("/predict")
def predict(request: PredictionRequest):

    try:

        input_data = pd.DataFrame(
            [request.model_dump()],
            columns=FEATURE_COLUMNS
        )

        prediction = app.state.model.predict(input_data)

        predicted_cash = float(prediction[0])

        return {
            "predicted_withdrawal_thousand_rs": round(
                predicted_cash, 2
            ),
            "unit": "thousand INR"
        }

    except Exception as e:

        raise HTTPException(
            status_code=500,
            detail=f"Prediction failed: {str(e)}"
        )
