import os
import sys

from fastapi import FastAPI
from pydantic import BaseModel

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"))

from agents.orchestrator import run_pipeline

app = FastAPI()

class DisputeCase(BaseModel):

    reason_code: str
    product_category: str
    payment_method: str
    order_value: float
    time_to_dispute_days: int
    days_remaining_to_respond: int
    delivery_confirmed: str
    communication_logs_present: bool
    device_ip_match_score: float
    listing_accuracy_score: float
    customer_prior_disputes: int
    customer_account_age_days: int


@app.get("/")
def root():
    return {"status":"running"}

@app.post("/evaluate-case")
def evaluate_case(case:DisputeCase):

    result = run_pipeline(case.model_dump())

    if result.get("error"):
        return {"error":case["error"]}

    return {
        "prediction": result["prediction_result"],
        "decision": result["decision"],
    }