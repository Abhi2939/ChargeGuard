import os
import sys
import joblib

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.abspath(os.path.join(BASE_DIR, ".."))

if SRC_DIR not in sys.path:
    sys.path.insert(0,SRC_DIR)

from model.explain import load_raw_model,generate_shap,explain_case
from agents.ingestion import ingest_case

ARTIFACTS_DIR = os.path.join(SRC_DIR, "model", "artifacts")

_calibrated_model = None
_raw_model = None
_explainer = None
_feature_columns = None

def _load_models():

    global _calibrated_model,_raw_model,_explainer,_feature_columns

    if _calibrated_model is None:
        _calibrated_model = joblib.load(os.path.join(ARTIFACTS_DIR,"calibrated_model.pkl"))
        _raw_model = load_raw_model()
        _explainer = generate_shap(_raw_model)
        _feature_columns = joblib.load(os.path.join(ARTIFACTS_DIR, "feature_columns.pkl"))

def predict_case(case:dict,top_n_explanation: int = 4) -> dict:

    _load_models()

    case_encoded = ingest_case(case,feature_columns=_feature_columns)

    calibrated_probability = float(_calibrated_model.predict_proba(case_encoded)[:,1][0])

    explanation = explain_case(
        _explainer,
        case_encoded,
        _feature_columns,
        top_n=top_n_explanation
    )

    return {
        "calibrated_probability": calibrated_probability,
        "explanation": explanation,
    }


if __name__ == "__main__":
    sample_case = {
        "reason_code": "item_not_received",
        "product_category": "electronics",
        "payment_method": "card",
        "order_value": 4500,
        "time_to_dispute_days": 5,
        "days_remaining_to_respond": 12,
        "delivery_confirmed": "not_shipped",
        "communication_logs_present": True,
        "device_ip_match_score": 0.4,
        "listing_accuracy_score": 0.7,
        "customer_prior_disputes": 0,
        "customer_account_age_days": 300,
    }

    result = predict_case(sample_case)
    print(f"Calibrated P(win): {result['calibrated_probability']:.3f}")
    print("Explanation:")
    for line in result["explanation"]:
        print(" ", line)

