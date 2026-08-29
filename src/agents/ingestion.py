import os
import joblib
import pandas as pd

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ARTIFACTS_DIR = os.path.join(BASE_DIR, "..", "model", "artifacts")

CATEGORICAL_COLS = ["reason_code", "product_category", "payment_method", "delivery_confirmed"]

REQUIRED_FIELDS = [
    "reason_code",
    "product_category",
    "payment_method",
    "order_value",
    "time_to_dispute_days",
    "days_remaining_to_respond",
    "delivery_confirmed",
    "communication_logs_present",
    "device_ip_match_score",
    "listing_accuracy_score",
    "customer_prior_disputes",
    "customer_account_age_days",
]

class CaseValidationError(Exception):
    """Raised when an incoming case is missing required fields or has bad values."""
    pass

def _load_feature_cols():
    return joblib.load(os.path.join(ARTIFACTS_DIR,"feature_columns.pkl"))

def validate_case(case:dict):

    missing = [f for f in REQUIRED_FIELDS if f not in case]

    if missing:
        raise CaseValidationError(f"Missing Feature column:{missing}")
    return True

def ingest_case(case:dict,feature_columns=None) -> pd.DataFrame:

    validate_case(case)

    if feature_columns is None:
        feature_columns = _load_feature_cols()

    case_df = pd.DataFrame([case])
    case_encoded = pd.get_dummies(case_df, columns=CATEGORICAL_COLS, drop_first=False)

    case_final = case_encoded.reindex(columns=feature_columns,fill_value=0)

    return case_final