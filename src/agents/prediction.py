import os
import sys
import joblib

from model.explain import load_raw_model,generate_shap,explain_case
from agents.ingestion import ingest_case

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.abspath(os.path.join(BASE_DIR, ".."))

if SRC_DIR not in sys.path:
    sys.path.insert(0,SRC_DIR)

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



