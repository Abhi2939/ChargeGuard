import os
import joblib
import numpy as np
import shap


ARTIFACTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "artifacts")

def load_raw_model():
    return joblib.load(os.path.join(ARTIFACTS_DIR, "raw_model.pkl"))

def generate_shap(raw_model):
    return shap.TreeExplainer(raw_model)

def humanize_feature_name(feature_name):
    """Convert a one-hot or raw column name into a readable label."""
    label_map = {
        "order_value": "Order value",
        "time_to_dispute_days": "Days until dispute was filed",
        "days_remaining_to_respond": "Days left to respond",
        "communication_logs_present": "Communication logs present",
        "device_ip_match_score": "Device/IP match score",
        "listing_accuracy_score": "Listing accuracy score",
        "customer_prior_disputes": "Customer's prior dispute count",
        "customer_account_age_days": "Customer account age",
    }
    if feature_name in label_map:
        return label_map[feature_name]
    
    prefixes = {
        "reason_code_": "Reason",
        "product_category_": "Product category",
        "payment_method_": "Payment method",
        "delivery_confirmed_": "Delivery status",
    }

    for prefix, label in prefixes.items():
        if feature_name.startswith(prefix):
            value = feature_name[len(prefix):].replace("_", " ")
            return f"{label}: {value}"

    return feature_name 


def explain_case(explainer, case_features_df, feature_names, top_n=4):

    shap_values = explainer(case_features_df)
    vals = shap_values.values[0]
    top_idx = np.argsort(-np.abs(vals))[:top_n]

    explanations = []
    for i in top_idx:
        readable = humanize_feature_name(feature_names[i])
        contribution = vals[i]
        direction = "increases" if contribution > 0 else "decreases"
        explanations.append(f"{readable} {direction} win likelihood ({contribution:+.3f})")

    return explanations
