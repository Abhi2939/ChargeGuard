import os
import json
from datetime import datetime,timezone

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
LOG_PATH = os.path.join(BASE_DIR, "..", "..", "reports", "audit_log.jsonl")


def log_decision(case:dict,prediction_results:dict,decision:dict):

    os.makedirs(os.path.dirname(LOG_PATH), exist_ok=True)

    record = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "case_input":case,
        "calibrated_probability":prediction_results["calibrated_probability"],
        "explanation": prediction_results["explanation"],
        "decision":decision["action"],
        "decision_reasoning": decision["reasoning"],
        "narration": decision.get("narration"),
    }

    with open(LOG_PATH,"a") as f:
        f.write(json.dumps(record) + "\n")

    return record

def read_audit_log():

    if not os.path.exists(LOG_PATH):
        return []

    with open(LOG_PATH, "r") as f:
        return [json.loads(line) for line in f if line.strip()]

