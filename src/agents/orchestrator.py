import os
import sys
from typing import TypedDict,Optional

from langgraph.graph import StateGraph,END,START

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.abspath(os.path.join(BASE_DIR, ".."))

if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from agents.prediction import predict_case
from agents.decision import decide_case
from agents.ingestion import validate_case,CaseValidationError
from utils.audit_log import log_decision

class DisputeState(TypedDict,total=False):

    case:dict

    prediction_result: Optional[dict]
    decision: Optional[dict]
    error: Optional[str]

def ingest_node(state:DisputeState) -> DisputeState:

    try:
        validate_case(state["case"])
    except CaseValidationError as e:
        state["error"] = str(e)

    return state

def predict_node(state:DisputeState) -> DisputeState:

    if state.get("error"):
        return state
    state["prediction_result"] = predict_case(state["case"])

    return state

def decide_node(state:DisputeState) -> DisputeState:

    if state.get("error"):
        return state
    state["decision"] = decide_case(state["case"],state["prediction_result"])

    return state

def audit_node(state:DisputeState) -> DisputeState:

    if state.get("error"):
        return state
    
    log_decision(state["case"],state["prediction_result"],state["decision"])
    return state


def build_graph():

    graph = StateGraph(DisputeState)

    graph.add_node("ingest",ingest_node)
    graph.add_node("predict",predict_node)
    graph.add_node("decide",decide_node)
    graph.add_node("log_audit",audit_node)


    graph.add_edge(START,"ingest")
    graph.add_edge("ingest","predict")
    graph.add_edge("predict","decide")
    graph.add_edge("decide","log_audit")
    graph.add_edge("log_audit",END)

    return graph.compile()

_compiled_graph = None

def run_pipeline(case:dict) -> DisputeState:

    global _compiled_graph
    if _compiled_graph is None:
        _compiled_graph = build_graph()

    result = _compiled_graph.invoke({"case": case})
    return result

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

    result = run_pipeline(sample_case)

    if result.get("error"):
        print("Pipeline error:", result["error"])
    else:
        print(f"Calibrated P(win): {result['prediction_result']['calibrated_probability']:.3f}")
        print("Explanation:")
        for line in result["prediction_result"]["explanation"]:
            print(" ", line)
        print(f"\nDecision: {result['decision']['action'].upper()}")
        print("Reasoning:", result["decision"]["reasoning"])
        print("Narration:", result["decision"]["narration"])

    