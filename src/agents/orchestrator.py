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
    state["decision"] = decide_case(state["case"])

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


    graph.add_edge(START,"ingest_node")
    graph.add_edge("ingest_node","predict_node")
    graph.add_edge("predict_node","decide_node")
    graph.add_edge("decide_node","audit_node")
    graph.add_edge("audit_node",END)

    return graph.compile()


    