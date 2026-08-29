import os
import sys
from typing import TypedDict,Optional

from langgraph.graph import StateGraph,END

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