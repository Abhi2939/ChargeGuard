import os
import sys
import json

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.abspath(os.path.join(BASE_DIR, "..", "src"))

if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from agents.orchestrator import run_pipeline

SAMPLE_CASES_PATH = os.path.join(BASE_DIR, "sample_cases.json")





