import os
import sys
import json

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.abspath(os.path.join(BASE_DIR, "..", "src"))

if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from agents.orchestrator import run_pipeline

SAMPLE_CASES_PATH = os.path.join(BASE_DIR, "sample_cases.json")

def print_divider():
    print("\n" + "=" * 70 + "\n")


def run_case(label: str,case: dict):

    print(f"CASE: {label}")
    print("-" * 70)
    print("Input:")
    for k, v in case.items():
        print(f"  {k}: {v}")

    result = run_pipeline(case)

    if result.get("error"):
        print(f"\n[ERROR] {result['error']}")
        return

    prediction = result["prediction_result"]
    decision = result["decision"]

    print(f"\nCalibrated P(win): {prediction['calibrated_probability']:.1%}")
    print("Top contributing factors:")
    for line in prediction["explanation"]:
        print(f"  - {line}")

    print(f"\nDecision: {decision['action'].upper()}")
    print(f"Reasoning: {decision['reasoning']}")
    print(f"\nNarration:\n  {decision['narration']}")

def main():
    with open(SAMPLE_CASES_PATH, "r") as f:
        demo_cases = json.load(f)

    print("CALIBRATED DISPUTE AGENT — DEMO WALKTHROUGH")
    print(f"Running {len(demo_cases)} curated cases through the full pipeline...")

    for entry in demo_cases:
        print_divider()
        run_case(entry["label"], entry["case"])

    print_divider()
    print("Demo complete. Full audit trail written to reports/audit_log.jsonl")


if __name__ == "__main__":
    main()



