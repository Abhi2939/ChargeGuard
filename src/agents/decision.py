import os
from groq import Groq
from dotenv import load_dotenv

FIGHT_THRESHOLD = 0.75
DROP_THRESHOLD = 0.40

load_dotenv()

def make_decision(calibrated_probability: float) -> dict:

    if calibrated_probability > FIGHT_THRESHOLD:
        action = "fight"
        reasoning = (
            f"Calibrated win probability ({calibrated_probability:.2%}) exceeds the "
            f"fight threshold ({FIGHT_THRESHOLD:.0%}) -- recommending the merchant "
            f"submit representment evidence."
        )
    elif calibrated_probability < DROP_THRESHOLD:
        action = "drop"
        reasoning = (
            f"Calibrated win probability ({calibrated_probability:.2%}) is below the "
            f"drop threshold ({DROP_THRESHOLD:.0%}) -- fighting this case is unlikely "
            f"to be worth the effort."
        )
    else:
        action = "review"
        reasoning = (
            f"Calibrated win probability ({calibrated_probability:.2%}) falls in the "
            f"borderline range ({DROP_THRESHOLD:.0%}-{FIGHT_THRESHOLD:.0%}) -- flagging "
            f"for human review rather than auto-deciding."
        )

    return {
        "action":action,
        "reasoning":reasoning
    }

def narrate_decision(case: dict,prediction_result: dict, decision: dict) -> str:

    api_key = os.environ.get("GROQ_API_KEY")
    explanation_text = ";".join(prediction_result["explanation"])

    if not api_key:
        return (
            f"[template fallback -- no GROQ_API_KEY set] "
            f"Recommendation: {decision['action'].upper()}. "
            f"{decision['reasoning']} Key factors: {explanation_text}."
        )

    try:
        client = Groq(api_key=api_key)

        prompt = (
            f"You are narrating a chargeback dispute decision for a merchant dashboard. "
            f"Do NOT change or second-guess the decision -- only explain it clearly and briefly.\n\n"
            f"Reason code: {case.get('reason_code')}\n"
            f"Order value: Rs {case.get('order_value')}\n"
            f"Calibrated win probability: {prediction_result['calibrated_probability']:.2%}\n"
            f"Decision: {decision['action'].upper()}\n"
            f"Decision logic: {decision['reasoning']}\n"
            f"Top contributing factors: {explanation_text}\n\n"
            f"Write a 2-3 sentence explanation for the merchant, in plain language."
        )

        response = client.chat.completions.create(
            model="openai/gpt-oss-120b",
            max_tokens=300,
            messages=[{"role": "user", "content": prompt}],
        )

        return response.choices[0].message.content

    except Exception as e:
        return (
            f"[narration unavailable: {e}] "
            f"Recommendation: {decision['action'].upper()}. {decision['reasoning']}"
        )

def decide_case(case: dict, prediction_result: dict) -> dict:
    """Full decision step: deterministic branch + LLM narration."""
    decision = make_decision(prediction_result["calibrated_probability"])
    decision["narration"] = narrate_decision(case, prediction_result, decision)
    return decision