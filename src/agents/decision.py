import os


FIGHT_THRESHOLD = 0.75
DROP_THRESHOLD = 0.40

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


