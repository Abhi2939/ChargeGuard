# ChargeGuard

A calibrated dispute agent that predicts whether a merchant will win a
chargeback dispute, with a trustworthy, calibrated confidence score — and
decides whether to auto-recommend fighting it, dropping it, or escalating to
human review.

Built for the Razorpay AI Buildathon 2026, Track 2 (AI Risk Manager).

## The problem

When a customer disputes a payment, the issuing bank reverses the money
immediately — before the merchant gets to respond. The merchant then has a
short window to submit evidence ("representment") and fight the chargeback.
Deciding whether a given dispute is worth fighting is currently a manual,
inconsistent judgment call, and getting it wrong is costly in both directions:
fighting a losing case wastes staff effort, and dropping a winnable one loses
real revenue.

ChargeGuard automates that decision — but the core design problem isn't
"predict who wins." It's "produce a confidence score that can actually be
trusted enough to act on automatically." A model that says "90% confident"
should be right about 90% of the time, or auto-acting on that number is
dangerous. That's what calibration solves, and it's the centerpiece of this
project.

## What it does

Given a dispute case (reason code, delivery status, evidence signals, customer
history), ChargeGuard:

1. Predicts the probability the merchant would win the dispute
2. Calibrates that probability so it reflects real-world frequency, not just
   raw model confidence
3. Explains which factors drove the prediction
4. Recommends fight / drop / human review based on fixed, auditable thresholds
5. Generates a plain-language explanation of the recommendation
6. Logs the full decision and reasoning to an audit trail

## Architecture