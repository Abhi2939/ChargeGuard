import numpy as np
import pandas as pd
from faker import Faker
import os

fake = Faker()
rng = np.random.default_rng(42)

N = 6000  # total synthetic cases

REASON_CODES = [
    "fraud_card_not_present",
    "item_not_received",
    "not_as_described",
    "duplicate_charge",
    "subscription_not_cancelled",
]

# Roughly realistic real-world skew: fraud and "not received" are most common
REASON_CODE_WEIGHTS = [0.28, 0.30, 0.20, 0.12, 0.10]

PRODUCT_CATEGORIES = ["electronics", "apparel", "digital_goods", "groceries", "home_goods", "beauty"]
PAYMENT_METHODS = ["card", "upi", "netbanking", "wallet"]


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def generate_case():
    reason_code = rng.choice(REASON_CODES, p=REASON_CODE_WEIGHTS)
    product_category = rng.choice(PRODUCT_CATEGORIES)
    payment_method = rng.choice(PAYMENT_METHODS, p=[0.45, 0.35, 0.10, 0.10])

    order_value = float(np.round(rng.lognormal(mean=6.8, sigma=0.9), 2))  # skewed, mostly Rs 300-15000
    order_value = min(order_value, 150000)

    time_to_dispute = int(rng.exponential(scale=10)) + 1  # days; most disputes come fast
    time_to_dispute = min(time_to_dispute, 120)

    days_remaining_to_respond = int(rng.integers(1, 22))  # 7-21 day window, some already tight

    # --- Evidence features (raw, noisy signals) ---
    delivery_confirmed = rng.choice(
        ["signed_confirmation", "delivered_no_signature", "in_transit", "not_shipped"],
        p=[0.45, 0.30, 0.15, 0.10],
    )
    communication_logs_present = rng.choice([True, False], p=[0.4, 0.6])
    device_ip_match_score = float(np.round(rng.beta(2, 2), 3))  # 0-1, weak prior
    listing_accuracy_score = float(np.round(rng.beta(3, 2), 3))  # 0-1, skewed decent

    customer_prior_disputes = int(rng.poisson(0.6))
    customer_account_age_days = int(rng.exponential(scale=400)) + 5

    # --- Latent "true" evidence strength, CONDITIONAL on reason_code ---
    # Each branch weights the features that actually matter for that dispute type.
    delivery_strength_map = {
        "signed_confirmation": 1.0,
        "delivered_no_signature": 0.45,
        "in_transit": -0.3,
        "not_shipped": -1.2,
    }
    delivery_strength = delivery_strength_map[delivery_confirmed]

    if reason_code == "item_not_received":
        latent = (
            1.8 * delivery_strength
            + 0.3 * communication_logs_present
            - 0.15 * np.log1p(time_to_dispute)
            + 0.05 * (listing_accuracy_score - 0.5)
        )
    elif reason_code == "not_as_described":
        latent = (
            1.6 * (listing_accuracy_score - 0.5)
            + 0.5 * communication_logs_present
            + 0.2 * delivery_strength
            - 0.1 * np.log1p(time_to_dispute)
        )
    elif reason_code == "fraud_card_not_present":
        latent = (
            2.0 * (device_ip_match_score - 0.5)
            + 0.4 * delivery_strength
            - 0.4 * customer_prior_disputes
            - 0.05 * np.log1p(time_to_dispute)
        )
    elif reason_code == "duplicate_charge":
        # mostly a paperwork/record-matching case -- less about physical evidence
        latent = (
            0.9 * communication_logs_present
            + 0.3 * (1 - min(customer_prior_disputes, 3) / 3)
            + 0.2 * delivery_strength * 0.3
        )
    else:  # subscription_not_cancelled
        latent = (
            0.7 * communication_logs_present
            - 0.3 * customer_prior_disputes
            + 0.15 * np.log1p(customer_account_age_days / 30)
            - 0.2
        )

    # Noise: this is what keeps the label from being a deterministic rule.
    # Without this, a classifier fits it almost perfectly and there's nothing
    # for calibration to visibly correct.
    noise = rng.normal(0, 0.9)
    true_win_prob = float(np.clip(sigmoid(latent + noise), 0.01, 0.99))

    merchant_won = bool(rng.random() < true_win_prob)

    return {
        "case_id": fake.uuid4(),
        "reason_code": reason_code,
        "product_category": product_category,
        "payment_method": payment_method,
        "order_value": order_value,
        "time_to_dispute_days": time_to_dispute,
        "days_remaining_to_respond": days_remaining_to_respond,
        "delivery_confirmed": delivery_confirmed,
        "communication_logs_present": communication_logs_present,
        "device_ip_match_score": device_ip_match_score,
        "listing_accuracy_score": listing_accuracy_score,
        "customer_prior_disputes": customer_prior_disputes,
        "customer_account_age_days": customer_account_age_days,
        # true_win_prob is included ONLY for your own debugging / sanity checks.
        # Drop it before training -- it's not something you'd have in real life,
        # and it would leak the answer straight into the model.
        "_true_win_prob_debug_only": round(true_win_prob, 4),
        "merchant_won": merchant_won,
    }


def main():
    rows = [generate_case() for _ in range(N)]
    df = pd.DataFrame(rows)

    print("=== Dataset shape ===")
    print(df.shape)

    print("\n=== Overall win rate ===")
    print(df["merchant_won"].mean().round(3))

    print("\n=== Win rate by reason_code (sanity check: should differ meaningfully) ===")
    print(df.groupby("reason_code")["merchant_won"].mean().round(3))

    print("\n=== Win rate by delivery_confirmed within item_not_received (should show clear gradient) ===")
    inr = df[df["reason_code"] == "item_not_received"]
    print(inr.groupby("delivery_confirmed")["merchant_won"].mean().round(3))

    print("\n=== Correlation check: device_ip_match_score vs win, within fraud cases ===")
    fraud = df[df["reason_code"] == "fraud_card_not_present"]
    print(fraud[["device_ip_match_score", "merchant_won"]].corr().iloc[0, 1].round(3))

    out_path = os.path.join(os.path.dirname(__file__), "dispute_dataset.csv")
    df.to_csv(out_path, index=False)
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
