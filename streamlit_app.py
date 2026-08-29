import requests
import streamlit as st

API_URL = "http://localhost:8000"

st.set_page_config(page_title="Calibrated Dispute Agent", page_icon="⚖️", layout="centered")

st.title("⚖️ Calibrated Dispute Agent")
st.caption(
    "Predicts whether a merchant will win a chargeback dispute, with a "
    "calibrated confidence score -- and recommends fight, drop, or human review."
)

with st.form("case_form"):
    st.subheader("Dispute case details")

    col1, col2 = st.columns(2)

    with col1:
        reason_code = st.selectbox(
            "Reason code",
            [
                "item_not_received",
                "not_as_described",
                "fraud_card_not_present",
                "duplicate_charge",
                "subscription_not_cancelled",
            ],
        )
        product_category = st.selectbox(
            "Product category",
            ["electronics", "apparel", "digital_goods", "groceries", "home_goods", "beauty"],
        )
        payment_method = st.selectbox("Payment method", ["card", "upi", "netbanking", "wallet"])
        order_value = st.number_input("Order value (Rs)", min_value=1.0, value=1500.0, step=100.0)
        time_to_dispute_days = st.number_input("Days until dispute was filed", min_value=0, value=5)
        days_remaining_to_respond = st.number_input("Days left to respond", min_value=0, value=14)

    with col2:
        delivery_confirmed = st.selectbox(
            "Delivery status",
            ["signed_confirmation", "delivered_no_signature", "in_transit", "not_shipped"],
        )
        communication_logs_present = st.checkbox("Communication logs present", value=False)
        device_ip_match_score = st.slider("Device/IP match score", 0.0, 1.0, 0.5)
        listing_accuracy_score = st.slider("Listing accuracy score", 0.0, 1.0, 0.5)
        customer_prior_disputes = st.number_input("Customer's prior dispute count", min_value=0, value=0)
        customer_account_age_days = st.number_input("Customer account age (days)", min_value=0, value=300)

    submitted = st.form_submit_button("Evaluate case")

if submitted:
    case_payload = {
        "reason_code": reason_code,
        "product_category": product_category,
        "payment_method": payment_method,
        "order_value": order_value,
        "time_to_dispute_days": time_to_dispute_days,
        "days_remaining_to_respond": days_remaining_to_respond,
        "delivery_confirmed": delivery_confirmed,
        "communication_logs_present": communication_logs_present,
        "device_ip_match_score": device_ip_match_score,
        "listing_accuracy_score": listing_accuracy_score,
        "customer_prior_disputes": customer_prior_disputes,
        "customer_account_age_days": customer_account_age_days,
    }

    try:
        with st.spinner("Running through ingestion, calibrated prediction, and decision agents..."):
            response = requests.post(f"{API_URL}/evaluate-case", json=case_payload, timeout=30)
    except requests.exceptions.ConnectionError:
        st.error(
            "Couldn't reach the API backend. Make sure it's running: "
            "`uvicorn main:app --reload`"
        )
        st.stop()

    if response.status_code != 200:
        st.error(f"API error ({response.status_code}): {response.json().get('detail')}")
        st.stop()

    result = response.json()
    prediction = result["prediction"]
    decision = result["decision"]

    st.divider()

    prob = prediction["calibrated_probability"]
    action = decision["action"]

    action_colors = {"fight": "🟢", "drop": "🔴", "review": "🟡"}
    st.subheader(f"{action_colors.get(action, '')} Decision: {action.upper()}")

    st.metric("Calibrated win probability", f"{prob:.1%}")
    st.progress(prob)

    st.markdown("**Reasoning:**")
    st.write(decision["reasoning"])

    st.markdown("**Top contributing factors:**")
    for line in prediction["explanation"]:
        st.markdown(f"- {line}")

    st.markdown("**Narration:**")
    st.info(decision["narration"])

    with st.expander("View raw case input"):
        st.json(case_payload)

st.divider()
if st.checkbox("Show recent audit trail"):
    try:
        audit_response = requests.get(f"{API_URL}/audit-log", params={"limit": 5}, timeout=10)
        entries = audit_response.json()["entries"]
        for entry in reversed(entries):
            with st.expander(f"{entry['timestamp']} — {entry['decision'].upper()}"):
                st.json(entry)
    except requests.exceptions.ConnectionError:
        st.warning("Backend not reachable.")