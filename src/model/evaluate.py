import os
import joblib

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.calibration import calibration_curve
from sklearn.metrics import confusion_matrix,classification_report,brier_score_loss,log_loss,precision_score, recall_score


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "..", "..", "data", "dispute_dataset.csv")
ARTIFACTS_DIR = os.path.join(BASE_DIR,"artifacts")
OUTPUT_DIR = os.path.join(BASE_DIR, "..", "..", "reports")

CATEGORICAL_COLS = ['reason_code','product_category','payment_method','delivery_confirmed']
DROP_COLS = ["case_id", "_true_win_prob_debug_only"]
TARGET_COL = "merchant_won"
RANDOM_STATE = 42

FP_COST_PER_CASE = 150

def load_artifacts():
    raw_model = joblib.load(os.path.join(ARTIFACTS_DIR, "raw_model.pkl"))
    cal_model = joblib.load(os.path.join(ARTIFACTS_DIR, "calibrated_model.pkl"))
    feature_columns = joblib.load(os.path.join(ARTIFACTS_DIR, "feature_columns.pkl"))
    return raw_model, cal_model, feature_columns


def load_test_split(feature_columns, path=DATA_PATH):

    df = pd.read_csv(path)
    df_clean = df.drop(columns=[c for c in DROP_COLS if c in df.columns])
    df_enc = pd.get_dummies(df_clean, columns=CATEGORICAL_COLS, drop_first=False)

    X = df_enc.reindex(columns=feature_columns, fill_value=0)
    y = df_enc[TARGET_COL].astype(int)

    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.4, stratify=y, random_state=RANDOM_STATE
    )
    X_cal, X_test, y_cal, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, stratify=y_temp, random_state=RANDOM_STATE
    )
    return X_test, y_test, df


def plot_reliability_diagrams(y_test, raw_probs, cal_probs, save_path=None):
    frac_pos_raw, mean_pred_raw = calibration_curve(y_test, raw_probs, n_bins=10, strategy="quantile")
    frac_pos_cal, mean_pred_cal = calibration_curve(y_test, cal_probs, n_bins=10, strategy="quantile")

    fig, axes = plt.subplots(1, 2, figsize=(12, 6), sharey=True)

    axes[0].plot([0, 1], [0, 1], linestyle="--", color="gray")
    axes[0].plot(mean_pred_raw, frac_pos_raw, marker="o", color="crimson")
    axes[0].set_title("BEFORE Calibration (Raw XGBoost)")
    axes[0].set_xlabel("Mean predicted probability")
    axes[0].set_ylabel("Actual win rate")
    axes[0].grid(alpha=0.3)

    axes[1].plot([0, 1], [0, 1], linestyle="--", color="gray")
    axes[1].plot(mean_pred_cal, frac_pos_cal, marker="o", color="seagreen")
    axes[1].set_title("AFTER Calibration (Platt Scaling)")
    axes[1].set_xlabel("Mean predicted probability")
    axes[1].grid(alpha=0.3)

    plt.suptitle("Reliability Diagram: Raw vs. Calibrated Model")
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved reliability diagram to {save_path}")
    plt.show()


def print_calibration_scores(y_test, raw_probs, cal_probs):
    print("=== Calibration comparison ===")
    print(f"Brier score  — raw: {brier_score_loss(y_test, raw_probs):.4f}   "
          f"calibrated: {brier_score_loss(y_test, cal_probs):.4f}")
    print(f"Log loss     — raw: {log_loss(y_test, raw_probs):.4f}   "
          f"calibrated: {log_loss(y_test, cal_probs):.4f}")

def print_classification_report(y_test, cal_probs, threshold=0.5):
    preds = (cal_probs > threshold).astype(int)
    print(f"\n=== Classification report (threshold={threshold}) ===")
    print(classification_report(y_test, preds, target_names=["lost (0)", "won (1)"]))
    return preds


def cost_analysis(df, y_test, preds, fp_cost_per_case=FP_COST_PER_CASE):
    tn, fp, fn, tp = confusion_matrix(y_test, preds).ravel()

    fn_mask = (preds == 0) & (y_test.values == 1)
    fn_indices = y_test.index[fn_mask]
    avg_order_value_fn = df.loc[fn_indices, "order_value"].mean() if fn_mask.sum() > 0 else 0

    total_fp_cost = fp * fp_cost_per_case
    total_fn_cost = avg_order_value_fn * fn

    print("\n=== False-positive cost analysis ===")
    print(f"False positives (fight, actually lost): {fp}  -> Rs {total_fp_cost:,.0f} wasted effort")
    print(f"False negatives (drop, actually would've won): {fn}  -> Rs {total_fn_cost:,.0f} lost revenue")

    return {"fp": fp, "fn": fn, "total_fp_cost": total_fp_cost, "total_fn_cost": total_fn_cost}


def threshold_sweep(df, y_test, cal_probs, thresholds=(0.3, 0.4, 0.5, 0.6, 0.75)):

    print(f"\n{'Threshold':>10} {'Precision':>10} {'Recall':>10} {'FP':>6} {'FN':>6} {'Total Cost (Rs)':>18}")
    for t in thresholds:
        p = (cal_probs > t).astype(int)
        prec = precision_score(y_test, p, zero_division=0)
        rec = recall_score(y_test, p, zero_division=0)
        tn_, fp_, fn_, tp_ = confusion_matrix(y_test, p).ravel()
        fn_idx = y_test.index[(p == 0) & (y_test.values == 1)]
        avg_ov = df.loc[fn_idx, "order_value"].mean() if len(fn_idx) > 0 else 0
        cost = fp_ * FP_COST_PER_CASE + avg_ov * fn_
        print(f"{t:>10} {prec:>10.3f} {rec:>10.3f} {fp_:>6} {fn_:>6} {cost:>18,.0f}")



def main():
    raw_model, cal_model, feature_columns = load_artifacts()
    X_test, y_test, df = load_test_split(feature_columns)

    raw_probs = raw_model.predict_proba(X_test)[:, 1]
    cal_probs = cal_model.predict_proba(X_test)[:, 1]

    print_calibration_scores(y_test, raw_probs, cal_probs)

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    plot_reliability_diagrams(
        y_test, raw_probs, cal_probs,
        save_path=os.path.join(OUTPUT_DIR, "reliability_before_after.png"),
    )

    preds = print_classification_report(y_test, cal_probs, threshold=0.5)
    cost_analysis(df, y_test, preds)
    threshold_sweep(df, y_test, cal_probs)


if __name__ == "__main__":
    main()