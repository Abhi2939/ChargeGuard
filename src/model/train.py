import os
import joblib

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report,brier_score_loss,log_loss
from xgboost import XGBClassifier
from sklearn.calibration import calibration_curve
from sklearn.calibration import CalibratedClassifierCV
from sklearn.frozen import FrozenEstimator


DATA_PATH = os.path.join(os.path.dirname("data/dispute_dataset.csv"))
ARTIFACTS_DIR = os.path.join(os.path.dirname("artifacts"))

CATEGORICAL_COLS = ["reason_code", "product_category", "payment_method", "delivery_confirmed"]
DROP_COLS = ["case_id", "_true_win_prob_debug_only"]
TARGET_COL = "merchant_won"

RANDOM_STATE = 42

def load_data(path=DATA_PATH):

    df = pd.read_csv(path)
    df = df.drop(columns=DROP_COLS,axis=1)

    df_enc = pd.get_dummies(df,columns=CATEGORICAL_COLS,drop_first=False)

    X = df_enc.drop(columns=[TARGET_COL])
    y = df_enc[TARGET_COL].astype(int)

    return X,y, df

def split_data(X,y):

    X_train,X_temp,y_train,y_temp = train_test_split(
        X,y,test_size=0.4,stratify=y,random_state=42
    )

    X_cal,X_test,y_cal,y_test = train_test_split(
        X_temp,y_temp,test_size=0.5,stratify=y_temp,random_state=42
    )

    return X_train,X_cal,X_test,y_train,y_cal,y_test

def train_raw_model(X_train,y_train):

    raw_model = XGBClassifier(
        n_estimators=200,
        max_depth=4,
        learning_rate=0.08,
        eval_metric='logloss',
        random_state=42
    )

    raw_model.fit(X_train,y_train)

    return raw_model

def train_calibrated_model(X_cal,y_cal,raw_model):

    try:
        cal_model = CalibratedClassifierCV(FrozenEstimator(raw_model),method="sigmoid")
    except ImportError:
        cal_model = CalibratedClassifierCV(raw_model, method="sigmoid", cv="prefit")

    cal_model.fit(X_cal, y_cal)
    return cal_model

def report_scores(y_test, raw_probs, cal_probs):

    print("=== Calibration comparison ===")
    print(f"Brier score  — raw: {brier_score_loss(y_test, raw_probs):.4f}   "
          f"calibrated: {brier_score_loss(y_test, cal_probs):.4f}")
    print(f"Log loss     — raw: {log_loss(y_test, raw_probs):.4f}   "
          f"calibrated: {log_loss(y_test, cal_probs):.4f}")

def save_artifacts(raw_model, cal_model, feature_columns):

    os.makedirs(ARTIFACTS_DIR, exist_ok=True)

    joblib.dump(raw_model, os.path.join(ARTIFACTS_DIR, "raw_model.pkl"))
    joblib.dump(cal_model, os.path.join(ARTIFACTS_DIR, "calibrated_model.pkl"))
    joblib.dump(feature_columns, os.path.join(ARTIFACTS_DIR, "feature_columns.pkl"))


def main():
    X, y, _ = load_and_prepare_data()
    X_train, X_cal, X_test, y_train, y_cal, y_test = split_data(X, y)

    print(f"Train: {X_train.shape[0]}  Calibration: {X_cal.shape[0]}  Test: {X_test.shape[0]}")

    raw_model = train_raw_model(X_train, y_train)
    cal_model = calibrate_model(raw_model, X_cal, y_cal)

    raw_probs = raw_model.predict_proba(X_test)[:, 1]
    cal_probs = cal_model.predict_proba(X_test)[:, 1]
    report_scores(y_test, raw_probs, cal_probs)

    save_artifacts(raw_model, cal_model, X.columns.tolist())


if __name__ == "__main__":
    main()

    


