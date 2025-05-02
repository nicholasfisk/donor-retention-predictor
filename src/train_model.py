"""
train_model.py
--------------
Train + evaluate donor-retention models on the augmented dataset
produced by data_generation.py.  Keeps the better of two pipelines
(LogisticRegression vs GradientBoosting) and writes it to
models/model.pkl   (models/ is git-ignored).

Run:
    python src/train_model.py \
        --input  data/processed/transfusion_augmented.csv \
        --model  models/model.pkl
"""

import argparse
from pathlib import Path
from sklearn.preprocessing import OneHotEncoder, StandardScaler
import sklearn
from packaging import version
import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

def build_preprocessor(df: pd.DataFrame) -> ColumnTransformer:
    numeric_cols = [
        "recency_months", "frequency", "monetary_cc", "time_months",
        "email_open_rate", "events_attended", "sms_opt_in", "last_campaign_days",
    ]
    categorical_cols = ["newsletter_segment"]

    # --- choose the right keyword depending on sklearn version ----------
    if version.parse(sklearn.__version__) >= version.parse("1.2"):
        ohe = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
    else:
        ohe = OneHotEncoder(sparse=False, handle_unknown="ignore")

    return ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numeric_cols),
            ("cat", ohe, categorical_cols),
        ]
    )

def main(args):
    df = pd.read_csv(args.input)

    # ------------------ train/test split -----------------------------
    X = df.drop(columns=["donated_again"])
    y = df["donated_again"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.30, stratify=y, random_state=42
    )

    # ------------------ preprocessing -------------------------------
    preprocessor = build_preprocessor(df)

    # Pipelines -------------------------------------------------------
    pipe_log = Pipeline(
        steps=[
            ("prep", preprocessor),
            ("clf", LogisticRegression(max_iter=1000, solver="lbfgs")),
        ]
    )

    pipe_gbm = Pipeline(
        steps=[
            ("prep", preprocessor),
            ("clf", GradientBoostingClassifier(random_state=42)),
        ]
    )

    # ------------------ fit both models -----------------------------
    print("Training Logistic Regression …")
    pipe_log.fit(X_train, y_train)
    print("Training Gradient Boosting …")
    pipe_gbm.fit(X_train, y_train)

    # ------------------ evaluate ------------------------------------
    def score_model(pipe, name):
        proba = pipe.predict_proba(X_test)[:, 1]
        auc = roc_auc_score(y_test, proba)
        preds = (proba >= 0.5).astype(int)
        print(f"\n== {name} ==")
        print(f"AUC:  {auc:.3f}")
        print(classification_report(y_test, preds, digits=3))
        return auc

    auc_log = score_model(pipe_log, "Logistic Regression")
    auc_gbm = score_model(pipe_gbm, "Gradient Boosting")

    # ------------------ choose + save -------------------------------
    best_pipe = pipe_gbm if auc_gbm >= auc_log else pipe_log
    best_name = "Gradient Boosting" if best_pipe is pipe_gbm else "Logistic Regression"

    args.model.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(best_pipe, args.model)
    print(f"\nSaved best model ({best_name}) → {args.model}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("data/processed/transfusion_augmented.csv"),
        help="CSV produced by data_generation.py",
    )
    parser.add_argument(
        "--model",
        type=Path,
        default=Path("models/model.pkl"),
        help="Path to write the trained model (.pkl)",
    )
    main(parser.parse_args())
