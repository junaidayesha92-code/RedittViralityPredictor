"""
train_model.py
----------------
Trains and evaluates two models for the virality classification task:
  1. Baseline: Logistic Regression on metadata only (no text) -- establishes
     the floor. If your fancy model barely beats this, that's worth knowing
     and saying out loud, not hiding.
  2. Main model: XGBoost on metadata + TF-IDF title features.

EVALUATION NOTES:
- The target is imbalanced (~10% viral by construction), so accuracy is a
  misleading headline metric -- a model that always predicts "not viral"
  gets 90% accuracy and is useless. We report precision/recall/F1/PR-AUC/ROC-AUC.
- Class imbalance is handled via XGBoost's scale_pos_weight rather than
  naive oversampling, to avoid duplicating (and overfitting on) the same
  rare examples.
- Train/test split is done BEFORE fitting the TF-IDF vectorizer, and the
  vectorizer is fit only on train, to avoid vocabulary leakage from test set.

Outputs saved to models/:
  - baseline_model.pkl, xgb_model.pkl, tfidf_vectorizer.pkl
  - feature_columns.json (needed by the Streamlit app to rebuild inputs)
  - metrics.json (for the README / resume bullet -- use real numbers, not vibes)
  - shap_summary.png
"""

import json
import pickle
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score, classification_report, precision_recall_curve,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

from features import (
    META_FEATURE_COLS, TEXT_FEATURE_COLS, add_meta_features, add_target,
    add_text_features, build_tfidf,
)

MODELS_DIR = Path(__file__).parent / "models"
MODELS_DIR.mkdir(exist_ok=True)
DATA_DIR = Path(__file__).parent / "data"


def load_and_split():
    """Load raw data, engineer non-text features + target, then split BEFORE fitting TF-IDF."""
    df = pd.read_csv(DATA_DIR / "reddit_posts_raw.csv")
    df = df.dropna(subset=["title", "subreddit", "score_at_24h"]).reset_index(drop=True)

    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
    analyzer = SentimentIntensityAnalyzer()
    df = add_text_features(df, analyzer)
    df = add_meta_features(df)
    df = add_target(df)

    train_df, test_df = train_test_split(
        df, test_size=0.2, random_state=42, stratify=df["is_viral"]
    )
    return train_df.reset_index(drop=True), test_df.reset_index(drop=True)


def assemble_matrix(df, subreddit_cols, tfidf_df):
    subreddit_dummies = pd.get_dummies(df["subreddit"], prefix="sub")
    subreddit_dummies = subreddit_dummies.reindex(columns=subreddit_cols, fill_value=0)
    feature_cols = TEXT_FEATURE_COLS + META_FEATURE_COLS
    X = pd.concat(
        [df[feature_cols].reset_index(drop=True), subreddit_dummies.reset_index(drop=True),
         tfidf_df.reset_index(drop=True)],
        axis=1,
    )
    return X


def main():
    train_df, test_df = load_and_split()
    print(f"Train: {len(train_df)} | Test: {len(test_df)} | "
          f"Train viral rate: {train_df['is_viral'].mean():.3f} | "
          f"Test viral rate: {test_df['is_viral'].mean():.3f}")

    # Fit TF-IDF on TRAIN ONLY
    tfidf_train, vectorizer = build_tfidf(train_df)
    tfidf_test, _ = build_tfidf(test_df, fit_vectorizer=vectorizer)

    subreddit_cols = sorted(pd.get_dummies(train_df["subreddit"], prefix="sub").columns)
    X_train = assemble_matrix(train_df, subreddit_cols, tfidf_train)
    X_test = assemble_matrix(test_df, subreddit_cols, tfidf_test)
    y_train, y_test = train_df["is_viral"], test_df["is_viral"]

    feature_columns = list(X_train.columns)

    # ---------------- Baseline: Logistic Regression, metadata only ----------------
    meta_cols = [c for c in feature_columns if not c.startswith("tfidf_")]
    scaler = StandardScaler()
    X_train_meta = scaler.fit_transform(X_train[meta_cols])
    X_test_meta = scaler.transform(X_test[meta_cols])

    baseline = LogisticRegression(max_iter=1000, class_weight="balanced")
    baseline.fit(X_train_meta, y_train)
    baseline_probs = baseline.predict_proba(X_test_meta)[:, 1]

    # ---------------- Main model: XGBoost, metadata + TF-IDF ----------------
    scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()
    xgb = XGBClassifier(
        n_estimators=300,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=scale_pos_weight,
        eval_metric="aucpr",
        random_state=42,
    )
    xgb.fit(X_train, y_train)
    xgb_probs = xgb.predict_proba(X_test)[:, 1]

    # ---------------- Evaluation ----------------
    def evaluate(name, y_true, probs, threshold=0.5):
        preds = (probs >= threshold).astype(int)
        report = classification_report(y_true, preds, output_dict=True, zero_division=0)
        roc_auc = roc_auc_score(y_true, probs)
        pr_auc = average_precision_score(y_true, probs)
        print(f"\n--- {name} ---")
        print(f"ROC-AUC: {roc_auc:.3f} | PR-AUC: {pr_auc:.3f}")
        print(f"Precision (viral): {report['1']['precision']:.3f} | "
              f"Recall (viral): {report['1']['recall']:.3f} | "
              f"F1 (viral): {report['1']['f1-score']:.3f}")
        return {
            "roc_auc": roc_auc, "pr_auc": pr_auc,
            "precision_viral": report["1"]["precision"],
            "recall_viral": report["1"]["recall"],
            "f1_viral": report["1"]["f1-score"],
        }

    baseline_metrics = evaluate("Baseline (Logistic Regression, metadata only)", y_test, baseline_probs)
    xgb_metrics = evaluate("Main model (XGBoost, metadata + TF-IDF)", y_test, xgb_probs)

    # ---------------- SHAP explainability ----------------
    print("\nComputing SHAP values...")
    explainer = shap.TreeExplainer(xgb)
    shap_values = explainer.shap_values(X_test)

    plt.figure()
    shap.summary_plot(shap_values, X_test, show=False, max_display=15)
    plt.tight_layout()
    plt.savefig(MODELS_DIR / "shap_summary.png", dpi=150)
    plt.close()
    print(f"Saved SHAP summary plot -> {MODELS_DIR / 'shap_summary.png'}")

    # ---------------- Save everything the Streamlit app needs ----------------
    with open(MODELS_DIR / "xgb_model.pkl", "wb") as f:
        pickle.dump(xgb, f)
    with open(MODELS_DIR / "baseline_model.pkl", "wb") as f:
        pickle.dump({"model": baseline, "scaler": scaler, "meta_cols": meta_cols}, f)
    with open(MODELS_DIR / "tfidf_vectorizer.pkl", "wb") as f:
        pickle.dump(vectorizer, f)
    with open(MODELS_DIR / "feature_columns.json", "w") as f:
        json.dump({"feature_columns": feature_columns, "subreddit_cols": subreddit_cols}, f, indent=2)
    with open(MODELS_DIR / "metrics.json", "w") as f:
        json.dump({"baseline": baseline_metrics, "xgboost": xgb_metrics,
                    "n_train": len(train_df), "n_test": len(test_df),
                    "viral_rate": float(y_test.mean())}, f, indent=2)
    # Explainer saved via shap.TreeExplainer is picklable and cheap to recompute from the model,
    # so we just re-instantiate it in the app from xgb_model.pkl rather than pickling it separately.

    print(f"\nSaved models + metrics -> {MODELS_DIR}/")
    print("\nSummary for your README:")
    print(json.dumps({"baseline": baseline_metrics, "xgboost": xgb_metrics}, indent=2))


if __name__ == "__main__":
    main()
