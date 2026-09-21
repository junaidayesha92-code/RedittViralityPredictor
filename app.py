"""
app.py
-------
Streamlit dashboard for the Reddit Virality Predictor.

Two tabs:
  1. Predict: enter a draft title + subreddit + posting time -> get a
     virality probability and the top SHAP factors driving that prediction.
  2. Explore: EDA on the training data (virality rate by hour/subreddit/
     title length, SHAP global summary).

Run locally:  streamlit run app.py
Deploy free:  push to GitHub, then deploy on share.streamlit.io (Streamlit
              Community Cloud) pointing at this file.
"""

import json
import pickle
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import shap
import streamlit as st
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

from features import META_FEATURE_COLS, TEXT_FEATURE_COLS

BASE_DIR = Path(__file__).parent
MODELS_DIR = BASE_DIR / "models"
DATA_DIR = BASE_DIR / "data"

st.set_page_config(page_title="Reddit Virality Predictor", page_icon="🚀", layout="wide")


@st.cache_resource
def load_artifacts():
    with open(MODELS_DIR / "xgb_model.pkl", "rb") as f:
        xgb_model = pickle.load(f)
    with open(MODELS_DIR / "tfidf_vectorizer.pkl", "rb") as f:
        vectorizer = pickle.load(f)
    with open(MODELS_DIR / "feature_columns.json") as f:
        cols_info = json.load(f)
    with open(MODELS_DIR / "metrics.json") as f:
        metrics = json.load(f)
    explainer = shap.TreeExplainer(xgb_model)
    return xgb_model, vectorizer, cols_info, metrics, explainer


@st.cache_data
def load_training_data():
    df = pd.read_csv(DATA_DIR / "reddit_posts_raw.csv")
    return df


analyzer = SentimentIntensityAnalyzer()


def build_single_row_features(title, subreddit, hour, dow, author_karma, author_age_days,
                                 flair_present, is_self, vectorizer, subreddit_cols):
    sentiments = analyzer.polarity_scores(title)
    row = {
        "title_len": len(title),
        "title_word_count": len(title.split()),
        "has_question_mark": int("?" in title),
        "has_number": int(any(c.isdigit() for c in title)),
        "caps_ratio": sum(1 for c in title if c.isupper()) / max(len(title), 1),
        "sentiment_compound": sentiments["compound"],
        "sentiment_pos": sentiments["pos"],
        "sentiment_neg": sentiments["neg"],
        "hour_posted_utc": hour,
        "day_of_week_posted": dow,
        "is_weekend": int(dow >= 5),
        "is_self": int(is_self),
        "author_karma_at_post": author_karma,
        "author_age_days_at_post": author_age_days,
        "has_flair": int(flair_present),
        "hour_sin": np.sin(2 * np.pi * hour / 24),
        "hour_cos": np.cos(2 * np.pi * hour / 24),
    }
    feature_cols = TEXT_FEATURE_COLS + META_FEATURE_COLS
    base_df = pd.DataFrame([{c: row[c] for c in feature_cols}])

    sub_dummies = pd.DataFrame([[0] * len(subreddit_cols)], columns=subreddit_cols)
    sub_col = f"sub_{subreddit}"
    if sub_col in sub_dummies.columns:
        sub_dummies[sub_col] = 1

    tfidf_vec = vectorizer.transform([title])
    tfidf_df = pd.DataFrame(
        tfidf_vec.toarray(),
        columns=[f"tfidf_{w}" for w in vectorizer.get_feature_names_out()],
    )

    X = pd.concat([base_df, sub_dummies, tfidf_df], axis=1)
    return X


# ---------------- Load everything ----------------
try:
    xgb_model, vectorizer, cols_info, metrics, explainer = load_artifacts()
    train_df = load_training_data()
    ARTIFACTS_LOADED = True
except FileNotFoundError:
    ARTIFACTS_LOADED = False

st.title("🚀 Reddit Virality Predictor")
st.caption(
    "Predicts whether a post will land in the top 10% of scores for its subreddit "
    "within 24 hours, using only features available at post time."
)

if not ARTIFACTS_LOADED:
    st.error(
        "Model artifacts not found. Run the pipeline first:\n\n"
        "```\npython generate_synthetic_data.py   # or collect_data.py for real data\n"
        "python train_model.py\n```"
    )
    st.stop()

tab_predict, tab_explore = st.tabs(["🎯 Predict", "📊 Explore the data"])

# ==================== PREDICT TAB ====================
with tab_predict:
    col_input, col_output = st.columns([1, 1.3])

    subreddits = sorted([c.replace("sub_", "") for c in cols_info["subreddit_cols"]])

    with col_input:
        st.subheader("Draft your post")
        title = st.text_input("Title", "What's the most useful thing you've automated with AI?")
        subreddit = st.selectbox("Subreddit", subreddits)
        post_time = st.time_input("Planned posting time (UTC)", datetime(2026, 1, 1, 18, 0).time())
        day_option = st.selectbox(
            "Day of week",
            ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"],
            index=4,
        )
        dow = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"].index(day_option)
        author_karma = st.number_input("Your account karma", min_value=0, value=5000, step=100)
        author_age_days = st.number_input("Account age (days)", min_value=1, value=730, step=30)
        is_self = st.checkbox("Text post (self post)", value=True)
        flair_present = st.checkbox("Has flair", value=False)

    X_input = build_single_row_features(
        title, subreddit, post_time.hour, dow, author_karma, author_age_days,
        flair_present, is_self, vectorizer, cols_info["subreddit_cols"],
    )
    X_input = X_input[cols_info["feature_columns"]]  # enforce exact column order

    prob = xgb_model.predict_proba(X_input)[0, 1]
    shap_values = explainer.shap_values(X_input)

    with col_output:
        st.subheader("Prediction")
        st.metric("Virality probability", f"{prob:.1%}",
                   help="Probability this post lands in the top 10% of scores for its subreddit within 24h")
        st.progress(min(float(prob), 1.0))

        if prob > 0.25:
            st.success("Well above the 10% baseline rate — strong signals detected.")
        elif prob > 0.12:
            st.info("Slightly above the 10% baseline rate.")
        else:
            st.warning("Below the 10% baseline rate for this subreddit.")

        st.markdown("**Top factors driving this prediction**")
        shap_row = pd.Series(shap_values[0], index=X_input.columns).sort_values(key=abs, ascending=False)
        top_factors = shap_row.head(8)
        factor_df = pd.DataFrame({
            "feature": top_factors.index,
            "impact": top_factors.values,
        })
        factor_df["direction"] = np.where(factor_df["impact"] > 0, "↑ increases", "↓ decreases")
        fig = px.bar(
            factor_df.sort_values("impact"), x="impact", y="feature", orientation="h",
            color="impact", color_continuous_scale=["#d62728", "#2ca02c"],
            labels={"impact": "SHAP value (impact on prediction)", "feature": ""},
        )
        fig.update_layout(showlegend=False, coloraxis_showscale=False, height=350)
        st.plotly_chart(fig, use_container_width=True)
        st.caption(
            "SHAP values show how much each feature pushed the prediction up or down "
            "from the average. tfidf_* features reflect specific words/phrases in the title."
        )

# ==================== EXPLORE TAB ====================
with tab_explore:
    st.subheader("Model performance")
    m1, m2, m3 = st.columns(3)
    m1.metric("XGBoost ROC-AUC", f"{metrics['xgboost']['roc_auc']:.3f}")
    m2.metric("XGBoost PR-AUC", f"{metrics['xgboost']['pr_auc']:.3f}")
    m3.metric("Baseline ROC-AUC", f"{metrics['baseline']['roc_auc']:.3f}")
    st.caption(
        f"Evaluated on a held-out test set of {metrics['n_test']} posts "
        f"({metrics['viral_rate']:.1%} viral rate). Baseline = logistic regression on "
        f"metadata only. Main model adds TF-IDF title features via XGBoost."
    )

    st.divider()
    st.subheader("Virality rate by posting hour (UTC)")
    hourly = train_df.copy()
    hourly["hour"] = pd.to_datetime(hourly["created_utc"], unit="s").dt.hour
    hourly_rate = hourly.groupby("hour")["score_at_24h"].apply(
        lambda s: (s >= s.quantile(0.9)).mean()
    ).reset_index(name="viral_rate")
    fig_hour = px.bar(hourly_rate, x="hour", y="viral_rate",
                       labels={"hour": "Hour posted (UTC)", "viral_rate": "Viral rate"})
    st.plotly_chart(fig_hour, use_container_width=True)

    col_a, col_b = st.columns(2)
    with col_a:
        st.subheader("Median score by subreddit")
        sub_median = train_df.groupby("subreddit")["score_at_24h"].median().reset_index()
        fig_sub = px.bar(sub_median.sort_values("score_at_24h"), x="score_at_24h", y="subreddit",
                          orientation="h", labels={"score_at_24h": "Median score at 24h"})
        st.plotly_chart(fig_sub, use_container_width=True)

    with col_b:
        st.subheader("Title length distribution")
        train_df["title_len"] = train_df["title"].str.len()
        fig_len = px.histogram(train_df, x="title_len", nbins=40,
                                labels={"title_len": "Title length (characters)"})
        st.plotly_chart(fig_len, use_container_width=True)

    st.divider()
    st.subheader("Global feature importance (SHAP)")
    st.image(str(MODELS_DIR / "shap_summary.png"), use_container_width=True)
