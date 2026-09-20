"""
features.py
------------
Turns raw collected posts into a modeling-ready feature matrix.

KEY DESIGN DECISION (defend this in interviews):
"Viral" is defined as being in the top VIRAL_PERCENTILE of score WITHIN
its own subreddit, not against a global threshold. A top-5% post in
r/MachineLearning (small, niche) and a top-5% post in r/AskReddit (huge,
default) represent very different absolute scores but the same relative
achievement. A global threshold would just learn "predict AskReddit."

LEAKAGE GUARD:
We only use features knowable AT POST TIME. score_at_24h, num_comments_at_24h,
and upvote_ratio_at_24h are used ONLY to construct the target label -- they
are dropped from the feature matrix before modeling. This is the single
most important thing to get right and the thing most similar student
projects get wrong.
"""

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

DATA_DIR = Path(__file__).parent / "data"
VIRAL_PERCENTILE = 0.90   # top 10% of a subreddit's posts = "viral"

TEXT_FEATURE_COLS = [
    "title_len", "title_word_count", "has_question_mark", "has_number",
    "caps_ratio", "sentiment_compound", "sentiment_pos", "sentiment_neg",
]
META_FEATURE_COLS = [
    "hour_posted_utc", "day_of_week_posted", "is_weekend", "is_self",
    "author_karma_at_post", "author_age_days_at_post", "has_flair",
    "hour_sin", "hour_cos",
]
LEAKY_COLS = ["score_at_24h", "num_comments_at_24h", "upvote_ratio_at_24h"]


def add_text_features(df: pd.DataFrame, analyzer: SentimentIntensityAnalyzer) -> pd.DataFrame:
    df["title_len"] = df["title"].str.len()
    df["title_word_count"] = df["title"].str.split().str.len()
    df["has_question_mark"] = df["title"].str.contains(r"\?").astype(int)
    df["has_number"] = df["title"].str.contains(r"\d").astype(int)
    df["caps_ratio"] = df["title"].apply(
        lambda t: sum(1 for c in t if c.isupper()) / max(len(t), 1)
    )

    sentiments = df["title"].apply(analyzer.polarity_scores)
    df["sentiment_compound"] = sentiments.apply(lambda s: s["compound"])
    df["sentiment_pos"] = sentiments.apply(lambda s: s["pos"])
    df["sentiment_neg"] = sentiments.apply(lambda s: s["neg"])
    return df


def add_meta_features(df: pd.DataFrame) -> pd.DataFrame:
    df["is_weekend"] = (df["day_of_week_posted"] >= 5).astype(int)
    df["is_self"] = df["is_self"].astype(int)
    df["has_flair"] = df["flair"].notna().astype(int)
    df["author_karma_at_post"] = df["author_karma_at_post"].fillna(
        df["author_karma_at_post"].median()
    )
    df["author_age_days_at_post"] = df["author_age_days_at_post"].fillna(
        df["author_age_days_at_post"].median()
    )
    # cyclical encoding so hour 23 and hour 0 are recognized as adjacent
    df["hour_sin"] = np.sin(2 * np.pi * df["hour_posted_utc"] / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour_posted_utc"] / 24)
    return df


def add_target(df: pd.DataFrame) -> pd.DataFrame:
    df["viral_threshold"] = df.groupby("subreddit")["score_at_24h"].transform(
        lambda s: s.quantile(VIRAL_PERCENTILE)
    )
    df["is_viral"] = (df["score_at_24h"] >= df["viral_threshold"]).astype(int)
    return df


def build_tfidf(df: pd.DataFrame, max_features: int = 150, fit_vectorizer: TfidfVectorizer = None):
    if fit_vectorizer is None:
        vectorizer = TfidfVectorizer(
            max_features=max_features, stop_words="english", ngram_range=(1, 2), min_df=3
        )
        tfidf_matrix = vectorizer.fit_transform(df["title"])
    else:
        vectorizer = fit_vectorizer
        tfidf_matrix = vectorizer.transform(df["title"])

    tfidf_df = pd.DataFrame(
        tfidf_matrix.toarray(),
        columns=[f"tfidf_{w}" for w in vectorizer.get_feature_names_out()],
        index=df.index,
    )
    return tfidf_df, vectorizer


def build_feature_matrix(raw_csv_path: Path = None, fit_vectorizer: TfidfVectorizer = None):
    """Main entry point: raw CSV -> (X, y, feature_names, vectorizer, full_df)."""
    if raw_csv_path is None:
        raw_csv_path = DATA_DIR / "reddit_posts_raw.csv"

    df = pd.read_csv(raw_csv_path)
    df = df.dropna(subset=["title", "subreddit", "score_at_24h"]).reset_index(drop=True)

    analyzer = SentimentIntensityAnalyzer()
    df = add_text_features(df, analyzer)
    df = add_meta_features(df)
    df = add_target(df)

    tfidf_df, vectorizer = build_tfidf(df, fit_vectorizer=fit_vectorizer)

    subreddit_dummies = pd.get_dummies(df["subreddit"], prefix="sub")

    feature_cols = TEXT_FEATURE_COLS + META_FEATURE_COLS
    X = pd.concat(
        [df[feature_cols].reset_index(drop=True), subreddit_dummies.reset_index(drop=True),
         tfidf_df.reset_index(drop=True)],
        axis=1,
    )
    y = df["is_viral"]

    # sanity check: make sure no leaky columns snuck into X
    assert not any(col in X.columns for col in LEAKY_COLS), "Leakage detected in feature matrix!"

    return X, y, df, vectorizer


if __name__ == "__main__":
    X, y, df, vectorizer = build_feature_matrix()
    print(f"Feature matrix: {X.shape}")
    print(f"Viral rate: {y.mean():.3f}")
    print(f"Viral rate by subreddit:\n{df.groupby('subreddit')['is_viral'].mean()}")
