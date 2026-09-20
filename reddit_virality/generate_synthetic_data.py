"""
generate_synthetic_data.py
---------------------------
Generates a realistic-but-synthetic dataset shaped exactly like what
collect_data.py produces, so you can build/test/demo the entire pipeline
today while your real 24h-observation-window collection runs in the
background (it needs real wall-clock time to accumulate).

IMPORTANT: This is scaffolding, not your final deliverable. Swap in
data/reddit_posts_raw.csv from collect_data.py once you have ~1-2 weeks
of real collection, then rerun features.py and train_model.py unchanged.

The generator deliberately encodes a few realistic, learnable signals
(title sentiment, posting hour, author karma, subreddit baseline rates)
plus substantial noise, so the downstream model has something genuine
to find without being a trivial 100%-accuracy toy problem.
"""

import random
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

random.seed(42)
np.random.seed(42)

DATA_DIR = Path(__file__).parent / "data"
DATA_DIR.mkdir(exist_ok=True)

analyzer = SentimentIntensityAnalyzer()

SUBREDDITS = {
    # name: (baseline_score_mean, baseline_score_std, size_factor)
    "dataisbeautiful": (150, 400, 1.0),
    "MachineLearning": (80, 250, 0.8),
    "explainlikeimfive": (400, 900, 1.4),
    "todayilearned": (600, 1500, 1.6),
    "AskReddit": (1200, 3500, 2.0),
}

FLAIRS = {
    "dataisbeautiful": ["OC", "Discussion", "Question", None],
    "MachineLearning": ["Research", "Discussion", "Project", "News", None],
    "explainlikeimfive": ["Physics", "Biology", "Technology", "Economics", None],
    "todayilearned": [None, "Discussion"],
    "AskReddit": [None],
}

TITLE_TEMPLATES = [
    "TIL that {fact}",
    "ELI5: Why does {thing} happen?",
    "What's the {superlative} thing you've ever {verb}?",
    "[OC] {metric} of {topic} over the last {n} years",
    "I built a {adjective} model to predict {topic}",
    "Can someone explain why {thing} is so {adjective}?",
    "{topic} just changed everything we knew about {field}",
    "Show HN style: I made a tool that {verb}",
    "New paper shows {topic} outperforms previous methods",
    "What is your opinion on {topic}?",
    "Breaking down {topic} in simple terms",
    "The truth about {topic} nobody talks about",
    "{n} things I learned after {verb} for a year",
    "Why is {thing} not more well known?",
    "Does anyone else think {topic} is overrated?",
]

WORDS = {
    "fact": ["octopuses have three hearts", "the Eiffel Tower grows in summer",
              "bananas are berries", "honey never spoils", "sharks predate trees"],
    "thing": ["jet lag", "the placebo effect", "inflation", "deja vu", "muscle memory"],
    "superlative": ["strangest", "best", "most terrifying", "most rewarding", "weirdest"],
    "verb": ["cooked", "built", "witnessed", "trained", "debugged", "automating my job"],
    "metric": ["Global CO2 emissions", "Average salaries", "Life expectancy", "Housing prices"],
    "topic": ["climate change", "large language models", "the housing market",
               "quantum computing", "renewable energy", "the stock market", "gradient descent"],
    "n": ["5", "10", "3", "7", "20"],
    "adjective": ["simple", "surprising", "controversial", "elegant", "overengineered"],
    "field": ["economics", "physics", "machine learning", "biology", "history"],
}


def make_title():
    template = random.choice(TITLE_TEMPLATES)
    filled = template
    for key, options in WORDS.items():
        if "{" + key + "}" in filled:
            filled = filled.replace("{" + key + "}", random.choice(options))
    if random.random() < 0.15:
        filled = filled.rstrip(".") + "?"
    if random.random() < 0.08:
        filled = filled.upper()
    return filled


def generate(n_posts=8000):
    rows = []
    start_date = datetime(2026, 5, 1, tzinfo=timezone.utc)

    for i in range(n_posts):
        sub = random.choice(list(SUBREDDITS.keys()))
        base_mean, base_std, size_factor = SUBREDDITS[sub]

        title = make_title()
        sentiment = analyzer.polarity_scores(title)["compound"]

        created = start_date + timedelta(
            days=random.uniform(0, 90),
            hours=random.uniform(0, 24),
        )
        hour = created.hour
        dow = created.weekday()

        author_karma = max(0, np.random.lognormal(mean=8, sigma=2))
        author_age_days = max(1, np.random.lognormal(mean=6.5, sigma=1.2))

        # --- learnable signal (real but not overwhelming -- this is not a toy 100%-acc problem) ---
        hour_boost = 1.6 if 12 <= hour <= 22 else 0.6            # evenings do notably better
        weekend_boost = 1.3 if dow >= 5 else 1.0
        sentiment_boost = 1 + 0.9 * sentiment                      # title sentiment matters a lot
        karma_boost = 1 + 0.35 * np.log1p(author_karma) / 10
        length_penalty = 1.0 if 20 <= len(title) <= 120 else 0.75
        question_boost = 1.35 if title.strip().endswith("?") else 1.0

        signal_multiplier = (
            hour_boost * weekend_boost * sentiment_boost *
            karma_boost * length_penalty * question_boost
        )

        # heavy-tailed score distribution (most posts flop, some go viral) -- like real Reddit
        base_draw = np.random.lognormal(mean=np.log(max(base_mean, 1)), sigma=1.0)
        score = max(0, int(base_draw * signal_multiplier * size_factor * random.uniform(0.6, 1.3)))
        comments = max(0, int(score * random.uniform(0.05, 0.25) + np.random.poisson(3)))
        upvote_ratio = float(np.clip(0.55 + 0.3 * sentiment + np.random.normal(0, 0.08), 0.5, 1.0))

        rows.append({
            "post_id": f"synthetic_{i}",
            "subreddit": sub,
            "title": title,
            "created_utc": created.timestamp(),
            "flair": random.choice(FLAIRS[sub]),
            "is_self": random.random() < 0.6,
            "author_karma_at_post": round(author_karma, 1),
            "author_age_days_at_post": round(author_age_days, 1),
            "hour_posted_utc": hour,
            "day_of_week_posted": dow,
            "collected_at": created.timestamp() + 60,
            "score_at_24h": score,
            "num_comments_at_24h": comments,
            "upvote_ratio_at_24h": round(upvote_ratio, 3),
        })

    return pd.DataFrame(rows)


if __name__ == "__main__":
    df = generate(n_posts=8000)
    out_path = DATA_DIR / "reddit_posts_raw.csv"
    df.to_csv(out_path, index=False)
    print(f"Generated {len(df)} synthetic posts -> {out_path}")
    print(df.groupby("subreddit")["score_at_24h"].describe()[["mean", "50%", "max"]])
