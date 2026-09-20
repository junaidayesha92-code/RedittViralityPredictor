"""
collect_data.py
----------------
Collects Reddit posts via PRAW and, after a fixed observation window,
snapshots their score/comment count to build a virality-prediction dataset.

WHY THIS EXISTS (design note for your README / interview talking points):
A common mistake in "predict virality" projects is scraping score/comments
at whatever moment you happen to run the scraper. That leaks time-since-post
into your target variable, because a 2-hour-old post and a 2-day-old post
are not comparable. This script instead:
  1. Records posts at (or near) submission time -> features_at_post_time
  2. Waits until each post is OBSERVATION_HOURS old
  3. Snapshots score/comments at that fixed horizon -> target variable

Run this in two modes:
  python collect_data.py --mode seed      # collect new posts today
  python collect_data.py --mode snapshot  # snapshot posts that have aged in

Schedule both via cron (or run manually) over several days to build a dataset.

SETUP:
  1. Create a Reddit app at https://www.reddit.com/prefs/apps (type: "script")
  2. Copy .env.example to .env and fill in your credentials
  3. pip install -r requirements.txt
"""

import argparse
import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import praw
from dotenv import load_dotenv

load_dotenv()

DATA_DIR = Path(__file__).parent / "data"
DATA_DIR.mkdir(exist_ok=True)
PENDING_PATH = DATA_DIR / "pending_posts.jsonl"   # posts awaiting snapshot
FINAL_PATH = DATA_DIR / "reddit_posts_raw.csv"     # posts with target captured

OBSERVATION_HOURS = 24          # fixed horizon for the target snapshot
SUBREDDITS = [
    "dataisbeautiful",
    "MachineLearning",
    "explainlikeimfive",
    "todayilearned",
    "AskReddit",
]
POSTS_PER_SUBREDDIT_PER_RUN = 100


def get_reddit_client():
    return praw.Reddit(
        client_id=os.environ["REDDIT_CLIENT_ID"],
        client_secret=os.environ["REDDIT_CLIENT_SECRET"],
        user_agent=os.environ.get("REDDIT_USER_AGENT", "virality-research-script/0.1"),
    )


def seed(reddit):
    """Pull fresh 'new' posts and record their at-post-time features."""
    rows = []
    for sub_name in SUBREDDITS:
        subreddit = reddit.subreddit(sub_name)
        for submission in subreddit.new(limit=POSTS_PER_SUBREDDIT_PER_RUN):
            author = submission.author
            try:
                author_karma = author.link_karma + author.comment_karma if author else None
                author_age_days = (
                    (datetime.now(timezone.utc).timestamp() - author.created_utc) / 86400
                    if author else None
                )
            except Exception:
                # Suspended/deleted authors throw on attribute access
                author_karma, author_age_days = None, None

            rows.append(
                {
                    "post_id": submission.id,
                    "subreddit": sub_name,
                    "title": submission.title,
                    "created_utc": submission.created_utc,
                    "flair": submission.link_flair_text,
                    "is_self": submission.is_self,
                    "author_karma_at_post": author_karma,
                    "author_age_days_at_post": author_age_days,
                    "hour_posted_utc": datetime.fromtimestamp(
                        submission.created_utc, tz=timezone.utc
                    ).hour,
                    "day_of_week_posted": datetime.fromtimestamp(
                        submission.created_utc, tz=timezone.utc
                    ).weekday(),
                    "collected_at": time.time(),
                }
            )

    with open(PENDING_PATH, "a") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")

    print(f"Seeded {len(rows)} posts -> {PENDING_PATH}")


def snapshot(reddit):
    """Check pending posts; if they've aged past OBSERVATION_HOURS, snapshot the target."""
    if not PENDING_PATH.exists():
        print("No pending posts file found. Run --mode seed first.")
        return

    pending = [json.loads(line) for line in open(PENDING_PATH)]
    still_pending, ready = [], []
    now = time.time()

    for p in pending:
        age_hours = (now - p["created_utc"]) / 3600
        if age_hours >= OBSERVATION_HOURS:
            ready.append(p)
        else:
            still_pending.append(p)

    finalized = []
    for p in ready:
        try:
            submission = reddit.submission(id=p["post_id"])
            p["score_at_24h"] = submission.score
            p["num_comments_at_24h"] = submission.num_comments
            p["upvote_ratio_at_24h"] = submission.upvote_ratio
            finalized.append(p)
        except Exception as e:
            print(f"Skipping {p['post_id']}: {e}")

    # Append finalized rows to the CSV
    if finalized:
        df_new = pd.DataFrame(finalized)
        if FINAL_PATH.exists():
            df_existing = pd.read_csv(FINAL_PATH)
            df_out = pd.concat([df_existing, df_new], ignore_index=True)
        else:
            df_out = df_new
        df_out.to_csv(FINAL_PATH, index=False)

    # Rewrite pending file with only still-pending posts
    with open(PENDING_PATH, "w") as f:
        for p in still_pending:
            f.write(json.dumps(p) + "\n")

    print(f"Finalized {len(finalized)} posts -> {FINAL_PATH}")
    print(f"Still pending (not aged in yet): {len(still_pending)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["seed", "snapshot"], required=True)
    args = parser.parse_args()

    reddit = get_reddit_client()
    if args.mode == "seed":
        seed(reddit)
    else:
        snapshot(reddit)
