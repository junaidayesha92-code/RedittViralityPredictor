# Reddit Virality Predictor

Predicts whether a Reddit post will land in the **top 10% of scores for its subreddit**
within 24 hours, using only features available **at post time** — then explains *why*
via SHAP, and lets you test draft titles interactively in a live dashboard.

**[Live demo](https://junaidayesha92-code-redittviralitypre-reddit-viralityapp-t0z8iu.streamlit.app/)** · Built with Python, XGBoost, SHAP, Streamlit

> **Note:** the deployed demo is currently trained on a synthetic dataset (same schema
> as real Reddit data, generated to unblock development while a real 24-hour data
> collection window runs — see [Data](#data)). Treat predictions as a demonstration
> of the pipeline, not as real Reddit findings, until the model is retrained on
> collected data.

## The problem

Most "predict virality" student projects have a subtle but serious bug: they scrape
score/comments whenever the scraper happens to run, which leaks *time-since-posting*
into the label (a 2-hour-old post looks "less viral" than a 2-day-old one purely
because it's had less time to accumulate votes). They also often threshold "viral"
globally, which mostly just teaches the model to recognize which subreddit a post
is from (r/AskReddit posts get 100x the upvotes of r/MachineLearning posts as a
matter of subreddit size, not content quality).

This project fixes both:
- **Fixed observation window**: every post's target is captured at exactly 24 hours
  after submission, not "whenever I scraped it."
- **Per-subreddit relative threshold**: "viral" = top 10% *within that subreddit's*
  score distribution, so the label reflects relative performance, not subreddit size.

## What it does

1. **Collects** Reddit posts via the Reddit API (PRAW), recording only features
   knowable at submission time (title, subreddit, flair, author karma/age, time posted).
2. **Waits** 24 hours, then snapshots score/comments as the prediction target.
3. **Engineers features**: title sentiment (VADER), title length/structure, TF-IDF
   text features, cyclical hour-of-day encoding, author metadata.
4. **Trains two models**:
   - Baseline: logistic regression on metadata only
   - Main model: XGBoost on metadata + TF-IDF title features, with class-imbalance
     handling via `scale_pos_weight`
5. **Explains predictions** with SHAP — both globally (which features matter overall)
   and per-prediction (why *this* title got *this* score).
6. **Serves it live** via a Streamlit dashboard: type a draft title, pick a subreddit
   and posting time, and see a predicted virality probability with the top
   contributing factors.

## Results

*(Numbers below are from the included synthetic dataset — see [Data](#data) for why,
and swap in `models/metrics.json` after training on real collected data.)*

| Model | ROC-AUC | PR-AUC | Precision (viral) | Recall (viral) |
|---|---|---|---|---|
| Baseline (logistic regression, metadata only) | 0.774 | 0.246 | 0.19 | 0.81 |
| XGBoost (metadata + TF-IDF) | 0.740 | 0.229 | 0.21 | 0.63 |

**Honest finding, not a cherry-picked one**: the metadata-only baseline actually edges
out the text-augmented XGBoost model on ROC-AUC here. That's a legitimate result worth
reporting rather than hiding — it suggests posting time and author reputation carry
more signal than title wording for this dataset, which is a reasonable prior for how
Reddit's ranking algorithm and default-subreddit dynamics actually work. XGBoost still
wins on precision/F1, so which model you'd ship depends on whether false positives or
false negatives matter more for the use case.

Class imbalance (10% positive rate by construction) means **accuracy is not reported
as a headline metric** — a model that always predicts "not viral" would score 90%
accuracy while being useless.

## Data

Real Reddit data requires a genuine 24-hour wall-clock observation window per post,
which doesn't fit a same-day setup. This repo ships with:
- `collect_data.py` — the real PRAW-based collector (seed + snapshot modes), ready
  to run against your own Reddit API credentials
- `generate_synthetic_data.py` — a synthetic data generator that produces data in the
  *exact same schema*, with realistic heavy-tailed score distributions and genuine
  (moderate, not perfect) signal in the underlying features, so the full pipeline
  is testable and demoable immediately

**To switch to real data**: run `collect_data.py --mode seed` daily for a couple of
weeks, then `collect_data.py --mode snapshot` to finalize posts as they age past 24h.
Once `data/reddit_posts_raw.csv` has real rows, `train_model.py` and `app.py` work
unchanged — the schema is identical.

## Project structure

```
reddit_virality/
├── collect_data.py            # Real Reddit data collection (PRAW)
├── generate_synthetic_data.py # Synthetic data generator (same schema)
├── features.py                # Feature engineering + target definition
├── train_model.py             # Model training, evaluation, SHAP
├── app.py                     # Streamlit dashboard
├── requirements.txt
├── .env.example                # Reddit API credential template
├── data/
│   └── reddit_posts_raw.csv   # Raw collected/generated posts
└── models/                    # Saved models, vectorizer, metrics (generated)
```

## Running it

```bash
pip install -r requirements.txt

# Option A: use the included synthetic dataset (already generated)
# Option B: collect real data (needs Reddit API credentials, see .env.example)
python collect_data.py --mode seed
# ... wait 24h+ ...
python collect_data.py --mode snapshot

# Train
python train_model.py

# Launch dashboard
streamlit run app.py
```

## Deployment

Live at: **https://junaidayesha92-code-redittviralitypre-reddit-viralityapp-t0z8iu.streamlit.app/**

Deployed free on [Streamlit Community Cloud](https://share.streamlit.io). To deploy
your own copy:
1. Push this repo to GitHub
2. Connect the repo at share.streamlit.io, point at `app.py`
3. Add Reddit API secrets under app settings if using live collection

## Design decisions worth asking about in an interview

- **Why 24-hour fixed window instead of "current score"?** Avoids leaking
  time-since-post into the label — see [The problem](#the-problem).
- **Why per-subreddit percentile instead of a global score threshold?** A global
  threshold mostly encodes "which subreddit is this," not "is this a strong post
  for its context." See [The problem](#the-problem).
- **Why is TF-IDF fit only on the training set?** Fitting on the full dataset before
  splitting would leak test-set vocabulary into the model's feature space.
- **Why `scale_pos_weight` instead of oversampling the minority class?** Oversampling
  duplicates the same rare positive examples, which encourages overfitting to them;
  reweighting the loss function achieves a similar imbalance correction without that risk.
- **Why report PR-AUC alongside ROC-AUC?** With a 10% positive rate, ROC-AUC can look
  deceptively good since the negative class dominates; PR-AUC is more sensitive to
  performance on the minority (viral) class, which is the one that actually matters here.

## What I'd do with more time

- Real embeddings (sentence-transformers) instead of TF-IDF for richer title semantics
- Track prediction performance over time as subreddit norms drift (concept drift)
- A/B-style backtesting: replay historical posts through the model in strict
  time order to check for any residual leakage
- Expand beyond title text to thumbnail/image features for image-heavy subreddits
