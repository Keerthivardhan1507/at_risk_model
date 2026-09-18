# Game Analytics & Retention Modeling

Analyzes game event data to understand player engagement, retention, and
spending behavior, and builds a machine learning model to identify players
who may be at risk based on inactivity.

## Contents

- Exploratory Data Analysis (EDA)
- Player-level feature engineering
- At-risk player classification
- Cross-validation and model evaluation
- Hyperparameter tuning
- MySQL database integration
- FastAPI backend with a prediction API

## Project Structure

```text
game_assignment/
├── data/
│   └── game_events.csv
├── notebook/
│   ├── 1.eda.ipynb
│   ├── database/
│   │   └── player_features.csv
│   └── models/
│       ├── at_risk_model.pkl
│       └── model_metadata.pkl
├── main.py
├── model.py
├── database.py
├── load_database.py
├── requirements.txt
├── README.md
└── .env
```

## Dataset

4,518 events from 180 players over a 14-day period, including:

- `session_start`
- `session_end`
- `level_start`
- `level_complete`
- `level_fail`
- `purchase`

---

## Setup and Installation

### 1. Create and activate a virtual environment

```bash
python -m venv .venv
.venv\Scripts\activate
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Configure environment variables

Create a `.env` file:

```
DATABASE_URL=mysql+pymysql://username:password@localhost:3306/game_analytics
```

> `.env` contains database credentials and should never be committed to
> version control. It's already covered by `.gitignore`.

### 4. Create the MySQL database

```sql
CREATE DATABASE game_analytics;
```

### 5. Load the database

```bash
python load_database.py
```

### 6. Start the FastAPI application

```bash
uvicorn main:app --reload
```

API documentation is available through the FastAPI Swagger interface.

---

## Part 1 — Exploratory Data Analysis

Metrics calculated: Daily Active Users (DAU), level completion rate, average
session duration, total revenue, paying players, ARPPU, and D1 retention.

### Key Results

| Metric | Value |
|---|---|
| Average DAU | ~44 |
| Maximum DAU | 66 |
| Total revenue | $169.80 |
| Paying players | 11 |
| ARPPU | $15.44 |

### Session Duration

Session duration was calculated by pairing `session_start` and `session_end`
events using `session_id`. All 758 sessions in the dataset have both start
and end events. Overall mean session duration was approximately 453 seconds
(~7.55 minutes).

### D1 Retention Definition

D1 retention was calculated using a cohort-based approach: for each player,
the date of their first recorded event was treated as their cohort date, and
a player was considered retained on D1 if they generated at least one event
on the following calendar day. The final cohort day was excluded, since the
dataset does not contain the following day for that cohort.

### Deeper Analysis

**Android vs. iOS session duration** — A Welch two-sample t-test compared
mean session duration between Android and iOS users. No statistically
significant difference was found at the 5% significance level.

**Engagement vs. spending** — Pearson correlation examined associations
between engagement features and total spending. Sessions count showed the
strongest positive association with total spend, though the relationship
was weak. These are associations only, not evidence of causation.

---

## Part 2 — Feature Engineering

One row was created per player, with the following features:

- `sessions_count_total`
- `avg_session_duration_sec`
- `levels_completed`
- `levels_failed`
- `purchases_count`
- `total_spend`
- `days_since_install`
- `days_since_last_active`

### At-Risk Definition

A player is labeled **at risk** when `days_since_last_active > 3`. This
threshold was chosen because the dataset spans only 14 days, and a 3-day
inactivity window gives a practical separation while keeping a reasonably
balanced target distribution.

### Leakage Check

`days_since_last_active` is excluded from the model predictors because it
directly defines the target label — including it would let the model
reproduce the labeling rule instead of learning meaningful behavioral
patterns.

---

## Model

A **Random Forest Classifier**, evaluated with 5-fold `StratifiedKFold`
cross-validation on accuracy, precision, recall, F1-score, and ROC-AUC.

### Hyperparameter Tuning

`max_depth` was tuned across several candidate values. The selected model:

- `n_estimators = 200`
- `max_depth = 5`
- `class_weight = balanced`
- `random_state = 42`

### Cross-Validated Results

| Metric | Score |
|---|---|
| Accuracy | 0.794 |
| Precision | 0.816 |
| Recall | 0.818 |
| F1-score | 0.810 |
| ROC-AUC | 0.872 |

---

## Part 3 — FastAPI

### `GET /metrics/summary`

Returns the main game analytics KPIs: total players, total revenue, paying
players, ARPPU, DAU trend, level completion rate by level, and D1 retention
by cohort.

### `GET /players/{id}/risk`

Returns a stored player's features plus their at-risk prediction and risk
probability. Returns HTTP 404 if the player does not exist.

### `POST /predict`

Accepts an arbitrary player feature set and returns an at-risk prediction
and risk probability. Accepts the same seven features used in training:

- `sessions_count_total`
- `avg_session_duration_sec`
- `levels_completed`
- `levels_failed`
- `purchases_count`
- `total_spend`
- `days_since_install`

`days_since_last_active` is intentionally excluded, since it directly
defines the at-risk label and would cause target leakage.

---

## Database

MySQL is used to persist the engineered player-level feature table.

- **Database:** `game_analytics`
- **Table:** `player_features`

---

## Known Limitations

- The dataset contains only 180 players.
- The observation period is only 14 days.
- The at-risk label is a manually defined inactivity proxy, not a genuine
  future churn outcome.
- There is no separate future holdout dataset.
- `days_since_install` has high feature importance and may reflect player
  lifecycle effects within the short observation window.
- Correlation analysis identifies associations only and does not establish
  causality.
- The model should be validated on a larger future dataset, with a genuine
  future-based churn or retention outcome, before use in production or
  automated product decisions.