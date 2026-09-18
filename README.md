# Game Analytics & Retention Modeling

## Project Overview

This project analyzes game event data to understand player engagement,
retention, and spending behavior. It also builds a machine learning model
to identify players who may be at risk based on inactivity.

The project includes:

- Exploratory Data Analysis (EDA)
- Player-level feature engineering
- At-risk player classification
- Cross-validation and model evaluation
- Hyperparameter tuning
- MySQL database integration
- FastAPI backend
- Prediction API

---

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

The dataset contains game events including:

- session_start
- session_end
- level_start
- level_complete
- level_fail
- purchase

The dataset contains 4,518 events from 180 players over a 14-day period.

---

## Part 1 — Exploratory Data Analysis

The following metrics were calculated:

- Daily Active Users (DAU)
- Level completion rate
- Average session duration
- Total revenue
- Paying players
- ARPPU
- D1 retention

### Key Results

- Average DAU: approximately 44
- Maximum DAU: 66
- Total revenue: $169.80
- Paying players: 11
- ARPPU: $15.44

### Session Duration

Session duration was calculated by pairing `session_start` and `session_end`
events using `session_id`.

The dataset contains 758 sessions, and all 758 sessions have both start and
end events.

The overall mean session duration was approximately 453 seconds
(about 7.55 minutes).

### Deeper Analysis

#### Android vs iOS Session Duration

A Welch two-sample t-test was used to compare mean session duration
between Android and iOS users.

The test did not provide sufficient statistical evidence of a difference
in mean session duration at the 5% significance level.

#### Engagement vs Spending

Pearson correlation was used to examine associations between player
engagement features and total spending.

Sessions count showed the strongest positive association with total spend,
although the relationship was weak.

These results represent associations and should not be interpreted as
evidence of causation.

---

## Part 2 — Feature Engineering

One row was created for each player.

Features:

- sessions_count_total
- avg_session_duration_sec
- levels_completed
- levels_failed
- purchases_count
- total_spend
- days_since_install
- days_since_last_active

### At-Risk Definition

A player is labelled at risk when:

days_since_last_active > 3

This threshold was selected because the dataset covers only 14 days and
a 3-day inactivity period provides a practical separation while retaining
a reasonably balanced target distribution.

### Leakage Check

`days_since_last_active` was not included in the model predictors because
it directly defines the target label.

Including it would allow the model to reproduce the labeling rule instead
of learning meaningful behavioral patterns.

---

## Model

A Random Forest Classifier was used.

Cross-validation was performed using 5-fold StratifiedKFold.

The following metrics were evaluated:

- Accuracy
- Precision
- Recall
- F1-score
- ROC-AUC

### Hyperparameter Tuning

The `max_depth` parameter was tuned using several candidate values.

The selected model used:

- n_estimators = 200
- max_depth = 5
- class_weight = balanced
- random_state = 42

After tuning, the cross-validated results were approximately:

- Accuracy: 0.794
- Precision: 0.816
- Recall: 0.818
- F1-score: 0.810
- ROC-AUC: 0.872

---

## Model Limitations

The model should not be treated as a production-ready churn model.

The dataset contains only 180 players and covers a short 14-day period.
The `at_risk` target is also a manually defined proxy based on inactivity
rather than an observed future churn outcome.

The model should therefore be validated using a larger future dataset and
a genuine future-based churn or retention outcome before being used for
automated product decisions.

---

## Part 3 — FastAPI

The API provides three main endpoints.
### GET /metrics/summary

Returns the main game analytics KPIs:

- Total players
- Total revenue
- Paying players
- ARPPU
- DAU trend
- Level completion rate by level
- D1 retention by cohort

### GET /players/{id}/risk

Returns the stored player's features together with:

- At-risk prediction
- Risk probability

Returns HTTP 404 when the player does not exist.

### POST /predict

Accepts an arbitrary player feature set and returns:

- At-risk prediction
- Risk probability
The request accepts the same seven model features used during training:

- sessions_count_total
- avg_session_duration_sec
- levels_completed
- levels_failed
- purchases_count
- total_spend
- days_since_install

`days_since_last_active` is intentionally not required for prediction because
it directly defines the at-risk label and would cause target leakage.

---

### D1 Retention Definition

D1 retention was calculated using a cohort-based approach.

For each player, the date of their first recorded event was treated as their
cohort date. A player was considered retained on D1 if they generated at least
one event on the following calendar day.

The final cohort day was excluded because the dataset does not contain the
following day for that cohort.


## Database

MySQL is used to persist the engineered player-level feature table.

Database:

game_analytics

Table:

player_features

---

The .env file contains database credentials and should not be committed
to version control.


**Do not actually include `.env` in the GitHub repository.** Your `.gitignore` already handles that.

---

## 11. Add setup/run instructions

This is currently the biggest missing practical section.

Add:

```markdown
## Setup and Installation

### 1. Create and activate virtual environment

```bash
python -m venv .venv
.venv\Scripts\activate
2. Install dependencies
pip install -r requirements.txt
3. Configure environment variables

Create a .env file:

DATABASE_URL=mysql+pymysql://username:password@localhost:3306/game_analytics
4. Create the MySQL database

Create a database named:

CREATE DATABASE game_analytics;
5. Load the database
python load_database.py
6. Start the FastAPI application
uvicorn main:app --reload

The API documentation is available through the FastAPI Swagger interface.


---

## 12. Add a Known Limitations section

Your existing Model Limitations is good. I'd rename it:

```markdown
## Known Limitations

- The dataset contains only 180 players.
- The observation period is only 14 days.
- The at-risk label is a manually defined inactivity proxy rather than a
  genuine future churn outcome.
- There is no separate future holdout dataset.
- `days_since_install` has high feature importance and may reflect player
  lifecycle effects within the short observation window.
- Correlation analysis identifies associations and does not establish
  causality.
- The model should be validated on a larger future dataset before production
  deployment.



