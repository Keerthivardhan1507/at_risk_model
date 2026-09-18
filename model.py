"""
Feature engineering and at-risk player model.

This script follows the feature engineering and Random Forest logic
used in the EDA notebook for the Game Analytics & Retention assessment.
"""

import os
import pickle

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_validate


DATA_PATH = os.path.join("data", "game_events.csv")
MODEL_DIR = os.path.join("notebook", "models")
FEATURES_DIR = os.path.join("notebook", "database")

FEATURE_COLUMNS = [
    "sessions_count_total",
    "avg_session_duration_sec",
    "levels_completed",
    "levels_failed",
    "purchases_count",
    "total_spend",
    "days_since_install",
]


def build_player_features(df: pd.DataFrame) -> pd.DataFrame:
    """Build the player-level feature table using vectorized groupby operations."""

    df = df.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"])

    # Session-level duration
    
    session_events = df[
    df["event_name"].isin(["session_start", "session_end"])
].copy()

    sessions = (
    session_events
    .groupby(["player_id", "session_id", "event_name"])["timestamp"]
    .min()
    .unstack("event_name")
    .reset_index()
)

    sessions["duration_sec"] = (
    sessions["session_end"] - sessions["session_start"]
).dt.total_seconds()

    session_features = (
    sessions.groupby("player_id")["duration_sec"]
    .agg(
        sessions_count_total="count",
        avg_session_duration_sec="mean",
    )
    .reset_index()
)

    # Level features
    level_features = (
        df.assign(
            completed=(df["event_name"] == "level_complete").astype(int),
            failed=(df["event_name"] == "level_fail").astype(int),
        )
        .groupby("player_id")
        .agg(
            levels_completed=("completed", "sum"),
            levels_failed=("failed", "sum"),
        )
        .reset_index()
    )

    # Purchase features
    purchase_features = (
        df.assign(
            purchase_flag=(df["event_name"] == "purchase").astype(int),
            purchase_revenue=df["revenue"].fillna(0),
        )
        .groupby("player_id")
        .agg(
            purchases_count=("purchase_flag", "sum"),
            total_spend=("purchase_revenue", "sum"),
        )
        .reset_index()
    )

    # Activity/lifecycle features
    player_activity = (
        df.groupby("player_id")["timestamp"]
        .agg(
            first_event_time="min",
            last_event_time="max",
        )
        .reset_index()
    )

    dataset_latest_timestamp = df["timestamp"].max()

    player_activity["days_since_install"] = (
        dataset_latest_timestamp - player_activity["first_event_time"]
    ).dt.total_seconds() / (24 * 60 * 60)

    player_activity["days_since_last_active"] = (
        dataset_latest_timestamp - player_activity["last_event_time"]
    ).dt.total_seconds() / (24 * 60 * 60)

    # Merge all player-level features
    player_features = (
        df[["player_id"]]
        .drop_duplicates()
        .merge(session_features, on="player_id", how="left")
        .merge(level_features, on="player_id", how="left")
        .merge(purchase_features, on="player_id", how="left")
        .merge(player_activity, on="player_id", how="left")
    )

    zero_features = [
        "sessions_count_total",
        "avg_session_duration_sec",
        "levels_completed",
        "levels_failed",
        "purchases_count",
        "total_spend",
    ]

    player_features[zero_features] = (
        player_features[zero_features].fillna(0)
    )

    # At-risk definition used in the assessment
    player_features["at_risk"] = (
        player_features["days_since_last_active"] > 3
    ).astype(int)

    return player_features


def evaluate_and_train(player_features: pd.DataFrame):
    """Run 5-fold CV, light tuning, and train the final Random Forest."""

    X = player_features[FEATURE_COLUMNS]
    y = player_features["at_risk"]

    cv = StratifiedKFold(
        n_splits=5,
        shuffle=True,
        random_state=42,
    )

    scoring = [
        "accuracy",
        "precision",
        "recall",
        "f1",
        "roc_auc",
    ]

    # Baseline model
    baseline_model = RandomForestClassifier(
        n_estimators=200,
        random_state=42,
        class_weight="balanced",
    )

    baseline_scores = cross_validate(
        baseline_model,
        X,
        y,
        cv=cv,
        scoring=scoring,
    )

    print("Baseline CV results:")
    for metric in scoring:
        print(
            f"{metric}: "
            f"{baseline_scores['test_' + metric].mean():.4f} "
            f"+/- {baseline_scores['test_' + metric].std():.4f}"
        )

    # Light hyperparameter tuning
    depths = [3, 5, 10, 15, None]
    tuning_results = []

    for depth in depths:
        tuned_model = RandomForestClassifier(
            n_estimators=200,
            max_depth=depth,
            random_state=42,
            class_weight="balanced",
        )

        scores = cross_validate(
            tuned_model,
            X,
            y,
            cv=cv,
            scoring=scoring,
        )

        tuning_results.append(
            {
                "max_depth": depth,
                "accuracy": scores["test_accuracy"].mean(),
                "precision": scores["test_precision"].mean(),
                "recall": scores["test_recall"].mean(),
                "f1": scores["test_f1"].mean(),
                "roc_auc": scores["test_roc_auc"].mean(),
            }
        )

    tuning_results_df = pd.DataFrame(tuning_results)
    print("\nHyperparameter tuning:")
    print(tuning_results_df)

    # Selected model: max_depth=5 based on the notebook's CV results
    final_model = RandomForestClassifier(
        n_estimators=200,
        max_depth=5,
        random_state=42,
        class_weight="balanced",
    )

    final_model.fit(X, y)

    return final_model, baseline_scores, tuning_results_df


def save_outputs(player_features, final_model):
    """Persist player features, model, and model metadata."""

    os.makedirs(MODEL_DIR, exist_ok=True)
    os.makedirs(FEATURES_DIR, exist_ok=True)

    features_path = os.path.join(FEATURES_DIR, "player_features.csv")
    model_path = os.path.join(MODEL_DIR, "at_risk_model.pkl")
    metadata_path = os.path.join(MODEL_DIR, "model_metadata.pkl")

    player_features.to_csv(features_path, index=False)

    with open(model_path, "wb") as file:
        pickle.dump(final_model, file)

    feature_metadata = {
        "features": FEATURE_COLUMNS,
        "at_risk_rule": "days_since_last_active > 3",
        "max_depth": 5,
    }

    with open(metadata_path, "wb") as file:
        pickle.dump(feature_metadata, file)

    print(f"\nPlayer features saved to: {features_path}")
    print(f"Model saved to: {model_path}")
    print(f"Model metadata saved to: {metadata_path}")


def main():
    df = pd.read_csv(DATA_PATH)
    player_features = build_player_features(df)

    print("Player feature table shape:", player_features.shape)
    print("\nTarget distribution:")
    print(player_features["at_risk"].value_counts())

    final_model, _, _ = evaluate_and_train(player_features)

    print("\nFinal model feature importance:")
    importance = pd.Series(
        final_model.feature_importances_,
        index=FEATURE_COLUMNS,
    ).sort_values(ascending=False)
    print(importance)

    save_outputs(player_features, final_model)


if __name__ == "__main__":
    main()