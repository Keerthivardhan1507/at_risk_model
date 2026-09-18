import os
from pathlib import Path

import pandas as pd
from sqlalchemy import create_engine, inspect
from dotenv import load_dotenv


# Load environment variables
load_dotenv()

DATABASE_URL = os.getenv("DATABASE_URL")

if not DATABASE_URL:
    raise ValueError("DATABASE_URL is not set in the .env file")

engine = create_engine(DATABASE_URL)

BASE_DIR = Path(__file__).resolve().parent

FEATURES_PATH = BASE_DIR / "notebook" / "database" / "player_features.csv"
EVENTS_PATH = BASE_DIR / "data" / "game_events.csv"


def initialize_database():
    """
    Create and populate the required MySQL tables if they do not exist.
    """

    inspector = inspect(engine)
    existing_tables = inspector.get_table_names()

    # -----------------------------------------
    # 1. Load player features
    # -----------------------------------------

    if "player_features" not in existing_tables:

        df_features = pd.read_csv(FEATURES_PATH)

        df_features.to_sql(
            "player_features",
            con=engine,
            if_exists="replace",
            index=False
        )

        print("Player features table created.")
        print(f"Player feature rows inserted: {len(df_features)}")

    else:
        print("player_features table already exists. Skipping load.")

    # -----------------------------------------
    # 2. Load raw game events
    # -----------------------------------------

    if "game_events" not in existing_tables:

        df_events = pd.read_csv(EVENTS_PATH)

        df_events.to_sql(
            "game_events",
            con=engine,
            if_exists="replace",
            index=False
        )

        print("Game events table created.")
        print(f"Game event rows inserted: {len(df_events)}")

    else:
        print("game_events table already exists. Skipping load.")


# Allow manual execution
if __name__ == "__main__":
    initialize_database()