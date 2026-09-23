from pathlib import Path
from fastapi import FastAPI,HTTPException
from sqlalchemy import text
from database import engine
import pickle
from pydantic import BaseModel
from contextlib import asynccontextmanager
from load_database import initialize_database

@asynccontextmanager
async def lifespan(app: FastAPI):
    print("Initializing database...")
    initialize_database()
    print("Database initialization complete.")
    yield

#----------------------------------------
# FASTAPI APPLICATION
#----------------------------------------

app = FastAPI(
    title="Game Analytics API",
    description="API for game player analytics and retention risk prediction",
    version="1.0",
    lifespan=lifespan
)

# --------------------------------------------------
# Load model once when application starts
# --------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "notebook" / "models" / "at_risk_model.pkl"

with open(MODEL_PATH, "rb") as file:
    model = pickle.load(file)


# --------------------------------------------------
# Pydantic models
# --------------------------------------------------

class PlayerFeatures(BaseModel):
    sessions_count_total: int
    avg_session_duration_sec: float
    levels_completed: int
    levels_failed: int
    purchases_count: int
    total_spend: float
    days_since_install: float
    
class predictionResponse(BaseModel):
    at_risk: int
    risk_probability: float
    risk_probability_percent: float
    
class MetricsSummaryResponse(BaseModel):
    total_players: int
    total_revenue: float
    paying_players: int
    arppu: float
    at_risk_players: int

# --------------------------------------------------
# Home endpoint
# --------------------------------------------------

@app.get("/")
def home():
    return {
        "message": "Game Analytics API is running"
    }
    
# --------------------------------------------------
# GET /metrics/summary
# --------------------------------------------------


@app.get("/metrics/summary")
def metrics_summary():
    
    with engine.connect() as connection:
        
        # -----------------------------------------
        # Total players
        # -----------------------------------------
        
        total_players = connection.execute(
            text("SELECT COUNT(DISTINCT player_id) FROM game_events")
        ).scalar()
        
        total_revenue = connection.execute(
            text("""SELECT COALESCE(SUM(revenue), 0) 
                 from game_events 
                 where event_name = 'purchase'
                 """)
        ).scalar()
        
        paying_players = connection.execute(
            text("""
                 select count(DISTINCT player_id)
                 from game_events
                 where event_name = 'purchase'
                 """)
        ).scalar()
        
        # -----------------------------------------
        # DAU trend
        # -----------------------------------------
        dau_result = connection.execute(
            text("""
                 select
                        DATE(timestamp) as event_date,
                        count(DISTINCT player_id) as dau
                    from game_events
                    group by DATE(timestamp)
                    order by event_date
                 """)
        ).mappings().all()
        
        level_result = connection.execute(
            text("""
                SELECT
                    level_number,
                    SUM(
                        CASE
                            WHEN event_name = 'level_complete'
                            THEN 1
                            ELSE 0
                        END
                    ) AS completed,

                    SUM(
                        CASE
                            WHEN event_name = 'level_fail'
                            THEN 1
                            ELSE 0
                        END
                    ) AS failed

                FROM game_events

                WHERE event_name IN (
                    'level_complete',
                    'level_fail'
                )

                GROUP BY level_number
                ORDER BY level_number
            """)
        ).mappings().all()
        
        retention_result = connection.execute(
            text("""
                WITH first_events AS (
                    SELECT
                        player_id,
                        DATE(MIN(timestamp)) AS cohort_date
                    FROM game_events
                    GROUP BY player_id
                ),

                retained_players AS (
                    SELECT DISTINCT
                        f.player_id,
                        f.cohort_date
                    FROM first_events f
                    JOIN game_events e
                        ON e.player_id = f.player_id
                        AND DATE(e.timestamp) =
                            DATE_ADD(
                                f.cohort_date,
                                INTERVAL 1 DAY
                            )
                )

                SELECT
                    f.cohort_date,
                    COUNT(*) AS cohort_players,
                    COUNT(r.player_id) AS retained_players

                FROM first_events f

                LEFT JOIN retained_players r
                    ON f.player_id = r.player_id
                    AND f.cohort_date = r.cohort_date
                WHERE f.cohort_date < (SELECT MAX(cohort_date) FROM first_events)
                GROUP BY f.cohort_date
                ORDER BY f.cohort_date
            """)
        ).mappings().all()
        
        # -----------------------------------------
        # Calculate ARPPU
        # -----------------------------------------

    total_revenue = float(total_revenue or 0)
    paying_players = int(paying_players or 0)

    arppu = (
        total_revenue / paying_players
        if paying_players > 0
        else 0
    )

    # -----------------------------------------
    # Format DAU
    # -----------------------------------------

    dau_trend = [
        {
            "date": str(row["event_date"]),
            "dau": int(row["dau"])
        }
        for row in dau_result
    ]

    # -----------------------------------------
    # Format level completion
    # -----------------------------------------

    level_completion = []

    for row in level_result:

        completed = int(row["completed"] or 0)
        failed = int(row["failed"] or 0)

        attempts = completed + failed

        completion_rate = (
            completed / attempts
            if attempts > 0
            else 0
        )

        level_completion.append({
            "level": int(row["level_number"]),
            "completed": completed,
            "failed": failed,
            "completion_rate": round(
                completion_rate, 4
            ),
            "completion_rate_percent": round(
                completion_rate * 100, 2
            )
        })

    # -----------------------------------------
    # Format D1 retention
    # -----------------------------------------

    d1_retention = []

    for row in retention_result:

        cohort_players = int(row["cohort_players"])
        retained_players = int(row["retained_players"])

        retention_rate = (
            retained_players / cohort_players
            if cohort_players > 0
            else 0
        )

        d1_retention.append({
            "cohort_date": str(row["cohort_date"]),
            "cohort_players": cohort_players,
            "retained_players": retained_players,
            "d1_retention": round(
                retention_rate, 4
            ),
            "d1_retention_percent": round(
                retention_rate * 100, 2
            )
        })

    # -----------------------------------------
    # Return API response
    # -----------------------------------------

    return {
        "total_players": int(total_players),
        "total_revenue": round(total_revenue, 2),
        "paying_players": paying_players,
        "arppu": round(arppu, 2),

        "dau_trend": dau_trend,

        "level_completion": level_completion,

        "d1_retention": d1_retention
    }
    
# --------------------------------------------------
# GET /players/{player_id}/risk
# --------------------------------------------------


@app.get("/players/{player_id}/risk")
def player_risk(player_id: int):

    # Get player features from MySQL
    with engine.connect() as connection:

        query = text("""
            SELECT
                player_id,
                sessions_count_total,
                avg_session_duration_sec,
                levels_completed,
                levels_failed,
                purchases_count,
                total_spend,
                days_since_install,
                days_since_last_active
            FROM player_features
            WHERE player_id = :player_id
        """)

        result = connection.execute(
            query,
            {"player_id": player_id}
        ).mappings().first()

    # Player not found
    if result is None:
        raise HTTPException(
            status_code=404,
            detail=f"Player with ID {player_id} not found"
        )

    # Convert database result to dictionary
    player = dict(result)

    # Features in exactly the same order used during training
    input_data = [[
        player["sessions_count_total"],
        player["avg_session_duration_sec"],
        player["levels_completed"],
        player["levels_failed"],
        player["purchases_count"],
        player["total_spend"],
        player["days_since_install"]
    ]]

    # Model prediction
    prediction = model.predict(input_data)[0]
    probability = model.predict_proba(input_data)[0][1]

    return {
        "player_id": player["player_id"],
        "features": {
            "sessions_count_total": player["sessions_count_total"],
            "avg_session_duration_sec": player["avg_session_duration_sec"],
            "levels_completed": player["levels_completed"],
            "levels_failed": player["levels_failed"],
            "purchases_count": player["purchases_count"],
            "total_spend": player["total_spend"],
            "days_since_install": player["days_since_install"],
            "days_since_last_active": player["days_since_last_active"]
        },
        "at_risk": int(prediction),
        "risk_probability": round(float(probability), 4),
        "risk_probability_percent": round(float(probability) * 100, 2)
    }
    
# --------------------------------------------------
# POST /predict
# --------------------------------------------------
    
@app.post("/predict")
def predict_risk(features: PlayerFeatures):

    input_data = [[
        features.sessions_count_total,
        features.avg_session_duration_sec,
        features.levels_completed,
        features.levels_failed,
        features.purchases_count,
        features.total_spend,
        features.days_since_install
    ]]

    prediction = model.predict(input_data)[0]
    probability = model.predict_proba(input_data)[0][1]

    return {
    "at_risk": int(prediction),
    "risk_probability": round(float(probability), 4),
    "risk_probability_percent": round(float(probability) * 100, 2)
}
