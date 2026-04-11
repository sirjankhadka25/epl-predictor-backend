import os
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

if not os.path.exists("xgb_model.pkl"):
    print("Model not found — training now...")
    from train import train_and_save
    train_and_save()

from data import get_fixtures, get_recent_results, get_past_matches
from model import predict

app = FastAPI(title="EPL Predictor API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def root():
    return {"status": "EPL Predictor API is running"}

@app.get("/fixtures")
def fixtures():
    try:
        return get_fixtures()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/predict/{home_team}/{away_team}")
def predict_match(home_team: str, away_team: str):
    try:
        recent = {
            home_team: get_recent_results(home_team),
            away_team: get_recent_results(away_team),
        }
        result = predict(home_team, away_team, recent)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/results")
def results():
    try:
        past = get_past_matches(limit=30)
        enriched = []
        for m in past:
            try:
                recent = {
                    m["homeTeam"]: get_recent_results(m["homeTeam"]),
                    m["awayTeam"]: get_recent_results(m["awayTeam"]),
                }
                pred = predict(m["homeTeam"], m["awayTeam"], recent)
                m["prediction"]    = pred["prediction"]
                m["confidence"]    = pred["confidence"]
                m["probabilities"] = pred["probabilities"]
                m["correct"]       = pred["prediction"] == m["actual"]
            except:
                m["prediction"]    = None
                m["confidence"]    = None
                m["probabilities"] = None
                m["correct"]       = None
            enriched.append(m)
        return enriched
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))