from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from data import get_fixtures, get_recent_results
from model import predict

app = FastAPI(title="EPL Predictor API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
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