import joblib
import numpy as np
import os

BASE_DIR  = os.path.dirname(__file__)

xgb_model = joblib.load(os.path.join(BASE_DIR, "xgb_model.pkl"))
scaler     = joblib.load(os.path.join(BASE_DIR, "scaler.pkl"))
features   = joblib.load(os.path.join(BASE_DIR, "features.pkl"))

RESULT_MAP   = {0: "A", 1: "D", 2: "H"}
RESULT_LABEL = {"H": "Home win", "D": "Draw", "A": "Away win"}

def avg_goals(matches, team):
    vals = []
    for m in matches:
        score = m.get("score", {}).get("fullTime", {})
        if m["homeTeam"]["name"] == team:
            vals.append(score.get("home") or 0)
        else:
            vals.append(score.get("away") or 0)
    return np.mean(vals) if vals else 1.2

def build_features(home_team: str, away_team: str, recent_results: dict):
    home_goals = avg_goals(recent_results.get(home_team, []), home_team)
    away_goals = avg_goals(recent_results.get(away_team, []), away_team)

    feat = {
        "ht_goal_diff":        home_goals - away_goals,
        "shot_diff":           home_goals * 5 - away_goals * 5,
        "shot_on_target_diff": home_goals * 3 - away_goals * 3,
        "corner_diff":         home_goals * 2 - away_goals * 2,
        "foul_diff":           0.0,
        "yellow_diff":         0.0,
        "red_diff":            0.0,
        "HST":                 home_goals * 3,
        "AST":                 away_goals * 3,
        "HF":                  10.0,
        "AF":                  10.0,
    }

    vector = np.array([[feat[f] for f in features]])
    return vector

def predict(home_team: str, away_team: str, recent_results: dict):
    X         = build_features(home_team, away_team, recent_results)
    probs     = xgb_model.predict_proba(X)[0]
    pred_idx  = int(np.argmax(probs))
    pred_code = RESULT_MAP[pred_idx]

    return {
        "prediction": pred_code,
        "label":      RESULT_LABEL[pred_code],
        "confidence": round(float(probs[pred_idx]) * 100, 1),
        "probabilities": {
            "home": round(float(probs[2]) * 100, 1),
            "draw": round(float(probs[1]) * 100, 1),
            "away": round(float(probs[0]) * 100, 1),
        }
    }