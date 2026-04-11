import requests
import os
import time
from dotenv import load_dotenv

load_dotenv()

API_KEY  = os.getenv("FOOTBALL_API_KEY")
BASE_URL = "https://api.football-data.org/v4"
HEADERS  = {"X-Auth-Token": API_KEY}
EPL_ID   = 2021

# Simple cache — stores (data, timestamp)
_cache = {}
CACHE_TTL = 300  # 5 minutes

def cached_get(url, params=None):
    key = url + str(params)
    now = time.time()
    if key in _cache:
        data, ts = _cache[key]
        if now - ts < CACHE_TTL:
            print(f"Cache hit: {url}")
            return data
    response = requests.get(url, headers=HEADERS, params=params)
    response.raise_for_status()
    data = response.json()
    _cache[key] = (data, now)
    return data

def get_fixtures():
    url    = f"{BASE_URL}/competitions/{EPL_ID}/matches"
    data   = cached_get(url, params={"status": "SCHEDULED"})
    matches = data.get("matches", [])
    fixtures = []
    for m in matches[:20]:
        fixtures.append({
            "id":       m["id"],
            "homeTeam": m["homeTeam"]["name"],
            "awayTeam": m["awayTeam"]["name"],
            "date":     m["utcDate"],
            "matchday": m["matchday"],
        })
    return fixtures

def get_recent_results(team_name: str, limit: int = 5):
    url     = f"{BASE_URL}/competitions/{EPL_ID}/matches"
    data    = cached_get(url, params={"status": "FINISHED"})
    matches = data.get("matches", [])
    team_matches = [
        m for m in matches
        if m["homeTeam"]["name"] == team_name
        or m["awayTeam"]["name"] == team_name
    ]
    return team_matches[-limit:]

def get_past_matches(limit: int = 30):
    url     = f"{BASE_URL}/competitions/{EPL_ID}/matches"
    data    = cached_get(url, params={"status": "FINISHED"})
    matches = data.get("matches", [])

    past = []
    for m in matches[-limit:]:
        score      = m.get("score", {}).get("fullTime", {})
        home_goals = score.get("home", 0) or 0
        away_goals = score.get("away", 0) or 0

        if home_goals > away_goals:
            actual = "H"
        elif away_goals > home_goals:
            actual = "A"
        else:
            actual = "D"

        past.append({
            "id":        m["id"],
            "homeTeam":  m["homeTeam"]["name"],
            "awayTeam":  m["awayTeam"]["name"],
            "date":      m["utcDate"],
            "matchday":  m["matchday"],
            "homeGoals": home_goals,
            "awayGoals": away_goals,
            "actual":    actual,
        })
    return past