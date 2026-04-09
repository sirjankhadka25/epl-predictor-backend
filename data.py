import requests
import os
from dotenv import load_dotenv

load_dotenv()

API_KEY  = os.getenv("FOOTBALL_API_KEY")
BASE_URL = "https://api.football-data.org/v4"
HEADERS  = {"X-Auth-Token": API_KEY}
EPL_ID   = 2021

def get_fixtures():
    url      = f"{BASE_URL}/competitions/{EPL_ID}/matches"
    params   = {"status": "SCHEDULED"}
    response = requests.get(url, headers=HEADERS, params=params)
    response.raise_for_status()
    matches  = response.json().get("matches", [])

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
    url      = f"{BASE_URL}/competitions/{EPL_ID}/matches"
    params   = {"status": "FINISHED"}
    response = requests.get(url, headers=HEADERS, params=params)
    response.raise_for_status()
    matches  = response.json().get("matches", [])

    team_matches = [
        m for m in matches
        if m["homeTeam"]["name"] == team_name
        or m["awayTeam"]["name"] == team_name
    ]
    return team_matches[-limit:]