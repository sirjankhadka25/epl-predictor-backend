import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier
import joblib
import os

def train_and_save():
    print("Training model...")

    seasons = [
        "https://www.football-data.co.uk/mmz4281/0001/E0.csv",
        "https://www.football-data.co.uk/mmz4281/0102/E0.csv",
        "https://www.football-data.co.uk/mmz4281/0203/E0.csv",
        "https://www.football-data.co.uk/mmz4281/0304/E0.csv",
        "https://www.football-data.co.uk/mmz4281/0405/E0.csv",
        "https://www.football-data.co.uk/mmz4281/0506/E0.csv",
        "https://www.football-data.co.uk/mmz4281/0607/E0.csv",
        "https://www.football-data.co.uk/mmz4281/0708/E0.csv",
        "https://www.football-data.co.uk/mmz4281/0809/E0.csv",
        "https://www.football-data.co.uk/mmz4281/0910/E0.csv",
        "https://www.football-data.co.uk/mmz4281/1011/E0.csv",
        "https://www.football-data.co.uk/mmz4281/1112/E0.csv",
        "https://www.football-data.co.uk/mmz4281/1213/E0.csv",
        "https://www.football-data.co.uk/mmz4281/1314/E0.csv",
        "https://www.football-data.co.uk/mmz4281/1415/E0.csv",
        "https://www.football-data.co.uk/mmz4281/1516/E0.csv",
        "https://www.football-data.co.uk/mmz4281/1617/E0.csv",
        "https://www.football-data.co.uk/mmz4281/1718/E0.csv",
        "https://www.football-data.co.uk/mmz4281/1819/E0.csv",
        "https://www.football-data.co.uk/mmz4281/1920/E0.csv",
        "https://www.football-data.co.uk/mmz4281/2021/E0.csv",
        "https://www.football-data.co.uk/mmz4281/2122/E0.csv",
        "https://www.football-data.co.uk/mmz4281/2223/E0.csv",
        "https://www.football-data.co.uk/mmz4281/2324/E0.csv",
        "https://www.football-data.co.uk/mmz4281/2425/E0.csv",
    ]

    dfs = []
    for url in seasons:
        try:
            df = pd.read_csv(url, encoding='latin-1', on_bad_lines='skip')
            dfs.append(df)
            print(f"Loaded {url.split('/')[-2]}")
        except Exception as e:
            print(f"Skipped {url}: {e}")

    raw_df = pd.concat(dfs, ignore_index=True)

    cols = ['FTHG','FTAG','HTHG','HTAG','HS','AS','HST','AST','HC','AC','HF','AF','HY','AY','HR','AR','FTR']
    df = raw_df[cols].copy().dropna()
    df['Result'] = df['FTR'].map({'H': 2, 'D': 1, 'A': 0})

    df['ht_goal_diff']        = df['HTHG'] - df['HTAG']
    df['shot_diff']           = df['HS']   - df['AS']
    df['shot_on_target_diff'] = df['HST']  - df['AST']
    df['corner_diff']         = df['HC']   - df['AC']
    df['foul_diff']           = df['HF']   - df['AF']
    df['yellow_diff']         = df['HY']   - df['AY']
    df['red_diff']            = df['HR']   - df['AR']

    features = [
        'ht_goal_diff', 'shot_diff', 'shot_on_target_diff',
        'corner_diff', 'foul_diff', 'yellow_diff', 'red_diff',
        'HST', 'AST', 'HF', 'AF'
    ]

    X = df[features]
    y = df['Result']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)

    model = XGBClassifier(
        n_estimators=300,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.8,
        random_state=42,
        eval_metric='mlogloss',
        verbosity=0
    )
    model.fit(X_train, y_train)

    joblib.dump(model,    'xgb_model.pkl')
    joblib.dump(scaler,   'scaler.pkl')
    joblib.dump(features, 'features.pkl')
    print("Model saved!")

if __name__ == "__main__":
    train_and_save()