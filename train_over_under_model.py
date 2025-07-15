# train_over_under_model.py
import pandas as pd
import requests
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import joblib

TRAINING_DATA_URL = "http://localhost:8080/api/v1/ml-data/training-set"
MODEL_FILENAME = "over_under_v2_h2h.joblib"  # <-- NYTT MODELLNAVN

CANONICAL_FEATURES = [
    'homeAvgShotsOnGoal', 'homeAvgShotsOffGoal', 'homeAvgCorners', 'homeInjuries',
    'homePlayersAvgRating', 'homePlayersAvgGoals',
    'awayAvgShotsOnGoal', 'awayAvgShotsOffGoal', 'awayAvgCorners', 'awayInjuries',
    'awayPlayersAvgRating', 'awayPlayersAvgGoals',
    'h2hHomeWinPercentage', 'h2hAwayWinPercentage', 'h2hDrawPercentage', 'h2hAvgGoals'
]


def fetch_training_data():
    print("Henter treningsdata for Over/Under-modell (med H2H)...")
    try:
        response = requests.get(TRAINING_DATA_URL, timeout=3600)
        response.raise_for_status()
        data = response.json()
        print(f"Mottok {len(data)} rader med data.")
        return data
    except requests.exceptions.RequestException as e:
        print(f"KUNNE IKKE HENTE DATA: {e}")
        return None


def train_model():
    training_data = fetch_training_data()
    if not training_data or len(training_data) < 50:
        print("Avslutter trening: Ikke nok data.")
        return

    df = pd.DataFrame(training_data)

    df = df.dropna(subset=['goalsHome', 'goalsAway'])
    df['total_goals'] = df['goalsHome'] + df['goalsAway']
    df['is_over_2_5'] = (df['total_goals'] > 2.5).astype(int)

    features_to_drop = ['fixtureId', 'season', 'leagueId', 'result', 'goalsHome', 'goalsAway', 'total_goals',
                        'is_over_2_5']
    features = df.drop(columns=features_to_drop, errors='ignore').reindex(columns=CANONICAL_FEATURES, fill_value=0.0)
    target = df['is_over_2_5']

    print("\nFeatures som brukes for trening (med H2H):")
    print(features.columns.tolist())

    X_train, X_test, y_train, y_test = train_test_split(
        features, target, test_size=0.2, random_state=42, stratify=target
    )

    print("\nStarter modelltrening med XGBoost...")
    model = xgb.XGBClassifier(
        objective='binary:logistic',
        eval_metric='logloss',
        use_label_encoder=False
    )
    model.fit(X_train, y_train)
    print("Modelltrening fullført.")

    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"\nModellens nøyaktighet på testsettet: {accuracy * 100:.2f}%")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=['Under 2.5', 'Over 2.5']))

    joblib.dump(model, MODEL_FILENAME)
    print(f"\nModell '{MODEL_FILENAME}' er lagret.")


if __name__ == '__main__':
    train_model()