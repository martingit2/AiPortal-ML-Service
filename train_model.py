# train_model.py
import pandas as pd
import requests
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import joblib
import numpy as np

# URL til backend-endepunktet som serverer det rike treningsdatasettet
TRAINING_DATA_URL = "http://localhost:8080/api/v1/ml-data/training-set"
MODEL_FILENAME = "football_predictor_v2.joblib"

def fetch_training_data():
    """Henter treningsdata fra Spring Boot-backenden."""
    print("Henter treningsdata fra backend...")
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
    """Hovedfunksjon for å hente data, trene, evaluere og lagre modellen."""
    training_data = fetch_training_data()
    if not training_data or len(training_data) < 50:
        print("Avslutter trening: Ikke nok data.")
        return

    df = pd.DataFrame(training_data)
    print("\nBeskrivelse av mottatt data:")
    print(df.describe())

    if df['homePlayersAvgRating'].sum() == 0 or df['awayPlayersAvgRating'].sum() == 0:
        print("\nADVARSEL: Spiller-spesifikke features (rating/mål) er bare nuller.")
        print("Dette indikerer at datainnsamlingen for spillerstatistikk ikke er fullført.")

    df['result_binary'] = df['result'].apply(lambda x: 1 if x == 'HOME_WIN' else 0)
    features = df.drop(columns=['fixtureId', 'season', 'leagueId', 'result', 'result_binary'])
    target = df['result_binary']

    print("\nFeatures som brukes for trening:")
    print(features.columns.tolist())

    X_train, X_test, y_train, y_test = train_test_split(
        features, target, test_size=0.2, random_state=42, stratify=target
    )
    print(f"\nData splittet: {len(X_train)} rader for trening, {len(X_test)} for testing.")

    print("\nStarter modelltrening med XGBoost...")
    model = xgb.XGBClassifier(eval_metric='logloss', use_label_encoder=False)
    model.fit(X_train, y_train)
    print("Modelltrening fullført.")

    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"\nModellens nøyaktighet på testsettet: {accuracy * 100:.2f}%")

    feature_importances = pd.DataFrame(model.feature_importances_, index=features.columns,
                                       columns=['importance']).sort_values('importance', ascending=False)
    print("\nFeature Importance:")
    print(feature_importances)

    print(f"\nLagrer trent modell til fil: {MODEL_FILENAME}")
    joblib.dump(model, MODEL_FILENAME)
    print(f"Modell lagret som '{MODEL_FILENAME}'. Restart Flask-appen for å laste inn den nye modellen.")

if __name__ == '__main__':
    train_model()