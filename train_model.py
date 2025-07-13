# train_model.py
import pandas as pd
import requests
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
import joblib
import numpy as np

TRAINING_DATA_URL = "http://localhost:8080/api/v1/ml-data/training-set"
MODEL_FILENAME = "football_predictor_v1.joblib"


def fetch_training_data():
    print("Henter treningsdata fra backend...")
    try:
        response = requests.get(TRAINING_DATA_URL, timeout=60)
        response.raise_for_status()
        data = response.json()
        print(f"Mottok {len(data)} rader med data.")
        return data
    except requests.exceptions.RequestException as e:
        print(f"KUNNE IKKE HENTE DATA: {e}")
        return None


def train_model():
    training_data = fetch_training_data()
    if not training_data or len(training_data) < 10:
        print("Avslutter trening: Ikke nok data for å lage en meningsfull modell.")
        return

    df = pd.DataFrame(training_data)
    print("Konverterte data til DataFrame. Eksempel:")
    print(df.head())

    # --- START PÅ MIDLERTIDIG ENDRING FOR SMÅ DATASETT ---
    # Vi gjør om problemet til et binært problem: Hjemmeseier (1) eller ikke (0)

    print("\nADVARSEL: Kjører i midlertidig 'binær' modus på grunn av lite data.")
    df['result_binary'] = df['result'].apply(lambda x: 1 if x == 'HOME_WIN' else 0)

    features = df.drop(columns=['fixtureId', 'season', 'leagueId', 'result', 'result_binary'])
    target = df['result_binary']

    # Her trenger vi ikke LabelEncoder siden vi har laget 0 og 1 manuelt
    target_encoded = target

    print("\nFeatures som brukes for trening:")
    print(features.columns.tolist())
    print("\nTarget-klasser (binær): [0: Not Home Win, 1: Home Win]")
    # --- SLUTT PÅ MIDLERTIDIG ENDRING ---

    X_train, X_test, y_train, y_test = train_test_split(
        features, target_encoded, test_size=0.2, random_state=42
    )
    print(f"\nData splittet: {len(X_train)} rader for trening, {len(X_test)} for testing.")

    print("\nStarter modelltrening med XGBoost...")

    # Vi bruker nå en standard binær klassifiserer
    model = xgb.XGBClassifier(eval_metric='logloss')

    model.fit(X_train, y_train)
    print("Modelltrening fullført.")

    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"\nModellens nøyaktighet på testsettet: {accuracy * 100:.2f}%")

    feature_importances = pd.DataFrame(model.feature_importances_, index=features.columns,
                                       columns=['importance']).sort_values('importance', ascending=False)
    print("\nFeature Importance:")
    print(feature_importances)

    print(f"\nLagrer modell til fil: {MODEL_FILENAME}")
    joblib.dump(model, MODEL_FILENAME)
    # Vi trenger ikke lagre label_encoder for denne binære modellen
    print("Modell lagret. Klar til bruk i Flask-appen.")


if __name__ == '__main__':
    train_model()