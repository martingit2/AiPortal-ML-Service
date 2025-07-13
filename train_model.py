# train_model.py
import pandas as pd
import requests
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
import joblib
import numpy as np

# URL til backend-endepunktet som serverer treningsdata
TRAINING_DATA_URL = "http://localhost:8080/api/v1/ml-data/training-set"

# Navn på filen vi vil lagre den ferdige modellen som
MODEL_FILENAME = "football_predictor_v1.joblib"


def fetch_training_data():
    """Henter treningsdata fra Spring Boot-backenden."""
    print("Henter treningsdata fra backend... (Dette kan ta 30-60 minutter, vær tålmodig)")
    try:
        # Setter timeout til 1 time (3600 sekunder) for å være helt sikker
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
    if not training_data or len(training_data) < 20:  # Øker minimumskravet litt
        print("Avslutter trening: Ikke nok data for å lage en meningsfull modell.")
        return

    df = pd.DataFrame(training_data)
    print("Konverterte data til DataFrame. Eksempel:")
    print(df.head())
    print("\nBeskrivelse av data:")
    print(df.describe())  # Gir en statistisk oversikt over dataen

    # For nå, la oss holde oss til den binære modellen til vi har verifisert pipelinen
    print("\nADVARSEL: Kjører i midlertidig 'binær' modus (HOME_WIN vs NOT_HOME_WIN).")
    df['result_binary'] = df['result'].apply(lambda x: 1 if x == 'HOME_WIN' else 0)

    features = df.drop(columns=['fixtureId', 'season', 'leagueId', 'result', 'result_binary'])
    target = df['result_binary']

    print("\nFeatures som brukes for trening:")
    print(features.columns.tolist())

    # Vi kan nå prøve stratify igjen, siden vi forventer mer data
    counts = np.bincount(target)
    min_class_count = counts.min() if len(counts) > 0 else 0

    stratify_param = None
    if min_class_count >= 2:
        print("\nNok data i alle klasser for stratifisering.")
        stratify_param = target
    else:
        print(f"\nADVARSEL: Minste klasse har kun {min_class_count} medlem(mer). Deaktiverer 'stratify'.")

    X_train, X_test, y_train, y_test = train_test_split(
        features, target, test_size=0.2, random_state=42, stratify=stratify_param
    )
    print(f"\nData splittet: {len(X_train)} rader for trening, {len(X_test)} for testing.")

    print("\nStarter modelltrening med XGBoost...")
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
    print("Modell lagret. Klar til bruk i Flask-appen.")


if __name__ == '__main__':
    train_model()