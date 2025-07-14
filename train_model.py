# train_model.py
import pandas as pd
import requests
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import LabelEncoder  # NY IMPORT
import joblib
import numpy as np

TRAINING_DATA_URL = "http://localhost:8080/api/v1/ml-data/training-set"
# NYTT MODELLNAVN FOR MULTIKLASSE-MODELLEN
MODEL_FILENAME = "football_predictor_v3_multiclass.joblib"
# NY FIL FOR Å LAGRE LABEL ENCODEREN
ENCODER_FILENAME = "result_encoder.joblib"


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
    """Hovedfunksjon for å trene en MULTIKLASSE-modell."""
    training_data = fetch_training_data()
    if not training_data or len(training_data) < 50:
        print("Avslutter trening: Ikke nok data.")
        return

    df = pd.DataFrame(training_data)

    # --- VIKTIG ENDRING: Gå fra binær til multiklasse-target ---
    # Vi bruker LabelEncoder for å konvertere "HOME_WIN", "DRAW", "AWAY_WIN"
    # til numeriske verdier (0, 1, 2).
    le = LabelEncoder()
    df['result_encoded'] = le.fit_transform(df['result'])

    # Skriv ut mappingen slik at vi vet hva tallene betyr
    print("\nLabel Encoding Mapping:")
    for index, class_label in enumerate(le.classes_):
        print(f"{class_label} -> {index}")

    features = df.drop(columns=['fixtureId', 'season', 'leagueId', 'result', 'result_encoded'])
    target = df['result_encoded']

    print("\nFeatures som brukes for trening:")
    print(features.columns.tolist())

    # Splitt data for trening og testing
    X_train, X_test, y_train, y_test = train_test_split(
        features, target, test_size=0.2, random_state=42, stratify=target
    )
    print(f"\nData splittet: {len(X_train)} rader for trening, {len(X_test)} for testing.")

    print("\nStarter modelltrening med XGBoost for multiklasse...")
    # For multiklasse spesifiserer vi objective='multi:softprob' og num_class
    model = xgb.XGBClassifier(
        objective='multi:softprob',
        num_class=len(le.classes_),  # Forteller modellen hvor mange klasser det er
        eval_metric='mlogloss',
        use_label_encoder=False
    )

    model.fit(X_train, y_train)
    print("Modelltrening fullført.")

    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"\nModellens nøyaktighet på testsettet: {accuracy * 100:.2f}%")

    # Vis en mer detaljert rapport for multiklasse
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=le.classes_))

    feature_importances = pd.DataFrame(model.feature_importances_, index=features.columns,
                                       columns=['importance']).sort_values('importance', ascending=False)
    print("\nFeature Importance:")
    print(feature_importances)

    print(f"\nLagrer trent modell til fil: {MODEL_FILENAME}")
    joblib.dump(model, MODEL_FILENAME)

    print(f"Lagrer LabelEncoder til fil: {ENCODER_FILENAME}")
    joblib.dump(le, ENCODER_FILENAME)  # Vi MÅ lagre encoderen for å kunne dekode senere

    print(f"Modeller lagret. Restart Flask-appen for å laste inn den nye '{MODEL_FILENAME}'-modellen.")


if __name__ == '__main__':
    train_model()