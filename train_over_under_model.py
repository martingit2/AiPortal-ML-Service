# train_over_under_model.py
import pandas as pd
import requests
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, log_loss
import joblib
import json

TRAINING_DATA_URL = "http://localhost:8080/api/v1/ml-data/training-set"
REGISTER_MODEL_URL = "http://localhost:8080/api/v1/models/register"
MODEL_FILENAME = "over_under_v3_possession.joblib"

CANONICAL_FEATURES = [
    'homeAvgShotsOnGoal', 'homeAvgShotsOffGoal', 'homeAvgCorners', 'homeInjuries',
    'homePlayersAvgRating', 'homePlayersAvgGoals', 'homeAvgPossession',
    'awayAvgShotsOnGoal', 'awayAvgShotsOffGoal', 'awayAvgCorners', 'awayInjuries',
    'awayPlayersAvgRating', 'awayPlayersAvgGoals', 'awayAvgPossession',
    'h2hHomeWinPercentage', 'h2hAwayWinPercentage', 'h2hDrawPercentage', 'h2hAvgGoals'
]


def fetch_training_data():
    """Henter treningsdata fra Spring Boot-backenden."""
    print("Henter treningsdata for Over/Under-modell (med Possession)...")
    try:
        response = requests.get(TRAINING_DATA_URL, timeout=120)
        response.raise_for_status()
        data = response.json()
        print(f"Mottok {len(data)} rader med data.")
        return data
    except requests.exceptions.RequestException as e:
        print(f"KUNNE IKKE HENTE DATA: {e}")
        return None


def register_model_results(model_name, market_type, accuracy, log_loss_score, report, importances):
    """Sender modellens resultater til backend for lagring."""
    payload = {
        "modelName": model_name,
        "marketType": market_type,
        "accuracy": accuracy,
        "logLoss": log_loss_score,
        "classificationReport": report,
        "featureImportances": importances.to_json(orient='split')
    }
    try:
        response = requests.post(REGISTER_MODEL_URL, json=payload, timeout=10)
        response.raise_for_status()
        print(f"Resultater for modell '{model_name}' ble registrert i databasen.")
    except requests.exceptions.RequestException as e:
        print(f"ADVARSEL: Kunne ikke registrere modellresultater. Feil: {e}")


def train_model():
    """Hovedfunksjon for å trene en BINÆR modell for Over/Under 2.5 mål."""
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

    print("\nFeatures som brukes for trening (med Possession):")
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

    y_pred_proba = model.predict_proba(X_test)
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    loss = log_loss(y_test, y_pred_proba)

    print(f"\nModellens nøyaktighet på testsettet: {accuracy * 100:.2f}%")
    print(f"Log Loss på testsettet: {loss:.4f}")

    report_string = classification_report(y_test, y_pred, target_names=['Under 2.5', 'Over 2.5'])
    print("\nClassification Report:")
    print(report_string)

    feature_importances = pd.DataFrame(
        model.feature_importances_,
        index=features.columns,
        columns=['importance']
    ).sort_values('importance', ascending=False)

    print("\nFeature Importance:")
    print(feature_importances.head(10))

    joblib.dump(model, MODEL_FILENAME)
    print(f"\nModell '{MODEL_FILENAME}' er lagret.")

    # Registrer resultatene
    register_model_results(
        model_name=MODEL_FILENAME,
        market_type="OVER_UNDER_2.5",
        accuracy=accuracy,
        log_loss_score=loss,
        report=report_string,
        importances=feature_importances
    )


if __name__ == '__main__':
    train_model()