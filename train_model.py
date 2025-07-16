# train_model.py
import pandas as pd
import requests
import xgboost as xgb
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, classification_report, log_loss
from sklearn.preprocessing import LabelEncoder
from sklearn.calibration import CalibratedClassifierCV
import joblib
import json

TRAINING_DATA_URL = "http://localhost:8080/api/v1/ml-data/training-set"
REGISTER_MODEL_URL = "http://localhost:8080/api/v1/models/register"
MODEL_FILENAME = "football_predictor_v5_h2h.joblib"
ENCODER_FILENAME = "result_encoder_v5.joblib"

CANONICAL_FEATURES = [
    'homeAvgShotsOnGoal', 'homeAvgShotsOffGoal', 'homeAvgCorners', 'homeInjuries',
    'homePlayersAvgRating', 'homePlayersAvgGoals', 'homeAvgPossession',
    'awayAvgShotsOnGoal', 'awayAvgShotsOffGoal', 'awayAvgCorners', 'awayInjuries',
    'awayPlayersAvgRating', 'awayPlayersAvgGoals', 'awayAvgPossession',
    'h2hHomeWinPercentage', 'h2hAwayWinPercentage', 'h2hDrawPercentage', 'h2hAvgGoals'
]


def fetch_training_data():
    """Henter treningsdata fra Spring Boot-backenden."""
    print("Henter treningsdata for Kampvinner-modell (med H2H)...")
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
        "featureImportances": importances.to_json(orient='split')  # Konverterer til JSON-streng
    }
    try:
        response = requests.post(REGISTER_MODEL_URL, json=payload, timeout=10)
        response.raise_for_status()
        print(f"Resultater for modell '{model_name}' ble registrert i databasen.")
    except requests.exceptions.RequestException as e:
        print(f"ADVARSEL: Kunne ikke registrere modellresultater. Feil: {e}")


def train_model():
    """Hovedfunksjon for å TUNE og KALIBRERE en multiklasse-modell."""
    training_data = fetch_training_data()
    if not training_data or len(training_data) < 100:
        print("Avslutter trening: Ikke nok data.")
        return

    df = pd.DataFrame(training_data)

    le = LabelEncoder()
    df['result_encoded'] = le.fit_transform(df['result'])

    features_to_drop = ['fixtureId', 'season', 'leagueId', 'result', 'result_encoded', 'goalsHome', 'goalsAway']
    features = df.drop(columns=features_to_drop, errors='ignore').reindex(columns=CANONICAL_FEATURES, fill_value=0.0)
    target = df['result_encoded']

    print("\nFeatures som brukes for trening (med H2H & Possession):")
    print(features.columns.tolist())

    X_train, X_test, y_train, y_test = train_test_split(
        features, target, test_size=0.2, random_state=42, stratify=target
    )

    param_grid = {
        'max_depth': [3, 5, 7],
        'n_estimators': [100, 200],
        'learning_rate': [0.05, 0.1]
    }

    print("\nStarter hyperparameter-tuning med GridSearchCV...")
    xgb_model = xgb.XGBClassifier(
        objective='multi:softprob',
        eval_metric='mlogloss',
        use_label_encoder=False,
        random_state=42
    )

    grid_search = GridSearchCV(
        estimator=xgb_model,
        param_grid=param_grid,
        scoring='neg_log_loss',
        cv=3,
        verbose=2,
        n_jobs=-1
    )

    grid_search.fit(X_train, y_train)

    print(f"\nTuning fullført. Beste parametere funnet: {grid_search.best_params_}")

    best_xgb_model = grid_search.best_estimator_

    print("\nStarter kalibrering av den beste modellen...")
    calibrated_model = CalibratedClassifierCV(
        best_xgb_model,
        method='isotonic',
        cv='prefit'
    )

    calibrated_model.fit(X_train, y_train)
    print("Kalibrering fullført.")

    print("\n--- Evaluering av endelig H2H/Possession-modell ---")

    y_pred_proba = calibrated_model.predict_proba(X_test)
    y_pred = calibrated_model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    loss = log_loss(y_test, y_pred_proba)

    print(f"Nøyaktighet på testsettet: {accuracy * 100:.2f}%")
    print(f"Log Loss på testsettet: {loss:.4f} (lavere er bedre)")

    report_string = classification_report(y_test, y_pred, target_names=le.classes_)
    print("\nClassification Report:")
    print(report_string)

    feature_importances = pd.DataFrame(
        best_xgb_model.feature_importances_,
        index=features.columns,
        columns=['importance']
    ).sort_values('importance', ascending=False)

    print("\nFeature Importance:")
    print(feature_importances.head(10))

    print(f"\nLagrer endelig modell til fil: {MODEL_FILENAME}")
    joblib.dump(calibrated_model, MODEL_FILENAME)
    joblib.dump(le, ENCODER_FILENAME)
    print("Modeller lagret.")

    # Registrer resultatene i databasen
    register_model_results(
        model_name=MODEL_FILENAME,
        market_type="MATCH_WINNER",
        accuracy=accuracy,
        log_loss_score=loss,
        report=report_string,
        importances=feature_importances
    )


if __name__ == '__main__':
    train_model()