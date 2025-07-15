# train_model.py
import pandas as pd
import requests
import xgboost as xgb
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, classification_report, log_loss
from sklearn.preprocessing import LabelEncoder
from sklearn.calibration import CalibratedClassifierCV
import joblib

TRAINING_DATA_URL = "http://localhost:8080/api/v1/ml-data/training-set"
MODEL_FILENAME = "football_predictor_v5_h2h.joblib"  # <-- NYTT MODELLNAVN
ENCODER_FILENAME = "result_encoder_v5.joblib"  # <-- NY ENCODER-FIL

CANONICAL_FEATURES = [
    'homeAvgShotsOnGoal', 'homeAvgShotsOffGoal', 'homeAvgCorners', 'homeInjuries',
    'homePlayersAvgRating', 'homePlayersAvgGoals',
    'awayAvgShotsOnGoal', 'awayAvgShotsOffGoal', 'awayAvgCorners', 'awayInjuries',
    'awayPlayersAvgRating', 'awayPlayersAvgGoals',
    'h2hHomeWinPercentage', 'h2hAwayWinPercentage', 'h2hDrawPercentage', 'h2hAvgGoals'
]


def fetch_training_data():
    print("Henter treningsdata for Kampvinner-modell (med H2H)...")
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
    if not training_data or len(training_data) < 100:
        print("Avslutter trening: Ikke nok data.")
        return

    df = pd.DataFrame(training_data)

    le = LabelEncoder()
    df['result_encoded'] = le.fit_transform(df['result'])

    features_to_drop = ['fixtureId', 'season', 'leagueId', 'result', 'result_encoded', 'goalsHome', 'goalsAway']
    # Bruk .reindex for å sikre at alle kanoniske features er tilstede, selv om noen mangler i df
    features = df.drop(columns=features_to_drop, errors='ignore').reindex(columns=CANONICAL_FEATURES, fill_value=0.0)
    target = df['result_encoded']

    print("\nFeatures som brukes for trening (med H2H):")
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
        estimator=xgb_model, param_grid=param_grid, scoring='neg_log_loss', cv=3, verbose=2, n_jobs=-1
    )
    grid_search.fit(X_train, y_train)

    print(f"\nTuning fullført. Beste parametere funnet: {grid_search.best_params_}")

    best_xgb_model = grid_search.best_estimator_

    print("\nStarter kalibrering av den beste modellen...")
    calibrated_model = CalibratedClassifierCV(best_xgb_model, method='isotonic', cv='prefit')
    calibrated_model.fit(X_train, y_train)
    print("Kalibrering fullført.")

    print("\n--- Evaluering av endelig H2H-modell ---")
    y_pred_proba = calibrated_model.predict_proba(X_test)
    y_pred = calibrated_model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    loss = log_loss(y_test, y_pred_proba)

    print(f"Nøyaktighet på testsettet: {accuracy * 100:.2f}%")
    print(f"Log Loss på testsettet: {loss:.4f} (lavere er bedre)")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=le.classes_))

    print(f"\nLagrer endelig modell til fil: {MODEL_FILENAME}")
    joblib.dump(calibrated_model, MODEL_FILENAME)
    joblib.dump(le, ENCODER_FILENAME)
    print("\nProsessen er fullført. Restart Flask-appen.")


if __name__ == '__main__':
    train_model()