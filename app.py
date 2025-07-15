# app.py
from flask import Flask, request, jsonify
from textblob import TextBlob
import re
import joblib
import pandas as pd
import os

app = Flask(__name__)

# --- MODELL-LASTING VED OPPSTART ---
# Oppdaterte filnavn for å laste de nye H2H-modellene
MODEL_V3_FILENAME = "football_predictor_v5_h2h.joblib"
ENCODER_FILENAME = "result_encoder_v5.joblib"
MODEL_OU_FILENAME = "over_under_v2_h2h.joblib"

model_v3 = None
encoder_v3 = None
model_ou = None

# Den kanoniske feature-listen som nå inkluderer H2H
CANONICAL_FEATURES = [
    'homeAvgShotsOnGoal', 'homeAvgShotsOffGoal', 'homeAvgCorners', 'homeInjuries',
    'homePlayersAvgRating', 'homePlayersAvgGoals',
    'awayAvgShotsOnGoal', 'awayAvgShotsOffGoal', 'awayAvgCorners', 'awayInjuries',
    'awayPlayersAvgRating', 'awayPlayersAvgGoals',
    'h2hHomeWinPercentage', 'h2hAwayWinPercentage', 'h2hDrawPercentage', 'h2hAvgGoals'
]

# Last inn kampvinner-modellen
if os.path.exists(MODEL_V3_FILENAME):
    print(f"Laster inn lagret modell fra: {MODEL_V3_FILENAME}")
    model_v3 = joblib.load(MODEL_V3_FILENAME)
    print("Kampvinner-modell (H2H) lastet inn.")
else:
    print(f"ADVARSEL: Modellfilen '{MODEL_V3_FILENAME}' ble ikke funnet.")

if os.path.exists(ENCODER_FILENAME):
    print(f"Laster inn lagret encoder fra: {ENCODER_FILENAME}")
    encoder_v3 = joblib.load(ENCODER_FILENAME)
    print("Encoder lastet inn. Klasse-rekkefølge:", encoder_v3.classes_)
else:
    print(f"ADVARSEL: Encoder-filen '{ENCODER_FILENAME}' ble ikke funnet.")

if os.path.exists(MODEL_OU_FILENAME):
    print(f"Laster inn lagret Over/Under-modell fra: {MODEL_OU_FILENAME}")
    model_ou = joblib.load(MODEL_OU_FILENAME)
    print("Over/Under-modell (H2H) lastet inn.")
else:
    print(f"ADVARSEL: Over/Under-modellfilen '{MODEL_OU_FILENAME}' ble ikke funnet.")


def get_input_dataframe(request_data):
    if not request_data:
        return None, "Request body mangler"
    try:
        temp_df = pd.DataFrame([request_data])
        final_df = temp_df.reindex(columns=CANONICAL_FEATURES, fill_value=0.0)

        for col in final_df.columns:
            final_df[col] = pd.to_numeric(final_df[col], errors='coerce').fillna(0.0)

        return final_df, None

    except Exception as e:
        return None, f"Generisk feil ved laging av DataFrame: {str(e)}"


@app.route('/predict/match_outcome', methods=['POST'])
def predict_match_outcome():
    if model_v3 is None or encoder_v3 is None:
        return jsonify({'error': 'Kampvinner-modell eller encoder er ikke lastet inn.'}), 503

    input_df, error = get_input_dataframe(request.get_json())
    if error:
        return jsonify({'error': error}), 400

    try:
        probabilities = model_v3.predict_proba(input_df)[0]
        response_probabilities = {
            klass.lower(): float(prob)
            for klass, prob in zip(encoder_v3.classes_, probabilities)
        }

        response_data = {
            "prediction_model_version": MODEL_V3_FILENAME,
            "probabilities": response_probabilities
        }

        return jsonify(response_data)
    except Exception as e:
        print(f"Prediction Error in /match_outcome: {e}")
        return jsonify({'error': f'En feil oppstod: {str(e)}'}), 500


@app.route('/predict/over_under', methods=['POST'])
def predict_over_under():
    if model_ou is None:
        return jsonify({'error': 'Over/Under-modellen er ikke lastet inn.'}), 503

    input_df, error = get_input_dataframe(request.get_json())
    if error:
        return jsonify({'error': error}), 400

    try:
        probabilities = model_ou.predict_proba(input_df)[0]
        response_data = {
            "prediction_model_version": MODEL_OU_FILENAME,
            "probabilities": {
                "under_2_5": float(probabilities[0]),
                "over_2_5": float(probabilities[1])
            }
        }
        return jsonify(response_data)
    except Exception as e:
        print(f"Prediction Error in /over_under: {e}")
        return jsonify({'error': f'En feil oppstod under prediksjon: {str(e)}'}), 500


@app.route('/extract-insights', methods=['POST'])
def extract_insights():
    data = request.get_json()
    if not data or 'text' not in data:
        return jsonify({'error': 'Request body må inneholde "text"-felt'}), 400
    text = data['text']
    blob = TextBlob(text)
    score = blob.sentiment.polarity
    label = 'NEUTRAL'
    if score > 0.1:
        label = 'POSITIVE'
    elif score < -0.1:
        label = 'NEGATIVE'

    ENTITY_KEYWORDS = {
        "GOALS_OVER_UNDER": [r'\bover\b', r'\bunder\b', 'o/u'],
        "CORNERS": ['corner', 'corners', 'hjørnespark'],
        "CARDS": ['card', 'cards', 'gult', 'rødt', 'kort'],
        "TEAM_RESULT": ['vinner', 'seier', 'tap', 'uavgjort'],
        "TEAM_NEWS_NEGATIVE": ['skade', 'skadet', 'suspensjon'],
        "TEAM_NEWS_POSITIVE": ['tilbake fra skade', 'full tropp']
    }
    entities = []
    for entity, keywords in ENTITY_KEYWORDS.items():
        if any(re.search(r'\b' + kw + r'\b', text, re.IGNORECASE) for kw in keywords):
            entities.append(entity)

    return jsonify({
        'general_sentiment': {'label': label, 'score': score},
        'entities_found': entities
    })


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001, debug=True)