# app.py
from flask import Flask, request, jsonify
from textblob import TextBlob
import re
import joblib
import pandas as pd
import numpy as np
import os

app = Flask(__name__)

MODEL_FILENAME = "football_predictor_v2.joblib"
model = None

EXPECTED_FEATURES = [
    'homeAvgShotsOnGoal', 'homeAvgShotsOffGoal', 'homeAvgCorners', 'homeInjuries',
    'homePlayersAvgRating', 'homePlayersAvgGoals',
    'awayAvgShotsOnGoal', 'awayAvgShotsOffGoal', 'awayAvgCorners', 'awayInjuries',
    'awayPlayersAvgRating', 'awayPlayersAvgGoals'
]

if os.path.exists(MODEL_FILENAME):
    print(f"Laster inn lagret modell fra: {MODEL_FILENAME}")
    model = joblib.load(MODEL_FILENAME)
    print("Modell lastet inn.")
else:
    print(f"ADVARSEL: Modellfilen '{MODEL_FILENAME}' ble ikke funnet. /predict endepunktet vil ikke fungere.")

ENTITY_KEYWORDS = {
    "GOALS_OVER_UNDER": [r'\bover\b', r'\bunder\b', 'o/u'],
    "CORNERS": ['corner', 'corners', 'hjørnespark'],
    "CARDS": ['card', 'cards', 'gult', 'rødt', 'kort'],
    "TEAM_RESULT": ['vinner', 'seier', 'tap', 'uavgjort'],
    "TEAM_NEWS_NEGATIVE": ['skade', 'skadet', 'suspensjon'],
    "TEAM_NEWS_POSITIVE": ['tilbake fra skade', 'full tropp']
}

@app.route('/predict/match_outcome', methods=['POST'])
def predict_match_outcome():
    if model is None:
        return jsonify({'error': 'Modellen er ikke lastet inn. Kjør treningsskriptet først.'}), 503
    data = request.get_json()
    if not data:
        return jsonify({'error': 'Request body mangler'}), 400
    try:
        ordered_data = {feature: data.get(feature, 0) for feature in EXPECTED_FEATURES}
        input_df = pd.DataFrame([ordered_data], columns=EXPECTED_FEATURES)
        print(f"Mottok data for prediksjon: {input_df.to_dict('records')}")
        probabilities = model.predict_proba(input_df)
        home_win_prob = probabilities[0][1]
        not_home_win_prob = 1 - home_win_prob
        draw_prob = not_home_win_prob * 0.4
        away_win_prob = not_home_win_prob * 0.6
        response_data = {
            "prediction_model_version": "football_predictor_v2",
            "probabilities": {
                "home_win": float(home_win_prob),
                "draw": float(draw_prob),
                "away_win": float(away_win_prob)
            }
        }
        print(f"Returnerer prediksjon: {response_data}")
        return jsonify(response_data)
    except Exception as e:
        print(f"FEIL under prediksjon: {e}")
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
    if score > 0.1: label = 'POSITIVE'
    elif score < -0.1: label = 'NEGATIVE'
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