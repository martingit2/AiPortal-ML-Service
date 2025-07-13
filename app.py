# app.py
from flask import Flask, request, jsonify
from textblob import TextBlob
import re
import joblib
import pandas as pd
import numpy as np
import os

app = Flask(__name__)

# --- MODELL-LASTING VED OPPSTART ---
MODEL_FILENAME = "football_predictor_v1.joblib"
model = None

if os.path.exists(MODEL_FILENAME):
    print(f"Laster inn lagret modell fra: {MODEL_FILENAME}")
    model = joblib.load(MODEL_FILENAME)
    print("Modell lastet inn.")
else:
    print(f"ADVARSEL: Modellfilen '{MODEL_FILENAME}' ble ikke funnet. /predict endepunktet vil ikke fungere.")

ENTITY_KEYWORDS = {
    "GOALS_OVER_UNDER": [r'\bover\b', r'\bunder\b', 'o/u', r'o\d\.\d', r'u\d\.\d', 'mange mål', 'få mål', 'målfest',
                         'tett kamp', 'high scoring', 'low scoring'],
    "TEAM_GOALS": ['scorer', 'score', 'mål', 'nettet', 'clean sheet', 'holde nullen', 'scorer ikke', 'failed to score'],
    "BOTH_TEAMS_TO_SCORE": ['btts', 'begge lag scorer', 'both teams to score', 'mål i begge ender'],
    "PLAYER_TO_SCORE": ['målscorer', 'scorer i dag', 'to score', 'anytime scorer', 'første mål', 'siste mål',
                        'first goalscorer'],
    "CORNERS": ['corner', 'corners', 'hjørnespark'],
    "CARDS": ['card', 'cards', 'yellow', 'red', 'gult', 'rødt', 'kort', 'booking', 'utvisning', 'stygg kamp',
              'mange frispark', 'sendt av banen'],
    "TEAM_RESULT": ['vinner', 'seier', 'tap', 'taper', 'uavgjort', 'draw', 'lett match', 'knuser', 'dominerer',
                    'ydmyket', 'walkover'],
    "PLAYER_PERFORMANCE": ['i form', 'on fire', 'strålende', 'dårlig form', 'ute av det', 'usynlig', 'outstanding',
                           'poor performance', 'man of the match', 'motm'],
    "TEAM_NEWS_NEGATIVE": ['skade', 'skadet', 'injury', 'injured', 'suspensjon', 'suspensjoner', 'suspension',
                           'mangler', 'ute', 'krise', 'uro', 'tvilsom', 'doubtful', 'sidelined'],
    "TEAM_NEWS_POSITIVE": ['tilbake fra skade', 'full tropp', 'fit', 'klar til kamp', 'fit again', 'squad is ready']
}


@app.route('/predict/match_outcome', methods=['POST'])
def predict_match_outcome():
    """
    Mottar features for en enkelt, kommende kamp og returnerer
    modellens predikerte sannsynligheter.
    """
    if model is None:
        return jsonify({'error': 'Modellen er ikke lastet inn. Kjør treningsskriptet først.'}), 503

    data = request.get_json()
    if not data:
        return jsonify({'error': 'Request body mangler'}), 400

    try:
        input_df = pd.DataFrame([data])

        print(f"Mottok data for prediksjon: {input_df.to_dict('records')}")

        probabilities = model.predict_proba(input_df)

        home_win_prob = probabilities[0][1]
        not_home_win_prob = 1 - home_win_prob
        draw_prob = not_home_win_prob * 0.4
        away_win_prob = not_home_win_prob * 0.6

        # Konverter NumPy-typer til standard Python-typer før serialisering
        response_data = {
            "prediction_model_version": "football_predictor_v1_binary",
            "probabilities": {
                "home_win": float(home_win_prob),
                "draw": float(draw_prob),
                "away_win": float(away_win_prob)
            }
        }

        print(f"Returnerer prediksjon: {response_data}")

        return jsonify(response_data)

    except Exception as e:
        return jsonify({'error': f'En feil oppstod under prediksjon: {str(e)}'}), 500


@app.route('/extract-insights', methods=['POST'])
def extract_insights():
    data = request.get_json()
    if not data or 'text' not in data:
        return jsonify({'error': 'Request body må inneholde et "text"-felt'}), 400
    text_to_analyze = data['text']
    text_lower = text_to_analyze.lower()
    blob = TextBlob(text_to_analyze)
    sentiment_score = blob.sentiment.polarity
    sentiment_label = 'NEUTRAL'
    if sentiment_score > 0.1:
        sentiment_label = 'POSITIVE'
    elif sentiment_score < -0.1:
        sentiment_label = 'NEGATIVE'
    general_sentiment = {'label': sentiment_label, 'score': sentiment_score}
    entities_found = []
    found_entity_types = set()
    for entity_type, keywords in ENTITY_KEYWORDS.items():
        mentioned_terms = []
        count = 0
        for keyword in keywords:
            matches = re.findall(keyword, text_lower, re.IGNORECASE)
            if matches:
                count += len(matches)
                mentioned_terms.append(keyword.replace(r'\b', ''))
        if count > 0 and entity_type not in found_entity_types:
            entities_found.append({
                "entity": entity_type,
                "count": count,
                "mentioned_terms": sorted(list(set(mentioned_terms)))
            })
            found_entity_types.add(entity_type)
    response_data = {
        'text': text_to_analyze,
        'general_sentiment': general_sentiment,
        'entities_found': entities_found
    }
    return jsonify(response_data)


@app.route('/analyze-sentiment', methods=['POST'])
def analyze_sentiment_legacy():
    data = request.get_json()
    if not data or 'text' not in data:
        return jsonify({'error': 'Request body må inneholde et "text"-felt'}), 400
    text_to_analyze = data['text']
    blob = TextBlob(text_to_analyze)
    sentiment_score = blob.sentiment.polarity
    sentiment_label = 'NEUTRAL'
    if sentiment_score > 0.1:
        sentiment_label = 'POSITIVE'
    elif sentiment_score < -0.1:
        sentiment_label = 'NEGATIVE'
    response_data = {
        'text': text_to_analyze,
        'sentiment': sentiment_label,
        'score': sentiment_score
    }
    return jsonify(response_data)


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001, debug=True)