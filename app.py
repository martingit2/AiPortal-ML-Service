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
MODEL_FILENAME = "football_predictor_v3_multiclass.joblib"
ENCODER_FILENAME = "result_encoder.joblib"
model = None
encoder = None

EXPECTED_FEATURES = [
    'homeAvgShotsOnGoal', 'homeAvgShotsOffGoal', 'homeAvgCorners', 'homeInjuries',
    'homePlayersAvgRating', 'homePlayersAvgGoals',
    'awayAvgShotsOnGoal', 'awayAvgShotsOffGoal', 'awayAvgCorners', 'awayInjuries',
    'awayPlayersAvgRating', 'awayPlayersAvgGoals'
]

# Last inn modellen
if os.path.exists(MODEL_FILENAME):
    print(f"Laster inn lagret modell fra: {MODEL_FILENAME}")
    model = joblib.load(MODEL_FILENAME)
    print("Modell lastet inn.")
else:
    print(f"ADVARSEL: Modellfilen '{MODEL_FILENAME}' ble ikke funnet.")

# Last inn encoderen
if os.path.exists(ENCODER_FILENAME):
    print(f"Laster inn lagret encoder fra: {ENCODER_FILENAME}")
    encoder = joblib.load(ENCODER_FILENAME)
    print("Encoder lastet inn. Klasse-rekkefølge:", encoder.classes_)
else:
    print(f"ADVARSEL: Encoder-filen '{ENCODER_FILENAME}' ble ikke funnet.")

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
    """
    Mottar features for en enkelt, kommende kamp og returnerer
    modellens predikerte sannsynligheter for H, D, A.
    """
    if model is None or encoder is None:
        return jsonify({'error': 'Modell eller encoder er ikke lastet inn. Kjør treningsskriptet først.'}), 503

    data = request.get_json()
    if not data:
        return jsonify({'error': 'Request body mangler'}), 400

    try:
        ordered_data = {feature: data.get(feature, 0) for feature in EXPECTED_FEATURES}
        input_df = pd.DataFrame([ordered_data], columns=EXPECTED_FEATURES)

        print(f"Mottok data for prediksjon: {input_df.to_dict('records')}")

        # predict_proba for en multiklasse-modell returnerer en liste av lister
        # f.eks. [[prob_class_0, prob_class_1, prob_class_2]]
        probabilities = model.predict_proba(input_df)[0]

        # Finn indeksen for hvert utfall basert på den lagrede encoderen
        # Bruker .tolist() for å kunne bruke .index() metoden
        class_order = encoder.classes_.tolist()
        away_win_index = class_order.index('AWAY_WIN')
        draw_index = class_order.index('DRAW')
        home_win_index = class_order.index('HOME_WIN')

        # Hent sannsynlighetene direkte fra riktig indeks
        response_data = {
            "prediction_model_version": "football_predictor_v3_multiclass",
            "probabilities": {
                "home_win": float(probabilities[home_win_index]),
                "draw": float(probabilities[draw_index]),
                "away_win": float(probabilities[away_win_index])
            }
        }

        print(f"Returnerer prediksjon: {response_data}")
        return jsonify(response_data)

    except Exception as e:
        print(f"FEIL under prediksjon: {e}")
        return jsonify({'error': f'En feil oppstod under prediksjon: {str(e)}'}), 500


@app.route('/extract-insights', methods=['POST'])
def extract_insights():
    # Denne funksjonen er uendret
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