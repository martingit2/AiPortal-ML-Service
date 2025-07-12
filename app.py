# app.py
from flask import Flask, request, jsonify
from textblob import TextBlob
import re  # Importerer 'regular expressions' for mer avansert søk

app = Flask(__name__)

# OMFATTENDE LISTE MED SØKEORD OG MØNSTRE FOR Å IDENTIFISERE ENTITETER
# 're.IGNORECASE' vil bli brukt, så vi trenger ikke bekymre oss for store/små bokstaver.
ENTITY_KEYWORDS = {
    # Nøkkelord relatert til totalt antall mål
    "GOALS_OVER_UNDER": [r'\bover\b', r'\bunder\b', 'o/u', r'o\d\.\d', r'u\d\.\d', 'mange mål', 'få mål', 'målfest',
                         'tett kamp', 'high scoring', 'low scoring'],

    # Nøkkelord for om et lag scorer eller ikke
    "TEAM_GOALS": ['scorer', 'score', 'mål', 'nettet', 'clean sheet', 'holde nullen', 'scorer ikke', 'failed to score'],

    # Nøkkelord spesifikt for "Begge Lag Scorer"
    "BOTH_TEAMS_TO_SCORE": ['btts', 'begge lag scorer', 'both teams to score', 'mål i begge ender'],

    # Nøkkelord for spillerscoringer
    "PLAYER_TO_SCORE": ['målscorer', 'scorer i dag', 'to score', 'anytime scorer', 'første mål', 'siste mål',
                        'first goalscorer'],

    # Nøkkelord for hjørnespark
    "CORNERS": ['corner', 'corners', 'hjørnespark'],

    # Nøkkelord for kort (gule/røde)
    "CARDS": ['card', 'cards', 'yellow', 'red', 'gult', 'rødt', 'kort', 'booking', 'utvisning', 'stygg kamp',
              'mange frispark', 'sendt av banen'],

    # Nøkkelord om kampresultat
    "TEAM_RESULT": ['vinner', 'seier', 'tap', 'taper', 'uavgjort', 'draw', 'lett match', 'knuser', 'dominerer',
                    'ydmyket', 'walkover'],

    # Nøkkelord om en spillers generelle form/prestasjon
    "PLAYER_PERFORMANCE": ['i form', 'on fire', 'strålende', 'dårlig form', 'ute av det', 'usynlig', 'outstanding',
                           'poor performance', 'man of the match', 'motm'],

    # Nøkkelord om negative lagnyheter (skader/suspensjoner)
    "TEAM_NEWS_NEGATIVE": ['skade', 'skadet', 'injury', 'injured', 'suspensjon', 'suspensjoner', 'suspension',
                           'mangler', 'ute', 'krise', 'uro', 'tvilsom', 'doubtful', 'sidelined'],

    # Nøkkelord om positive lagnyheter
    "TEAM_NEWS_POSITIVE": ['tilbake fra skade', 'full tropp', 'fit', 'klar til kamp', 'fit again', 'squad is ready',
                           'return from injury']
}


@app.route('/extract-insights', methods=['POST'])
def extract_insights():
    """
    Dette nye endepunktet analyserer tekst for både generelt sentiment og
    for å identifisere spesifikke, pre-definerte betting-relaterte entiteter.
    """
    data = request.get_json()
    if not data or 'text' not in data:
        return jsonify({'error': 'Request body må inneholde et "text"-felt'}), 400

    text_to_analyze = data['text']
    text_lower = text_to_analyze.lower()

    # 1. Generell Sentimentanalyse (med TextBlob som før)
    blob = TextBlob(text_to_analyze)
    sentiment_score = blob.sentiment.polarity
    sentiment_label = 'NEUTRAL'
    if sentiment_score > 0.1:
        sentiment_label = 'POSITIVE'
    elif sentiment_score < -0.1:
        sentiment_label = 'NEGATIVE'

    general_sentiment = {
        'label': sentiment_label,
        'score': sentiment_score
    }

    # 2. Entitet/Keyword Extraction
    entities_found = []
    found_entity_types = set()

    for entity_type, keywords in ENTITY_KEYWORDS.items():
        mentioned_terms = []
        count = 0
        for keyword in keywords:
            # Bruker re.findall for å telle alle forekomster (også overlappende)
            matches = re.findall(keyword, text_lower, re.IGNORECASE)
            if matches:
                count += len(matches)
                # Legger til det originale søkeordet (uten regex-spesialtegn)
                mentioned_terms.append(keyword.replace(r'\b', ''))

        if count > 0 and entity_type not in found_entity_types:
            entities_found.append({
                "entity": entity_type,
                "count": count,
                "mentioned_terms": sorted(list(set(mentioned_terms)))  # Unike, sorterte termer
            })
            found_entity_types.add(entity_type)

    # 3. Bygg den endelige, rike responsen
    response_data = {
        'text': text_to_analyze,
        'general_sentiment': general_sentiment,
        'entities_found': entities_found
    }

    return jsonify(response_data)


@app.route('/analyze-sentiment', methods=['POST'])
def analyze_sentiment_legacy():
    """
    Beholder det gamle endepunktet for bakoverkompatibilitet og enkel testing.
    Dette kan fjernes senere.
    """
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