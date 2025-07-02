# app.py
from flask import Flask, request, jsonify
from textblob import TextBlob

# Opprett Flask-applikasjonen
app = Flask(__name__)


# Definer et API-endepunkt på /analyze-sentiment som kun godtar POST-forespørsler
@app.route('/analyze-sentiment', methods=['POST'])
def analyze_sentiment():
    # Hent JSON-data fra forespørselen
    data = request.get_json()

    # Sjekk om dataen er gyldig og inneholder en 'text'-nøkkel
    if not data or 'text' not in data:
        # Returner en feilmelding hvis data mangler
        return jsonify({'error': 'Request body må inneholde et "text"-felt'}), 400

    text_to_analyze = data['text']

    # Opprett et TextBlob-objekt med teksten
    blob = TextBlob(text_to_analyze)

    # Utfør sentimentanalyse. .polarity gir en score fra -1 (negativ) til 1 (positiv)
    sentiment_score = blob.sentiment.polarity

    # Kategoriser scoren til en enkel label
    sentiment_label = 'NEUTRAL'
    if sentiment_score > 0.1:  # Terskler kan justeres
        sentiment_label = 'POSITIVE'
    elif sentiment_score < -0.1:
        sentiment_label = 'NEGATIVE'

    # Lag respons-objektet
    response_data = {
        'text': text_to_analyze,
        'sentiment': sentiment_label,
        'score': sentiment_score
    }

    # Returner resultatet som JSON med en 200 OK statuskode
    return jsonify(response_data)


# Denne blokken kjøres kun når du starter scriptet direkte (f.eks. med `python app.py`)
if __name__ == '__main__':
    # Kjører appen på port 5001 med debug-modus på.
    # host='0.0.0.0' gjør den tilgjengelig fra andre maskiner/containere på nettverket.
    app.run(host='0.0.0.0', port=5001, debug=True)