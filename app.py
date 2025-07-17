# app.py (KORRIGERT IGJEN FOR TYPE-FEIL)

from flask import Flask, request, jsonify
import joblib
import pandas as pd
import os
import numpy as np
from sklearn.calibration import CalibratedClassifierCV

app = Flask(__name__)

# --- DYNAMISK MODELL-LASTING (forblir lik, den fungerte) ---
MODELS_DIR = "."
loaded_models = {}
canonical_features = []

print("--- Starter dynamisk modell-lasting ---")
canonical_model_found = False
model_files = sorted([f for f in os.listdir(MODELS_DIR) if f.endswith(".joblib")], reverse=True)

for filename in model_files:
    if filename.startswith("football_predictor") and not canonical_model_found:
        try:
            model = joblib.load(os.path.join(MODELS_DIR, filename))
            base_estimator = model.estimator if isinstance(model, CalibratedClassifierCV) else model
            if hasattr(base_estimator, 'feature_names_in_'):
                canonical_features = base_estimator.feature_names_in_
                print(f"✔️ Canonical features satt fra '{filename}': {len(canonical_features)} features.")
                canonical_model_found = True
        except Exception as e:
            print(f"❌ Feil ved lasting av canonical-modell '{filename}': {e}")

print("\n--- Laster alle modeller ---")
for filename in model_files:
    if filename not in loaded_models:
        try:
            model = joblib.load(os.path.join(MODELS_DIR, filename))
            loaded_models[filename] = model
            print(f"✔️ Lastet inn modell: {filename}")
        except Exception as e:
            print(f"❌ Kunne ikke laste {filename}: {e}")

if (isinstance(canonical_features, np.ndarray) and canonical_features.size == 0) or \
        (isinstance(canonical_features, list) and not canonical_features):
    print("ADVARSEL: 'canonical_features' ble ikke satt. Sjekk at minst én gyldig modell finnes.")


def get_input_dataframe(features_dict):
    if not features_dict:
        return None, "Mangler 'features' i request body."
    if (isinstance(canonical_features, np.ndarray) and canonical_features.size == 0) or \
            (isinstance(canonical_features, list) and not canonical_features):
        return None, "Serverfeil: 'canonical_features' er ikke definert. Kan ikke prosessere request."
    try:
        temp_df = pd.DataFrame([features_dict])
        final_df = temp_df.reindex(columns=canonical_features, fill_value=0.0)
        for col in final_df.columns:
            final_df[col] = pd.to_numeric(final_df[col], errors='coerce').fillna(0.0)
        return final_df, None
    except Exception as e:
        return None, f"Feil ved bygging av DataFrame: {str(e)}"


# --- OPPDATERT, ROBUST PREDIKSJONSLOGIKK ---
@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    if not data or 'modelName' not in data or 'features' not in data:
        return jsonify({'error': "Request body må inneholde 'modelName' og 'features'."}), 400

    model_name = data['modelName']
    features = data['features']

    if model_name not in loaded_models:
        return jsonify({'error': f"Modell '{model_name}' er ikke lastet inn eller finnes ikke."}), 404

    model = loaded_models[model_name]
    input_df, error = get_input_dataframe(features)
    if error:
        return jsonify({'error': error}), 400

    try:
        # Encoder-filer skal ikke brukes til prediksjon
        if 'encoder' in model_name:
            return jsonify({
                "model_used": model_name,
                "probabilities": {"classes": model.classes_.tolist()}
            })

        probabilities = model.predict_proba(input_df)[0]

        # Standardiser outputen til "class_0", "class_1", etc. for alle modeller.
        # Dette er den mest robuste løsningen.
        response_probs = {
            f"class_{i}": float(prob)
            for i, prob in enumerate(probabilities)
        }

        response_data = {
            "model_used": model_name,
            "probabilities": response_probs
        }
        return jsonify(response_data)

    except Exception as e:
        print(f"Prediction Error for model {model_name}: {e}")
        return jsonify({'error': f'En feil oppstod under prediksjon: {str(e)}'}), 500


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001, debug=True)