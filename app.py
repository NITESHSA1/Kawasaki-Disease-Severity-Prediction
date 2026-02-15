# app.py - Kawasaki Disease Predictor (API + Web UI Complete)

import os
import numpy as np
import pandas as pd
from flask import Flask, request, jsonify, render_template_string
from flask_cors import CORS
import joblib
from datetime import datetime

# Use project folder (script location) instead of hardcoded path
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Model paths
ARTIFACTS = {
    "xgb": os.path.join(BASE_DIR, "KD_final_xgb_model.joblib"),
    "ada": os.path.join(BASE_DIR, "KD_final_ada_model.joblib"),
    "encoder": os.path.join(BASE_DIR, "KD_final_label_encoder.joblib"),
    "weights": os.path.join(BASE_DIR, "KD_final_ensemble_weights_XGB_ADA.npz"),
    "features_csv": os.path.join(BASE_DIR, "KD_MGWO_REDUCED_MATRIX.csv")
}

# Global models (loaded once)
print("🚀 Initializing Kawasaki Disease Ensemble...")
models = {}
try:
    models["xgb"] = joblib.load(ARTIFACTS["xgb"])
    models["ada"] = joblib.load(ARTIFACTS["ada"])
    models["encoder"] = joblib.load(ARTIFACTS["encoder"])
    weights = np.load(ARTIFACTS["weights"])
    w_xgb, w_ada = float(weights["w_xgb"]), float(weights["w_ada"])

    # Load features
    df_meta = pd.read_csv(ARTIFACTS["features_csv"])
    FEATURE_NAMES = df_meta.columns[:-1].tolist()
    CLASSES = models["encoder"].classes_.tolist()
    print(f"✅ Success: {len(FEATURE_NAMES)} features, classes: {CLASSES}")
except Exception as e:
    print(f"❌ Load error: {e}")
    FEATURE_NAMES, CLASSES = [], []

app = Flask(__name__)
CORS(app)

@app.route('/', methods=['GET'])
def home():
    """API Documentation"""
    return jsonify({
        "name": "Kawasaki Disease Severity Predictor",
        "version": "3.0",
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "endpoints": {
            "docs": "/",
            "health": "/health",
            "predict_api": "/predict (POST JSON)",
            "web_ui": "/ui"
        },
        "features": FEATURE_NAMES,
        "classes": CLASSES,
        "example_features": dict(zip(FEATURE_NAMES, np.random.rand(len(FEATURE_NAMES))))
    })

@app.route('/health', methods=['GET'])
def health():
    return jsonify({"status": "ok", "features": len(FEATURE_NAMES)})

@app.route('/predict', methods=['POST'])
def predict_api():
    """JSON API endpoint"""
    data = request.get_json(silent=True)
    if not data or "features" not in data:
        return jsonify({"error": "Missing 'features'"}), 400

    features = data["features"]
    missing = [f for f in FEATURE_NAMES if f not in features]
    if missing:
        return jsonify({"error": "Missing", "need": missing}), 400

    try:
        X = np.array([float(features[f]) for f in FEATURE_NAMES]).reshape(1, -1)
    except Exception:
        return jsonify({"error": "Invalid numbers"}), 400

    p_xgb = models["xgb"].predict_proba(X)[0]
    p_ada = models["ada"].predict_proba(X)[0]
    p_ens = w_xgb * p_xgb + w_ada * p_ada

    return jsonify({
        "severity": models["encoder"].inverse_transform([np.argmax(p_ens)])[0],
        "confidence": float(np.max(p_ens)),
        "probabilities": {c: float(p) for c, p in zip(CLASSES, p_ens)},
        "models": {
            "xgb": CLASSES[np.argmax(p_xgb)],
            "ada": CLASSES[np.argmax(p_ada)]
        }
    })

@app.route('/ui', methods=['GET', 'POST'])
def web_ui():
    """Interactive Web Form"""
    prediction = None
    if request.method == 'POST':
        try:
            features = {f: float(request.form.get(f, 0)) for f in FEATURE_NAMES}
            X = np.array([features[f] for f in FEATURE_NAMES]).reshape(1, -1)

            p_xgb = models["xgb"].predict_proba(X)[0]
            p_ada = models["ada"].predict_proba(X)[0]
            p_ens = w_xgb * p_xgb + w_ada * p_ada

            pred_class = models["encoder"].inverse_transform([np.argmax(p_ens)])[0]
            probs = {c: p for c, p in zip(CLASSES, p_ens)}

            prediction = {
                'class': pred_class,
                'confidence': f"{np.max(p_ens):.1%}",
                'probs': probs,
                'features': features
            }
        except Exception as e:
            prediction = {'error': str(e)}

    # Generate form fields
    form_fields = ''.join(f'''
        <div class="field">
            <label>{f}:</label>
            <input type="number" step="0.01" min="0" max="1" name="{f}" 
                   value="{0.5 if request.method=='GET' else prediction.get('features', {}).get(f, 0.5)}">
        </div>''' for f in FEATURE_NAMES)

    probs_html = ''
    if prediction and 'probs' in prediction:
        probs_html = '<ul>' + ''.join(
            f'<li>{c}: <strong>{v:.1%}</strong></li>' for c, v in prediction['probs'].items()
        ) + '</ul>'

    result_html = f'''
        <div class="result {'error' if prediction and 'error' in prediction else prediction['class'].lower()}">
            <h2>{prediction.get('error', f"{prediction['class']} ({prediction['confidence']})") if prediction else ''}</h2>
            {probs_html}
        </div>''' if prediction else ''

    return render_template_string(f'''
    <!DOCTYPE html>
    <html>
    <head>
        <title>Kawasaki Disease Predictor</title>
        <style>
            body {{ font-family: Arial; max-width: 800px; margin: 50px auto; padding: 20px; }}
            .field {{ margin: 10px 0; }} label {{ display: inline-block; width: 250px; font-weight: bold; }}
            input {{ width: 100px; padding: 5px; }} button {{ padding: 10px 20px; background: #007bff; color: white; border: none; cursor: pointer; }}
            .result {{ padding: 20px; border-radius: 10px; margin: 20px 0; text-align: center; }}
            .mild {{ background: #d4edda; color: #155724; }} .severe {{ background: #f8d7da; color: #721c24; }}
            .error {{ background: #fff3cd; color: #856404; }} h1 {{ color: #333; }} h2 {{ margin: 0; }}
            .info {{ background: #e7f3ff; padding: 15px; border-left: 4px solid #007bff; margin: 20px 0; }}
        </style>
    </head>
    <body>
        <h1>🏥 Kawasaki Disease Severity Predictor</h1>
        <p class="info">
            <strong>Ensemble Model:</strong> XGBoost + AdaBoost (MGWO Feature Selection)<br>
            Classes: Mild, Severe, Unknown | <a href="/">API Docs</a> | <a href="/health">Health</a>
        </p>
        
        <form method="POST">
            {form_fields}
            <br><button type="submit">🔮 Predict Severity</button>
        </form>
        
        {result_html}
        
        <footer style="margin-top: 50px; text-align: center; color: #666;">
            Anurag Institutions ML Project | Powered by Flask & scikit-learn
        </footer>
    </body>
    </html>
    ''')

@app.errorhandler(404)
def not_found(e):
    return jsonify({"error": "Use /ui (web) or /predict (API)"}), 404

if __name__ == "__main__":
    print("🌐 Full app ready!")
    print("📱 Web UI: http://127.0.0.1:5000/ui")
    print("📊 API:    http://127.0.0.1:5000/")
    app.run(host="0.0.0.0", port=5000, debug=True)
