# application.py
from flask import Flask, request, render_template, jsonify
import os, traceback

# Try to import both pickle and joblib loaders
try:
    import joblib
except Exception:
    joblib = None
import pickle

# Flask WSGI callable for Render/Gunicorn
application = Flask(__name__)
app = application

MODEL_LOCAL_PATH = os.path.join(os.path.dirname(__file__), "sentiment_model.pkl")

# Helper to load model (tries pickle then joblib)
def load_model(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Model file not found at {path}")

    errors = {}
    # Try pickle
    try:
        with open(path, "rb") as f:
            m = pickle.load(f)
        return m
    except Exception as e:
        errors["pickle"] = str(e)

    # Try joblib
    if joblib is not None:
        try:
            m = joblib.load(path)
            return m
        except Exception as e:
            errors["joblib"] = str(e)
    raise RuntimeError(f"Failed to load model. Attempts: {errors}")

# Load at startup
try:
    model = load_model(MODEL_LOCAL_PATH)
    MODEL_LOAD_ERROR = None
except Exception as e:
    model = None
    MODEL_LOAD_ERROR = str(e)
    # store traceback for logs
    print("Model load failed:", MODEL_LOAD_ERROR)
    traceback.print_exc()

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    if model is None:
        return jsonify({"error": "Model not loaded", "details": MODEL_LOAD_ERROR}), 500

    # Accept either 'text' or 'input_text' or 'user_text'
    text = request.form.get("text") or request.form.get("input_text") or request.form.get("user_text") or ""
    text = text.strip()
    if not text:
        return jsonify({"error": "No text provided"}), 400

    try:
        pred = model.predict([text])
        pred_value = pred[0]
        response = {"prediction": str(pred_value)}

        # include confidence if available
        try:
            if hasattr(model, "predict_proba"):
                probs = model.predict_proba([text])[0]
                # pick highest prob
                confidence = float(max(probs))
                response["confidence"] = round(confidence, 4)
        except Exception:
            # ignore probability issues
            pass

        return jsonify(response)
    except Exception as e:
        return jsonify({"error": "Prediction failed", "details": str(e)}), 500

if __name__ == "__main__":
    # Local debug server (not used on Render)
    application.run(host="0.0.0.0", port=5000, debug=True)
