# test_load_model.py
import pickle, joblib, os, sys

MODEL_PATH = "sentiment_model.pkl"

print("Exists:", os.path.exists(MODEL_PATH))
print("Size (bytes):", os.path.getsize(MODEL_PATH) if os.path.exists(MODEL_PATH) else "N/A")

# Try pickle
try:
    with open(MODEL_PATH, "rb") as f:
        m = pickle.load(f)
    print("Loaded with pickle. Type:", type(m))
    print("Has predict():", hasattr(m, "predict"))
    print("Has predict_proba():", hasattr(m, "predict_proba"))
except Exception as e:
    print("Pickle load failed:", e)

# Try joblib
try:
    import joblib
    m = joblib.load(MODEL_PATH)
    print("Loaded with joblib. Type:", type(m))
    print("Has predict():", hasattr(m, "predict"))
    print("Has predict_proba():", hasattr(m, "predict_proba"))
except Exception as e:
    print("Joblib load failed:", e)
