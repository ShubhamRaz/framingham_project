from flask import Flask, request, render_template, jsonify
import pickle
import numpy as np
import traceback
import os

app = Flask(__name__)

# ------------- CONFIG ----------------
FEATURE_NAMES = [
    "male", "age", "currentSmoker", "cigsPerDay", "BPMeds",
    "prevalentStroke", "prevalentHyp", "diabetes", "totChol",
    "sysBP", "diaBP", "BMI", "heartRate", "glucose"
]

MODEL_PATH = os.path.join(os.path.dirname(__file__), "trained_model/model.pkl")
SCALER_PATH = os.path.join(os.path.dirname(__file__), "trained_model/scaler.pkl")

# If you prefer automatic imputation for missing items, set True and supply DEFAULTS.
USE_DEFAULTS = False
DEFAULTS = {
    "diaBP": 80,
    "cigsPerDay": 0
}
# --------------------------------------

# Load model and scaler (fail gracefully)
model = None
scaler = None
try:
    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)
except Exception as e:
    print("Warning: could not load model.pkl:", e)

try:
    with open(SCALER_PATH, "rb") as f:
        scaler = pickle.load(f)
except Exception as e:
    print("Warning: could not load scaler.pkl:", e)


def prepare_inputs(form):
    """
    Read values from form-like mapping and return numpy array shape (1, n_features).
    Raises ValueError with an aggregated message if missing/invalid fields are found.
    """
    values = []
    missing = []
    invalid = []

    for feat in FEATURE_NAMES:
        raw = form.get(feat)
        if raw is None or str(raw).strip() == "":
            if USE_DEFAULTS and feat in DEFAULTS:
                raw = str(DEFAULTS[feat])
            else:
                missing.append(feat)
                continue

        s = str(raw).strip()
        # Accept common boolean words
        if s.lower() in ["yes", "y", "true", "t"]:
            s = "1"
        if s.lower() in ["no", "n", "false", "f"]:
            s = "0"

        try:
            fv = float(s)
        except Exception:
            invalid.append((feat, raw))
            continue

        values.append(fv)

    if missing or invalid:
        parts = []
        if missing:
            parts.append("Missing fields: " + ", ".join(missing))
        if invalid:
            parts.append("Invalid values: " + ", ".join(f"{k}={v!r}" for k, v in invalid))
        raise ValueError("; ".join(parts))

    # final sanity check: number of features
    if len(values) != len(FEATURE_NAMES):
        raise ValueError("Unexpected number of parsed features; expected "
                         f"{len(FEATURE_NAMES)} got {len(values)}.")

    return np.array(values).reshape(1, -1)


def get_prediction(arr, threshold=0.5):
    """
    Scale inputs, compute probability (best-effort), and return dict with probability and label.
    """
    if model is None:
        raise RuntimeError("Model not loaded (model.pkl).")
    # scale if scaler available
    X = arr
    if scaler is not None:
        X = scaler.transform(arr)

    # try predict_proba, then decision_function, then predict
    try:
        prob = float(model.predict_proba(X)[0][1])
    except Exception:
        try:
            score = model.decision_function(X)[0]
            prob = float(1.0 / (1.0 + np.exp(-score)))  # crude sigmoid
        except Exception:
            pred = model.predict(X)[0]
            prob = float(pred)

    label = 1 if prob >= float(threshold) else 0
    return {"probability": prob, "label": int(label)}


@app.route("/", methods=["GET"])
def home():
    # initial render: no prediction
    return render_template("index.html", prediction_text=None, inputs=None, threshold="0.5")


@app.route("/predict", methods=["POST"])
def predict():
    try:
        threshold_raw = request.form.get("threshold", "0.5").strip()
        try:
            threshold = float(threshold_raw) if threshold_raw != "" else 0.5
        except Exception:
            threshold = 0.5

        arr = prepare_inputs(request.form)  # may raise ValueError
        res = get_prediction(arr, threshold=threshold)
        prob = res["probability"]
        label = res["label"]

        risk_text = "HIGH RISK (likely to develop CHD in 10 years)" if label == 1 else "LOW RISK"
        prob_pct = f"{prob*100:.1f}%"
        advice = ("Recommend clinical follow-up and lifestyle interventions. This is a screening "
                  "prediction, not a diagnosis.") if label == 1 else "Continue healthy lifestyle; consult clinician for personalised advice."

        return render_template(
            "index.html",
            prediction_text=True,
            risk_text=risk_text,
            probability_text=f"Predicted probability: {prob_pct}",
            advice=advice,
            threshold=str(threshold),
            inputs={k: request.form.get(k, "") for k in FEATURE_NAMES}
        )

    except Exception as exc:
        tb = traceback.format_exc()
        # Return the user's provided inputs back to the template to refill the form
        return render_template(
            "index.html",
            prediction_text=False,
            error=str(exc),
            tb=tb,
            inputs={k: request.form.get(k, "") for k in FEATURE_NAMES},
            threshold=request.form.get("threshold", "0.5")
        )


@app.route("/api/predict", methods=["POST"])
def api_predict():
    data = request.get_json(force=True, silent=True)
    if not isinstance(data, dict):
        return jsonify({"success": False, "error": "Invalid JSON body."}), 400

    try:
        threshold = float(data.get("threshold", 0.5))
    except Exception:
        threshold = 0.5

    # Build a simple dict that mimics request.form for prepare_inputs
    form_like = {}
    for feat in FEATURE_NAMES:
        if feat not in data:
            if USE_DEFAULTS and feat in DEFAULTS:
                form_like[feat] = str(DEFAULTS[feat])
            else:
                return jsonify({"success": False, "error": f"Missing feature in JSON: {feat}"}), 400
        else:
            form_like[feat] = str(data[feat])

    try:
        arr = prepare_inputs(form_like)
        res = get_prediction(arr, threshold)
        return jsonify({"success": True, "probability": res["probability"], "label": res["label"]})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 400


if __name__ == "__main__":
    app.run(debug=True)
