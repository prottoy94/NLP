from flask import Blueprint, jsonify, render_template, request

from model.predictor import DuplicateQuestionPredictor


app_routes = Blueprint("app_routes", __name__)
predictor = DuplicateQuestionPredictor()


@app_routes.get("/")
def index():
    return render_template("index.html")


@app_routes.get("/health")
def health():
    return jsonify({"status": "ok", "model_loaded": predictor.is_ready})


@app_routes.post("/api/predict")
def predict():
    payload = request.get_json(silent=True) or {}
    question1 = str(payload.get("question1", "")).strip()
    question2 = str(payload.get("question2", "")).strip()

    if not question1 or not question2:
        return jsonify({"error": "Both questions are required."}), 400

    try:
        result = predictor.predict(question1, question2)
        return jsonify(result)
    except Exception as exc:
        return jsonify({"error": str(exc)}), 500
