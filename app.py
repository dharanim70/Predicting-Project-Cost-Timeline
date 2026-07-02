from flask import Flask, render_template, request
import joblib
import pandas as pd
import os
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)

# Load trained models at startup
try:
    cost_model = joblib.load("cost_model.pkl")
    timeline_model = joblib.load("time_model.pkl")
    logger.info("Models loaded successfully.")
except Exception as e:
    logger.error(f"Failed to load models: {e}")
    cost_model = None
    timeline_model = None


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    if cost_model is None or timeline_model is None:
        return render_template("index.html", error="Models are not available. Please retrain using train_model.py.")

    try:
        project_size = int(request.form.get("project_size", "").strip())
        team_size = int(request.form.get("team_size", "").strip())
        complexity = int(request.form.get("complexity", "").strip())
        experience = int(request.form.get("experience", "").strip())
    except (ValueError, AttributeError):
        return render_template("index.html", error="All fields must be valid numbers.")

    if not (1 <= complexity <= 5):
        return render_template("index.html", error="Complexity must be between 1 and 5.")
    if team_size <= 0 or project_size <= 0 or experience < 0:
        return render_template("index.html", error="Project Size and Team Size must be positive; Experience cannot be negative.")

    try:
        data = pd.DataFrame([{
            "Project_Size": project_size,
            "Team_Size": team_size,
            "Complexity": complexity,
            "Experience": experience
        }])
        predicted_cost = cost_model.predict(data)[0]
        predicted_timeline = timeline_model.predict(data)[0]
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        return render_template("index.html", error="Prediction failed. Please check your inputs and try again.")

    return render_template(
        "index.html",
        cost=round(predicted_cost, 2),
        timeline=round(predicted_timeline, 2),
        has_result=True
    )


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)