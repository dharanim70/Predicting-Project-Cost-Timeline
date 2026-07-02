from flask import Flask, render_template, request
import joblib
import pandas as pd
import os

app = Flask(__name__)

# Load trained models
cost_model = joblib.load("cost_model.pkl")
timeline_model = joblib.load("time_model.pkl")


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    project_size = int(request.form["project_size"])
    team_size = int(request.form["team_size"])
    complexity = int(request.form["complexity"])
    experience = int(request.form["experience"])

    data = pd.DataFrame([{
        "Project_Size": project_size,
        "Team_Size": team_size,
        "Complexity": complexity,
        "Experience": experience
    }])

    predicted_cost = cost_model.predict(data)[0]
    predicted_timeline = timeline_model.predict(data)[0]

    return render_template(
        "index.html",
        cost=round(predicted_cost, 2),
        timeline=round(predicted_timeline, 2)
    )


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)