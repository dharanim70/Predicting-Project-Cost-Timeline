from flask import Flask, render_template, request
import joblib
import pandas as pd

app = Flask(__name__)

# Load trained models
cost_model = joblib.load("cost_model.pkl")
timeline_model = joblib.load("time_model.pkl")


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():

    team_size = int(request.form["team_size"])
    number_of_tasks = int(request.form["number_of_tasks"])
    project_complexity = int(request.form["project_complexity"])
    resource_requirements = int(request.form["resource_requirements"])

    data = pd.DataFrame([[team_size,
                          number_of_tasks,
                          project_complexity,
                          resource_requirements]],
                        columns=[
                            "Team_Size",
                            "Number_of_Tasks",
                            "Project_Complexity",
                            "Resource_Requirements"
                        ])

    predicted_cost = cost_model.predict(data)[0]
    predicted_timeline = timeline_model.predict(data)[0]

    return render_template(
        "index.html",
        predicted_cost=round(predicted_cost, 2),
        predicted_timeline=round(predicted_timeline, 2)
    )


if __name__ == "__main__":
    app.run(debug=True)