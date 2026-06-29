import pandas as pd
from sklearn.ensemble import RandomForestRegressor
import joblib

data = pd.read_csv("dataset/projects.csv")

X = data[["Project_Size","Team_Size","Complexity","Experience"]]
y_cost = data["Cost"]
y_time = data["Timeline"]

cost_model = RandomForestRegressor(random_state=42)
time_model = RandomForestRegressor(random_state=42)

cost_model.fit(X, y_cost)
time_model.fit(X, y_time)

joblib.dump(cost_model, "cost_model.pkl")
joblib.dump(time_model, "time_model.pkl")

print("Model Trained Successfully!")
