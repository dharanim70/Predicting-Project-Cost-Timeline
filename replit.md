# Predicting Project Cost and Timeline Using Machine Learning

A Flask web app that uses pre-trained ML models to predict software project cost and timeline.

## How to run

The app starts automatically via the **Start application** workflow, which runs:

```
python app.py
```

The server listens on port 5000. Open the preview pane to use the app.

## Project structure

- `app.py` — Flask server with `/` (home) and `/predict` (POST) routes
- `train_model.py` — Script to retrain models from the dataset
- `cost_model.pkl` / `time_model.pkl` — Pre-trained Random Forest models
- `dataset/projects.csv` — Training data
- `templates/index.html` — Web UI
- `static/style.css` — Styles

## Inputs

| Field | Description |
|-------|-------------|
| Project Size | Numeric size of the project |
| Team Size | Number of team members |
| Complexity | Complexity level (numeric) |
| Experience | Team experience level (numeric) |

## Retraining models

```
python train_model.py
```

## Dependencies

Listed in `requirements.txt`: Flask, pandas, scikit-learn, joblib, numpy.
