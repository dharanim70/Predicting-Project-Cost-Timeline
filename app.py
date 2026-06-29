from flask import Flask, request

app = Flask(__name__)

@app.route('/')
def home():
    return """
    <h1>Predicting Project Cost and Timeline</h1>
    <p>Machine Learning Mini Project</p>
    """

if __name__ == "__main__":
    app.run(debug=True)
