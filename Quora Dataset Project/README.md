# Quora Duplicate Question Detector

A web application that predicts whether two questions have the same intended meaning. The project combines NLP feature engineering with a Random Forest classifier and serves predictions through a Flask application.

The main model-development work is in `final_duplicate_project.ipynb`. The deployable application is contained in the `webapp` directory.

## Features

- Compares a pair of natural-language questions
- Preprocesses text and expands common contractions
- Generates basic word, token, length, fuzzy matching, and bag-of-words features
- Returns a duplicate/not-duplicate label, probability, and confidence
- Provides a responsive browser interface
- Exposes health and prediction endpoints
- Uses MVC-style separation between routes, prediction logic, and browser UI code

## Project Structure

```text
Quora Dataset Project/
|-- final_duplicate_project.ipynb    # Main data analysis and model notebook
|-- duplicate_question_project.ipynb # Additional project notebook
|-- quora_duplicate_questions.tsv    # Source dataset
|-- rf_model.pkl                     # Serialized Random Forest model
|-- count_vectorizer.pkl             # Serialized CountVectorizer
|-- README.md
`-- webapp/
    |-- app.py                       # Flask application factory and startup
    |-- requirements.txt             # Python dependencies
    |-- controller/
    |   `-- routes.py                # HTTP routes and API handling
    |-- model/
    |   `-- predictor.py             # Preprocessing and prediction logic
    |-- static/
    |   |-- css/styles.css           # Application styling
    |   `-- js/
    |       |-- model.js             # Browser API client
    |       |-- view.js              # DOM rendering
    |       `-- controller.js        # UI event coordination
    `-- templates/
        `-- index.html               # Main web page
```

## How It Works

1. The Flask controller accepts two questions from the browser.
2. The predictor lowercases the text, removes HTML, expands contractions, normalizes symbols, and removes non-word characters.
3. The predictor constructs 22 engineered features, including:
   - Shared-word ratios
   - Common first and last words
   - Token and word-count differences
   - Longest-common-substring length
   - Sequence and token fuzzy-match scores
4. The CountVectorizer transforms both cleaned questions into bag-of-words vectors.
5. The feature array is passed to the serialized Random Forest model.
6. The API returns the predicted class and, when supported by the model, duplicate probability and confidence.

## Requirements

- Python 3.9 or newer
- Windows PowerShell or another command terminal
- The existing `rf_model.pkl` and `count_vectorizer.pkl` files

The web application does not train the model at runtime. Both model artifacts must remain in the project root, alongside the `webapp` directory.

## Installation

Open PowerShell in the project directory and create a virtual environment:

```powershell
cd "d:\NLP\Quora Dataset Project"
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r webapp\requirements.txt
```

If PowerShell prevents virtual environment activation, enable local script execution for the current user:

```powershell
Set-ExecutionPolicy -Scope CurrentUser -ExecutionPolicy RemoteSigned
```

## Running the Application

Start the Flask server from the project root:

```powershell
python webapp\app.py
```

Open <http://127.0.0.1:5000> in a browser.

To enable Flask debug mode:

```powershell
$env:FLASK_DEBUG="1"
python webapp\app.py
```

To use a different local port:

```powershell
$env:PORT="8080"
python webapp\app.py
```

## API

### Health Check

`GET /health`

Example response:

```json
{
  "status": "ok",
  "model_loaded": true
}
```

### Predict Duplicate Questions

`POST /api/predict`

Request body:

```json
{
  "question1": "What is the capital of India?",
  "question2": "What is the current capital of India?"
}
```

Example response:

```json
{
  "prediction": 1,
  "label": "Duplicate",
  "duplicate_probability": 0.72,
  "confidence": 0.72,
  "cleaned_questions": {
    "question1": "what is the capital of india",
    "question2": "what is the current capital of india"
  }
}
```

Missing questions return HTTP `400`. Prediction and artifact-loading failures return HTTP `500`.

## Prediction Model

The deployed model is the serialized Random Forest artifact named `rf_model.pkl`. The application also requires `count_vectorizer.pkl` to transform question text consistently with the training pipeline.

The optional `fuzzywuzzy` dependency provides fuzzy string matching. The predictor includes an internal fallback implementation, so prediction can still run if `fuzzywuzzy` is not available.

## Notes

- The Flask server is intended for local development or an environment that has suitable serialized-model compatibility.
- Predictions can differ across scikit-learn versions if the serialized model and runtime environment use incompatible package versions.
- Reusing the exact environment used to train the model artifacts is recommended for stable deployment.
- The dataset, notebook, and model files should be kept together unless the paths in `model/predictor.py` are updated.
