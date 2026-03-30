# ML Model FastAPI Project

## 📌 Project Overview

This project implements an end-to-end machine learning pipeline using the Census Income dataset.

It includes:

* Data processing and model training
* Model evaluation (including slice metrics)
* REST API using FastAPI
* CI pipeline with GitHub Actions
* Deployment on Heroku

---

## 🔗 Links

* GitHub Repository:
  https://github.com/gardnerlingjia/ML_model_FASTAPI

* Live API:
  https://ml-fastapi-lingjia-5e077637aff3.herokuapp.com

* API Docs (Swagger):
  https://ml-fastapi-lingjia-5e077637aff3.herokuapp.com/docs

---

## ⚙️ Environment Setup

```bash
python -m venv .venv
.venv\Scripts\activate   # Windows
pip install -r requirements.txt
```

Python version used: **3.11**

---

## 🤖 Model

* Algorithm: Random Forest Classifier
* Framework: scikit-learn
* Task: Predict income (>50K or <=50K)

---

## 📊 Model Evaluation

Metrics used:

* Precision
* Recall
* F1 Score

Additionally:

* Slice-based evaluation implemented
* Output saved to `slice_output.txt`

---

## 🚀 API

### GET `/`

Returns welcome message

### POST `/inference`

Performs prediction

Example request:

```json
{
  "age": 37,
  "workclass": "Private",
  "fnlgt": 77516,
  "education": "Bachelors",
  "education-num": 13,
  "marital-status": "Married-civ-spouse",
  "occupation": "Tech-support",
  "relationship": "Husband",
  "race": "White",
  "sex": "Male",
  "capital-gain": 2174,
  "capital-loss": 0,
  "hours-per-week": 40,
  "native-country": "United-States"
}
```

---

## 🧪 Testing

* Unit tests implemented using pytest
* Tests include:

  * Model functions
  * API endpoints (GET + POST for both prediction outcomes)

---

## 🔄 Continuous Integration

* GitHub Actions pipeline runs:

  * `flake8`
  * `pytest`

✔ All checks passing

---

## 📸 Screenshots

* `example.png` → FastAPI docs with example request
* `continuous_integration.png` → CI pipeline passing

---

## 📁 Project Structure

```
starter/
  ├── starter/
  │   ├── ml/
  │   ├── train_model.py
  │   └── main.py
  ├── test_main.py
  └── test_model.py
```

---

## ⚠️ Notes

This project is for educational purposes and not intended for production use without further validation and fairness analysis.
