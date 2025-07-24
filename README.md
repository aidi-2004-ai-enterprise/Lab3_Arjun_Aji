# 🐧 Penguins Classification - Lab 3 (Arjun Aji)

This project uses **XGBoost** to train a machine learning model that classifies penguin species based on features like bill length, flipper length, and body mass. The model is deployed using **FastAPI**, allowing prediction via an interactive web interface or API endpoint.

---

## 📁 Project Structure
```bash
Lab3_Arjun_Aji/
├── app/
│ └── main.py # FastAPI application
├── data/
│ └── penguins.csv # Dataset
├── model/
│ └── model.pkl # Trained XGBoost model
├── train.py # Model training script
├── pyproject.toml # Project dependencies (PEP 621 style)
├── .gitignore
└── README.md # Project overview

```

---

## 🚀 Features

- Train a classification model using **XGBoost**
- Serve predictions via **FastAPI**
- Type hinting and Pydantic models for input validation
- Local virtual environment setup
- Minimal dependencies managed via `pyproject.toml`

---

## 📦 Dependencies

Managed in `pyproject.toml` using [UV](https://github.com/astral-sh/uv) or [Poetry].

Essential libraries:

- `xgboost`
- `pandas`
- `scikit-learn`
- `fastapi`
- `uvicorn`
- `pydantic`

---

## 🛠️ Setup Instructions

### 1. Clone the repo

```bash
git clone https://github.com/aidi-2004-ai-enterprise/Lab3_Arjun_Aji.git
cd Lab3_Arjun_Aji
```
2. Create virtual environment
```bash

python -m venv .venv
# Activate it:
# On Windows:
.venv\Scripts\activate
# On macOS/Linux:
source .venv/bin/activate
```
3. Install dependencies
If you're using uv:

```bash

uv pip install -r requirements.txt
# OR
uv pip install -r pyproject.toml
```
Or install manually:

```bash

pip install -r requirements.txt
```
📊 Training the Model
```bash

python train.py
```


🌐 Running the FastAPI Server
```bash

uvicorn app.main:app --reload
Open in browser: http://127.0.0.1:8000

Swagger UI: http://127.0.0.1:8000/docs
```
📥 Sample Input (POST)
```bash
{
  "island": "Torgersen",
  "bill_length_mm": 39.1,
  "bill_depth_mm": 18.7,
  "flipper_length_mm": 181,
  "body_mass_g": 3750,
  "sex": "Male"
}
```
📌 Notes
Do not push .venv folder to GitHub.

Avoid committing large model binaries directly.

Add .venv and /model/*.pkl to your .gitignore.

