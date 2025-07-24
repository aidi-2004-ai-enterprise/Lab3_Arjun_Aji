

# app/main.py

from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import xgboost as xgb
import numpy as np
import json
import os

app = FastAPI()

# Define request model
class PenguinFeatures(BaseModel):
    bill_length_mm: float
    bill_depth_mm: float
    flipper_length_mm: float
    body_mass_g: float
    sex_female: int
    sex_male: int
    island_Biscoe: int
    island_Dream: int
    island_Torgersen: int

# Load model and metadata
model_path = "app/data/model.json"
meta_path = "app/data/metadata.json"

if not os.path.exists(model_path) or not os.path.exists(meta_path):
    raise FileNotFoundError("Model or metadata not found. Train the model first.")

booster = xgb.Booster()
booster.load_model(model_path)

with open(meta_path, "r") as f:
    metadata = json.load(f)

model_columns = metadata["features"]
class_names = metadata["classes"]

@app.get("/")
def home():
    return {"message": "Penguin species classifier is running!"}

@app.post("/predict")
def predict_species(features: PenguinFeatures):
    try:
        # Convert input to DataFrame
        input_df = pd.DataFrame([features.dict()])
        input_df = input_df[model_columns]  # Ensure correct order

        dmatrix = xgb.DMatrix(input_df)
        prediction = booster.predict(dmatrix)
        class_index = int(np.argmax(prediction))
        predicted_species = class_names[class_index]

        return {"predicted_species": predicted_species}
    except Exception as e:
        return {"error": str(e)}
