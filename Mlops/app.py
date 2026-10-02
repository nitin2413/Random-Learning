import joblib
from pathlib import Path
from fastapi import FastAPI

app = FastAPI()

MODEL_PATH = Path(__file__).parent / "model.pkl"
model = joblib.load(MODEL_PATH)

@app.get("/health")
def health():
    return {"status" : "ok"}

@app.post("/predict")
def predict(data: dict):
    input = [data["input"]]
    prediction = model.predict(input).tolist()
    return {"prediction": prediction}

