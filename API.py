from fastapi import FastAPI
import joblib
import numpy as np

app = FastAPI()
model = joblib.load("offset_predictor.pkl")

@app.get("/predict")
def predict(shell_travel_time: float, distance: float, angle: float, enemy_max_speed: float):
    X = np.array([[shell_travel_time, distance, angle, enemy_max_speed]])
    prediction = model.predict(X)[0]
    return {"offset_x": prediction}