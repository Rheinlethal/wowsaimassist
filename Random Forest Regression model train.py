import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
import joblib

# Load data
data = pd.read_csv("excel datasets tembakan.csv")  # file berisi kolom sesuai formatmu

X = data[["shell_travel_time", "distance", "angle", "enemy_max_speed"]]
y = data["offset_x"]

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Save model
joblib.dump(model, "offset_predictor.pkl")

# Test akurasi
print("Score:", model.score(X_test, y_test))