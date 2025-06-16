import fastf1
import pandas as pd
import numpy as np
import requests
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import GridSearchCV, KFold
from sklearn.metrics import mean_absolute_error, make_scorer
from sklearn.ensemble import GradientBoostingRegressor
import matplotlib.pyplot as plt

# Pull Canada 2024 race data
session_2024 = fastf1.get_session(2024, 'Canada', 'R')
session_2024.load()

# Filter valid race laps only (no out-laps, safety cars, pits, etc.)
laps_2024 = session_2024.laps.pick_quicklaps().copy()

# Convert lap times to seconds
for col in ["LapTime", "Sector1Time", "Sector2Time", "Sector3Time"]:
    laps_2024[f"{col} (s)"] = laps_2024[col].dt.total_seconds()

# Compute sector averages per driver
sector_times = laps_2024.groupby("Driver").agg({
    "Sector1Time (s)": "mean",
    "Sector2Time (s)": "mean",
    "Sector3Time (s)": "mean"
}).reset_index()

# Load 2025 Canada qualifying data
session_qualy = fastf1.get_session(2025, 'Canada', 'Q')
session_qualy.load()
laps_qualy = session_qualy.laps.pick_quicklaps().copy()
laps_qualy['QualifyingTime (s)'] = laps_qualy['LapTime'].dt.total_seconds()

# Get best quali time per driver
qualifying = laps_qualy.groupby('Driver')['QualifyingTime (s)'].min().reset_index()
pole_time = qualifying['QualifyingTime (s)'].min()
qualifying['QualiDeltaToPole (s)'] = qualifying['QualifyingTime (s)'] - pole_time

# Pull race-day weather forecast
API_KEY = "e70daeaec703afb866ed8107b1c6cf8b"
weather_url = f"http://api.openweathermap.org/data/2.5/forecast?lat=45.5019&lon=-73.5674&appid={API_KEY}&units=metric"
weather = requests.get(weather_url).json()

# Extract single forecast time
forecast_time = "2025-06-15 13:00:00"
forecast = next((f for f in weather['list'] if f['dt_txt'] == forecast_time), None)

# Apply weather data globally across qualifying dataset
rain = (forecast['pop'] if forecast else 0) * 100
temp = (forecast['main']['temp'] if forecast else 20)
qualifying['RainProbability'] = rain
qualifying['TemperatureNorm'] = temp - 20

# Simple team map
team_map = {
    "VER": "Red Bull", "PER": "Red Bull", "NOR": "McLaren", "PIA": "McLaren",
    "LEC": "Ferrari", "SAI": "Ferrari", "HAM": "Mercedes", "RUS": "Mercedes",
    "GAS": "Alpine", "OCO": "Alpine", "ALO": "Aston Martin", "STR": "Aston Martin",
    "TSU": "Racing Bulls", "RIC": "Racing Bulls", "HUL": "Haas", "MAG": "Haas",
    "BOT": "Kick Sauber", "ZHO": "Kick Sauber", "SAR": "Williams", "ALB": "Williams"
}
qualifying["Team"] = qualifying["Driver"].map(team_map)

# Merge sector averages into qualifying data
data = qualifying.merge(sector_times, on="Driver", how="left")

# Fill missing sector data with sector medians (for new drivers)
for s in ["Sector1Time (s)", "Sector2Time (s)", "Sector3Time (s)"]:
    median_value = sector_times[s].median()
    data[s].fillna(median_value, inplace=True)

# Encode categorical variables
data['Driver_encoded'] = LabelEncoder().fit_transform(data['Driver'])
data['Team_encoded'] = LabelEncoder().fit_transform(data['Team'])

# Features used for modeling
features = [
    "QualiDeltaToPole (s)",
    "Sector1Time (s)", "Sector2Time (s)", "Sector3Time (s)",
    "RainProbability", "TemperatureNorm",
    "Driver_encoded", "Team_encoded"
]
X = data[features]

# Build race pace target from cleaned 2024 race data
y_raw = laps_2024.groupby("Driver")["LapTime (s)"].mean()
data["RaceTimeTarget"] = data["Driver"].map(y_raw)
y = data["RaceTimeTarget"].fillna(y_raw.mean())

# Nested CV with hyperparameter tuning
param_grid = {
    "n_estimators": [100, 300, 500],
    "learning_rate": [0.01, 0.05, 0.1],
    "max_depth": [3, 4, 5],
    "subsample": [0.8, 1.0]
}
mae = make_scorer(mean_absolute_error, greater_is_better=False)
outer_cv = KFold(n_splits=5, shuffle=True, random_state=7)
outer_mae = []

for train_idx, test_idx in outer_cv.split(X):
    grid = GridSearchCV(GradientBoostingRegressor(random_state=7), param_grid, scoring=mae, cv=3, n_jobs=-1)
    grid.fit(X.iloc[train_idx], y.iloc[train_idx])
    y_pred = grid.best_estimator_.predict(X.iloc[test_idx])
    fold_mae = mean_absolute_error(y.iloc[test_idx], y_pred)
    outer_mae.append(fold_mae)
    print(f"Fold MAE: {fold_mae:.3f}")

print(f"\nNested CV MAE: {np.mean(outer_mae):.3f} ± {np.std(outer_mae):.3f} seconds")

# Train final model on full data
final_grid = GridSearchCV(GradientBoostingRegressor(random_state=7), param_grid, scoring=mae, cv=5, n_jobs=-1)
final_grid.fit(X, y)
model = final_grid.best_estimator_

importances = model.feature_importances_
feature_names = [
    "QualiDeltaToPole (s)",
    "Sector1Time (s)", "Sector2Time (s)", "Sector3Time (s)",
    "RainProbability", "TemperatureNorm",
    "Driver_encoded", "Team_encoded"
]

# Plot feature importances
plt.figure(figsize=(8, 5))
plt.barh(feature_names, importances, color="skyblue")
plt.xlabel("Feature Importance")
plt.title("Feature Importance - 2025 Canada GP Race Pace Model")
plt.gca().invert_yaxis()
plt.tight_layout()
plt.show()

# Apply output calibration
y_pred_raw = model.predict(X)
shift = y.mean() - np.mean(y_pred_raw)
y_pred_calibrated = y_pred_raw + shift
data["PredictedRaceTime (s)"] = y_pred_calibrated

# Display predictions
results = data.sort_values("PredictedRaceTime (s)").reset_index(drop=True)
print("\n🏁 2025 Canada GP Predicted Results 🏁")
print(results[["Driver", "PredictedRaceTime (s)"]])

print("\n🏁 2025 Canada GP Podium:🏁\n")
for pos, row in results.head(3).iterrows():
    medals = ["🥇", "🥈", "🥉"]
    print(f"{medals[pos]} {row['Driver']} ({row['PredictedRaceTime (s)']:.3f} s)")
