🏎️ F1 Race Pace Predictor — 2025 Canadian GP
🚀 Project Overview
This project builds a full machine learning pipeline to simulate and predict race pace outcomes for the 2025 Canadian Grand Prix using real F1 race data.

The model combines:

FastF1 API (2024 Canada GP race data + 2025 Canada GP qualifying data)

OpenWeatherMap API (real weather forecasts)

Full ML pipeline: feature engineering, hyperparameter tuning, evaluation, and calibration

⚙️ Tech Stack
Python 3.10+

FastF1 API (live race/qualifying data ingestion)

scikit-learn (modeling, hyperparameter tuning, nested cross-validation)

Gradient Boosting Regressor

OpenWeatherMap API (live weather data)

pandas / numpy / matplotlib (data prep & visualization)

🔬 Key ML Pipeline Steps
✅ Cleaned race laps using .pick_quicklaps() (filtered out pit stops, out laps, safety cars, and DNFs)

✅ Feature engineering:

Sector 1, 2, 3 averages per driver

Qualifying delta-to-pole

Normalized weather data (rain probability & temperature)

Driver/team categorical encodings

✅ Trained Gradient Boosting model with nested cross-validation for robust model evaluation on small dataset

✅ Applied post-prediction calibration to align output lap times with realistic race pace (~75 seconds per lap)

📊 Sample Output
python-repl

🏁 2025 Canada GP Predicted Results 🏁

Driver    PredictedRaceTime (s)
------    ---------------------
HAM       ~75.5
VER       ~75.6
NOR       ~75.7
...

✅ Predictions now align with real-world F1 race pace at Montreal.

🔎 Key Challenges I Faced
Raw lap data includes invalid laps (pit-ins, out-laps, formation laps, SC laps) which heavily distort target race pace. Fixing this with FastF1’s lap filtering was critical.

Normalizing weather data prevented non-sector features from dominating lap time predictions.

Handling missing supervision for drivers who had no valid race data (e.g. DNFs like Leclerc) without allowing them to distort fold-level evaluation.

Applied nested cross-validation to avoid overfitting and give a stable estimate of generalization error.

🔄 Possible Next Steps
Add multi-season historical data for stronger supervision across all drivers

Incorporate tire compounds, safety car timing, and pit stop strategies

Build blended models that simulate quali-to-race delta shifts

Use grouped CV to stabilize fold variance for small datasets

📦 Install Instructions
git clone https://github.com/ec1s123/f1-canada-predictor.git

cd f1-canada-predictor

pip install -r requirements.txt

🔑 API Keys Required

You will need an OpenWeatherMap API key.

Add your API key to the API_KEY = "YOUR_API_KEY" line in the code.

📄 File Structure
├── canada_gp_predictor.py    # Full ML pipeline code
├── README.md                 # Project documentation (this file)
├── requirements.txt          # Package dependencies
├── feature_importance.png    # Optional feature importance visual

⚠ License
For educational purposes only. Not affiliated with Formula 1 or any F1 teams.

