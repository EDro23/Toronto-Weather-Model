# Toronto Weather Prediction Model

A machine-learning model that classifies daily Toronto weather into three categories:

| Class | Label | Description |
|-------|-------|-------------|
| 0 | **Dry** | No precipitation |
| 1 | **Rain** | Rain recorded |
| 2 | **Snow** | Snow recorded |

Two classifiers are trained and compared:
- **Random Forest** (~76% accuracy)
- **Logistic Regression** (~78% accuracy)

---

## Project Structure

```
Toronto-Weather-Model/
├── data/
│   └── raw/
│       └── weatherstats_toronto_daily.csv   # Historical Toronto weather (32 K+ rows)
├── model/
│   └── weather_model.pkl                    # Saved models, scaler & feature list
├── notebooks/
│   └── Toronto-Weather-Prediction-Model.ipynb  # Original Jupyter exploration notebook
├── src/
│   ├── main.py                              # Training pipeline + prediction function
│   └── requirements.txt                    # Python dependencies
└── README.md
```

---

## Quick Start

### 1. Install dependencies

```bash
pip install -r src/requirements.txt
```

### 2. Train the model

This reads the raw CSV, trains both classifiers, evaluates them, and saves the model bundle to `model/weather_model.pkl`.

```bash
python src/main.py
```

Expected output:

```
Loading data ...
  32,036 rows, 72 columns

Preparing features ...
  Features : 57 columns
  Target distribution:
Dry     19624
Rain     8447
Snow     3965

Splitting data (70% train / 30% test) ...
  Train samples : 22,425
  Test  samples : 9,611

Training models ...
  Done.

==================================================
  Random Forest
==================================================
Accuracy : 0.7636
...

==================================================
  Logistic Regression
==================================================
Accuracy : 0.7814
...

Models saved to: model/weather_model.pkl
```

---

## Making Predictions

After training, use the `predict()` function to classify a new day's weather. Pass a dictionary of feature values; any missing features default to `0`.

```python
from src.main import predict

result = predict({
    "avg_temperature": 3.0,
    "max_relative_humidity": 85.0,
    "avg_wind_speed": 20.0,
    "min_visibility": 10.0,
    "snow_on_ground": 5.0,
})

print(result["label"])          # e.g. "Snow"
print(result["class"])          # e.g. 2
print(result["probabilities"])  # e.g. {'Dry': 0.05, 'Rain': 0.10, 'Snow': 0.85}
```

By default the Random Forest model is used. To use Logistic Regression:

```python
result = predict(features, model_key="logistic_regression")
```

---

## Feature Engineering

The model uses **57 weather features** derived from the raw dataset. The following columns are excluded:

- **Date / time fields**: `date`, `sunrise_*`, `sunset_*`
- **High-NaN columns**: `solar_radiation`, `*_cloud_cover_4`
- **Target-leakage columns**: `rain`, `snow`, `precipitation` (these directly define the target)

The target label is created from the raw data:
- `snow > 0` → class 2 (Snow)
- `rain > 0` → class 1 (Rain)
- otherwise → class 0 (Dry)

---

## Data

Source: [weatherstats.ca](https://toronto.weatherstats.ca/download.html) – daily weather statistics for Toronto.

The raw CSV contains 74 attributes including temperature, humidity, dew point, wind speed/gust, pressure, visibility, health index, UV forecasts, cloud cover, and degree-day accumulators.

---

## Dependencies

| Package | Version |
|---------|---------|
| pandas | ≥ 1.3.0 |
| numpy | ≥ 1.21.0 |
| scikit-learn | ≥ 1.0.0 |
| joblib | ≥ 1.1.0 |
