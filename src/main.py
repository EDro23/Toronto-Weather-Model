"""
Toronto Weather Prediction Model
=================================
Classifies daily Toronto weather into three categories:
  0 = Dry   (no precipitation)
  1 = Rain
  2 = Snow

Usage
-----
Train and save models:
    python src/main.py

Predict from code:
    from src.main import predict
    result = predict({"avg_temperature": 5, "max_relative_humidity": 80, ...})
    print(result)  # {"label": "Rain", "class": 1, "probabilities": {...}}
"""

import os
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(_ROOT, "data", "raw", "weatherstats_toronto_daily.csv")
MODEL_PATH = os.path.join(_ROOT, "model", "weather_model.pkl")

# Columns removed before training
_DROP_ALWAYS = [
    "date",
    "sunrise_hhmm", "sunrise_unixtime", "sunrise_f",
    "sunset_hhmm", "sunset_unixtime", "sunset_f",
]
_DROP_NAN_COLS = [
    "solar_radiation",
    "max_cloud_cover_4", "avg_hourly_cloud_cover_4",
    "avg_cloud_cover_4", "min_cloud_cover_4",
]
_LEAKAGE_COLS = ["rain", "snow", "precipitation"]

LABEL_MAP = {0: "Dry", 1: "Rain", 2: "Snow"}

# ---------------------------------------------------------------------------
# Data loading & preparation
# ---------------------------------------------------------------------------

def load_data(path: str = DATA_PATH) -> pd.DataFrame:
    """Load raw CSV and strip whitespace from column names."""
    df = pd.read_csv(path)
    df.columns = df.columns.str.strip()
    return df


def build_target(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create the 'target' column from rain / snow amounts.
    Priority: Snow > Rain > Dry
    """
    df = df.copy()
    df["target"] = 0
    df.loc[df["rain"] > 0, "target"] = 1
    df.loc[df["snow"] > 0, "target"] = 2
    return df


def prepare_features(df: pd.DataFrame):
    """
    Drop non-informative, leakage, and high-NaN columns.
    Returns X (DataFrame) and y (Series).
    """
    df = build_target(df)

    cols_to_drop = [c for c in _DROP_ALWAYS + _DROP_NAN_COLS + _LEAKAGE_COLS
                    if c in df.columns]
    X = df.drop(columns=cols_to_drop + ["target"])
    y = df["target"]

    # Fill any remaining numeric NaNs with column medians
    X = X.fillna(X.median(numeric_only=True))

    return X, y


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train(X_train: pd.DataFrame, y_train: pd.Series):
    """
    Train a RandomForestClassifier and a LogisticRegression.
    Returns (rf_model, lr_model, scaler).
    """
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)

    rf = RandomForestClassifier(
        n_estimators=150,
        random_state=42,
        n_jobs=-1,
        class_weight="balanced",
    )
    rf.fit(X_train_scaled, y_train)

    lr = LogisticRegression(max_iter=1000, random_state=42)
    lr.fit(X_train_scaled, y_train)

    return rf, lr, scaler


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def evaluate(name: str, model, scaler: StandardScaler,
             X_test: pd.DataFrame, y_test: pd.Series) -> None:
    """Print accuracy, confusion matrix, and classification report."""
    X_scaled = scaler.transform(X_test)
    y_pred = model.predict(X_scaled)

    print(f"\n{'='*50}")
    print(f"  {name}")
    print(f"{'='*50}")
    print(f"Accuracy : {accuracy_score(y_test, y_pred):.4f}")
    print(f"\nConfusion Matrix:\n{confusion_matrix(y_test, y_pred)}")
    print(f"\nClassification Report (0=Dry, 1=Rain, 2=Snow):")
    print(classification_report(y_test, y_pred, target_names=["Dry", "Rain", "Snow"]))


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------

def save_model(rf, lr, scaler: StandardScaler,
               feature_columns: list, path: str = MODEL_PATH) -> None:
    """Serialize models, scaler, and feature column list to disk."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    bundle = {
        "random_forest": rf,
        "logistic_regression": lr,
        "scaler": scaler,
        "feature_columns": feature_columns,
    }
    joblib.dump(bundle, path, compress=3)
    print(f"\nModels saved to: {path}")


def load_model(path: str = MODEL_PATH) -> dict:
    """Load the saved model bundle from disk."""
    if not os.path.exists(path) or os.path.getsize(path) == 0:
        raise FileNotFoundError(
            f"No trained model found at '{path}'. "
            "Run 'python src/main.py' to train and save the model first."
        )
    return joblib.load(path)


# ---------------------------------------------------------------------------
# Prediction
# ---------------------------------------------------------------------------

def predict(features: dict, model_key: str = "random_forest",
            path: str = MODEL_PATH) -> dict:
    """
    Predict weather type from a dictionary of feature values.

    Parameters
    ----------
    features : dict
        Feature name → value pairs. Missing features are filled with 0.
    model_key : str
        "random_forest" (default) or "logistic_regression".
    path : str
        Path to the saved model bundle.

    Returns
    -------
    dict with keys:
        "class"         : int (0=Dry, 1=Rain, 2=Snow)
        "label"         : str ("Dry", "Rain", or "Snow")
        "probabilities" : dict of {label: probability}
    """
    bundle = load_model(path)
    model = bundle[model_key]
    scaler = bundle["scaler"]
    feature_columns = bundle["feature_columns"]

    row = pd.DataFrame([{col: features.get(col, 0) for col in feature_columns}])
    row_scaled = scaler.transform(row)

    pred_class = int(model.predict(row_scaled)[0])
    proba = model.predict_proba(row_scaled)[0]
    probabilities = {LABEL_MAP[i]: round(float(p), 4) for i, p in enumerate(proba)}

    return {
        "class": pred_class,
        "label": LABEL_MAP[pred_class],
        "probabilities": probabilities,
    }


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def main():
    print("Loading data ...")
    df = load_data()
    print(f"  {len(df):,} rows, {len(df.columns)} columns")

    print("\nPreparing features ...")
    X, y = prepare_features(df)
    print(f"  Features : {X.shape[1]} columns")
    print(f"  Target distribution:\n{y.value_counts().rename(LABEL_MAP).to_string()}")

    print("\nSplitting data (70% train / 30% test) ...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )
    print(f"  Train samples : {len(X_train):,}")
    print(f"  Test  samples : {len(X_test):,}")

    print("\nTraining models ...")
    rf, lr, scaler = train(X_train, y_train)
    print("  Done.")

    evaluate("Random Forest", rf, scaler, X_test, y_test)
    evaluate("Logistic Regression", lr, scaler, X_test, y_test)

    save_model(rf, lr, scaler, list(X.columns))

    # Quick smoke-test of the predict() function
    print("\nSample prediction (using first test row):")
    sample = X_test.iloc[0].to_dict()
    result = predict(sample)
    print(f"  → {result['label']} (class {result['class']})")
    print(f"  Probabilities: {result['probabilities']}")


if __name__ == "__main__":
    main()
