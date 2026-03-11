import pickle

import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

DATASET_PATH = "women_travel_safety_dataset_updated.csv"
MODEL_PATH = "gradient_boosting_model.pkl"
SCALER_PATH = "scaler.pkl"
FEATURES_PATH = "feature_columns.pkl"
TARGET_COLUMN = "safety_score"
FEATURE_COLUMNS = [
    "street_light",
    "bus_stops",
    "shops",
    "poi",
    "parking_slots",
    "police_distance",
    "hospital_distance",
    "emergency_count",
    "travel_hour",
]


def main():
    data = pd.read_csv(DATASET_PATH)

    if "id" in data.columns:
        data = data.drop(columns=["id"])

    required_columns = FEATURE_COLUMNS + [TARGET_COLUMN]
    missing_columns = [c for c in required_columns if c not in data.columns]
    if missing_columns:
        raise ValueError(f"Dataset is missing required columns: {missing_columns}")

    clean_data = data[required_columns].dropna().copy()
    if clean_data.empty:
        raise ValueError("Dataset has no valid rows after dropping missing values.")

    x = clean_data[FEATURE_COLUMNS]
    y = clean_data[TARGET_COLUMN]

    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.2, random_state=42
    )

    scaler = StandardScaler()
    x_train_scaled = scaler.fit_transform(x_train)
    x_test_scaled = scaler.transform(x_test)

    model = GradientBoostingRegressor(random_state=42)
    model.fit(x_train_scaled, y_train)

    predictions = model.predict(x_test_scaled)
    mae = mean_absolute_error(y_test, predictions)
    r2 = r2_score(y_test, predictions)

    with open(MODEL_PATH, "wb") as f:
        pickle.dump(model, f)
    with open(SCALER_PATH, "wb") as f:
        pickle.dump(scaler, f)
    with open(FEATURES_PATH, "wb") as f:
        pickle.dump(FEATURE_COLUMNS, f)

    print("Model trained successfully")
    print("Rows used:", len(clean_data))
    print("Features:", FEATURE_COLUMNS)
    print("MAE:", round(mae, 3))
    print("R2:", round(r2, 3))
    print(
        f"Saved: {MODEL_PATH}, {SCALER_PATH}, {FEATURES_PATH}"
    )


if __name__ == "__main__":
    main()
