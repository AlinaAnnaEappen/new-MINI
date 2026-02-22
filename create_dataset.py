import pandas as pd
import numpy as np

np.random.seed(42)
n = 1000  # dataset size

# -----------------------------
# Generate Raw Features
# -----------------------------
data = pd.DataFrame({
    "street_light": np.random.randint(0, 40, n),
    "bus_stops": np.random.randint(0, 15, n),
    "shops": np.random.randint(0, 30, n),
    "poi": np.random.randint(0, 25, n),
    "parking_slots": np.random.randint(0, 50, n),
    "police_distance": np.random.uniform(200, 7000, n),
    "hospital_distance": np.random.uniform(200, 7000, n),
    "emergency_count": np.random.randint(0, 10, n),
    "travel_hour": np.random.randint(0, 24, n)
})

# -----------------------------
# Scoring Rules
# -----------------------------
def police_score(d):
    if d <= 3000:
        return 1
    elif d <= 5000:
        return 0.6
    return 0

def hospital_score(d):
    if d <= 2000:
        return 1
    elif d <= 5000:
        return 0.5
    return 0

def bus_score(c):
    if c >= 5:
        return 1
    elif c >= 2:
        return 0.6
    return 0.2

def light_score(c):
    if c >= 20:
        return 1
    elif c >= 10:
        return 0.6
    return 0.2

def parking_score(c):
    if 5 <= c <= 20:
        return 1
    elif 1 <= c < 5:
        return 0.6
    elif 21 <= c <= 40:
        return 0.6
    elif c == 0:
        return 0.2
    return 0.4

def time_score(hour):
    if 6 <= hour < 18:
        return 1.0
    elif 18 <= hour < 21:
        return 0.8
    elif 21 <= hour < 24:
        return 0.6
    return 0.4

# -----------------------------
# Apply Scores
# -----------------------------
data["police_score"] = data["police_distance"].apply(police_score)
data["hospital_score"] = data["hospital_distance"].apply(hospital_score)
data["bus_score"] = data["bus_stops"].apply(bus_score)
data["light_score"] = data["street_light"].apply(light_score)
data["parking_score"] = data["parking_slots"].apply(parking_score)
data["time_score"] = data["travel_hour"].apply(time_score)

data["activity_score"] = (data["shops"] + data["poi"]) / 55
data["emergency_score"] = 1 - (data["emergency_count"] / 10)

# -----------------------------
# Final Safety Score (1–100)
# -----------------------------
data["base_safety"] = (
    0.25 * data["emergency_score"] +
    0.20 * ((data["light_score"] + data["bus_score"]) / 2) +
    0.20 * data["activity_score"] +
    0.15 * data["parking_score"] +
    0.20 * ((data["police_score"] + data["hospital_score"]) / 2)
)

data["safety_score"] = data["base_safety"] * data["time_score"] * 100
data["safety_score"] = data["safety_score"].clip(1, 100)

# Keep final columns
data = data[[
    "street_light",
    "bus_stops",
    "shops",
    "poi",
    "parking_slots",
    "police_distance",
    "hospital_distance",
    "emergency_count",
    "travel_hour",
    "safety_score"
]]

data.to_csv("safety_dataset.csv", index=False)

print("Dataset created successfully!")
print(data.head())