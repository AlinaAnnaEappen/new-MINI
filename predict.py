import os
import pickle
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime

import pandas as pd
from flask import Flask, jsonify, render_template, request

from fetch_features import get_coordinates, get_nearest_facilities, get_safety_features

app = Flask(__name__)

# -----------------------------
# Load Model + Scaler
# -----------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "gradient_boosting_model.pkl")
SCALER_PATH = os.path.join(BASE_DIR, "scaler.pkl")
FEATURES_PATH = os.path.join(BASE_DIR, "feature_columns.pkl")
DEFAULT_FEATURE_COLUMNS = [
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

with open(MODEL_PATH, "rb") as f:
    model = pickle.load(f)

with open(SCALER_PATH, "rb") as f:
    scaler = pickle.load(f)

if os.path.exists(FEATURES_PATH):
    with open(FEATURES_PATH, "rb") as f:
        FEATURE_COLUMNS = pickle.load(f)
else:
    FEATURE_COLUMNS = DEFAULT_FEATURE_COLUMNS

LOOKUP_CACHE = {}
LOOKUP_CACHE_TTL_SECONDS = 600


# -----------------------------
# Helpers
# -----------------------------
def clamp(value, low, high):
    return max(low, min(high, value))


def to_float(value):
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def safety_label(prob):
    if prob < 35:
        return "UNSAFE"
    if prob < 65:
        return "CAUTIOUS"
    return "SAFE"


def cache_get_or_compute(key, compute_fn):
    now_ts = time.time()
    cached = LOOKUP_CACHE.get(key)
    if cached and (now_ts - cached["ts"] <= LOOKUP_CACHE_TTL_SECONDS):
        return cached["value"]

    value = compute_fn()
    LOOKUP_CACHE[key] = {"ts": now_ts, "value": value}
    return value


def normalize_features(raw_features, travel_hour, facilities):
    street_light = int(clamp(to_float(raw_features.get("street_lighting")) or 0, 0, 40))
    bus_stops = int(clamp(to_float(raw_features.get("bus_stop_count")) or 0, 0, 15))
    shops = int(clamp(to_float(raw_features.get("nearby_shops")) or 0, 0, 30))
    poi = int(clamp(to_float(raw_features.get("poi_count")) or 0, 0, 25))
    parking_slots = int(clamp(to_float(raw_features.get("parking_slots")) or 0, 0, 50))
    emergency_support_count = int(
        clamp(to_float(raw_features.get("emergency_count")) or 0, 0, 10)
    )

    police_info = (facilities or {}).get("police_station") or {}
    hospital_info = (facilities or {}).get("hospital") or {}

    police_distance_km = to_float(police_info.get("distance_km"))
    if police_distance_km is None:
        police_distance_m = 3500.0
    else:
        police_distance_m = police_distance_km * 1000.0
    police_distance_m = clamp(police_distance_m, 200.0, 7000.0)

    hospital_distance_km = to_float(raw_features.get("hospital_distance"))
    nearest_hospital_km = to_float(hospital_info.get("distance_km"))
    if hospital_distance_km is not None and nearest_hospital_km is not None:
        hospital_distance_km = min(hospital_distance_km, nearest_hospital_km)
    elif hospital_distance_km is None:
        hospital_distance_km = nearest_hospital_km

    if hospital_distance_km is None:
        hospital_distance_m = 3500.0
    else:
        hospital_distance_m = hospital_distance_km * 1000.0
    hospital_distance_m = clamp(hospital_distance_m, 200.0, 7000.0)

    # Model was trained with "higher emergency_count = higher risk".
    # Live OSM feature currently counts nearby emergency facilities (higher is better),
    # so convert support count into a risk-like proxy to keep semantics aligned.
    emergency_support_effective = emergency_support_count
    if police_distance_m <= 3000:
        emergency_support_effective += 2
    if hospital_distance_m <= 2500:
        emergency_support_effective += 2
    emergency_support_effective = int(clamp(emergency_support_effective, 0, 10))
    emergency_count = int(clamp(10 - emergency_support_effective, 0, 10))

    return {
        "street_light": street_light,
        "bus_stops": bus_stops,
        "shops": shops,
        "poi": poi,
        "parking_slots": parking_slots,
        "police_distance": round(police_distance_m, 2),
        "hospital_distance": round(hospital_distance_m, 2),
        "emergency_count": emergency_count,
        "travel_hour": int(clamp(travel_hour, 0, 23)),
    }


def apply_risk_adjustments(prob, features, is_night, is_weekend):
    adjusted_prob = prob
    risk_points = 0

    if is_night == 1:
        adjusted_prob -= 8.0
        risk_points += 1
    if features["street_light"] <= 1:
        adjusted_prob -= 10.0
        risk_points += 1
    if features["police_distance"] >= 6000.0:
        adjusted_prob -= 8.0
        risk_points += 1
    if features["hospital_distance"] >= 6000.0:
        adjusted_prob -= 6.0
        risk_points += 1
    if features["bus_stops"] <= 1:
        adjusted_prob -= 5.0
        risk_points += 1
    if features["shops"] <= 2:
        adjusted_prob -= 4.0
        risk_points += 1
    if features["poi"] <= 1:
        adjusted_prob -= 3.0
        risk_points += 1
    if features["emergency_count"] >= 7:
        adjusted_prob -= 7.0
        risk_points += 1
    if is_weekend == 1 and is_night == 1:
        adjusted_prob -= 3.0
        risk_points += 1

    # Positive calibration for clearly well-served, daytime urban areas.
    if is_night == 0 and features["street_light"] >= 8:
        adjusted_prob += 3.0
    if features["bus_stops"] >= 4:
        adjusted_prob += 1.0
    if features["shops"] >= 12 and features["poi"] >= 8:
        adjusted_prob += 2.0
    if features["police_distance"] <= 3000.0 and features["hospital_distance"] <= 3000.0:
        adjusted_prob += 3.0

    force_unsafe = (
        risk_points >= 6
        or (is_night == 1 and features["street_light"] <= 1 and features["police_distance"] >= 6000.0)
    )

    adjusted_prob = clamp(adjusted_prob, 0.0, 100.0)
    if force_unsafe:
        adjusted_prob = min(adjusted_prob, 35.0)

    return adjusted_prob, force_unsafe, risk_points


# -----------------------------
# Pages Routes
# -----------------------------
@app.route("/")
def home():
    return render_template("frontend.html")


@app.route("/advisory")
def advisory():
    return render_template("advisory.html")


@app.route("/aboutus")
def aboutus():
    return render_template("aboutus.html")


@app.route("/contact")
def contact():
    return render_template("contact.html")


@app.route("/feedback")
def feedback():
    return render_template("feedback.html")


@app.route("/map")
def map_page():
    return render_template("map.html")


# -----------------------------
# Predict Route (API)
# -----------------------------
@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json(silent=True) or request.form

    place = (data.get("place") or "").strip()
    date_str = (data.get("date") or "").strip()
    time_str = (data.get("time") or "").strip()

    if not place or not date_str or not time_str:
        return jsonify({"error": "Missing place/date/time"}), 400

    if ":" not in time_str:
        return jsonify({"error": "Invalid time format. Use HH:MM"}), 400

    time_parts = time_str.split(":")
    if len(time_parts) < 
