import joblib
import folium
from feature_extractor import extract_features, haversine  # make sure haversine is imported

# Load model once
model = joblib.load("safety_model.pkl")


def predict_safety(place: str, travel_hour: int):
    """
    Predict safety score and return structured result with distances and map
    """

    result = extract_features(place, travel_hour)

    if result is None:
        return {"success": False, "error": "Location not found"}

    features, lat, lon, hospital, police = result

    # -----------------------------
    # Predict Safety Score
    # -----------------------------
    probability = model.predict_proba(features)[0][1] * 100
    safety_score = round(probability, 2)

    # Risk Level & Color
    if safety_score < 40:
        risk_level = "UNSAFE"
        color = "red"
    elif 40 <= safety_score < 70:
        risk_level = "CAUTION"
        color = "orange"
    else:
        risk_level = "SAFE"
        color = "green"

    # -----------------------------
    # Calculate Distances
    # -----------------------------
    police_distance = None
    hospital_distance = None

    if police["lat"] is not None:
        police_distance = round(haversine(lat, lon, police["lat"], police["lon"]) / 1000, 2)  # in km

    if hospital["lat"] is not None:
        hospital_distance = round(haversine(lat, lon, hospital["lat"], hospital["lon"]) / 1000, 2)  # in km

    # -----------------------------
    # Generate Dynamic Map
    # -----------------------------
    m = folium.Map(location=[lat, lon], zoom_start=14)

    folium.Marker(
        [lat, lon],
        popup=f"Location: {place}<br>Safety Score: {safety_score}%",
        icon=folium.Icon(color=color)
    ).add_to(m)

    if police["lat"] is not None:
        folium.Marker(
            [police["lat"], police["lon"]],
            popup=(
                f"Police: {police['name']}<br>"
                f"Phone: {police['phone']}<br>"
                f"Distance: {police_distance} km"
            ),
            icon=folium.Icon(color="blue")
        ).add_to(m)

    if hospital["lat"] is not None:
        folium.Marker(
            [hospital["lat"], hospital["lon"]],
            popup=(
                f"Hospital: {hospital['name']}<br>"
                f"Phone: {hospital['phone']}<br>"
                f"Distance: {hospital_distance} km"
            ),
            icon=folium.Icon(color="red")
        ).add_to(m)

    map_html = m._repr_html_()

    # -----------------------------
    # Return structured response
    # -----------------------------
    return {
        "success": True,
        "safety_score": safety_score,
        "risk_level": risk_level,
        "coordinates": {"lat": lat, "lon": lon},
        "police": {**police, "distance_km": police_distance},
        "hospital": {**hospital, "distance_km": hospital_distance},
        "map_html": map_html
    }
