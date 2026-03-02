import joblib
import folium
from feature_extractor import extract_features


# -----------------------------
# Load ML Model
# -----------------------------
model = joblib.load("safety_model.pkl")


# -----------------------------
# User Input
# -----------------------------
place = input("Enter Place: ")
travel_hour = int(input("Enter Travel Hour (0-23): "))


# -----------------------------
# Extract Features + Contacts
# -----------------------------
result = extract_features(place, travel_hour)

if result is None:
    print("Location not found.")
    exit()

features, lat, lon, hospital, police = result


# -----------------------------
# Predict Safety
# -----------------------------
probability = model.predict_proba(features)[0][1] * 100
safety_score = round(probability, 2)

print(f"\nSafety Score: {safety_score}%")

if safety_score < 40:
    print("⚠️ UNSAFE AREA")
elif 40 <= safety_score < 70:
    print("⚡ CAUTION ADVISED")
else:
    print("🟢 SAFE AREA")


# -----------------------------
# Display Emergency Contacts
# -----------------------------
print("\n--- Emergency Contacts Nearby ---")

print("\nNearest Police Station:")
print("Name:", police["name"])
print("Phone:", police["phone"])

print("\nNearest Hospital:")
print("Name:", hospital["name"])
print("Phone:", hospital["phone"])


# -----------------------------
# Generate Map
# -----------------------------
m = folium.Map(location=[lat, lon], zoom_start=14)

# User marker
folium.Marker(
    [lat, lon],
    popup="User Location",
    icon=folium.Icon(color="green")
).add_to(m)

# Police marker
if police["lat"] is not None:
    folium.Marker(
        [police["lat"], police["lon"]],
        popup=f"Police: {police['name']} | Phone: {police['phone']}",
        icon=folium.Icon(color="blue")
    ).add_to(m)

# Hospital marker
if hospital["lat"] is not None:
    folium.Marker(
        [hospital["lat"], hospital["lon"]],
        popup=f"Hospital: {hospital['name']} | Phone: {hospital['phone']}",
        icon=folium.Icon(color="red")
    ).add_to(m)

m.save("safety_map.html")

print("\nMap saved as safety_map.html (Open in browser)")
