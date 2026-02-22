import joblib
from feature_extractor import extract_features
from geopy.geocoders import Nominatim

# -----------------------------
# Step 1: Load the trained model
# -----------------------------
model = joblib.load("safety_model.pkl")  # GradientBoostingClassifier

# -----------------------------
# Step 2: Get user input
# -----------------------------
place = input("Enter Place: ")
travel_hour = int(input("Enter Travel Hour (0-23): "))

# -----------------------------
# Step 3: Get coordinates (optional, for display)
# -----------------------------
geolocator = Nominatim(user_agent="safety_predictor")
location = geolocator.geocode(place)

if location:
    lat, lon = location.latitude, location.longitude
    print(f"Coordinates of {place}: {lat}, {lon}")
else:
    print(f"Could not find coordinates for {place}.")
    lat, lon = None, None

# -----------------------------
# Step 4: Extract features (keep existing feature extractor)
# -----------------------------
features = extract_features(place, travel_hour)

# -----------------------------
# Step 5: Predict probability and safety score
# -----------------------------
if features is not None:
    probability = model.predict_proba(features)[0][1] * 100  # class 1 = SAFE
    safety_score = round(probability, 2)

    print(f"\nSafety Score: {safety_score}%")

    # -----------------------------
    # Step 6: Risk classification
    # -----------------------------
    if safety_score < 40:
        print("⚠️ UNSAFE AREA")
    elif 40 <= safety_score < 70:
        print("⚡ CAUTION ADVISED")
    else:
        print("🟢 SAFE AREA")
else:
    print("Could not extract features. Please check the place name or input format.")