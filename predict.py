import joblib
import pandas as pd

model = joblib.load("safety_model.pkl")

sample_input = pd.DataFrame([{
    "street_light": 25,
    "bus_stops": 6,
    "shops": 15,
    "poi": 10,
    "parking_slots": 12,
    "police_distance": 2000,
    "hospital_distance": 1500,
    "emergency_count": 2,
    "travel_hour": 14
}])

prediction = model.predict(sample_input)[0]

print("Safety Score:", round(prediction, 2), "%")

if prediction >= 70:
    print("Status: SAFE AREA")
elif prediction >= 40:
    print("Status: MODERATE RISK")
else:
    print("Status: HIGH RISK")