import osmnx as ox
import pandas as pd
from math import radians, cos, sin, asin, sqrt

ox.settings.log_console = False
ox.settings.use_cache = True

LOCAL_RADIUS = 1000
POLICE_RADIUS = 5000


# ---------------------------
# HAVERSINE
# ---------------------------

def haversine(lat1, lon1, lat2, lon2):
    R = 6371000
    dlat = radians(lat2 - lat1)
    dlon = radians(lon2 - lon1)
    a = sin(dlat/2)**2 + cos(radians(lat1)) * cos(radians(lat2)) * sin(dlon/2)**2
    c = 2 * asin(sqrt(a))
    return R * c


# ---------------------------
# GEOCODE
# ---------------------------

def get_coordinates(place):
    try:
        return ox.geocode(place)
    except:
        return None, None


# ---------------------------
# EXTRACT FEATURES
# ---------------------------

def extract_features(place, hour):

    lat, lon = get_coordinates(place)

    if lat is None:
        print("Location not found.")
        return None

    print("Coordinates:", lat, lon)

    # -------- 1KM FEATURES (INCLUDING HOSPITAL) --------
    tags_local = {
        "highway": ["street_lamp", "bus_stop"],
        "shop": True,
        "amenity": True
    }

    try:
        gdf_local = ox.features_from_point(
            (lat, lon),
            tags=tags_local,
            dist=LOCAL_RADIUS
        )
    except:
        gdf_local = pd.DataFrame()

    street_light = len(gdf_local[gdf_local.get("highway") == "street_lamp"])
    bus_stops = len(gdf_local[gdf_local.get("highway") == "bus_stop"])
    shops = len(gdf_local[gdf_local.get("shop").notna()])
    poi = len(gdf_local[gdf_local.get("amenity").notna()])
    parking_slots = len(gdf_local[gdf_local.get("amenity") == "parking"])

    emergency_count = len(
        gdf_local[gdf_local.get("amenity").isin(["police", "hospital", "fire_station"])]
    )

    # Hospital distance (within 1km)
    hospital_distance = 9999
    hospital_places = gdf_local[gdf_local.get("amenity") == "hospital"]

    if not hospital_places.empty:
        hospital_distance = min(
            haversine(lat, lon,
                      row.geometry.centroid.y,
                      row.geometry.centroid.x)
            for _, row in hospital_places.iterrows()
        )
        hospital_distance = round(hospital_distance, 2)

    # -------- POLICE 5KM --------
    police_distance = 9999

    try:
        police_gdf = ox.features_from_point(
            (lat, lon),
            tags={"amenity": "police"},
            dist=POLICE_RADIUS
        )

        if not police_gdf.empty:
            police_distance = min(
                haversine(lat, lon,
                          row.geometry.centroid.y,
                          row.geometry.centroid.x)
                for _, row in police_gdf.iterrows()
            )
            police_distance = round(police_distance, 2)

    except:
        pass

    data = {
        "street_light": street_light,
        "bus_stops": bus_stops,
        "shops": shops,
        "poi": poi,
        "parking_slots": parking_slots,
        "police_distance": police_distance,
        "hospital_distance": hospital_distance,
        "emergency_count": emergency_count,
        "travel_hour": hour
    }

    return pd.DataFrame([data])