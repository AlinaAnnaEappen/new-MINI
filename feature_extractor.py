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

    # -------- 1KM FEATURES --------
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

    # -------- HOSPITAL (within 1km) --------
    hospital_distance = 9999
    nearest_hospital = {
        "name": "Not Found",
        "phone": "Not Available",
        "lat": None,
        "lon": None
    }

    hospital_places = gdf_local[gdf_local.get("amenity") == "hospital"]

    if not hospital_places.empty:
        nearest_row = min(
            hospital_places.iterrows(),
            key=lambda x: haversine(
                lat, lon,
                x[1].geometry.centroid.y,
                x[1].geometry.centroid.x
            )
        )[1]

        hospital_distance = haversine(
            lat, lon,
            nearest_row.geometry.centroid.y,
            nearest_row.geometry.centroid.x
        )

        hospital_distance = round(hospital_distance, 2)

        nearest_hospital = {
            "name": nearest_row.get("name", "Unknown"),
            "phone": nearest_row.get("phone", "Not Available"),
            "lat": nearest_row.geometry.centroid.y,
            "lon": nearest_row.geometry.centroid.x
        }

    # -------- POLICE (within 5km) --------
    police_distance = 9999
    nearest_police = {
        "name": "Not Found",
        "phone": "Not Available",
        "lat": None,
        "lon": None
    }

    try:
        police_gdf = ox.features_from_point(
            (lat, lon),
            tags={"amenity": "police"},
            dist=POLICE_RADIUS
        )

        if not police_gdf.empty:
            nearest_row = min(
                police_gdf.iterrows(),
                key=lambda x: haversine(
                    lat, lon,
                    x[1].geometry.centroid.y,
                    x[1].geometry.centroid.x
                )
            )[1]

            police_distance = haversine(
                lat, lon,
                nearest_row.geometry.centroid.y,
                nearest_row.geometry.centroid.x
            )

            police_distance = round(police_distance, 2)

            nearest_police = {
                "name": nearest_row.get("name", "Unknown"),
                "phone": nearest_row.get("phone", "Not Available"),
                "lat": nearest_row.geometry.centroid.y,
                "lon": nearest_row.geometry.centroid.x
            }

    except:
        pass

    # -------- MODEL FEATURES --------
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

    return pd.DataFrame([data]), lat, lon, nearest_hospital, nearest_police
