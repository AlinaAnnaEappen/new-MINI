import requests
import math
import time
import re
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor

# Overpass endpoints (fallback order)
OVERPASS_URLS = [
    "https://overpass.kumi.systems/api/interpreter",
    "https://overpass.private.coffee/api/interpreter",
]
NOMINATIM_URL = "https://nominatim.openstreetmap.org/search"
OVERPASS_TIMEOUT = 6
NOMINATIM_TIMEOUT = 10
OVERPASS_RETRIES = 1
COORDS_REGEX = re.compile(
    r"^\s*(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)\s*$"
)
MAX_COORD_CACHE = 512
COORD_CACHE = {}
OVERPASS_COOLDOWN_SECONDS = 90
OVERPASS_FAIL_UNTIL = {}
STATE_NORMALIZATION_RULES = [
    (r"\bandhra\s*pradesh\b", "andhra pradesh"),
    (r"\barunachal\s*pradesh\b", "arunachal pradesh"),
    (r"\bhimachal\s*pradesh\b", "himachal pradesh"),
    (r"\bmadhya\s*pradesh\b", "madhya pradesh"),
    (r"\buttar\s*pradesh\b", "uttar pradesh"),
    (r"\btamil\s*nadu\b", "tamil nadu"),
    (r"\bwest\s*bengal\b", "west bengal"),
    (r"\bjharkhand\b", "jharkhand"),
    (r"\bchhattisgarh\b", "chhattisgarh"),
    (r"\bnew\s*delhi\b", "delhi"),
    (r"\bjammu\s*(?:and|&)?\s*kashmir\b", "jammu and kashmir"),
    (r"\bandaman\s*(?:and|&)?\s*nicobar\b", "andaman and nicobar"),
    (r"\bdadra\s*(?:and|&)?\s*nagar\s*haveli\b", "dadra and nagar haveli"),
    (r"\bdaman\s*(?:and|&)?\s*diu\b", "daman and diu"),
]


# -------------------------
# Convert Location -> Lat/Lon
# -------------------------
def _parse_coordinates(location_name):
    if not location_name:
        return None, None

    match = COORDS_REGEX.match(str(location_name))
    if not match:
        return None, None

    try:
        lat = float(match.group(1))
        lon = float(match.group(2))
    except ValueError:
        return None, None

    if -90 <= lat <= 90 and -180 <= lon <= 180:
        return lat, lon
    return None, None


def _normalize_location_text(location_name):
    text = str(location_name or "").strip()
    text = text.replace("\t", " ")
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"\s*,\s*", ", ", text)
    for pattern, normalized in STATE_NORMALIZATION_RULES:
        text = re.sub(pattern, normalized, text, flags=re.IGNORECASE)
    return text.strip(" ,")


def _build_query_variants(location_name):
    variants = []
    base = _normalize_location_text(location_name)
    if not base:
        return variants

    variants.append(base)
    token_parts = base.split()
    skip_trailing = {"district", "city", "town", "state"}
    if (
        "," not in base
        and len(token_parts) == 2
        and token_parts[-1].lower() not in skip_trailing
    ):
        variants.append(f"{' '.join(token_parts[:-1])}, {token_parts[-1]}")
    if "," not in base and len(token_parts) >= 3:
        variants.append(f"{' '.join(token_parts[:-2])}, {' '.join(token_parts[-2:])}")

    base_lower = base.lower()
    if "district" in base_lower:
        no_district = re.sub(r"\bdistrict\b", "", base, flags=re.IGNORECASE)
        no_district = _normalize_location_text(no_district)
        if no_district:
            variants.append(no_district)

    with_country = []
    for query in variants:
        if "india" not in query.lower():
            with_country.append(f"{query}, India")
    variants.extend(with_country)

    unique_variants = []
    seen = set()
    for query in variants:
        key = query.lower()
        if key in seen:
            continue
        seen.add(key)
        unique_variants.append(query)
    return unique_variants


def _nominatim_search(query, india_only=False):
    params = {
        "q": query,
        "format": "jsonv2",
        "addressdetails": 1,
        "limit": 8,
    }
    if india_only:
        params["countrycodes"] = "in"

    headers = {
        "User-Agent": "women-safety-index",
        "Accept-Language": "en",
    }
    response = requests.get(
        NOMINATIM_URL,
        params=params,
        headers=headers,
        timeout=NOMINATIM_TIMEOUT,
    )
    response.raise_for_status()
    return response.json()


def _score_candidate(row, original_query):
    score = 0.0

    try:
        score += float(row.get("importance", 0.0)) * 10.0
    except (TypeError, ValueError):
        pass

    place_type = (row.get("type") or "").lower()
    place_class = (row.get("class") or "").lower()
    type_weight = {
        "city": 4.0,
        "town": 3.5,
        "village": 3.0,
        "suburb": 2.5,
        "county": 3.0,
        "district": 3.5,
        "state_district": 3.5,
        "administrative": 2.5,
    }
    score += type_weight.get(place_type, 0.0)

    if place_class == "boundary" and place_type == "administrative":
        score += 2.0

    country_code = (row.get("address", {}).get("country_code") or "").lower()
    if country_code == "in":
        score += 2.5

    display_name = (row.get("display_name") or "").lower()
    terms = [
        t for t in re.split(r"[\s,.-]+", original_query.lower())
        if len(t) > 2 and t not in {"india", "district"}
    ]
    for term in terms:
        if term in display_name:
            score += 1.0

    return score


def _pick_best_candidate(rows, original_query):
    best = None
    best_score = float("-inf")

    for row in rows:
        try:
            lat = float(row.get("lat"))
            lon = float(row.get("lon"))
        except (TypeError, ValueError):
            continue

        if not (-90 <= lat <= 90 and -180 <= lon <= 180):
            continue

        score = _score_candidate(row, original_query)
        if score > best_score:
            best_score = score
            best = (lat, lon)

    return best


def _cache_coordinate(cache_key, lat_lon):
    if cache_key in COORD_CACHE:
        COORD_CACHE[cache_key] = lat_lon
        return
    if len(COORD_CACHE) >= MAX_COORD_CACHE:
        COORD_CACHE.clear()
    COORD_CACHE[cache_key] = lat_lon


def get_coordinates(location_name):
    cache_key = _normalize_location_text(location_name).lower()
    cached = COORD_CACHE.get(cache_key)
    if cached is not None:
        return cached

    parsed_lat, parsed_lon = _parse_coordinates(location_name)
    if parsed_lat is not None and parsed_lon is not None:
        _cache_coordinate(cache_key, (parsed_lat, parsed_lon))
        return parsed_lat, parsed_lon

    query_variants = _build_query_variants(location_name)
    if not query_variants:
        return None, None

    query_errors = []
    for india_only in (True, False):
        for query in query_variants:
            try:
                rows = _nominatim_search(query, india_only=india_only)
            except requests.exceptions.RequestException as e:
                query_errors.append((query, india_only, str(e)))
                continue

            if not rows:
                continue

            best = _pick_best_candidate(rows, query)
            if best is not None:
                _cache_coordinate(cache_key, best)
                return best

    if query_errors:
        failed = query_errors[0]
        print(
            "Geocoding error:",
            f"query='{failed[0]}' india_only={failed[1]}",
            failed[2],
        )

    return None, None


# -------------------------
# Distance Function
# -------------------------
def haversine(lat1, lon1, lat2, lon2):
    r = 6371000
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = (
        math.sin(dlat / 2) ** 2
        + math.cos(math.radians(lat1))
        * math.cos(math.radians(lat2))
        * math.sin(dlon / 2) ** 2
    )
    return 2 * r * math.atan2(math.sqrt(a), math.sqrt(1 - a))


def _run_overpass_query(query, label, max_endpoints=None):
    data = None
    last_error = None
    endpoints = OVERPASS_URLS if max_endpoints is None else OVERPASS_URLS[:max_endpoints]
    now_ts = time.time()
    active_endpoints = [e for e in endpoints if OVERPASS_FAIL_UNTIL.get(e, 0) <= now_ts]
    if not active_endpoints:
        active_endpoints = endpoints

    for attempt in range(OVERPASS_RETRIES):
        for endpoint in active_endpoints:
            try:
                response = requests.post(endpoint, data=query, timeout=OVERPASS_TIMEOUT)
                response.raise_for_status()
                data = response.json()
                OVERPASS_FAIL_UNTIL[endpoint] = 0
                return data
            except requests.exceptions.RequestException as e:
                last_error = e
                OVERPASS_FAIL_UNTIL[endpoint] = time.time() + OVERPASS_COOLDOWN_SECONDS
        if attempt < OVERPASS_RETRIES - 1:
            time.sleep(0.4 + attempt * 0.3)

    if data is None:
        print(f"Overpass API error for {label}: {last_error}")
    return data


def _element_point(el):
    lat = el.get("lat")
    lon = el.get("lon")
    if lat is not None and lon is not None:
        return lat, lon

    center = el.get("center", {})
    lat = center.get("lat")
    lon = center.get("lon")
    if lat is not None and lon is not None:
        return lat, lon

    return None, None


def _nearest_from_data(data, origin_lat, origin_lon):
    nearest_item = None
    for el in data.get("elements", []):
        lat2, lon2 = _element_point(el)
        if lat2 is None or lon2 is None:
            continue

        dist_km = round(haversine(origin_lat, origin_lon, lat2, lon2) / 1000, 2)
        tags = el.get("tags", {})
        candidate = {
            "name": tags.get("name") or "Unknown",
            "distance_km": dist_km,
            "lat": round(float(lat2), 6),
            "lon": round(float(lon2), 6),
        }

        if nearest_item is None or candidate["distance_km"] < nearest_item["distance_km"]:
            nearest_item = candidate

    return nearest_item


def _fetch_nearest_for_selectors(lat, lon, label, radii, selectors):
    for radius in radii:
        for selector in selectors:
            query = f"""
            [out:json][timeout:30];
            {selector}(around:{radius},{lat},{lon});
            out center;
            """
            data = _run_overpass_query(query, f"{label}-{radius}")
            if data is None:
                return None

            nearest_item = _nearest_from_data(data, lat, lon)
            if nearest_item is not None:
                return nearest_item
    return None


def _nominatim_nearest(lat, lon, query_text, radius_km):
    lat_delta = radius_km / 111.0
    lon_delta = radius_km / max(1e-6, 111.0 * math.cos(math.radians(lat)))
    left = lon - lon_delta
    right = lon + lon_delta
    top = lat + lat_delta
    bottom = lat - lat_delta

    params = {
        "q": query_text,
        "format": "json",
        "limit": 10,
        "viewbox": f"{left},{top},{right},{bottom}",
        "bounded": 1,
    }
    headers = {"User-Agent": "women-safety-index"}

    try:
        response = requests.get(
            NOMINATIM_URL,
            params=params,
            headers=headers,
            timeout=NOMINATIM_TIMEOUT,
        )
        response.raise_for_status()
        results = response.json()
    except requests.exceptions.RequestException:
        return None

    nearest_item = None
    for row in results:
        try:
            lat2 = float(row.get("lat"))
            lon2 = float(row.get("lon"))
        except (TypeError, ValueError):
            continue

        dist_km = round(haversine(lat, lon, lat2, lon2) / 1000, 2)
        item = {
            "name": row.get("display_name", "Unknown").split(",")[0] or "Unknown",
            "distance_km": dist_km,
            "lat": round(lat2, 6),
            "lon": round(lon2, 6),
        }
        if nearest_item is None or item["distance_km"] < nearest_item["distance_km"]:
            nearest_item = item

    return nearest_item


def get_nearest_facilities(lat, lon, facility_radius=5000, bus_radius=3000):
    facility_radius = int(facility_radius)
    bus_radius = int(bus_radius)

    facility_radii = [
        min(1200, facility_radius),
        min(3000, facility_radius),
        min(5000, max(5000, facility_radius)),
        7000,
    ]
    bus_radii = [
        min(800, bus_radius),
        min(2000, bus_radius),
        min(4000, max(4000, bus_radius)),
    ]
    facility_radii = sorted({r for r in facility_radii if r > 0})
    bus_radii = sorted({r for r in bus_radii if r > 0})

    def fetch_police():
        return _fetch_nearest_for_selectors(
            lat,
            lon,
            "police",
            facility_radii,
            ['nwr["amenity"="police"]'],
        )

    def fetch_hospital():
        return _fetch_nearest_for_selectors(
            lat,
            lon,
            "hospital",
            facility_radii,
            ['nwr["amenity"="hospital"]'],
        )

    def fetch_bus_stop():
        return _fetch_nearest_for_selectors(
            lat,
            lon,
            "bus_stop",
            bus_radii,
            ['nwr["highway"="bus_stop"]'],
        )

    def fetch_hotel():
        return _fetch_nearest_for_selectors(
            lat,
            lon,
            "hotel",
            facility_radii,
            [
                'nwr["tourism"="hotel"]',
                'nwr["amenity"="hotel"]',
            ],
        )

    with ThreadPoolExecutor(max_workers=4) as executor:
        f_police = executor.submit(fetch_police)
        f_hospital = executor.submit(fetch_hospital)
        f_bus = executor.submit(fetch_bus_stop)
        f_hotel = executor.submit(fetch_hotel)

        nearest = {
            "police_station": f_police.result(),
            "hospital": f_hospital.result(),
            "bus_stop": f_bus.result(),
            "hotel": f_hotel.result(),
        }

    if nearest["police_station"] is None:
        nearest["police_station"] = _nominatim_nearest(lat, lon, "police station", 6)
    if nearest["hospital"] is None:
        nearest["hospital"] = _nominatim_nearest(lat, lon, "hospital", 6)
    if nearest["bus_stop"] is None:
        nearest["bus_stop"] = _nominatim_nearest(lat, lon, "bus stop", 4)
    if nearest["hotel"] is None:
        nearest["hotel"] = _nominatim_nearest(lat, lon, "hotel", 6)

    return nearest


# -------------------------
# Nearest Amenity (Police / Hospital)
# -------------------------
def get_nearest_amenity(lat, lon, amenity, radius=3000):
    query = f"""
    [out:json][timeout:60];
    node["amenity"="{amenity}"](around:{radius},{lat},{lon});
    out;
    """

    data = _run_overpass_query(query, amenity)
    if data is None:
        return None

    min_distance_m = float("inf")

    for el in data.get("elements", []):
        lat2 = el.get("lat")
        lon2 = el.get("lon")
        if lat2 is not None and lon2 is not None:
            dist_m = haversine(lat, lon, lat2, lon2)
            min_distance_m = min(min_distance_m, dist_m)

    if min_distance_m == float("inf"):
        return None

    return round(min_distance_m / 1000, 2)


# -------------------------
# General Safety Features
# -------------------------
def get_safety_features(lat, lon, travel_hour, is_night, is_weekend):
    radius_general = 900

    query = f"""
    [out:json][timeout:25];
    (
      nwr(around:{radius_general},{lat},{lon})["shop"];
      nwr(around:{radius_general},{lat},{lon})["highway"="street_lamp"];
      nwr(around:{radius_general},{lat},{lon})["highway"="bus_stop"];
      nwr(around:{radius_general},{lat},{lon})["amenity"~"hospital|police|fire_station|parking"];
    );
    out center;
    """

    data = _run_overpass_query(query, "safety features")
    if data is None:
        return None

    street_lighting = 0
    nearby_shops = 0
    bus_stop_count = 0
    parking_slots = 0
    hospital_distance_m = float("inf")
    poi_count = 0
    emergency_count = 0
    road_type_score = 0.6

    for el in data.get("elements", []):
        tags = el.get("tags", {})
        lat2 = el.get("lat") or el.get("center", {}).get("lat")
        lon2 = el.get("lon") or el.get("center", {}).get("lon")

        if lat2 is None or lon2 is None:
            continue

        dist_m = haversine(lat, lon, lat2, lon2)

        if tags.get("highway") == "street_lamp":
            street_lighting += 1

        if tags.get("shop"):
            nearby_shops += 1

        if tags.get("highway") == "bus_stop":
            bus_stop_count += 1

        amenity = tags.get("amenity")
        if amenity:
            poi_count += 1
            if amenity in {"police", "hospital", "fire_station"}:
                emergency_count += 1
            if amenity == "parking":
                parking_slots += 1
            if amenity == "hospital":
                hospital_distance_m = min(hospital_distance_m, dist_m)

    hospital_distance_km = (
        round(hospital_distance_m / 1000, 2)
        if hospital_distance_m != float("inf")
        else None
    )

    return {
        "travel_hour": travel_hour,
        "is_night": is_night,
        "is_weekend": is_weekend,
        "street_lighting": street_lighting,
        "road_type_score": road_type_score,
        "nearby_shops": nearby_shops,
        "bus_stop_count": bus_stop_count,
        "parking_slots": parking_slots,
        "poi_count": poi_count,
        "emergency_count": emergency_count,
        "commercial_density": round(min(nearby_shops / 20.0, 1.0), 3),
        "hospital_distance": hospital_distance_km,
    }


# -------------------------
# MAIN PROGRAM
# -------------------------
if __name__ == "__main__":
    print("Women Travel Safety Index System")
    print("----------------------------------")

    location = input("Enter Location Name: ")
    date_input = input("Enter Date (DD-MM-YYYY): ")
    time_input = input("Enter Time (Example: 9 AM or 6 PM): ")

    try:
        user_date = datetime.strptime(date_input, "%d-%m-%Y")
        today = datetime.today().date()

        if user_date.date() < today:
            print("Invalid date. Date cannot be in the past.")
            raise SystemExit

    except ValueError:
        print("Invalid date format. Use DD-MM-YYYY")
        raise SystemExit

    try:
        parsed_time = datetime.strptime(time_input.strip().upper(), "%I %p")
        travel_hour = parsed_time.hour
    except ValueError:
        print("Invalid time format. Use like '9 AM' or '6 PM'")
        raise SystemExit

    is_night = 1 if travel_hour >= 19 or travel_hour <= 5 else 0
    is_weekend = 1 if user_date.weekday() >= 5 else 0

    print("\nGetting coordinates...")
    lat, lon = get_coordinates(location)

    if lat is None:
        print("Location not found.")
        raise SystemExit

    print(f"Location found: {lat}, {lon}\n")

    print("Fetching nearest police station...")
    police_distance = get_nearest_amenity(lat, lon, "police", radius=5000)

    time.sleep(2)

    print("Fetching safety features...")
    features = get_safety_features(lat, lon, travel_hour, is_night, is_weekend)

    if features:
        features["police_distance"] = police_distance

        print("\n------ SAFETY FEATURES ------")
        for key, value in features.items():
            print(f"{key}: {value}")
    else:
        print("Could not fetch safety data.")
