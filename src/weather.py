"""
Fetch real current weather from Open-Meteo (free, no API key required).
Replaces manual weather sliders in the app.
"""

import requests

_GEOCODE_URL = "https://geocoding-api.open-meteo.com/v1/search"
_WEATHER_URL = "https://api.open-meteo.com/v1/forecast"


def geocode(location: str):
    """
    Convert a city/location string to (lat, lon, display_name).
    Returns None if not found or request fails.
    """
    try:
        resp = requests.get(_GEOCODE_URL,
                            params={"name": location, "count": 1},
                            timeout=5)
        results = resp.json().get("results", [])
        if not results:
            return None
        r = results[0]
        display = f"{r.get('name', location)}, {r.get('country', '')}"
        return r["latitude"], r["longitude"], display
    except Exception:
        return None


def fetch_weather(lat: float, lon: float):
    """
    Fetch current temperature, humidity, and days since last rain.
    Returns dict or None on failure.
    """
    try:
        resp = requests.get(_WEATHER_URL, params={
            "latitude":       lat,
            "longitude":      lon,
            "current":        "temperature_2m,relative_humidity_2m",
            "daily":          "precipitation_sum",
            "past_days":      14,
            "forecast_days":  1,
            "timezone":       "auto",
        }, timeout=8)
        data = resp.json()

        current  = data["current"]
        temp     = float(current["temperature_2m"])
        humidity = int(current["relative_humidity_2m"])

        # Count consecutive dry days from most recent backwards
        daily_precip = data["daily"]["precipitation_sum"]
        days_since_rain = 0
        for precip in reversed(daily_precip):
            if precip is not None and precip > 0.1:
                break
            days_since_rain += 1
        days_since_rain = min(days_since_rain, 30)

        return {
            "temperature_c":   round(temp, 1),
            "humidity_pct":    humidity,
            "days_since_rain": days_since_rain,
        }
    except Exception:
        return None
