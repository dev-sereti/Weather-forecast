import openmeteo_requests
import pandas as pd
import requests_cache
from retry_requests import retry
import os

# Location names matched by index to latitude/longitude order
location_names = ["VFfarms", "KC-Kagano", "KC-Kigembe"]

# Setup the Open-Meteo API client with cache and retry on error
cache_session = requests_cache.CachedSession('.cache', expire_after=3600)
retry_session = retry(cache_session, retries=5, backoff_factor=0.2)
openmeteo = openmeteo_requests.Client(session=retry_session)

url = "https://api.open-meteo.com/v1/forecast"
params = {
    "latitude": [0.5603, 2.3328, 2.7334],
    "longitude": [34.0623, 29.0934, 23.11111],
    "hourly": ["wind_speed_180m", "wind_direction_180m", "temperature_180m", "rain", "relative_humidity_2m", "wind_gusts_10m"],
    "timezone": "auto",
    "past_days": 1,
}
responses = openmeteo.weather_api(url, params=params)

# Output Excel path
output_dir = r"C:\Users\VF4492\OneDrive - Victory Farms Ltd\Technology and Innovation - Power BI\Weather"
os.makedirs(output_dir, exist_ok=True)
output_file = os.path.join(output_dir, "weather_data.xlsx")

all_dataframes = {}

for i, response in enumerate(responses):
    location_name = location_names[i]

    print(f"\nLocation: {location_name}")
    print(f"Coordinates: {response.Latitude()}°N {response.Longitude()}°E")
    print(f"Elevation: {response.Elevation()} m asl")
    print(f"Timezone: {response.Timezone()}{response.TimezoneAbbreviation()}")
    print(f"Timezone difference to GMT+0: {response.UtcOffsetSeconds()}s")

    hourly = response.Hourly()
    hourly_wind_speed_180m = hourly.Variables(0).ValuesAsNumpy()
    hourly_wind_direction_180m = hourly.Variables(1).ValuesAsNumpy()
    hourly_temperature_180m = hourly.Variables(2).ValuesAsNumpy()
    hourly_rain = hourly.Variables(3).ValuesAsNumpy()
    hourly_relative_humidity_2m = hourly.Variables(4).ValuesAsNumpy()
    hourly_wind_gusts_10m = hourly.Variables(5).ValuesAsNumpy()

    hourly_data = {
        "date": pd.date_range(
            start=pd.to_datetime(hourly.Time(), unit="s", utc=True),
            end=pd.to_datetime(hourly.TimeEnd(), unit="s", utc=True),
            freq=pd.Timedelta(seconds=hourly.Interval()),
            inclusive="left"
        ).tz_convert(response.Timezone().decode())
    }

    hourly_data["location"] = location_name
    hourly_data["latitude"] = response.Latitude()
    hourly_data["longitude"] = response.Longitude()
    hourly_data["wind_speed_180m"] = hourly_wind_speed_180m
    hourly_data["wind_direction_180m"] = hourly_wind_direction_180m
    hourly_data["temperature_180m"] = hourly_temperature_180m
    hourly_data["rain"] = hourly_rain
    hourly_data["relative_humidity_2m"] = hourly_relative_humidity_2m
    hourly_data["wind_gusts_10m"] = hourly_wind_gusts_10m

    hourly_dataframe = pd.DataFrame(data=hourly_data)
    hourly_dataframe["date"] = hourly_dataframe["date"].dt.tz_localize(None)  # Remove tz for Excel

    print(f"\nHourly data for {location_name}\n", hourly_dataframe)

    all_dataframes[location_name] = hourly_dataframe

# Save to Excel — one sheet per location
with pd.ExcelWriter(output_file, engine="openpyxl") as writer:
    for location_name, df in all_dataframes.items():
        df.to_excel(writer, sheet_name=location_name, index=False)

print(f"\nData saved to: {output_file}")