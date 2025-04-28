import os
import csv
import requests
from enum import Enum
from datetime import datetime, timedelta
from utils import read_env

read_env()


class WeatherQuery(Enum):
    CURRENT = (
        "current",
        os.getenv('CURRENT_URL'),
        {
            "apiKey": os.getenv("API_KEY"),
            "format": "json",
            "stationId": "ITURIN3276",  # Torino center (Piazza Castello) station id
            "units": "m"
        }
    )
    FORECAST = (
        "forecast",
        os.getenv('FORECAST_URL'),
        {
            "apiKey": os.getenv("API_KEY"),
            "format": "json",
            "postalKey": "10121:IT",  # Hardcoded for Torino center (Piazza Castello)
            "units": "e",
            "language": "en-US"
        }
    )

    def __init__(self, query_name: str, url: str, params: dict):
        self.query_name = query_name
        self.url = url
        self.params = params

    def __str__(self):
        return self.query_name


class WeatherClient:
    """
    WeatherClient uses the WeatherQuery enum to fetch weather data and generate a CSV file.
    The CSV "weather.csv" contains hourly weather data (from the current hour onward)
    for Torino center (Piazza Castello), with columns for datetime, a binary rain flag,
    and a generic weather condition.
    """

    def __init__(self):
        self.current_query = WeatherQuery.CURRENT
        self.forecast_query = WeatherQuery.FORECAST

    def fetch_data(self, query: WeatherQuery):
        response = requests.get(query.url, params=query.params)
        if response.status_code == 200:
            return response.json()
        else:
            print(f"Error fetching {query} data: {response.status_code} - {response.text}")
            return None

    def get_current_condition(self):
        """
        Analyzes the current observation data to decide if it is raining.
        It uses metrics like accumulated precipitation, humidity, temperature–dew point closeness,
        and solar radiation to determine the condition.
        Returns a tuple: (rain_flag, condition) where condition is "rain" if raining, else "clear".
        """
        current_data = self.fetch_data(self.current_query)
        if not current_data:
            return 0, "clear"
        try:
            observations = current_data.get("observations", [])
            if not observations:
                print("No current observations available.")
                return 0, "clear"
            obs = observations[0]
            metric = obs.get("metric", {})
            precip_total = float(metric.get("precipTotal", 0))
            temp = float(metric.get("temp", 0))
            dewpt = float(metric.get("dewpt", 0))
            humidity = float(obs.get("humidity", 0))
            solar_rad = float(obs.get("solarRadiation", 0))

            if precip_total > 0.1 or (humidity >= 90 and abs(temp - dewpt) < 1 and solar_rad < 10):
                return 1, "rain"
            else:
                return 0, "clear"
        except Exception as e:
            print(f"Error parsing current observation: {e}")
            return 0, "clear"

    def determine_condition(self, narrative: str, rain_flag: int):
        """
        Determines a generic weather condition ("rain", "cloud", or "clear") based on the forecast narrative.
        If rain_flag is 1, returns "rain". Otherwise, it looks for keywords in the narrative.
        """
        if rain_flag == 1:
            return "rain"
        if narrative:
            narrative_lower = narrative.lower()
            if "rain" in narrative_lower:
                return "rain"
            elif "cloud" in narrative_lower:
                return "cloud"
            elif "sun" in narrative_lower:
                return "clear"
        return "clear"

    def create_hourly_csv(self, threshold: int = 50):
        """
        Creates a CSV file with columns: 'datetime', 'rain', and 'weather_condition'.
        For each hour (from 00:00 to 23:00) for each day in the forecast data,
        only rows from the current time onward are included.
          - For the current day, the current observation data is used.
          - For future days, "day" is defined as 07:00 <= hour < 19:00; otherwise, it's "night".
            The corresponding forecast index is:
              • 2 * day_offset for day,
              • 2 * day_offset + 1 for night.
        The datetime is formatted as '%Y-%m-%d %H:%M:%S'.
        """
        forecast_data = self.fetch_data(self.forecast_query)
        if not forecast_data:
            print("Forecast data is not available.")
            return

        current_rain, current_condition = self.get_current_condition()

        valid_dates = forecast_data.get("validTimeLocal", [])
        daypart_info = forecast_data.get("daypart", [{}])[0]
        precip_list = daypart_info.get("precipChance", [])
        narrative_list = daypart_info.get("narrative", [])
        num_days = len(valid_dates)
        now = datetime.now()

        csv_filename = os.path.join(os.getenv('DATA_PATH'), 'weather.csv')

        with open(csv_filename, "w", newline="", encoding="utf-8") as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(["datetime", "rain", "weather_condition"])
            for i in range(num_days):
                date_str = valid_dates[i][:10]
                day_date = datetime.strptime(date_str, "%Y-%m-%d")
                for hour in range(24):
                    current_dt = day_date + timedelta(hours=hour)
                    # For the current day, include hours starting from the current hour (include the current hour)
                    if current_dt.date() == now.date() and current_dt.hour < now.hour:
                        continue
                    dt_str = current_dt.strftime('%Y-%m-%d %H:%M:%S')
                    if i == 0:
                        rain = current_rain
                        condition = current_condition
                    else:
                        # Define "day" period as 07:00 to 19:00; otherwise "night"
                        if 7 <= current_dt.hour < 19:
                            index = 2 * i
                        else:
                            index = 2 * i + 1
                        if index >= len(precip_list) or precip_list[index] is None:
                            rain = 0
                            condition = "clear"
                        else:
                            precip = precip_list[index]
                            rain = 1 if precip >= threshold else 0
                            narrative = narrative_list[index] if index < len(narrative_list) and narrative_list[
                                index] else ""
                            condition = self.determine_condition(narrative, rain)
                    writer.writerow([dt_str, rain, condition])


if __name__ == "__main__":
    # Update weather file - todo: this should be done by the server
    w = WeatherClient()
    w.create_hourly_csv()
