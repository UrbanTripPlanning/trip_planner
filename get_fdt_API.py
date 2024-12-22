import requests
import xml.etree.ElementTree as ET
import folium
from datetime import datetime
import time
import os

DATA_URL = "https://opendata.5t.torino.it/get_fdt"
SAVE_DIRECTORY = "./"

# Create a directory to save the maps, if it doesn't already exist
os.makedirs(SAVE_DIRECTORY, exist_ok=True)


def fetch_and_save_map():
    try:
        # Perform the HTTP request
        response = requests.get(DATA_URL)
        response.raise_for_status()

        # Parse the XML
        root = ET.fromstring(response.content)

        # Create the map
        torino_map = folium.Map(location=[45.0703, 7.6869])

        # XML namespace specified in the feed
        namespace = {'ns': 'https://simone.5t.torino.it/ns/traffic_data.xsd'}

        # Iterate through traffic data
        for fdt in root.findall('ns:FDT_data', namespace):
            lat = float(fdt.get('lat', '0'))
            lng = float(fdt.get('lng', '0'))
            road_name = fdt.get('Road_name', 'Unknown')
            direction = fdt.get('direction', 'Unknown')
            accuracy = fdt.get('accuracy', '0')

            speedflow = fdt.find('ns:speedflow', namespace)
            flow = speedflow.get('flow', '0') if speedflow is not None else '0'
            speed = speedflow.get('speed', '0') if speedflow is not None else '0'

            # Popup information for each marker
            popup_info = f"""
            <b>Road Name:</b> {road_name}<br>
            <b>Direction:</b> {direction}<br>
            <b>Accuracy:</b> {accuracy}%<br>
            <b>Flow:</b> {flow} vehicles<br>
            <b>Speed:</b> {speed} km/h
            """
            folium.Marker(
                location=[lat, lng],
                popup=folium.Popup(popup_info, max_width=300),
                icon=folium.Icon(color="blue", icon="info-sign")
            ).add_to(torino_map)

        # Save the map with a timestamp
        timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
        file_path = os.path.join(SAVE_DIRECTORY, f"./get_fdt_output_files/torino_traffic_map_{timestamp}.html")
        torino_map.save(file_path)
        print(f"Map created and saved as {file_path}")

    except requests.RequestException as e:
        print(f"Error during data download: {e}")
    except ET.ParseError as e:
        print(f"Error during XML parsing: {e}")
    except Exception as e:
        print(f"General error: {e}")

if __name__ == '__main__':
    print("Starting periodic map saving...")
    while True:
        fetch_and_save_map()
        # Wait for 5 minutes (300 seconds) before running again
        time.sleep(300)
