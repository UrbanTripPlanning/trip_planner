import geopandas as gpd
import pandas as pd
import matplotlib.pyplot as plt
import logging
from datetime import datetime
import os

# Configure logging to save to a file
os.makedirs("output", exist_ok=True)  # Create logging directory if it doesn't exist
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(f"output/export_file_analysis_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.log"),
        logging.StreamHandler()
    ]
)

# Define file paths (update these paths as per your file locations)
shp_file_path = "./export_5t/RR_SVR_intersezione_area_studio_Polito_ridotto.shp"
dbf_file_path = "./export_5t/RR_SVR_intersezione_area_studio_Polito_ridotto.dbf"
shx_file_path = "./export_5t/RR_SVR_intersezione_area_studio_Polito_ridotto.shx"
cpg_file_path = "./export_5t/RR_SVR_intersezione_area_studio_Polito_ridotto.cpg"
prj_file_path = "./export_5t/RR_SVR_intersezione_area_studio_Polito_ridotto.prj"
qmd_file_path = "./export_5t/RR_SVR_intersezione_area_studio_Polito_ridotto.qmd"
csv_network_path = "./export_5t/RR_SVR_intersezione_area_studio_Polito.csv"
csv_velocity_path = "./export_5t/export_polito_velocita_medie_feriali_nov2024.csv"
# image_path = "./export_5t/area_studio_Polito.png"


# 1. Extract and interpret the shapefile
logging.info("\n1) --- Processing Shapefile ---")
shapefile_data = gpd.read_file(shp_file_path)
logging.info("File Type: Main shapefile.")
logging.info("Represents a road network as a series of LINESTRING geometries.")
logging.info("Each segment connects two nodes (tail and head) and represents a specific road segment.")
logging.info(f"Number of geometries: {len(shapefile_data)}")
logging.info(f"Columns in shapefile: {shapefile_data.columns}")
logging.info("Example of a segment's geometry:")
logging.info(shapefile_data.geometry.iloc[0])
logging.info("Utility: Spatial representation of the road network for geospatial analysis and GIS visualization.")

# Convert shapefile geometries to GeoDataFrame
geometries = shapefile_data.geometry
shapefile_data.plot(figsize=(10, 8), edgecolor="black")
plt.title("Visualization of Road Network")
plt.savefig("road_network_visualization.png")
logging.info("Plot saved as 'road_network_visualization.png'.")

# 2. Extract and interpret the .dbf file
logging.info("\n2) --- Processing .dbf File ---")
dbf_data = gpd.read_file(dbf_file_path)
logging.info("File Type: Tabular file accompanying the .shp.")
logging.info("Provides descriptive alphanumeric data for each road segment.")
logging.info("Main columns include 'idno', 'tail', 'head', 'leng', and 'lane'.")
logging.info(f"Number of rows in .dbf: {len(dbf_data)}")
logging.info(f"Columns in .dbf: {dbf_data.columns}")
logging.info("Data Statistics:")
logging.info(f"Average length: {dbf_data['leng'].mean():.2f} units.")
logging.info(f"Average number of lanes: {dbf_data['lane'].mean():.2f}.")
logging.info("Example of a row:")
logging.info(dbf_data.iloc[0])
logging.info("Utility: Basis for road capacity calculations and tabular analysis.")

# 3. Process the .shx file (Spatial Index)
logging.info("\n3) --- Processing .shx File ---")
try:
    with open(shx_file_path, 'rb') as shx_file:
        logging.info("File Type: Spatial index file.")
        logging.info("Optimized structure enabling quick access to geometries in the .shp file.")
        logging.info(f".shx file '{shx_file_path}' successfully loaded.")
        logging.info("Utility: Improves efficiency during GIS analysis and visualization.")
except Exception as e:
    logging.error(f"Error loading .shx file: {e}")

# 4. Process the .cpg file (Encoding Information)
logging.info("\n4) --- Processing .cpg File ---")
try:
    with open(cpg_file_path, 'r') as cpg_file:
        encoding = cpg_file.read().strip()
        logging.info("File Type: Encoding information file.")
        logging.info(f"Specifies character encoding for text data: {encoding}")
except Exception as e:
    logging.error(f"Error loading .cpg file: {e}")

# 5. Process the .prj file (Projection Information)
logging.info("\n5) --- Processing .prj File ---")
try:
    with open(prj_file_path, 'r') as prj_file:
        projection = prj_file.read().strip()
        logging.info("File Type: Geographic reference system file.")
        logging.info("Specifies the WGS 1984 coordinate system.")
        logging.info("Projection information:")
        logging.info(projection)
except Exception as e:
    logging.error(f"Error loading .prj file: {e}")

# 6. Process the QGIS project file (.qmd)
logging.info("\n6) --- Processing .qmd File ---")
try:
    with open(qmd_file_path, 'r') as qmd_file:
        qmd_content = qmd_file.read()
        logging.info("File Type: GIS project file.")
        logging.info("May contain style configurations and display settings.")
        logging.info("Loaded successfully for QGIS project customization.")
except Exception as e:
    logging.error(f"Error loading .qmd file: {e}")

# 7. Process the CSV file for network data
logging.info("\n7) --- Processing Network CSV ---")
network_data = pd.read_csv(csv_network_path)
logging.info("File Type: CSV file with road network information.")
logging.info("Replicates data from the .dbf file.")
logging.info(f"Number of rows in network CSV: {len(network_data)}")
logging.info(f"Columns in network CSV: {network_data.columns}")
logging.info("Utility: Alternative tabular representation of network data.")

# Check if CSV and .dbf contain matching data
if set(dbf_data.columns).issubset(set(network_data.columns)):
    logging.info("Network CSV contains data consistent with the .dbf file.")
else:
    logging.warning("Network CSV has additional or missing columns compared to the .dbf file.")

# 8. Process the CSV file for velocity data
logging.info("\n8) --- Processing Velocity CSV ---")
velocity_data = pd.read_csv(csv_velocity_path)
logging.info("File Type: CSV file with traffic data.")
logging.info("Contains average velocities for each road segment.")
logging.info(f"Number of rows in velocity CSV: {len(velocity_data)}")
logging.info(f"Columns in velocity CSV: {velocity_data.columns}")
logging.info("Utility: Road performance analysis and identification of congested segments.")

# Merge velocity data with network data on 'idno'
logging.info("--- Merging Network and Velocity Data ---")
if 'idno' in network_data.columns and 'idno' in velocity_data.columns:
    merged_data = pd.merge(network_data, velocity_data, on="idno", how="inner")
    logging.info(f"Number of rows after merge: {len(merged_data)}")
    if len(merged_data) < min(len(network_data), len(velocity_data)):
        logging.warning("The merge resulted in fewer rows than expected. Some IDs might be unmatched.")
        unmatched_ids_network = set(network_data['idno']) - set(merged_data['idno'])
        unmatched_ids_velocity = set(velocity_data['idno']) - set(merged_data['idno'])
        logging.warning(f"Unmatched IDs in network data: {unmatched_ids_network}")
        logging.warning(f"Unmatched IDs in velocity data: {unmatched_ids_velocity}")
    logging.info("Merged Data Preview:")
    logging.info(merged_data.head())
else:
    logging.warning("Cannot merge: 'idno' column missing in one or both files.")

# 9. Display the image for area visualization
# logging.info("--- Displaying Area Image ---")
# image = Image.open(image_path)
# image.show()
# logging.info("The area of study map has been displayed. Use it as a visual reference.")

# Summary of data analysis
logging.info("\n--- Summary ---")
logging.info("1. Shapefile: Contains road network geometries and attributes.")
logging.info("2. .dbf File: Provides detailed attributes like segment length, lanes, and node connectivity.")
logging.info("3. .shx File: Spatial index used internally by GIS software.")
logging.info("4. .cpg File: Specifies character encoding for text data.")
logging.info("5. .prj File: Defines the projection system for spatial data.")
logging.info("6. QGIS Project (.qmd): Contains metadata for project configurations.")
logging.info("7. Network CSV: Alternative tabular representation of network data.")
logging.info("8. Velocity CSV: Contains traffic data, specifically average velocities on road segments.")
# logging.info("9. Area Image: Visual representation of the geographic study area.")
