import math
import pymongo
import overpy
import geopandas as gpd
import networkx as nx
from shapely.geometry import shape, LineString
import matplotlib.pyplot as plt


def haversine(coord1, coord2):
    """
    Calculate the Haversine distance (in meters) between two (lon, lat) pairs.
    """
    lon1, lat1 = coord1
    lon2, lat2 = coord2
    R = 6371000  # Earth radius in meters
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    delta_phi = math.radians(lat2 - lat1)
    delta_lambda = math.radians(lon2 - lon1)
    a = math.sin(delta_phi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(delta_lambda / 2) ** 2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    return R * c


# --- Part 1: Connect to MongoDB and retrieve DB road segments ---
MONGO_URI = "mongodb://urban_trip:urbantrip@43.157.33.33:27018/"
DB_NAME = "turin_go"
COLLECTION_NAME = "road"

client = pymongo.MongoClient(MONGO_URI)
db = client[DB_NAME]
db_docs = list(db[COLLECTION_NAME].find({}))
print(f"Retrieved {len(db_docs)} documents from the DB.")

# Build the DB graph by extracting endpoints from each road segment geometry.
db_graph = nx.Graph()
geom_list = []  # to later compute the bounding box
for doc in db_docs:
    if "geometry" not in doc:
        continue
    geom = shape(doc["geometry"])
    if geom.geom_type != "LineString":
        continue
    geom_list.append(geom)
    tail = (geom.coords[0][0], geom.coords[0][1])
    head = (geom.coords[-1][0], geom.coords[-1][1])
    db_graph.add_edge(tail, head, road_id=doc.get("road_id"), length=doc.get("length"))
print(f"DB graph has {db_graph.number_of_nodes()} nodes and {db_graph.number_of_edges()} edges.")

# Compute overall bounding box from DB segments.
gdf_db = gpd.GeoDataFrame({'geometry': geom_list})
minx, miny, maxx, maxy = gdf_db.total_bounds
print("DB Bounding box:")
print(f"  West (minx): {minx}")
print(f"  South (miny): {miny}")
print(f"  East (maxx): {maxx}")
print(f"  North (maxy): {maxy}")

# --- Part 2: Query OSM for highways within the bounding box ---
api = overpy.Overpass()
query = f"""
(
  way({miny},{minx},{maxy},{maxx})["highway"];
);
out body;
>;
out skel qt;
"""
osm_result = api.query(query)
osm_ways = osm_result.ways
print(f"OSM query returned {len(osm_ways)} ways.")

# Build the OSM graph, filtering unwanted features.
valid_highways = {
    "motorway", "motorway_link",
    "trunk", "trunk_link",
    "primary", "primary_link",
    "secondary", "secondary_link",
    "tertiary", "tertiary_link",
    "unclassified", "residential",
    "living_street", "service",
    "road"
}
osm_graph = nx.Graph()
for way in osm_ways:
    hw_value = way.tags.get("highway", "").lower()
    if hw_value not in valid_highways:
        continue
    if way.tags.get("area", "").lower() == "yes":
        continue
    coords = [(float(node.lon), float(node.lat)) for node in way.nodes]
    if len(coords) < 2:
        continue
    line = LineString(coords)
    if line.is_ring:
        continue
    tail = (line.coords[0][0], line.coords[0][1])
    head = (line.coords[-1][0], line.coords[-1][1])
    osm_graph.add_edge(tail, head, osm_id=way.id, tags=way.tags)
print(f"OSM graph has {osm_graph.number_of_nodes()} nodes and {osm_graph.number_of_edges()} edges.")

# --- Part 3: Compute Intersection Graph using Endpoint Tolerance ---
tolerance_m = 10  # Accept endpoints within 10 meters
intersect_graph = nx.Graph()
for (u_db, v_db, data_db) in db_graph.edges(data=True):
    for (u_osm, v_osm, data_osm) in osm_graph.edges(data=True):
        # Compare endpoints in both possible pairings.
        d1 = haversine(u_db, u_osm)
        d2 = haversine(v_db, v_osm)
        d3 = haversine(u_db, v_osm)
        d4 = haversine(v_db, u_osm)
        if (d1 < tolerance_m and d2 < tolerance_m) or (d3 < tolerance_m and d4 < tolerance_m):
            intersect_graph.add_edge(u_db, v_db, db_attrs=data_db, osm_attrs=data_osm)
            break
print("Intersection Graph (with tolerance of", tolerance_m, "meters):")
print(f"  Nodes: {intersect_graph.number_of_nodes()}")
print(f"  Edges: {intersect_graph.number_of_edges()}")

# --- Part 4: Plot the Graphs Side by Side and Save the Image ---
pos_db = {node: node for node in db_graph.nodes()}
pos_osm = {node: node for node in osm_graph.nodes()}
pos_int = {node: node for node in intersect_graph.nodes()}

fig, axs = plt.subplots(1, 3, figsize=(18, 6))
nx.draw_networkx_nodes(db_graph, pos_db, ax=axs[0], node_color='red', node_size=10)
nx.draw_networkx_edges(db_graph, pos_db, ax=axs[0], edge_color='red')
axs[0].set_title("DB Graph")
axs[0].set_xlabel("Longitude")
axs[0].set_ylabel("Latitude")

nx.draw_networkx_nodes(osm_graph, pos_osm, ax=axs[1], node_color='blue', node_size=10)
nx.draw_networkx_edges(osm_graph, pos_osm, ax=axs[1], edge_color='blue')
axs[1].set_title("OSM Graph")
axs[1].set_xlabel("Longitude")
axs[1].set_ylabel("Latitude")

nx.draw_networkx_nodes(intersect_graph, pos_int, ax=axs[2], node_color='green', node_size=10)
nx.draw_networkx_edges(intersect_graph, pos_int, ax=axs[2], edge_color='green')
axs[2].set_title("Intersection Graph")
axs[2].set_xlabel("Longitude")
axs[2].set_ylabel("Latitude")

plt.tight_layout()
plt.savefig("output/graphs_comparison.png")
plt.close()
print("Graphs saved to output/graphs_comparison.png")

# --- Part 5: Save Intersection Graph as a Shapefile ---
# Each edge in the intersection graph will be stored as a LineString with associated attributes.

records = []
for u, v, attrs in intersect_graph.edges(data=True):
    line = LineString([u, v])
    record = {
        "geometry": line,
        "db_road_id": attrs.get("db_attrs", {}).get("road_id", None),
        "db_length": attrs.get("db_attrs", {}).get("length", None),
        "osm_id": attrs.get("osm_attrs", {}).get("osm_id", None),
        # Store tags as JSON strings (you may adjust this as needed)
        "db_attrs": str(attrs.get("db_attrs", {})),
        "osm_attrs": str(attrs.get("osm_attrs", {}))
    }
    records.append(record)

# Create a GeoDataFrame from the records
gdf_intersect = gpd.GeoDataFrame(records, geometry="geometry", crs="EPSG:4326")
shapefile_path = "output/intersection_edges.gpkg"
gdf_intersect.to_file(shapefile_path, driver="GPKG")
print(f"Intersection edges saved to GeoPackage: {shapefile_path}")
