import os
import json
import pymongo
import networkx as nx
import numpy as np
from shapely.geometry import shape, Point
from shapely.ops import unary_union
from scipy.spatial import cKDTree
import matplotlib.pyplot as plt

# -------------------------------
# PARAMETERS
# -------------------------------
MONGO_URI     = "mongodb://urban_trip:urbantrip@43.157.33.33:27018/"
DB_NAME       = "turin_go"
COLLECTION    = "road"
BUFFER_DIST   = 1e-5       # tiny buffer to close gaps
NUM_SAMPLES   = 1000       # how many points to sample along the boundary
OUTPUT_DIR    = "output"   # directory for all outputs

# make sure output directory exists
os.makedirs(OUTPUT_DIR, exist_ok=True)

# -------------------------------
# 1) Load graph and collect LineString geometries
# -------------------------------
client = pymongo.MongoClient(MONGO_URI)
cursor = client[DB_NAME][COLLECTION].find()

G = nx.Graph()
line_geoms = []

for doc in cursor:
    geom = doc.get("geometry")
    if not geom:
        continue
    line = shape(geom)
    if line.geom_type != "LineString":
        continue
    start = tuple(line.coords[0])
    end   = tuple(line.coords[-1])
    G.add_edge(start, end)
    line_geoms.append(line)

# -------------------------------
# 2) Merge all segments and extract the outer boundary
# -------------------------------
merged = unary_union(line_geoms)
buffered = merged.buffer(BUFFER_DIST)
outer_ring = buffered.exterior  # closed LineString

# -------------------------------
# 3) Sample along the boundary, snap to nearest graph node
# -------------------------------
length = outer_ring.length
t_values = np.linspace(0, length, NUM_SAMPLES)

nodes_list = list(G.nodes())
kdtree = cKDTree(nodes_list)

snapped = []
for t in t_values:
    p = outer_ring.interpolate(t)
    _, idx = kdtree.query((p.x, p.y))
    snapped.append(nodes_list[idx])

# remove consecutive duplicates
perimeter_nodes = []
for node in snapped:
    if not perimeter_nodes or node != perimeter_nodes[-1]:
        perimeter_nodes.append(node)

# if empty, warn and exit
if not perimeter_nodes:
    raise RuntimeError("No perimeter nodes found – check graph / geometry.")

# save perimeter nodes to JSON
json_path = os.path.join(OUTPUT_DIR, "perimeter_nodes.json")
with open(json_path, "w") as f:
    json.dump(perimeter_nodes, f)
print(f"✅ Saved perimeter nodes ({len(perimeter_nodes)} points) to {json_path}")

# -------------------------------
# 4) Plot the graph and perimeter
# -------------------------------
pos = {node: node for node in G.nodes()}

plt.figure(figsize=(12, 8))

# draw base graph
nx.draw(
    G, pos,
    node_size=3,
    node_color="black",
    edge_color="gray",
    linewidths=0.5
)

# plot the perimeter outline
cycle = perimeter_nodes + [perimeter_nodes[0]]
xs, ys = zip(*cycle)
plt.plot(xs, ys, color="red", linewidth=2, label="Perimeter Outline")
plt.scatter(xs, ys, s=20, c="red", marker="o")

plt.legend(loc="upper left")
plt.axis("equal")
plt.tight_layout()

img_path = os.path.join(OUTPUT_DIR, "graph_with_perimeter.png")
plt.savefig(img_path, dpi=300)
plt.close()
print(f"✅ Saved graph + perimeter image to {img_path}")
