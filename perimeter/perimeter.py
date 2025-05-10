import pymongo
import networkx as nx
from shapely.geometry import shape, Point
from shapely.ops import unary_union
import matplotlib.pyplot as plt
import json
import numpy as np
from scipy.spatial import cKDTree

# --- 1) Costruzione del grafo e raccolta delle geometrie ---
MONGO_URI = "mongodb://urban_trip:urbantrip@43.157.33.33:27018/"
DB = pymongo.MongoClient(MONGO_URI)["turin_go"]["road"]

G = nx.Graph()
lines = []
for doc in DB.find():
    geom = doc.get("geometry")
    if not geom:
        continue
    ls = shape(geom)
    if ls.geom_type != "LineString":
        continue
    u = tuple(ls.coords[0])
    v = tuple(ls.coords[-1])
    G.add_edge(u, v)
    lines.append(ls)

# --- 2) Merge + buffer esterno per avere un contorno unico ---
merged = unary_union(lines)
buf_dist = 1e-5
buf = merged.buffer(buf_dist)
outer_ring = buf.exterior  # LineString

# --- 3) Campionamento di punti lungo l’anello + snap al nodo più vicino ---
length = outer_ring.length
N = 1000
ts = np.linspace(0, length, N)
# Prepara il KDTree sui nodi
coords = list(G.nodes())
tree = cKDTree(coords)

snapped = []
for t in ts:
    p = outer_ring.interpolate(t)
    _, idx = tree.query((p.x, p.y))
    snapped.append(coords[idx])

# Rimuovo duplicati consecutivi, mantenendo l’ordine
perim_nodes_sorted = [snapped[0]]
for pt in snapped[1:]:
    if pt != perim_nodes_sorted[-1]:
        perim_nodes_sorted.append(pt)

# Salvo i nodi
with open("perimeter_nodes.json", "w") as f:
    json.dump(perim_nodes_sorted, f)

# --- 4) Plot finale ---
pos = {n: n for n in G.nodes()}

plt.figure(figsize=(12, 8))
nx.draw(
    G, pos,
    node_size=3, node_color="black",
    edge_color="gray", linewidths=0.5
)

# disegno la polilinea rossa passando per tutti i nodi di perimetro
cycle = perim_nodes_sorted + [perim_nodes_sorted[0]]
xs, ys = zip(*cycle)
plt.plot(xs, ys, color="red", linewidth=2, label="Exact Perimeter")

# evidenzio i nodi di perimetro
px, py = zip(*perim_nodes_sorted)
plt.scatter(px, py, s=20, c="red", marker="o")

plt.legend(loc="upper left")
plt.axis("equal")
plt.tight_layout()
plt.savefig("roads_graph_exact_perimeter.png", dpi=300)
plt.close()

print("✅ Grafico salvato in roads_graph_exact_perimeter.png")
print("✅ Nodi del perimetro salvati in perimeter_nodes.json")
