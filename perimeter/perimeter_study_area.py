import pymongo
import networkx as nx
from shapely.geometry import shape, Point, Polygon
from shapely.ops import unary_union
import matplotlib.pyplot as plt
import json
import numpy as np
from scipy.spatial import cKDTree

# -------------------------------
# Parametri
# -------------------------------
MONGO_URI = "mongodb://urban_trip:urbantrip@43.157.33.33:27018/"
DB_NAME = "turin_go"
COLLECTION = "road"
BUF_DIST = 1e-5  # buffer per unire gap
N_SAMPLES = 1000  # campioni lungo il bordo
SIMPL_TOL = 0.002  # tolleranza per simplificazione

# -------------------------------
# 1) Caricamento grafo e geometrie
# -------------------------------
client = pymongo.MongoClient(MONGO_URI)
cursor = client[DB_NAME][COLLECTION].find()

G = nx.Graph()
lines = []

for doc in cursor:
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

# -------------------------------
# 2) Merge + buffer per estrarre contorno unico
# -------------------------------
merged = unary_union(lines)
buff = merged.buffer(BUF_DIST)
outer_ring = buff.exterior  # LineString chiuso

# -------------------------------
# 3) Campionamento + snap al nodo più vicino
# -------------------------------
length = outer_ring.length
ts = np.linspace(0, length, N_SAMPLES)

coords = list(G.nodes())
tree = cKDTree(coords)

snapped = []
for t in ts:
    p = outer_ring.interpolate(t)
    _, idx = tree.query((p.x, p.y))
    snapped.append(coords[idx])

# rimuovo duplicati consecutivi
perim_nodes = [snapped[0]]
for pt in snapped[1:]:
    if pt != perim_nodes[-1]:
        perim_nodes.append(pt)

# salvo i nodi di perimetro “exact”
with open("perimeter_nodes.json", "w") as f:
    json.dump(perim_nodes, f)

# -------------------------------
# 4) Plot del perimetro esatto
# -------------------------------
pos = {n: n for n in G.nodes()}

plt.figure(figsize=(12, 8))
nx.draw(
    G, pos,
    node_size=3,
    node_color="black",
    edge_color="gray",
    linewidths=0.5
)

cycle_x, cycle_y = zip(*(perim_nodes + [perim_nodes[0]]))
plt.plot(cycle_x, cycle_y, color="red", linewidth=2, label="Exact Perimeter")
plt.scatter(cycle_x, cycle_y, s=20, c="red", marker="o")

plt.legend(loc="upper left")
plt.axis("equal")
plt.tight_layout()
plt.savefig("roads_graph_exact_perimeter.png", dpi=300)
plt.close()

print("✅ roads_graph_exact_perimeter.png salvato")
print(f"✅ perimeter_nodes.json ({len(perim_nodes)} nodi) salvato")

# -------------------------------
# 5) Costruzione e semplificazione del poligono
# -------------------------------
# chiudo l’anello
ring = perim_nodes + [perim_nodes[0]]
poly = Polygon(ring)

# semplifico
poly_s = poly.simplify(SIMPL_TOL, preserve_topology=True)
simp_coords = list(poly_s.exterior.coords)[:-1]

# salvo i vertici semplificati
with open("perimeter_simplified.json", "w") as f:
    json.dump(simp_coords, f)

# -------------------------------
# 6) Plot di confronto ereditiario
# -------------------------------
plt.figure(figsize=(10, 6))
# grafo di perimetro (solo perim nodes)
Gp = nx.Graph()
for n in perim_nodes:
    Gp.add_node(n)
posp = {n: n for n in Gp.nodes()}

nx.draw(
    Gp, posp,
    node_size=3,
    node_color="black",
    edge_color="gray",
    linewidths=0.5
)

# perimetro originale (lightgray)
xs_o, ys_o = zip(*ring)
plt.plot(xs_o, ys_o, color="lightgray", linewidth=1, label="Original Exact")

# perimetro semplificato (rosso)
simp_ring = simp_coords + [simp_coords[0]]
xs_s, ys_s = zip(*simp_ring)
plt.plot(xs_s, ys_s, color="red", linewidth=2, label=f"Semplificato tol={SIMPL_TOL}")

plt.axis("equal")
plt.legend()
plt.tight_layout()
plt.savefig("perimeter_simplified.png", dpi=300)
plt.close()

print("✅ perimeter_simplified.png salvato")
print(f"✅ perimeter_simplified.json ({len(simp_coords)} nodi) salvato")

# -------------------------------
# 7) Plot finale: grafo + poligono semplificato
# -------------------------------
plt.figure(figsize=(12, 8))
nx.draw(
    G, pos,
    node_size=3,
    node_color="black",
    edge_color="lightgray",
    linewidths=0.5
)

# riempio il poligono semplificato
xs_p, ys_p = poly_s.exterior.xy
plt.fill(xs_p, ys_p, facecolor="red", alpha=0.2, label="Area semplificata")
plt.plot(xs_p, ys_p, color="red", linewidth=2)

plt.axis("equal")
plt.legend(loc="upper left")
plt.tight_layout()
plt.savefig("graph_with_simplified_polygon.png", dpi=300)
plt.close()

print("✅ graph_with_simplified_polygon.png salvato")
