import json
from shapely.geometry import Polygon, LineString
import matplotlib.pyplot as plt
import networkx as nx

# --- 1) Carico i nodi di perimetro ottenuti prima ---
with open("perimeter_nodes.json") as f:
    perim = json.load(f)

# Assicurati che sia chiuso
perim_ring = perim + [perim[0]]

# --- 2) Costruisco un Poligono e lo semplifico ---
poly = Polygon(perim_ring)

# Scegli una tolleranza:
# valori piccoli restano molto simili all'originale, valori maggiori semplificano di più.
tolerance = 0.002

poly_s = poly.simplify(tolerance, preserve_topology=True)

# Estraggo i nuovi vertici (senza ripetere l'ultimo)
simp_coords = list(poly_s.exterior.coords)[:-1]

# Salvo i vertici semplificati
with open("perimeter_simplified.json", "w") as f:
    json.dump(simp_coords, f)


# --- 3) Plot di confronto ---
# (rappresento sotto sia il grafo che la semplificazione)
# Ricostruisco la posizione dei nodi del grafo per il plot
# (usa lo stesso G che usavi prima; qui per brevità rifaccio veloce)
G = nx.Graph()
for x, y in perim:
    G.add_node((x, y))  # solo per demo, in realtà disegni l'intero G

pos = {n: n for n in G.nodes()}

plt.figure(figsize=(10, 6))
nx.draw(G, pos, node_size=3, node_color="black", edge_color="gray", linewidths=0.5)

# disegno il perimetro original-exact
xs_o, ys_o = zip(*(perim_ring))
plt.plot(xs_o, ys_o, color="lightgray", linewidth=1, label="Original Exact")

# disegno il perimetro semplificato
simp_ring = simp_coords + [simp_coords[0]]
xs_s, ys_s = zip(*simp_ring)
plt.plot(xs_s, ys_s, color="red", linewidth=2, label=f"Semplificato (tol={tolerance})")

plt.axis("equal")
plt.legend()
plt.tight_layout()
plt.savefig("perimeter_simplified.png", dpi=300)
plt.close()

print("✅ Semplificazione completata.")
print(f"- Vertici originali: {len(perim)}")
print(f"- Vertici semplificati: {len(simp_coords)}")
print("✅ Salvato perimeter_simplified.png e perimeter_simplified.json")
