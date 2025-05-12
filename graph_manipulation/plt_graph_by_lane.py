import pymongo
import networkx as nx
from shapely.geometry import shape
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

# --- Connessione a MongoDB e creazione del grafo ---
MONGO_URI = "mongodb://urban_trip:urbantrip@43.157.33.33:27018/"
DB_NAME = "turin_go"
COLLECTION_NAME = "road"

client = pymongo.MongoClient(MONGO_URI)
db = client[DB_NAME]
db_docs = list(db[COLLECTION_NAME].find({}))
print(f"Retrieved {len(db_docs)} documents from the DB.")

# Creazione del grafo: memorizziamo, per ogni edge, la geometria, i road_id (in una lista) e il campo 'lane'
db_graph = nx.Graph()

for doc in db_docs:
    if "geometry" not in doc:
        continue
    geom = shape(doc["geometry"])
    if geom.geom_type != "LineString":
        continue

    # Estrae il road_id e il numero di corsie (lane). Se non presente, assume 1 corsia
    this_road_id = doc.get("road_id")
    lane_value = doc.get("lane", 1)

    # Definizione dei nodi (tail e head) dalla geometria
    tail = (geom.coords[0][0], geom.coords[0][1])
    head = (geom.coords[-1][0], geom.coords[-1][1])

    # Se già esiste un edge tra tail e head, aggiorna i campi
    if db_graph.has_edge(tail, head):
        edge_data = db_graph.edges[tail, head]
        if this_road_id not in edge_data.get("road_ids", []):
            edge_data["road_ids"].append(this_road_id)
        edge_data["lane"] = max(edge_data.get("lane", lane_value), lane_value)
    else:
        db_graph.add_edge(tail, head, geometry=geom, road_ids=[this_road_id], lane=lane_value)

print(f"Originario graph has {db_graph.number_of_nodes()} nodes and {db_graph.number_of_edges()} edges.")

# --- Plot del grafo originario ---
# Costruisci la mappa delle posizioni: i nodi sono già le coordinate (tuple)
pos_origin = {node: node for node in db_graph.nodes()}

# Raggruppa gli edge in base al campo lane
edges_by_lane = {}
for u, v, data in db_graph.edges(data=True):
    lane = data.get("lane", 1)
    edges_by_lane.setdefault(lane, []).append((u, v))

# Definisci una mappa colori per i diversi valori di lane
color_mapping = {1: 'blue', 2: 'red', 3: 'green', 4: 'orange', 5: 'purple'}

# Crea il plot
fig, ax = plt.subplots(figsize=(12, 8))
nx.draw_networkx_nodes(db_graph, pos_origin, ax=ax, node_color='black', node_size=3)

# Disegna gli edge raggruppati per lane
for lane, edges in edges_by_lane.items():
    color = color_mapping.get(lane, 'black')
    nx.draw_networkx_edges(db_graph, pos_origin, edgelist=edges, ax=ax, edge_color=color, width=1)

# Crea handle per la legenda
legend_handles = [Line2D([0], [0], color=color_mapping.get(lane, 'black'), lw=2, label=f"Lane = {lane}")
                  for lane in sorted(edges_by_lane.keys())]

ax.legend(handles=legend_handles, loc='upper left')

ax.set_title("Grafo Originario colorato per numero di corsie (lane)")
ax.set_xlabel("Longitude")
ax.set_ylabel("Latitude")
ax.grid(True, linestyle='--', alpha=0.5)
ax.set_aspect('equal', adjustable='datalim')
plt.tight_layout()
plt.savefig("output/roads_by_lane.png", dpi=300)
plt.close()

print("Grafico salvato in output/roads_by_lane.png")
