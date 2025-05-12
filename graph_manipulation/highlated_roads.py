import pymongo
import networkx as nx
from shapely.geometry import shape
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

# --- Configurazione MongoDB e parametri ---
MONGO_URI = "mongodb://urban_trip:urbantrip@43.157.33.33:27018/"
DB_NAME = "turin_go"
COLLECTION_NAME = "road"
HIGHLIGHT_IDS = {1220209896, 1220302685, 822731287, 584402410}

# --- Connessione a MongoDB e caricamento documenti ---
client = pymongo.MongoClient(MONGO_URI)
collection = client[DB_NAME][COLLECTION_NAME]
db_docs = list(collection.find({}))

# --- Costruzione del grafo ---
G = nx.Graph()
for doc in db_docs:
    geom = doc.get("geometry")
    if not geom or geom.get("type") != "LineString":
        continue
    line = shape(geom)
    u = tuple(line.coords[0])
    v = tuple(line.coords[-1])
    rid = doc.get("road_id")
    if G.has_edge(u, v):
        if rid not in G[u][v]["road_ids"]:
            G[u][v]["road_ids"].append(rid)
    else:
        G.add_edge(u, v, road_ids=[rid])

# --- Preparazione per il disegno ---
pos = {node: node for node in G.nodes()}

# --- Creazione della figura ---
fig, ax = plt.subplots(figsize=(12, 8))

# Disegna nodi e archi base in grigio chiaro
nx.draw_networkx_nodes(
    G, pos,
    node_size=2,
    node_color='black',
    ax=ax
)
nx.draw_networkx_edges(
    G, pos,
    edgelist=G.edges(),
    edge_color='lightgray',
    width=1,
    ax=ax
)

# Seleziona ed evidenzia gli archi con road_id in HIGHLIGHT_IDS
highlight_edges = [
    (u, v) for u, v, data in G.edges(data=True)
    if any(rid in HIGHLIGHT_IDS for rid in data["road_ids"])
]
nx.draw_networkx_edges(
    G, pos,
    edgelist=highlight_edges,
    edge_color='yellow',
    width=2.5,
    ax=ax
)

# Aggiunge legenda per gli archi evidenziati
highlight_handle = Line2D(
    [0], [0],
    color='yellow',
    lw=2.5,
    label='Highlighted road segments'
)
ax.legend(
    handles=[highlight_handle],
    loc='upper left'
)

# Impostazioni finali del grafico
ax.set_title("Road Network\nHighlighted road segments in yellow")
ax.set_xlabel("Longitude")
ax.set_ylabel("Latitude")
ax.set_aspect('equal', adjustable='datalim')
ax.grid(False)
plt.tight_layout()

# Salva immagine
output_path = "../output/highlighted_roads.png"
plt.savefig(output_path, dpi=300)
plt.close()

print(f"Grafico salvato in: {output_path}")
