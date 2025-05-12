import pymongo
import networkx as nx
from shapely.geometry import shape, LineString
from shapely.ops import linemerge
import matplotlib.pyplot as plt

# --- Connessione a MongoDB e creazione del grafo ---
MONGO_URI = "mongodb://urban_trip:urbantrip@43.157.33.33:27018/"
DB_NAME = "turin_go"
COLLECTION_NAME = "road"

client = pymongo.MongoClient(MONGO_URI)
db = client[DB_NAME]
db_docs = list(db[COLLECTION_NAME].find({}))
print(f"Retrieved {len(db_docs)} documents from the DB.")

# Creazione del grafo: memorizziamo, per ogni edge, la geometria e il road_id (in una lista)
db_graph = nx.Graph()

for doc in db_docs:
    if "geometry" not in doc:
        continue
    geom = shape(doc["geometry"])
    if geom.geom_type != "LineString":
        continue

    # Il valore di road_id preso dal documento
    this_road_id = doc.get("road_id")

    # Definizione dei nodi (tail e head)
    tail = (geom.coords[0][0], geom.coords[0][1])
    head = (geom.coords[-1][0], geom.coords[-1][1])

    # Se già esiste un edge tra tail e head, aggiorna la lista dei road_id
    if db_graph.has_edge(tail, head):
        edge_data = db_graph.edges[tail, head]
        if this_road_id not in edge_data.get("road_ids", []):
            edge_data["road_ids"].append(this_road_id)
    else:
        db_graph.add_edge(tail, head, geometry=geom, road_ids=[this_road_id])

print(f"DB graph has {db_graph.number_of_nodes()} nodes and {db_graph.number_of_edges()} edges.")


# --- Funzione per semplificare il grafo contraendo i nodi di grado 2 ---
def simplify_graph(G):
    """
    Semplifica il grafo G contraendo i nodi di grado 2.
    Per ogni edge, mantiene:
      - 'geometry': la LineString unita,
      - 'road_ids': la lista dei road_id provenienti dai segmenti originali.
    """
    G_simple = G.copy()
    changed = True
    while changed:
        changed = False
        # Itera su una copia della lista dei nodi per evitare problemi durante la modifica
        for node in list(G_simple.nodes()):
            # Considera solo nodi di grado 2, che sono interni a un segmento continuo
            if G_simple.degree(node) == 2:
                neighbors = list(G_simple.neighbors(node))
                if len(neighbors) != 2:
                    continue
                u, v = neighbors

                # Recupera i dati degli edge incidenti
                edge_data_1 = G_simple.edges[u, node]
                edge_data_2 = G_simple.edges[node, v]
                geom1 = edge_data_1['geometry']
                geom2 = edge_data_2['geometry']

                # Assicuriamoci che le geometrie siano orientate correttamente per concatenarsi
                if geom1.coords[-1] != node:
                    geom1 = LineString(list(geom1.coords)[::-1])
                if geom2.coords[0] != node:
                    geom2 = LineString(list(geom2.coords)[::-1])

                # Unione delle coordinate, rimuovendo il duplicato nel punto di contatto
                new_coords = list(geom1.coords) + list(geom2.coords)[1:]
                new_geom = LineString(new_coords)

                # Unione dei road_id dei due edge, eliminando duplicati
                merged_road_ids = list(set(edge_data_1.get("road_ids", []) + edge_data_2.get("road_ids", [])))

                # Se esiste già un edge diretto tra u e v, aggiorna i road_id
                if G_simple.has_edge(u, v):
                    existing = G_simple.edges[u, v]
                    merged_road_ids = list(set(existing.get("road_ids", []) + merged_road_ids))
                    G_simple.edges[u, v].update({
                        "geometry": new_geom,
                        "road_ids": merged_road_ids
                    })
                else:
                    # Aggiungi il nuovo edge con la geometria unita e i road_id aggregati
                    G_simple.add_edge(u, v, geometry=new_geom, road_ids=merged_road_ids)

                # Rimuovi il nodo intermedio (quindi i due edge iniziali sono "fusi")
                G_simple.remove_node(node)
                changed = True
                break  # Riavvia il ciclo, dato che la topologia è cambiata
    return G_simple


# Applica la semplificazione del grafo
db_graph_simple = simplify_graph(db_graph)
print(f"Simplified graph has {db_graph_simple.number_of_nodes()} nodes and {db_graph_simple.number_of_edges()} edges.")

# --- Assegna un nuovo identificativo univoco a ciascuna strada semplificata e traccia il mapping ---
mapping_new_road = {}  # mapping tra new_road_id e i road_id originali
for idx, (u, v, data) in enumerate(db_graph_simple.edges(data=True)):
    new_road_id = idx  # Puoi usare una codifica diversa, se preferisci
    data["new_road_id"] = new_road_id
    mapping_new_road[new_road_id] = data["road_ids"]

print("Mapping (new_road_id -> road_ids):")
for rid, road_ids in mapping_new_road.items():
    print(f"Strada {rid}: road_ids {road_ids}")

# --- Visualizzazione del grafo semplificato ---
pos_simple = {node: node for node in db_graph_simple.nodes()}

fig, ax = plt.subplots(figsize=(12, 8))
nx.draw_networkx_nodes(db_graph_simple, pos_simple, ax=ax, node_color='blue', node_size=5)
nx.draw_networkx_edges(db_graph_simple, pos_simple, ax=ax, edge_color='gray', width=1)
ax.set_title("Grafo semplificato: Strade unite (solo road_id)")
ax.set_xlabel("Longitude")
ax.set_ylabel("Latitude")
ax.grid(True, linestyle='--', alpha=0.5)
ax.set_aspect('equal', adjustable='datalim')
plt.tight_layout()
plt.savefig("output/simplified_roads.png", dpi=300)
plt.close()
print("Grafo semplificato salvato in output/simplified_roads.png")
