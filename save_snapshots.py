import os
import asyncio
import pandas as pd
from datetime import datetime
import torch
import networkx as nx
from torch_geometric.data import Data

from utils import read_env, setup_logger_and_check_folders
read_env()
setup_logger_and_check_folders()
from modules.road_network import RoadNetwork

# Percorso del tuo file CSV
CSV_PATH = "./data/weather.csv"

# Dove salvare gli snapshot
OUT_DIR = "./data/snapshots"
os.makedirs(OUT_DIR, exist_ok=True)


def save_graph_snapshot(
        graph: nx.Graph,
        out_dir: str,
        tag: str = None
) -> None:
    """
    Save a complete graph snapshot (including all numeric node and edge attributes)
    as a PyTorch Geometric Data object.

    Args:
        graph:   A NetworkX graph with node and edge attributes.
        out_dir: Directory where the snapshot file will be saved.
        tag:     Optional string to identify the snapshot (e.g. '20250425T14h').

    This function:
      1. Enumerates nodes in a deterministic order and builds a node-index map.
      2. Constructs tensors for:
         - x          : node features (using node degree as placeholder).
         - edge_index : [2, E] long tensor of source/target indices.
         - edge_attr  : [E, F_edge] float tensor of all numeric edge attributes except 'weather_condition'.
      3. Records `edge_attr_keys` so downstream code can index by name.
      4. Saves the Data object as 'snapshot_<tag>.pt' under out_dir.

    Storing full snapshots duplicates static info, but avoids grouping logic.
    """
    # 1) Deterministic node ordering
    nodes = list(graph.nodes())
    node_to_idx = {node: i for i, node in enumerate(nodes)}

    # 2) Node feature tensor: degree as placeholder
    x = torch.tensor(
        [graph.degree[node] for node in nodes],
        dtype=torch.float
    ).unsqueeze(1)  # shape [N,1]

    # 3) Build edge_index and numeric edge attribute tensor
    edges = list(graph.edges(data=True))
    sources, targets = [], []
    attr_list = []

    # Determine keys to include: numeric attributes except 'weather_condition'
    if edges:
        first_attrs = edges[0][2]
        numeric_keys = [
            k for k, v in first_attrs.items()
            if k != 'weather_condition' and isinstance(v, (int, float, bool))
        ]
    else:
        numeric_keys = []

    for u, v, attr in edges:
        sources.append(node_to_idx[u])
        targets.append(node_to_idx[v])
        # collect numeric attributes in stable key order
        attr_list.append([float(attr.get(k, 0.0)) for k in numeric_keys])

    edge_index = torch.tensor([sources, targets], dtype=torch.long)  # [2, E]
    edge_attr = torch.tensor(attr_list, dtype=torch.float) if numeric_keys else None

    # 4) Create PyG Data and attach attribute keys
    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
    data.edge_attr_keys = numeric_keys

    # 5) Save Data to disk
    ts = tag or datetime.now().strftime("%Y%m%dT%H%M%S")
    os.makedirs(out_dir, exist_ok=True)
    filename = f"snapshot_{ts}.pt"
    path = os.path.join(out_dir, filename)
    torch.save(data, path)
    print(f"[graph_utils] Saved full graph snapshot → {path}")


async def save_for_datetime(weather: str, dt: datetime):
    """
    Build network at a given datetime and save a full snapshot.
    """
    net = RoadNetwork()
    await net.async_init(start_time=dt, end_time=None)
    net.build_graph()
    tag = f"{weather}_{dt:%Y%m%dT%H}"  # esempio: 'clear_20250429T14'
    save_graph_snapshot(net.graph, OUT_DIR, tag=tag)


async def main():
    # Carica il CSV
    df = pd.read_csv(CSV_PATH)
    df['datetime'] = pd.to_datetime(df['datetime'])

    for idx, row in df.iterrows():
        dt = row['datetime']
        weather = row['weather_condition']
        print(f"Saving snapshot for {weather} at {dt}")
        await save_for_datetime(weather, dt)


if __name__ == "__main__":
    asyncio.run(main())
