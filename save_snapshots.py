import os
import asyncio
import pandas as pd
from datetime import datetime
import torch
import networkx as nx
from torch_geometric.data import Data
from typing import Optional
from tqdm import tqdm
from utils import read_env, setup_logger_and_check_folders
read_env()
setup_logger_and_check_folders()
from modules.road_network import RoadNetwork

# === Configurazioni ===
CSV_PATH = "./data/weather.csv"
OUT_DIR = "./data/snapshots"
os.makedirs(OUT_DIR, exist_ok=True)


def save_graph_snapshot(
    graph: nx.Graph,
    out_dir: str,
    tag: Optional[str] = None
) -> None:
    """
    Convert a NetworkX graph into a PyTorch Geometric Data object
    and save it to disk. Only 'length', 'speed', and 'time' edge attributes are used.

    :param graph: NetworkX graph with edge attributes.
    :param out_dir: Destination directory for the snapshot.
    :param tag: Optional filename tag (e.g., 'clear_20250501T14').
    """

    # Fixed set of attributes expected from RoadNetwork
    edge_attr_keys = ['length', 'speed', 'time']

    # Create node index mapping
    nodes = list(graph.nodes())
    node_to_idx = {node: i for i, node in enumerate(nodes)}

    # Node features: placeholder = node degree
    x = torch.tensor(
        [graph.degree[node] for node in nodes],
        dtype=torch.float
    ).unsqueeze(1)  # [N, 1]

    # Build edges and edge attributes
    sources, targets, edge_attr_list = [], [], []

    for u, v, attr in graph.edges(data=True):
        sources.append(node_to_idx[u])
        targets.append(node_to_idx[v])
        edge_attr_list.append([
            float(attr.get(k, 0.0)) for k in edge_attr_keys
        ])

    edge_index = torch.tensor([sources, targets], dtype=torch.long)  # [2, E]
    edge_attr = torch.tensor(edge_attr_list, dtype=torch.float)      # [E, 3]

    # Build PyG Data object
    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
    data.edge_attr_keys = edge_attr_keys  # attach metadata

    # Generate filename
    tag = tag or datetime.now().strftime("%Y%m%dT%H%M%S")
    filename = f"snapshot_{tag}.pt"
    path = os.path.join(out_dir, filename)

    if os.path.exists(path):
        print(f"[SKIP] Snapshot {tag} already exists.")
        return

    torch.save(data, path)
    print(f"[snapshot] Saved → {path}")


async def save_for_datetime(weather: str, dt: datetime):
    """
    Build the road network at a specific datetime and save its graph snapshot.
    """
    try:
        net = RoadNetwork()
        await net.async_init(start_time=dt)
        net.build_graph()
        tag = f"{weather}_{dt:%Y%m%dT%H}"
        save_graph_snapshot(net.graph, OUT_DIR, tag=tag)
    except Exception as e:
        print(f"[ERROR] Failed to save snapshot for {dt}: {e}")


async def main():
    try:
        df = pd.read_csv(CSV_PATH, parse_dates=['datetime'])
    except FileNotFoundError:
        print(f"[ERROR] Weather file not found at {CSV_PATH}")
        return

    print(f"[INFO] Loaded {len(df)} entries from weather.csv")

    BATCH_SIZE = 20
    rows = list(df.itertuples(index=False))

    for i in tqdm(range(0, len(rows), BATCH_SIZE), desc="Saving snapshots"):
        batch = rows[i:i + BATCH_SIZE]
        tasks = [
            save_for_datetime(row.weather_condition, row.datetime)
            for row in batch
        ]
        await asyncio.gather(*tasks)

    print("[DONE] All snapshots processed.")


if __name__ == "__main__":
    asyncio.run(main())
