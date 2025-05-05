import os
import time
import json
import logging
import networkx as nx
import matplotlib.pyplot as plt
from enum import Enum
from datetime import datetime, timedelta
from typing import Tuple, Optional, List, Dict, Any

from modules.road_network import RoadNetwork
from utils import euclidean_distance, format_dt


class TransportMode(Enum):
    FOOT = ("Foot", 4.3 / 3.6)  # 4.3 km/h in m/s
    BIKE = ("Bike", 12 / 3.6)   # 12 km/h in m/s
    CAR = ("Car", None)         # Uses actual edge time or weight if GNN enabled

    def __init__(self, mode_name: str, default_speed: Optional[float]):
        self.mode_name = mode_name
        self.default_speed = default_speed

    def __str__(self) -> str:
        return self.mode_name


class RoutePlanner:
    """
    Computes optimal routes over a road network using Dijkstra or A*.
    Supports various transport modes and optional GNN-based edge weighting.
    """
    OUTPUT_DIR: str = os.getenv("CURRENT_OUT_PATH", "./output")

    def __init__(
        self,
        network: RoadNetwork,
        transport_mode: TransportMode = TransportMode.FOOT,
        algorithm_name: str = "A*",
        use_gnn: bool = False
    ) -> None:
        """
        Initialize the route planner with a network, transport mode and algorithm.

        :param network: A RoadNetwork instance with initialized graph.
        :param transport_mode: Selected mode of transport (foot, bike, car).
        :param algorithm_name: Pathfinding algorithm ('A*' or 'Dijkstra').
        :param use_gnn: Whether to use GNN-predicted edge weights (for cars).
        """
        self.network = network
        self.road_graph = network.graph
        self.transport_mode = transport_mode
        self.algorithm_name = algorithm_name
        self.use_gnn = use_gnn
        self.stats: Optional[Dict[str, Any]] = None

        os.makedirs(self.OUTPUT_DIR, exist_ok=True)
        logging.info(
            f"RoutePlanner initialized | Mode: {transport_mode.mode_name}, "
            f"Algorithm: {algorithm_name}, GNN: {use_gnn}"
        )

    def _select_cost_attribute(self) -> str:
        """
        Determine which edge attribute to use as pathfinding cost.
        """
        if self.transport_mode == TransportMode.CAR:
            return "weight" if self.use_gnn else "time"
        return "length"

    def _prepare_nodes(
        self,
        source_point: Tuple[float, float],
        target_point: Tuple[float, float]
    ) -> Tuple[Tuple[float, float], Tuple[float, float]]:
        """
        Snap input coordinates to the nearest nodes in the road graph.
        """
        src = self.network.snap_to_graph(source_point)
        tgt = self.network.snap_to_graph(target_point)
        return src, tgt

    def _run_path_algorithm(
        self,
        source: Tuple[float, float],
        target: Tuple[float, float],
        cost_attr: str
    ) -> List[Tuple[float, float]]:
        """
        Run shortest path algorithm (Dijkstra or A*) on the road graph.
        """
        if self.algorithm_name.lower() == "dijkstra":
            return nx.dijkstra_path(self.road_graph, source, target, weight=cost_attr)
        return nx.astar_path(
            self.road_graph, source, target,
            heuristic=euclidean_distance,
            weight=cost_attr
        )

    def _draw_network(
        self,
        ax: Any,
        path_edges: Optional[List[Tuple[Tuple[float, float], Tuple[float, float]]]] = None,
        path_nodes: Optional[List[Tuple[float, float]]] = None,
        path_color: str = "black"
    ) -> None:
        """
        Draw the road network and optional computed path.
        """
        pos = {node: node for node in self.road_graph.nodes()}
        nx.draw_networkx_edges(self.road_graph, pos, ax=ax, edge_color="lightgray", width=1)
        nx.draw_networkx_nodes(self.road_graph, pos, ax=ax, node_size=5, node_color="lightgray")

        if path_edges and path_nodes:
            nx.draw_networkx_edges(
                self.road_graph, pos, edgelist=path_edges, ax=ax,
                edge_color=path_color, width=3
            )
            nx.draw_networkx_nodes(
                self.road_graph, pos, nodelist=path_nodes, ax=ax,
                node_color=path_color, node_size=20
            )

    def compute(
        self,
        source_point: Tuple[float, float],
        target_point: Tuple[float, float],
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None
    ) -> Tuple[Optional[List[Tuple[float, float]]], Optional[Dict[str, Any]]]:
        """
        Compute the optimal route between two points and collect stats.

        :return: Tuple (path, statistics) or (None, None) if no path found.
        """
        if self.road_graph is None:
            raise RuntimeError("Road graph is not initialized.")

        cost_attr = self._select_cost_attribute()
        source, target = self._prepare_nodes(source_point, target_point)
        logging.info(f"Computing route from {source} to {target} using cost '{cost_attr}'")

        start = time.perf_counter()
        try:
            path = self._run_path_algorithm(source, target, cost_attr)
        except nx.NetworkXNoPath:
            logging.warning(f"No path found from {source} to {target} using {self.algorithm_name}")
            return None, None

        elapsed = time.perf_counter() - start
        logging.info(f"Route computed in {elapsed:.4f} seconds")

        self.stats = self.compute_statistics(path, start_time, end_time)
        return path, self.stats

    def plot_path(
        self,
        path: Optional[List[Tuple[float, float]]],
        path_color: str = "black"
    ) -> None:
        """
        Plot and save a figure of the computed path on the road graph.
        """
        if self.road_graph is None:
            raise RuntimeError("Road graph is not initialized.")

        fig, ax = plt.subplots(figsize=(10, 8))
        edges = list(zip(path, path[1:])) if path else None
        self._draw_network(ax, edges, path, path_color)

        ax.set_title(f"Route - {self.algorithm_name} ({self.transport_mode})")
        filename = f"route_{self.transport_mode.mode_name.lower()}.png"
        output_path = os.path.join(self.OUTPUT_DIR, filename)
        fig.savefig(output_path)
        plt.close(fig)

        logging.info(f"Saved route plot to '{output_path}'")

    def compute_statistics(
        self,
        path: List[Tuple[float, float]],
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None
    ) -> Dict[str, Any]:
        """
        Analyze the computed path to generate total length, duration, and time window.

        :return: Dictionary with statistics on the route.
        """
        if not path or self.road_graph is None:
            raise RuntimeError("Invalid state: missing path or road graph.")

        stats: Dict[str, Any] = {"length": 0.0, "duration": 0.0}
        for u, v in zip(path[:-1], path[1:]):
            edge = self.road_graph.get_edge_data(u, v) or {}
            length = edge.get("length", 0.0)
            if self.transport_mode == TransportMode.CAR:
                travel_time = edge.get("time", 0.0)
            else:
                speed = self.transport_mode.default_speed or 1.0
                travel_time = length / speed

            stats["length"] += length
            stats["duration"] += travel_time
            logging.debug(f"Edge {u} → {v}: length={length}, time={travel_time:.2f}s")

        now = datetime.now()
        if start_time and not end_time:
            end = start_time + timedelta(seconds=stats["duration"])
            stats["start_time"] = format_dt(start_time)
            stats["end_time"] = format_dt(end)
        elif end_time and not start_time:
            start = end_time - timedelta(seconds=stats["duration"])
            stats["start_time"] = format_dt(start)
            stats["end_time"] = format_dt(end_time)
        else:
            stats["start_time"] = format_dt(now)
            stats["end_time"] = format_dt(now + timedelta(seconds=stats["duration"]))

        # Round length and duration
        stats["length"] = round(stats["length"], 2)
        stats["duration"] = round(stats["duration"], 2)

        logging.info(f"Route statistics: {stats}")
        return stats

    def display_statistics(self) -> None:
        """
        Print the most recent route statistics to console.
        """
        if not self.stats:
            logging.warning("No statistics available to display.")
            print("No route has been computed yet.")
            return

        stats_json = json.dumps(self.stats, indent=4)
        print(f"{self.transport_mode} Route Statistics:")
        print(stats_json)
