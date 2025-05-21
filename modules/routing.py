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


class Algorithm(Enum):
    A = "A*"
    DIJKSTRA = "DIJKSTRA"

    def name(self):
        return self._name_.lower()


class TransportMode(Enum):
    FOOT = ("Foot", 4.3 / 3.6)  # 4.3 km/h → m/s
    BIKE = ("Bike", 15 / 3.6)  # 12 km/h → m/s
    CAR = ("Car", None)        # Use edge 'time' or 'weight'

    def __init__(self, mode_name: str, default_speed: Optional[float]):
        self.mode_name = mode_name
        self.default_speed = default_speed

    def __str__(self) -> str:
        return self.mode_name


class RoutePlanner:
    """
    Computes optimal routes on a road network using Dijkstra or A*.
    Supports FOOT and BIKE (fixed speeds, bidirectional edges) and CAR
    (using real edge times or GNN weights on directed edges).
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
        :param network: Initialized RoadNetwork (with DiGraph and 'pos' on each node).
        :param transport_mode: One of TransportMode.
        :param algorithm_name: 'A*' or 'Dijkstra'.
        :param use_gnn: If True and mode == CAR, use edge['weight'] instead of ['time'].
        """
        self.network = network
        self.road_graph: nx.DiGraph = network.graph
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
        Return the edge attribute used for path cost:
          - FOOT/BIKE → 'length'
          - CAR → 'weight' if use_gnn else 'time'
        """
        if self.transport_mode == TransportMode.CAR:
            return "weight" if self.use_gnn else "time"
        return "length"

    def _prepare_nodes(
        self,
        source_point: Tuple[float, float],
        target_point: Tuple[float, float]
    ) -> Tuple[int, int]:
        """
        Snap raw coords to the nearest graph node IDs.
        """
        src_id = self.network.snap_to_graph(source_point)
        tgt_id = self.network.snap_to_graph(target_point)
        return src_id, tgt_id

    def _run_path_algorithm(
        self,
        G: nx.Graph,
        source: int,
        target: int,
        cost_attr: str
    ) -> List[int]:
        """
        Execute the chosen pathfinding algorithm on graph G
        and return a list of node IDs.
        """
        if self.algorithm_name.lower() == "dijkstra":
            return nx.dijkstra_path(G, source, target, weight=cost_attr)

        # A* with Euclidean heuristic on node positions
        def heuristic(u: int, v: int) -> float:
            pu = G.nodes[u]["pos"]
            pv = G.nodes[v]["pos"]
            return euclidean_distance(pu, pv)

        return nx.astar_path(
            G, source, target,
            heuristic=heuristic,
            weight=cost_attr
        )

    def _draw_network(
        self,
        ax: plt.Axes,
        path_nodes: Optional[List[int]] = None,
        path_color: str = "black"
    ) -> None:
        """
        Draw the full network in light gray and optionally mark path_nodes.
        """
        pos = nx.get_node_attributes(self.road_graph, "pos")
        # draw undirected background
        G_und = self.road_graph.to_undirected()
        nx.draw_networkx_edges(G_und, pos, ax=ax, edge_color="lightgray", width=1, arrows=False)
        nx.draw_networkx_nodes(G_und, pos, ax=ax, node_size=5, node_color="lightgray")

        if path_nodes and len(path_nodes) > 1:
            edges = list(zip(path_nodes, path_nodes[1:]))
            nx.draw_networkx_edges(
                self.road_graph, pos,
                edgelist=edges,
                ax=ax, edge_color=path_color, width=3, arrows=False
            )
            nx.draw_networkx_nodes(
                self.road_graph, pos,
                nodelist=path_nodes,
                ax=ax, node_color=path_color, node_size=20
            )

    def compute(
        self,
        source_point: Tuple[float, float],
        target_point: Tuple[float, float],
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None
    ) -> Tuple[Optional[List[Tuple[float, float]]], Optional[Dict[str, Any]]]:
        """
        Compute the optimal route from source to target.
        CAR: directed; FOOT/BIKE: undirected.
        Returns (list of (lon,lat), stats) or (None, None).
        """
        if self.road_graph is None:
            raise RuntimeError("Road graph is not initialized.")

        cost_attr = self._select_cost_attribute()
        source_id, target_id = self._prepare_nodes(source_point, target_point)

        # choose graph topology
        if self.transport_mode == TransportMode.CAR:
            G = self.road_graph

        else:
            G = self.road_graph.to_undirected()

        logging.info(
            f"Routing ({self.transport_mode}) from {source_id} to {target_id} "
            f"on {'directed' if G is self.road_graph else 'undirected'} graph, cost='{cost_attr}'"
        )

        start = time.perf_counter()
        try:
            node_path = self._run_path_algorithm(G, source_id, target_id, cost_attr)
        except nx.NetworkXNoPath:
            logging.warning(
                f"No path found ({self.algorithm_name}) from {source_id} to {target_id}"
            )
            return None, None
        elapsed = time.perf_counter() - start
        logging.info(f"Path found in {elapsed:.4f}s")

        # compute stats on node path with correct graph
        self.stats = self.compute_statistics(node_path, start_time, end_time, graph=G)

        # map to coordinates
        coord_path = [G.nodes[n]["pos"] for n in node_path]
        return coord_path, self.stats

    def plot_path(
        self,
        path: Optional[List[Tuple[float, float]]],
        path_color: str = "red"
    ) -> None:
        """
        Plot the computed path on the network background.
        """
        if self.road_graph is None:
            raise RuntimeError("Road graph is not initialized.")
        if not path:
            logging.warning("No path to plot.")
            return

        fig, ax = plt.subplots(figsize=(10, 8))
        # background
        self._draw_network(ax, path_nodes=None, path_color=path_color)
        # overlay route
        xs, ys = zip(*path)
        ax.plot(xs, ys, linewidth=3, color=path_color, solid_capstyle="round")
        ax.set_title(f"Route {self.transport_mode} ({self.algorithm_name})")
        filename = f"route_{self.transport_mode.mode_name.lower()}.png"
        out_path = os.path.join(self.OUTPUT_DIR, filename)
        fig.savefig(out_path)
        plt.close(fig)
        logging.info(f"Saved route plot to '{out_path}'")

    def _calculate_delay(self, G: nx.DiGraph, path: List[int]) -> float:
        """
        Compute total signal delay along the path using Webster's uniform delay:
            d_base = ½ · C · (1 – g/C)²
        Total delay = number_of_signals · d_base
        """

        # 1) Compute degree only for nodes on the path
        node_degree = {
            node: len(set(G.predecessors(node)) | set(G.successors(node)))
            for node in path
        }

        # 2) Webster parameters
        INTERSECTION_THRESHOLD = 3  # degree ≥ 3 is signalized
        CYCLE_TIME = 90.0  # seconds
        GREEN_RATIO = 0.4  # fraction of cycle that is green

        # 3) Base delay per signal (Webster uniform delay, unsaturated)
        d_base = 0.5 * CYCLE_TIME * (1 - GREEN_RATIO) ** 2

        # 4) Count how many signals the path crosses
        signal_count = sum(
            1 for v in path[1:]  # skip the origin node
            if node_degree.get(v, 0) >= INTERSECTION_THRESHOLD
        )

        # 5) Total delay is simply count × d_base
        total_delay = signal_count * d_base

        logging.info(
            f"{signal_count} signals crossed, "
            f"d_base={d_base:.2f}s; total_delay={total_delay:.2f}s"
        )
        return total_delay

    def compute_statistics(
        self,
        path: List[int],
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        graph: Optional[nx.Graph] = None
    ) -> Dict[str, Any]:
        """
        From node-ID path on `graph`, compute total length, duration, and infer times.
        """
        G = graph or self.road_graph
        if not path or G is None:
            raise RuntimeError("Invalid state: missing path or graph.")

        total_length = 0.0
        total_duration = 0.0
        for u, v in zip(path[:-1], path[1:]):
            edge = G.get_edge_data(u, v, {}) or {}
            seg_len = edge.get("length", 0.0)
            if self.transport_mode == TransportMode.CAR:
                seg_time = edge.get("time", 0.0)
            else:
                speed = self.transport_mode.default_speed or 1.0
                seg_time = seg_len / speed
            total_length += seg_len
            total_duration += seg_time
            logging.debug(f"Edge {u}->{v}: length={seg_len}, time={seg_time:.2f}s")

        # apply traffic-light delay
        if self.transport_mode == TransportMode.CAR:
            total_duration += self._calculate_delay(G, path)

        now = datetime.now()
        if start_time and not end_time:
            st, et = start_time, start_time + timedelta(seconds=total_duration)
        elif end_time and not start_time:
            st, et = end_time - timedelta(seconds=total_duration), end_time
        else:
            st, et = now, now + timedelta(seconds=total_duration)

        stats = {
            "length": round(total_length, 2),
            "duration": round(total_duration, 2),
            "start_time": format_dt(st),
            "end_time": format_dt(et),
        }
        logging.info(f"Route statistics: {stats}")
        return stats

    def display_statistics(self) -> None:
        """
        Print the last computed route stats to the console in JSON.
        """
        if not self.stats:
            logging.warning("No statistics available; compute a route first.")
            print("No route has been computed yet.")
            return

        print(f"{self.transport_mode} Route Statistics:")
        print(json.dumps(self.stats, indent=4))
