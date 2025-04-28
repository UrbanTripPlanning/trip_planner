import os
import time
import json
import logging
import networkx as nx
import matplotlib.pyplot as plt
from enum import Enum
from datetime import datetime, timedelta
from typing import Tuple, Optional, List, Dict

from modules.road_network import RoadNetwork
from utils import euclidean_distance


class TransportMode(Enum):
    FOOT = ('Foot', 5 / 3.6)  # 5 km/h in m/s
    BIKE = ('Bike', 15 / 3.6)  # 15 km/h in m/s
    CAR = ('Car', None)  # For Car, we use the 'car_travel_time' field

    def __init__(self, mode_name: str, default_speed: Optional[float]):
        self.mode_name = mode_name
        self.default_speed = default_speed

    def __str__(self):
        return self.mode_name


class RoutePlanner:
    def __init__(
            self,
            network: RoadNetwork,
            transport_mode: TransportMode = TransportMode.FOOT,
            algorithm: str = 'A*',
            GNN: str = ''
    ):
        """
        Initialize the RoutePlanner.

        :param network: Instance of RoadNetwork.
        :param transport_mode: Transport mode (TransportMode Enum).
        :param algorithm: Algorithm to use ('A*' or 'Dijkstra').
        """
        self.network = network
        self.graph = network.graph
        self.transport_mode = transport_mode
        self.algorithm = algorithm
        self.GNN = GNN
        logging.info(
            f"Initialized RoutePlanner with transport_mode: {self.transport_mode.mode_name}, algorithm: {self.algorithm}")

    def compute(
            self,
            source_point: Tuple[float, float],
            target_point: Tuple[float, float]
    ) -> Tuple[Optional[List[Tuple[float, float]]], Optional[float]]:
        """
        Compute the route between source_point and target_point.

        The cost attribute is chosen based on the transport mode:
          - For Car: use 'car_travel_time'
          - For Foot/Bike: use 'length'

        Returns the computed path (a list of nodes) and the execution time.
        """
        # Select cost attribute based on transport mode.
        if self.transport_mode == TransportMode.CAR:
            if self.GNN:
                cost = 'weight'
            else:
                cost = 'car_travel_time'
            logging.info(f"Using cost attribute {cost} for Car mode.")
        else:
            cost = 'length'
            logging.info("Using cost attribute 'length' for Foot/Bike mode.")

        # Ensure the source and target nodes exist; if not, use the nearest node.
        source = self.network.ensure_node(source_point)
        target = self.network.ensure_node(target_point)
        logging.info(f"Computed source node: {source}, target node: {target}")

        start_time_compute = time.perf_counter()

        # Compute the route using the selected algorithm.
        if self.algorithm.lower() == 'dijkstra':
            try:
                path = nx.dijkstra_path(self.graph, source, target, weight=cost)
                logging.info(f"Dijkstra route found: {path}")
            except nx.NetworkXNoPath:
                logging.error("No route found using Dijkstra.")
                return None, None
        else:
            # Use A* algorithm by default.
            def heuristic(u: Tuple[float, float], v: Tuple[float, float]) -> float:
                return euclidean_distance(u, v)

            try:
                path = nx.astar_path(self.graph, source, target, heuristic=heuristic, weight=cost)
                logging.info(f"A* route found: {path}")
            except nx.NetworkXNoPath:
                logging.error("No route found using A*.")
                return None, None

        exec_time = time.perf_counter() - start_time_compute
        logging.info(f"Route computation time: {exec_time:.6f} seconds using {self.algorithm.capitalize()}")
        return path, exec_time

    def plot_path(self, path: Optional[List[Tuple[float, float]]], path_color: str = "black") -> None:
        """
        Plot the computed route on the network graph and save the plot.
        """
        pos = {node: node for node in self.graph.nodes()}
        plt.figure(figsize=(10, 8))
        # Draw background network edges and nodes.
        nx.draw_networkx_edges(self.graph, pos, edge_color='lightgray', width=1)
        nx.draw_networkx_nodes(self.graph, pos, node_size=5, node_color='lightgray')
        # Highlight the route.
        if path:
            path_edges = list(zip(path, path[1:]))
            nx.draw_networkx_edges(self.graph, pos, edgelist=path_edges, edge_color=path_color, width=3)
            nx.draw_networkx_nodes(self.graph, pos, nodelist=path, node_color=path_color, node_size=20)
        title = f"Best Path with {self.algorithm.capitalize()} - {self.transport_mode}"
        plt.title(title)
        filename = f'best_path_{self.transport_mode.mode_name.lower()}.png'
        plt.savefig(os.path.join(os.getenv("CURRENT_OUT_PATH"), filename))
        logging.info(f"Graph image saved as '{filename}' in directory '{os.getenv('CURRENT_OUT_PATH')}'")
        plt.close()

    def get_statistics(
            self,
            path: List[Tuple[float, float]],
            start_time: Optional[datetime] = None,
            end_time: Optional[datetime] = None
    ) -> Dict[str, float]:
        """
        Calculate route statistics by summing up edge attributes.

        For Car mode: sums the 'car_travel_time' values.
        For Foot/Bike mode: calculates travel time using the default speed.

        Depending on the time parameters provided:
          - If only start_time is provided: compute arrival time.
          - If only end_time is provided: compute the latest departure time.
          - If both are None: use current time as departure.

        Returns a dictionary with keys:
          - 'length': total route length,
          - 'duration': total travel duration,
          - and time information (start_time, end_time, deadline, or time margin).
        """
        stats = {'length': 0, 'duration': 0}
        # Iterate through each consecutive pair of nodes in the path.
        for i in range(len(path) - 1):
            u, v = path[i], path[i + 1]
            edge_data = self.graph.get_edge_data(u, v)
            length = edge_data.get('length', 0)
            if self.transport_mode == TransportMode.CAR:
                travel_time = edge_data.get('car_travel_time', 0)
            else:
                default_speed = self.transport_mode.default_speed
                travel_time = length / default_speed if default_speed else 0
            stats['length'] += length
            stats['duration'] += travel_time
            logging.debug(f"Edge {u} -> {v}: length = {length}, travel_time = {travel_time}")

        # Case: Only start_time provided.
        if start_time is not None and end_time is None:
            computed_end = start_time + timedelta(seconds=stats['duration'])
            stats['start_time'] = start_time.strftime('%Y-%m-%d %H:%M:%S')
            stats['end_time'] = computed_end.strftime('%Y-%m-%d %H:%M:%S')
            logging.info(f"Computed arrival time: {stats['end_time']} from departure at: {stats['start_time']}")
        # Case: Only end_time provided.
        elif end_time is not None and start_time is None:
            computed_start = end_time - timedelta(seconds=stats['duration'])
            stats['start_time'] = computed_start.strftime('%Y-%m-%d %H:%M:%S')
            stats['end_time'] = end_time.strftime('%Y-%m-%d %H:%M:%S')
            logging.info(f"Computed departure time: {stats['start_time']} to meet deadline: {stats['end_time']}")
        # Case: Neither provided (use current time).
        elif start_time is None and end_time is None:
            current_time = datetime.now()
            computed_end = current_time + timedelta(seconds=stats['duration'])
            stats['start_time'] = current_time.strftime('%Y-%m-%d %H:%M:%S')
            stats['end_time'] = computed_end.strftime('%Y-%m-%d %H:%M:%S')
            logging.info("No start_time or end_time provided; using current time as departure time.")

        # weather
        stats['weather_condition'] = edge_data.get('weather_condition', 'empty')

        # Round numeric values for clarity.
        for key, value in stats.items():
            if isinstance(value, (int, float)):
                stats[key] = round(value, 2)
        logging.info(f"Final route statistics: {stats}")
        return stats

    def display_statistics(
            self,
            path: List[Tuple[float, float]],
            start_time: Optional[datetime] = None,
            end_time: Optional[datetime] = None
    ) -> None:
        """
        Display and log the computed route statistics.
        """
        stats = self.get_statistics(path, start_time, end_time)
        stats_json = json.dumps(stats, indent=4)
        logging.info("Displaying route statistics:")
        logging.info(stats_json)
        print(f'{self.transport_mode} statistics:')
        print(stats_json)
