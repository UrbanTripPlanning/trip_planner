import os
import time
import math
import logging
import networkx as nx
import geopandas as gpd
import matplotlib.pyplot as plt
from functools import lru_cache
from typing import Tuple, Optional, List

from modules.compute_weights import ComputeWeights


@lru_cache(maxsize=None)
def euclidean_distance(p1: Tuple[float, float], p2: Tuple[float, float]) -> float:
    """
    Calculate the Euclidean distance between two points.

    :param p1: Tuple (x, y) representing the first point.
    :param p2: Tuple (x, y) representing the second point.
    :return: Euclidean distance.
    """
    return math.hypot(p1[0] - p2[0], p1[1] - p2[1])


class RoadNetwork:
    """
    Class to manage a road network loaded from a MongoDB collection via the asynchronous Database module.

    Handles:
      - Asynchronous initialization of the database connection.
      - Creating a GeoDataFrame from road data.
      - Building a NetworkX graph from road geometries.
      - Computing the shortest paths using Dijkstra and A* algorithms.
      - Visualizing the graph with highlighted paths.
    """

    def __init__(self, collection_name: str) -> None:
        """
        Initialize the RoadNetwork instance.

        :param collection_name: Collection name containing road data.
        """
        self.collection_name = collection_name
        self.gdf: Optional[gpd.GeoDataFrame] = None
        self.graph: Optional[nx.Graph] = None

    async def async_init(self):
        """
        Asynchronously initialize the database connection, load road data,
        and build the NetworkX graph.
        """
        cw = ComputeWeights()
        # Initialize the asynchronous DB connection
        await cw.initialize_db()
        # Query and load data from all collections (only road data is available)
        await cw.load_all_data()
        # Process and merge the data into one GeoDataFrame (only road data used)
        cw.merge_data()
        self.gdf = cw.get_geo_dataframe()
        # Build the graph from the GeoDataFrame
        self.build_graph()

    def build_graph(self) -> None:
        """
        Build a NetworkX graph from the GeoDataFrame.
        For each record with a LineString geometry, extract the start and end points,
        and add an edge with a weight based on the 'leng' attribute (if present) or the geometry's length.
        """
        self.graph = nx.Graph()
        try:
            for _, row in self.gdf.iterrows():
                geom = row['geometry']
                if geom.geom_type == 'LineString':
                    start = tuple(geom.coords[0])
                    end = tuple(geom.coords[-1])
                    weight = row.get('leng', geom.length)
                    self.graph.add_edge(start, end, weight=weight)
            logging.info(f"Graph built with {self.graph.number_of_nodes()} nodes and {self.graph.number_of_edges()} edges.")
        except Exception as e:
            logging.error(f"Error building graph: {e}")
            raise

    def find_nearest_node(self, point: Tuple[float, float]) -> Tuple[float, float]:
        """
        Find the node in the graph closest to the given point using Euclidean distance.

        :param point: Tuple (x, y) representing the query point.
        :return: The nearest node (x, y) in the graph.
        """
        nearest: Optional[Tuple[float, float]] = None
        min_dist = float('inf')
        for node in self.graph.nodes():
            dist = euclidean_distance(node, point)
            if dist < min_dist:
                min_dist = dist
                nearest = node
        if nearest is None:
            raise ValueError("No node found in the graph.")
        return nearest

    def _ensure_node(self, point: Tuple[float, float]) -> Tuple[float, float]:
        """
        Ensure that the point is a node in the graph. If not, return the nearest node.

        :param point: Tuple (x, y) representing the point.
        :return: A node (x, y) present in the graph.
        """
        if point not in self.graph.nodes():
            nearest = self.find_nearest_node(point)
            logging.info(f"Point {point} not found among nodes. Using nearest node: {nearest}")
            return nearest
        return point

    def compute_dijkstra(self, source_point: Tuple[float, float], target_point: Tuple[float, float]) -> Tuple[Optional[List[Tuple[float, float]]], Optional[float]]:
        """
        Compute the shortest path using Dijkstra's algorithm.
        If the source or target points are not present, they are replaced with the nearest nodes.

        :param source_point: Tuple (x, y) for the source.
        :param target_point: Tuple (x, y) for the target.
        :return: A tuple containing the path (list of nodes) and the execution time in seconds.
        """
        source = self._ensure_node(source_point)
        target = self._ensure_node(target_point)

        start_time = time.perf_counter()
        try:
            path = nx.dijkstra_path(self.graph, source, target, weight="weight")
            logging.info(f"Dijkstra path found: {path}")
        except nx.NetworkXNoPath:
            logging.error("No path found using Dijkstra's algorithm.")
            return None, None
        exec_time = time.perf_counter() - start_time
        logging.info(f"Dijkstra execution time: {exec_time:.6f} seconds")
        return path, exec_time

    def compute_astar(self, source_point: Tuple[float, float], target_point: Tuple[float, float]) -> Tuple[Optional[List[Tuple[float, float]]], Optional[float]]:
        """
        Compute the shortest path using the A* algorithm with Euclidean distance as the heuristic.
        If the source or target points are not present, they are replaced with the nearest nodes.

        :param source_point: Tuple (x, y) for the source.
        :param target_point: Tuple (x, y) for the target.
        :return: A tuple containing the path (list of nodes) and the execution time in seconds.
        """
        source = self._ensure_node(source_point)
        target = self._ensure_node(target_point)

        # Define the heuristic function explicitly
        def heuristic(u: Tuple[float, float], v: Tuple[float, float]) -> float:
            return euclidean_distance(u, v)

        start_time = time.perf_counter()
        try:
            path = nx.astar_path(self.graph, source, target, heuristic=heuristic, weight="weight")
            logging.info(f"A* path found: {path}")
        except nx.NetworkXNoPath:
            logging.error("No path found using A* algorithm.")
            return None, None
        exec_time = time.perf_counter() - start_time
        logging.info(f"A* execution time: {exec_time:.6f} seconds")
        return path, exec_time

    def plot_path(self, path: Optional[List[Tuple[float, float]]], title: str, filename: str, path_color: str = "red") -> None:
        """
        Plot the road network and highlight the specified path, then save the plot.

        :param path: List of nodes representing the path.
        :param title: Title for the plot.
        :param filename: File name to save the plot.
        :param path_color: Color to highlight the path.
        """
        pos = {node: node for node in self.graph.nodes()}
        plt.figure(figsize=(10, 8))
        # Draw network edges in light gray
        nx.draw_networkx_edges(self.graph, pos, edge_color='lightgray', width=1)
        nx.draw_networkx_nodes(self.graph, pos, node_size=5, node_color='lightgray')
        if path:
            path_edges = list(zip(path, path[1:]))
            nx.draw_networkx_edges(self.graph, pos, edgelist=path_edges, edge_color=path_color, width=3)
            nx.draw_networkx_nodes(self.graph, pos, nodelist=path, node_color=path_color, node_size=20)
        plt.title(title)
        plt.savefig(os.path.join(os.getenv("CURRENT_OUT_PATH"), filename))
        logging.info(f"Graph image saved as '{filename}'.")
        plt.close()  # Free the resources used by the plot
