import datetime
import logging
from typing import Tuple, Optional
import networkx as nx
import geopandas as gpd
from modules.road_data_processor import RoadDataProcessor
from utils import euclidean_distance


class RoadNetwork:
    """
    Class to manage a road network loaded from a MongoDB collection via the asynchronous Database module.

    Handles:
      - Asynchronous initialization of the database connection.
      - Creating a GeoDataFrame from road data.
      - Building a NetworkX graph from road geometries.
    """

    def __init__(self) -> None:
        """
        Initialize the RoadNetwork instance.
        """
        self.gdf: Optional[gpd.GeoDataFrame] = None
        self.graph: Optional[nx.Graph] = None
        self.processor = RoadDataProcessor()
        logging.info("RoadNetwork instance created. Processor initialized.")

    async def async_init(
            self,
            start_time: Optional[datetime.datetime] = None,
            end_time: Optional[datetime.datetime] = None
    ) -> None:
        """
        Asynchronously initialize the database connection, load road data,
        and build the NetworkX graph.
        """
        logging.info("Starting asynchronous initialization of RoadNetwork.")

        # Initialize the asynchronous DB connection.
        await self.processor.initialize_db()
        logging.info("Database connection initialized.")

        # Query and load data from all collections (only road data is available).
        await self.processor.load_all_data(start_time, end_time)
        logging.info("Road data loaded from database.")

        # Process and merge the data into one GeoDataFrame (only road data used).
        self.gdf = self.processor.build_network_geodataframe()
        if self.gdf is not None:
            logging.info(f"GeoDataFrame built with {len(self.gdf)} records.")
        else:
            logging.warning("GeoDataFrame is empty after processing.")

        # Build the graph from the GeoDataFrame.
        self.build_graph()
        logging.info("RoadNetwork asynchronous initialization complete.")

    def build_graph(self) -> None:
        """
        Build a NetworkX graph from the GeoDataFrame.
        Iterates through each road record and adds an edge using its geometry.
        """
        self.graph = nx.Graph()
        try:
            for index, row in self.gdf.iterrows():
                geom = row['geometry']
                # Process only LineString geometries.
                if geom.geom_type != 'LineString':
                    continue
                start = tuple(geom.coords[0])
                end = tuple(geom.coords[-1])
                # Use provided 'length' if available; otherwise, compute from geometry.
                length = row.get('length', geom.length)
                # Road classification based on number of lane
                lane: int = row.get('lane', 1)
                # Check if it is raining
                rain: float = row.get('rain', 0)
                # Calculate penalty based on the number of lane
                penalty = self._compute_penalty(lane)
                # Calculate car travel time if average speed is provided.
                default_car_avg_speed = row.get('avgSpeed', 0)
                car_avg_speed = default_car_avg_speed
                if rain:
                    car_avg_speed = default_car_avg_speed - penalty
                car_travel_time = length / (car_avg_speed / 3.6) if car_avg_speed > 0 else 0
                # Get weather condition
                weather_condition = row.get('weather_condition', 'empty')
                self.graph.add_edge(
                    start,
                    end,
                    length=length,
                    car_travel_time=car_travel_time,
                    weather_condition=weather_condition
                )
                logging.debug(
                    f"Edge aggiunto da {start} a {end}: length={length}, "
                    f"car_travel_time={car_travel_time}, weather_condition={weather_condition}"
                )
            logging.info(
                f"Graph built with {self.graph.number_of_nodes()} nodes and {self.graph.number_of_edges()} edges.")
        except Exception as e:
            logging.error(f"Error building graph: {e}")
            raise

    @staticmethod
    def _compute_penalty(lane: int) -> int:
        """
        Calcola una penalità basata sul numero di corsie.
        :param lane: Numero di corsie nel segmento.
        :return: Valore di penalità.
        """
        if lane == 1:
            return 8
        elif lane == 2:
            return 12
        else:
            return 20

    def _find_nearest_node(self, point: Tuple[float, float]) -> Tuple[float, float]:
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
            logging.error("No node found in the graph for the given point.")
            raise ValueError("No node found in the graph.")
        logging.debug(f"Nearest node to {point} is {nearest} with distance {min_dist}")
        return nearest

    def ensure_node(self, point: Tuple[float, float]) -> Tuple[float, float]:
        """
        Ensure that the point is a node in the graph. If not, return the nearest node.

        :param point: Tuple (x, y) representing the point.
        :return: A node (x, y) present in the graph.
        """
        if point not in self.graph.nodes():
            nearest = self._find_nearest_node(point)
            logging.info(f"Point {point} not found among nodes. Using nearest node: {nearest}")
            return nearest
        logging.debug(f"Point {point} exists in the graph.")
        return point
