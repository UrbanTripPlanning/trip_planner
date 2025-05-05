import os
import torch
import logging
import pandas as pd
import networkx as nx
import geopandas as gpd
from datetime import datetime
from typing import Tuple, Optional

from utils import euclidean_distance
from modules.road_data_processor import RoadDataProcessor
from modules.edge_weight_predictor import EdgeWeightPredictor


class RoadNetwork:
    """
    Manages a road network:
    - Loads traffic data via RoadDataProcessor.
    - Optionally applies GCN/STGCN to predict edge weights.
    - Builds a NetworkX graph with road segments and travel features.
    """
    DATA_PATH: str = os.getenv("DATA_PATH")

    def __init__(self, gnn_model: str = "") -> None:
        """
        Initialize RoadNetwork with optional GNN-based edge predictor.

        :param gnn_model: Model type ("GCN", "STGCN", or empty string to disable).
        """
        self.gnn_model = gnn_model
        self.processor: Optional[RoadDataProcessor] = None
        self.traffic_gdf: Optional[gpd.GeoDataFrame] = None
        self.graph: Optional[nx.Graph] = None
        self.predictor: Optional[EdgeWeightPredictor] = None

        if self.gnn_model:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self.predictor = EdgeWeightPredictor(self.gnn_model, self.device)
            logging.info(f"{self.gnn_model} model and scalers loaded.")

        logging.info("RoadNetwork instance created.")

    async def async_init(
        self,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None
    ) -> None:
        """
        Asynchronously initialize the network:
        - Load traffic data via RoadDataProcessor.
        - Build the underlying NetworkX graph.
        """
        self.processor = await RoadDataProcessor.async_init(start_time, end_time)
        await self.processor.load_all_data()
        self.traffic_gdf = self.processor.traffic_gdf
        self.build_graph()
        logging.info("RoadNetwork async initialization complete.")

    def build_graph(self) -> None:
        """
        Constructs the NetworkX graph from the GeoDataFrame.
        Uses predicted or rule-based weights depending on the configuration.
        """
        if self.traffic_gdf is None:
            raise RuntimeError("Traffic GeoDataFrame is not initialized.")

        self.graph = nx.Graph()

        for _, row in self.traffic_gdf.iterrows():
            geom = row.geometry
            if geom.geom_type != "LineString":
                continue

            u, v = tuple(geom.coords[0]), tuple(geom.coords[-1])
            speed = float(row.get("speed", 1))
            length = float(row.get("length", 1))
            time = float(row.get("time", 1))

            if self.gnn_model and self.predictor:
                weight = self.predictor.predict(
                    speed=speed,
                    length=length,
                    time=time,
                    u=u, v=v
                )
            else:
                weight = None

            self.graph.add_edge(
                u, v,
                speed=speed,
                length=length,
                time=time,
                weight=weight
            )

    def _get_nearest_node(self, point: Tuple[float, float]) -> Tuple[float, float]:
        """
        Finds the nearest graph node to a given point using Euclidean distance.

        :param point: Tuple (lon, lat) to search from.
        :return: The nearest node in the graph.
        """
        if self.graph is None or not self.graph.nodes:
            raise RuntimeError("Graph is empty. Cannot find nearest node.")

        nearest, min_dist = None, float("inf")
        for node in self.graph.nodes():
            d = euclidean_distance(node, point)
            if d < min_dist:
                min_dist, nearest = d, node
        return nearest

    def snap_to_graph(self, point: Tuple[float, float]) -> Tuple[float, float]:
        """
        Ensures a point belongs to the graph. If not, snaps to the nearest node.

        :param point: Tuple (lon, lat)
        :return: Valid node in the graph.
        """
        if self.graph is None:
            raise RuntimeError("Graph not initialized.")
        return point if point in self.graph else self._get_nearest_node(point)
