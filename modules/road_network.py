import os
import torch
import logging
import networkx as nx
import geopandas as gpd
from datetime import datetime
from typing import Tuple, Optional

from utils import euclidean_distance
from modules.road_data_processor import RoadDataProcessor
from GCN.inference import EdgeWeightPredictor
from STGCN.inference import LSTMEdgeWeightPredictor


class RoadNetwork:
    """
    Manages a directed road network:
      - Loads traffic data via RoadDataProcessor.
      - Optionally applies a GNN to predict or adjust edge weights.
      - Builds a NetworkX DiGraph with road segments and travel attributes.
    """
    DATA_PATH: str = os.getenv("DATA_PATH")

    def __init__(self, gnn_model: str = "") -> None:
        """
        Initialize RoadNetwork with optional GNN-based edge predictor.

        :param gnn_model: Model type ("GCN", "STGCN", or empty to disable).
        """
        self.gnn_model = gnn_model
        self.processor: Optional[RoadDataProcessor] = None
        self.traffic_gdf: Optional[gpd.GeoDataFrame] = None
        self.graph: Optional[nx.DiGraph] = None
        self.predictor: Optional[EdgeWeightPredictor] = None

        # Load GNN edge-weight predictor if requested
        if self.gnn_model== "GCN":
            print(f"cuda: {torch.cuda.is_available()}")
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self.predictor = EdgeWeightPredictor(model_path="./GCN/models/edge_autoencoder.pt", device=self.device)
            logging.info(f"{self.gnn_model} model and scalers loaded.")
        
        if self.gnn_model=='STGCN':
            print(f"cudaLSTM: {torch.cuda.is_available()}")
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self.predictor = LSTMEdgeWeightPredictor(model_path="./STGCN/models/lstm_autoencoder.pt", device=self.device)
            logging.info(f"{self.gnn_model} model and scalers loaded.")

        logging.info("RoadNetwork instance created.")

    async def async_init(
        self,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None
    ) -> None:
        """
        Asynchronously initialize the network:
          1. Load traffic data via RoadDataProcessor.
          2. Build the underlying directed graph.
        """
        self.processor = await RoadDataProcessor.async_init(start_time, end_time)
        await self.processor.load_all_data()
        self.traffic_gdf = self.processor.traffic_gdf
        self.build_graph()
        logging.info("RoadNetwork async initialization complete.")

    def build_graph(self) -> None:
        """
        Construct a directed NetworkX graph from the GeoDataFrame.
        Each DB record yields a one-way edge tail→head with its own attributes.
        """
        if self.traffic_gdf is None:
            raise RuntimeError("Traffic GeoDataFrame is not initialized.")

        # Use a directed graph so u→v ≠ v→u
        self.graph = nx.DiGraph()

        for _, row in self.traffic_gdf.iterrows():
            geom = row.geometry
            # Skip non-LineString geometries
            if geom.geom_type != "LineString":
                continue

            # Extract tail/head IDs and their coordinates
            tail_id = int(row["tail"])
            head_id = int(row["head"])
            lon_tail, lat_tail = geom.coords[0]
            lon_head, lat_head = geom.coords[-1]

            # Add each node once, storing its spatial position
            if tail_id not in self.graph:
                self.graph.add_node(tail_id, pos=(lon_tail, lat_tail))
            if head_id not in self.graph:
                self.graph.add_node(head_id, pos=(lon_head, lat_head))

            speed = float(row.get("speed", 1))
            length = float(row.get("length", 1))
            travel_time = float(row.get("time", 1))
            road_id = int(row["road_id"])

            # Add the one-way edge with all attributes
            self.graph.add_edge(
                tail_id,
                head_id,
                road_id=road_id,
                geometry=geom,
                speed=speed,
                length=length,
                time=travel_time,
            )

        # If using GNN-based weight prediction, overwrite or augment edge weights
        if (self.gnn_model=="STGCN" or self.gnn_model=="GCN") and self.predictor:
            weights = self.predictor.infer_edge_weights(self.graph)
            self.predictor.assign_weights_to_graph(self.graph, weights)
            
    def _get_nearest_node(self, point: Tuple[float, float]) -> int:
        """
        Find the nearest graph node to a given point using Euclidean distance.

        :param point: Tuple (lon, lat) to search from.
        :return: The node ID closest to the point.
        """
        if self.graph is None or not self.graph.nodes:
            raise RuntimeError("Graph is empty. Cannot find nearest node.")

        nearest_node = None
        min_dist = float("inf")
        for node, data in self.graph.nodes(data=True):
            node_point = data.get("pos")
            if node_point is None:
                continue
            d = euclidean_distance(node_point, point)
            if d < min_dist:
                min_dist = d
                nearest_node = node

        if nearest_node is None:
            raise RuntimeError(
                f"No graph node has a valid 'pos' attribute; "
                f"cannot snap point {point!r} to graph."
            )

        return nearest_node

    def snap_to_graph(self, point: Tuple[float, float]) -> int:
        """
        Ensure a point corresponds to a graph node. If not, snap to the nearest.

        :param point: Tuple (lon, lat)
        :return: Valid node ID in the graph.
        """
        if self.graph is None:
            raise RuntimeError("Graph not initialized.")

        # If already exactly at a node position, return that node
        for node, data in self.graph.nodes(data=True):
            if data.get("pos") == point:
                return node

        # Otherwise, find and return the nearest node
        return self._get_nearest_node(point)
