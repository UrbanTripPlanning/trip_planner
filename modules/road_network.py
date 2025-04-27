import os
import logging
import networkx as nx
import geopandas as gpd
import pandas as pd
import torch

from datetime import datetime
from typing import Tuple, Optional

from modules.road_data_processor import RoadDataProcessor
from modules.inferer import prepare_feature_vector, predict_travel_time, load_model_and_scalers
from utils import euclidean_distance

MIN_SPEED_KMH = 0.1  # km/h floor to avoid division-by-zero
FEATURE_KEYS = ["length", "hour", "lane", "rain", "avgSpeed"]


class RoadNetwork:
    """
    Manages a road network:
      - async init via RoadDataProcessor
      - optionally runs GNN inference to produce per-edge travel times
      - builds a NetworkX graph with either formula-based or model-based travel times
    """
    DATA_PATH: str = os.getenv("DATA_PATH", ".")

    def __init__(self, use_gnn_weights: bool = False) -> None:
        self.use_gnn_weights = use_gnn_weights
        self.processor: Optional[RoadDataProcessor] = None
        self.gdf: Optional[gpd.GeoDataFrame] = None
        self.graph: Optional[nx.Graph] = None

        # Initialize GNN model and scalers if requested
        if self.use_gnn_weights:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            self.model, self.feature_scaler, self.target_scaler = (
                load_model_and_scalers("./GCN/models", self.device)
            )
            logging.info("GNN model and scalers loaded.")

        logging.info("RoadNetwork instance created.")

    async def async_init(
            self,
            start_time: Optional[datetime] = None,
            end_time: Optional[datetime] = None
    ) -> None:
        # Load DB data and build GeoDataFrame
        self.processor = await RoadDataProcessor.async_init(start_time, end_time)
        await self.processor.load_all_data()
        self.gdf = self.processor.build_network_geodataframe()

        # Build NetworkX graph
        self.build_graph()
        logging.info("RoadNetwork async_init complete.")

    def build_graph(self) -> None:
        if self.gdf is None:
            raise RuntimeError("GeoDataFrame is not initialized.")
        self.graph = nx.Graph()

        for _, row in self.gdf.iterrows():
            geom = row.geometry
            if geom.geom_type != "LineString":
                continue
            u, v = tuple(geom.coords[0]), tuple(geom.coords[-1])
            length = row.get("length", geom.length)
            lane = int(row.get("lane", 1))
            rain = float(row.get("rain", 0))
            hour = int(row.get("hour", 0))
            avgSpeed = float(row.get("avgSpeed", 30.0))
            weather_condition = row.get("weather_condition", "clear")

            # Predict travel time (GNN)
            predicted_car_time = None
            if self.use_gnn_weights:
                feature_vector = prepare_feature_vector(length, hour, lane, rain, avgSpeed)
                predicted_car_time = predict_travel_time(
                    self.model, self.feature_scaler, self.target_scaler, feature_vector, self.device
                )

            # Formula-based travel time
            default_speed = avgSpeed
            rate = self._get_penalty_rate(lane, hour) if rain else 0.0
            car_speed = default_speed * (1 + rate)
            safe_speed = max(car_speed, MIN_SPEED_KMH)
            car_time = length / (safe_speed / 3.6)

            # Add edge to graph
            self.graph.add_edge(
                u, v,
                length=length,
                car_travel_time=car_time,
                predicted_car_travel_time=predicted_car_time,
                hour=hour,
                lane=lane,
                rain=rain,
                avgSpeed=avgSpeed,
                weather_condition=weather_condition,
            )

    def _load_penalties(self) -> pd.DataFrame:
        path = os.path.join(self.DATA_PATH, "penalties.csv")
        self._penalties_df = pd.read_csv(path)
        return self._penalties_df

    def _get_penalty_rate(self, lane: int, hour: int) -> float:
        df = self._load_penalties()
        try:
            rate = df.loc[df["hour"] == hour, str(lane)].iloc[0]
        except IndexError:
            raise ValueError(f"No penalty rate for hour={hour}, lane={lane}")
        return rate / 100

    def _find_nearest_node(self, point: Tuple[float, float]) -> Tuple[float, float]:
        nearest, min_dist = None, float("inf")
        for node in self.graph.nodes():
            d = euclidean_distance(node, point)
            if d < min_dist:
                min_dist, nearest = d, node
        if nearest is None:
            raise ValueError("No node found in the graph.")
        return nearest

    def ensure_node(self, point: Tuple[float, float]) -> Tuple[float, float]:
        return point if point in self.graph else self._find_nearest_node(point)
