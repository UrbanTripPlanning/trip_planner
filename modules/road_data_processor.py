import logging
import os
import pandas as pd
import geopandas as gpd
from datetime import datetime
from shapely.geometry import shape
from typing import Optional, List

from modules.db_manager import Database
from utils import get_time_info


class RoadDataProcessor:
    """
    Asynchronous data loader and processor for traffic network and weather data.
    """

    # Environment variable names for MongoDB collections and data path
    GRAPH_COLLECTION: str = os.getenv('GRAPH_COLLECTION')
    DATA_PATH: str = os.getenv('DATA_PATH')

    def __init__(self) -> None:
        if not all([self.GRAPH_COLLECTION, self.DATA_PATH]):
            raise EnvironmentError("One or more required environment variables are not set.")

        # Raw and processed data containers
        self.network_docs: List[dict] = []
        self.traffic_gdf: Optional[pd.DataFrame] = None

        # Time-related parameters
        self.start_time: Optional[datetime] = None
        self.end_time: Optional[datetime] = None
        self.reference_time: datetime = datetime.now()
        self.hour: int = 0
        self.weekday: int = 1
        self.month: int = 1
        self.rain_flag: bool = False

        logging.info("RoadDataProcessor instance created (not yet initialized).")

    @classmethod
    async def async_init(
        cls,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None
    ) -> "RoadDataProcessor":
        """
        Asynchronous factory method:
        1. Initializes database connection.
        2. Sets reference_time based on input.
        3. Extracts temporal parameters and computes rain condition.
        """
        instance = cls()
        await Database.initialize()
        logging.info("Database connection initialized.")

        instance.start_time = start_time
        instance.end_time = end_time

        # Choose one timestamp as the reference point
        if start_time is not None:
            instance.reference_time = start_time
        elif end_time is not None:
            instance.reference_time = end_time

        # Compute temporal fields
        hour, weekday, month = get_time_info(instance.reference_time)
        instance.hour = hour
        instance.weekday = weekday
        instance.month = month

        # Compute rain flag
        instance.rain_flag = instance.compute_rain_flag()

        logging.info(f"Reference time set to {instance.reference_time!r}")
        return instance

    def compute_rain_flag(self) -> int:
        """
        Loads weather data and determines the rain flag (0 or 1)
        for the timestamp closest to `reference_time`.
        """
        path = os.path.join(self.DATA_PATH, "weather.csv")
        if not os.path.isfile(path):
            logging.error(f"Missing weather file: {path}")
            raise FileNotFoundError(f"Weather file not found at {path}")

        df = pd.read_csv(path, index_col='datetime', parse_dates=True)
        idx = df.index.get_indexer([self.reference_time], method='nearest')[0]
        if idx < 0 or idx >= len(df):
            raise IndexError(f"No weather record near {self.reference_time!r}")
        rain = int(df.iloc[idx]['rain'])
        if rain not in (0, 1):
            raise ValueError(f"Unexpected rain flag value: {rain}")
        return rain

    async def query_traffic_data(self) -> List[dict]:
        """
        Queries traffic data from the graph collection filtered by hour, week, and month.
        Chooses fields based on rain condition.
        """
        speed = "$speed_clear" if not self.rain_flag else "$speed_rain"
        time = "$time_clear" if not self.rain_flag else "$time_rain"
        pipeline = [
            {"$match": {
                "hour": self.hour,
                "week": self.weekday,
                "month": self.month,
            }},
            {"$project": {
                "_id": 0,
                "road_id": 1,
                "speed": speed,
                "length": 1,
                "time": time,
                "geometry": 1,
            }}
        ]
        return await Database.query(
            collection=self.GRAPH_COLLECTION,
            method="aggregate",
            pipeline=pipeline
        )

    @staticmethod
    def build_traffic_geodataframe(docs: List[dict]) -> gpd.GeoDataFrame:
        """
        Converts raw traffic documents into a GeoDataFrame.
        Parses geometries using Shapely.
        """
        if not docs:
            logging.warning("No traffic data found; returning empty GeoDataFrame.")
            return gpd.GeoDataFrame(columns=['road_id', 'speed', 'length', 'time', 'road_type'], geometry=[])

        df = pd.DataFrame(docs)
        df['geometry'] = df['geometry'].apply(shape)
        return gpd.GeoDataFrame(df, geometry='geometry')

    async def load_all_data(self) -> None:
        """
        Main entry point to load and process traffic data.
        Queries the network and constructs a GeoDataFrame.
        """
        self.network_docs = await self.query_traffic_data()
        self.traffic_gdf = self.build_traffic_geodataframe(self.network_docs)
