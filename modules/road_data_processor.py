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

    # MongoDB collection name and local data path (from environment)
    GRAPH_COLLECTION: str = os.getenv('GRAPH_COLLECTION')
    DATA_PATH: str = os.getenv('DATA_PATH')

    def __init__(self) -> None:
        # Ensure required environment variables are set
        if not all([self.GRAPH_COLLECTION, self.DATA_PATH]):
            raise EnvironmentError("One or more required environment variables are not set.")

        # Raw documents fetched from the database
        self.network_docs: List[dict] = []
        # GeoDataFrame built from the documents
        self.traffic_gdf: Optional[gpd.GeoDataFrame] = None

        # Time window parameters
        self.start_time: Optional[datetime] = None
        self.end_time: Optional[datetime] = None
        # The timestamp against which we compute hour/week/month and rain
        self.reference_time: datetime = datetime.now()
        self.hour: int = 0
        self.weekday: int = 1
        self.month: int = 1

        # Flag indicating whether it's raining (True) or clear (False)
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
          1. Initialize database connection.
          2. Set reference_time from start_time/end_time.
          3. Compute temporal fields and rain_flag.
        """
        instance = cls()

        # Initialize the MongoDB connection
        await Database.initialize()
        logging.info("Database connection initialized.")

        # Assign time window if provided
        instance.start_time = start_time
        instance.end_time = end_time

        # Choose a reference timestamp
        if start_time is not None:
            instance.reference_time = start_time
        elif end_time is not None:
            instance.reference_time = end_time

        # Extract hour, weekday, month from reference_time
        hour, weekday, month = get_time_info(instance.reference_time)
        instance.hour = hour
        instance.weekday = weekday
        instance.month = month

        # Determine rain_flag by inspecting the nearest weather record
        instance.rain_flag = bool(instance.compute_rain_flag())

        logging.info(f"Reference time set to {instance.reference_time!r} "
                     f"(hour={instance.hour}, weekday={instance.weekday}, month={instance.month}, "
                     f"rain_flag={instance.rain_flag})")
        return instance

    def compute_rain_flag(self) -> int:
        """
        Load weather data and return the rain flag (0 or 1)
        for the record closest to self.reference_time.
        """
        weather_path = os.path.join(self.DATA_PATH, "weather.csv")
        if not os.path.isfile(weather_path):
            logging.error(f"Missing weather file: {weather_path}")
            raise FileNotFoundError(f"Weather file not found at {weather_path}")

        # Read the CSV with datetime index
        df = pd.read_csv(weather_path, index_col='datetime', parse_dates=True)

        # Find the index of the timestamp nearest to reference_time
        idx = df.index.get_indexer([self.reference_time], method='nearest')[0]
        if idx < 0 or idx >= len(df):
            raise IndexError(f"No weather record near {self.reference_time!r}")

        rain_val = df.iloc[idx]['rain']
        rain_int = int(rain_val)
        if rain_int not in (0, 1):
            raise ValueError(f"Unexpected rain flag value: {rain_val}")

        return rain_int

    async def query_traffic_data(self) -> List[dict]:
        """
        Query traffic data from MongoDB, filtering by hour, weekday, and month.
        Selects either clear or rain speed/time fields based on rain_flag.
        """
        speed_field = "$speed_clear" if not self.rain_flag else "$speed_rain"
        time_field = "$time_clear" if not self.rain_flag else "$time_rain"

        pipeline = [
            {"$match": {
                "hour": self.hour,
                "week": self.weekday,
                "month": self.month,
            }},
            {"$project": {
                "_id": 0,
                "road_id": 1,
                "tail": 1,
                "head": 1,
                "speed": speed_field,
                "length": 1,
                "time": time_field,
                "geometry": 1,
            }}
        ]

        docs = await Database.query(
            collection=self.GRAPH_COLLECTION,
            method="aggregate",
            pipeline=pipeline
        )
        logging.info(f"Query returned {len(docs)} traffic records.")
        return docs

    @staticmethod
    def build_traffic_geodataframe(docs: List[dict]) -> gpd.GeoDataFrame:
        """
        Convert raw traffic documents into a GeoDataFrame:
          - Parses each 'geometry' dict into a Shapely geometry.
          - Ensures the same schema when empty.
          - Assigns a defined CRS for spatial operations.
        """
        # Define expected columns
        cols = ['road_id', 'tail', 'head', 'speed', 'length', 'time', 'geometry']

        if not docs:
            logging.warning("No traffic data found; returning empty GeoDataFrame.")
            # Build an empty GeoDataFrame with correct schema and CRS
            empty = {col: [] for col in cols}
            return gpd.GeoDataFrame(empty, geometry='geometry', crs="EPSG:4326")

        # Load into a pandas DataFrame
        df = pd.DataFrame(docs)

        # Convert GeoJSON-like dicts to Shapely geometries
        df['geometry'] = df['geometry'].apply(shape)

        # Build GeoDataFrame with WGS84 CRS (adjust if needed)
        gdf = gpd.GeoDataFrame(df, geometry='geometry', crs="EPSG:4326")
        return gdf

    async def load_all_data(self) -> None:
        """
        Main entry point to load and prepare traffic data:
          1. Query MongoDB for traffic docs.
          2. Build the GeoDataFrame for downstream graph construction.
        """
        self.network_docs = await self.query_traffic_data()
        self.traffic_gdf = self.build_traffic_geodataframe(self.network_docs)
        logging.info(f"Loaded {len(self.network_docs)} traffic records into GeoDataFrame.")
