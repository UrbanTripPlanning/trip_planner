import logging
import os
import pandas as pd
import geopandas as gpd
from datetime import datetime
from shapely.geometry import shape
from typing import Optional, List, Dict, Any

from modules.db_manager import Database
from utils import get_hour


class RoadDataProcessor:
    """
    Asynchronous data loader and processor for road, traffic, and weather data.
    """

    # Environment variable names for MongoDB collections and data path
    ROAD_COLLECTION: str = os.getenv('ROAD_COLLECTION')
    TRAFFIC_COLLECTION: str = os.getenv('TRAFFIC_COLLECTION')
    DATA_PATH: str = os.getenv('DATA_PATH')

    def __init__(self) -> None:
        # Raw document storage
        self.road_docs: List[dict] = []
        self.traffic_docs: List[dict] = []

        # Processed data containers
        self.road_gdf: Optional[gpd.GeoDataFrame] = None
        self.traffic_df: Optional[pd.DataFrame] = None
        self.weather_ser: Optional[pd.Series] = None

        # Time-related parameters
        self.start_time: Optional[datetime] = None
        self.end_time: Optional[datetime] = None
        self.reference_time: datetime = datetime.now()
        self.hour: int = 0

        # Internal cache for the weather CSV
        self._weather_df: Optional[pd.DataFrame] = None

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
        2. Determine reference_time and hour to use for queries.
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

        # Compute the hour for traffic aggregation
        instance.hour = get_hour(instance.reference_time)
        logging.info(f"Reference time set to {instance.reference_time!r}, hour = {instance.hour}")

        return instance

    @staticmethod
    async def query_collection(
            collection: str,
            query: Optional[Dict[str, Any]] = None,
            method: str = "find",
            pipeline: Optional[List[Dict[str, Any]]] = None
    ) -> List[dict]:
        """
        Unified async interface to MongoDB:
        - find, find_one, or aggregate.
        Returns a list of documents (possibly empty).
        """
        if query is None:
            query = {}
        if method == "find":
            docs = await Database.find(collection, query)
        elif method == "find_one":
            doc = await Database.find_one(collection, query)
            docs = [doc] if doc else []
        elif method == "aggregate":
            docs = await Database.aggregate(collection, pipeline or [])
        else:
            raise ValueError(f"Unsupported query method: {method}")
        logging.info(f"{method.upper()} returned {len(docs)} documents from {collection}")
        return docs

    async def load_all_data(self) -> None:
        """
        Master loader that:
        1. Queries and processes road data.
        2. Processes weather data.
        3. Queries and processes traffic data.
        """
        # 1. Road data
        self.road_docs = await self.query_collection(self.ROAD_COLLECTION)
        self.road_gdf = self.process_road_data(self.road_docs)

        # 2. Weather data
        self.weather_ser = self.process_weather_data()

        # 3. Traffic data
        self.traffic_docs = await self._query_traffic_data()
        self.traffic_df = self.process_traffic_data(self.traffic_docs)

    async def _query_traffic_data(self) -> List[dict]:
        """
        Aggregate traffic data by 'hour' field and return raw documents.
        """
        pipeline = [
            {"$match": {"hour": self.hour}},
            {"$group": {
                "_id": "$road_id",
                "avgSpeed": {"$avg": "$avg_speed"}
            }}
        ]
        return await self.query_collection(
            collection=self.TRAFFIC_COLLECTION,
            method="aggregate",
            pipeline=pipeline
        )

    @staticmethod
    def process_road_data(docs: List[dict]) -> gpd.GeoDataFrame:
        """
        Convert raw road documents to a GeoDataFrame,
        parsing GeoJSON geometries into Shapely objects.
        """
        if not docs:
            logging.warning("No road documents provided; returning empty GeoDataFrame.")
            return gpd.GeoDataFrame()
        df = pd.DataFrame(docs)
        df['geometry'] = df['geometry'].apply(lambda g: shape(g))
        return gpd.GeoDataFrame(df, geometry='geometry')

    @staticmethod
    def process_traffic_data(docs: List[dict]) -> pd.DataFrame:
        """
        Convert raw traffic documents to a DataFrame and
        rename '_id' to 'road_id' for merging.
        """
        if not docs:
            logging.warning("No traffic documents provided; returning empty DataFrame.")
            return pd.DataFrame(columns=['road_id', 'avgSpeed'])
        df = pd.DataFrame(docs).rename(columns={'_id': 'road_id'})
        return df

    def _load_weather_df(self) -> pd.DataFrame:
        """
        Load and cache the weather CSV as a DataFrame indexed by datetime.
        """
        if self._weather_df is None:
            path = os.path.join(self.DATA_PATH, "weather.csv")
            if not os.path.isfile(path):
                raise FileNotFoundError(f"Weather file not found at {path}")
            df = pd.read_csv(path, index_col='datetime', parse_dates=True)
            self._weather_df = df
        return self._weather_df

    def process_weather_data(self) -> pd.Series:
        """
        Return the weather row closest to reference_time as a Series.
        """
        df = self._load_weather_df()
        idx = df.index.get_indexer([self.reference_time], method='nearest')[0]
        if idx < 0 or idx >= len(df):
            raise IndexError(f"No weather record near {self.reference_time!r}")
        return df.iloc[idx]

    def build_network_geodataframe(self) -> gpd.GeoDataFrame:
        """
        Merge road GeoDataFrame, traffic DataFrame, and weather Series
        into a single GeoDataFrame for analysis or export.
        """
        if self.road_gdf is None:
            raise RuntimeError("Road GeoDataFrame not initialized. Call load_all_data() first.")

        # Start from road GeoDataFrame
        network_gdf = self.road_gdf.copy()

        # Merge traffic data if available
        if self.traffic_df is not None and not self.traffic_df.empty:
            network_gdf = network_gdf.merge(self.traffic_df,
                                            on="road_id",
                                            how="left")

        # Add weather columns
        if self.weather_ser is not None:
            network_gdf['weather_condition'] = self.weather_ser['weather_condition']
            network_gdf['rain'] = self.weather_ser['rain']

        # Add hour column
        network_gdf['hour'] = self.hour

        return network_gdf
