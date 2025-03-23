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
    # Environment variables for the respective collections.
    ROAD_COLLECTION = os.getenv('ROAD_COLLECTION')
    TRAFFIC_COLLECTION = os.getenv('TRAFFIC_COLLECTION')
    WEATHER_COLLECTION = os.getenv('WEATHER_COLLECTION')

    def __init__(self):
        """
        Initialize the RoadDataProcessor instance.
        Sets up variables for road, traffic, and weather data, as well as the final GeoDataFrame.
        """
        self.road_data: Optional[List[dict]] = None
        self.traffic_data: Optional[List[dict]] = None
        self.weather_data: Optional[pd.DataFrame] = None
        self.geo_df: Optional[gpd.GeoDataFrame] = None
        logging.info("RoadDataProcessor instance created.")

    @staticmethod
    async def initialize_db() -> None:
        """
        Initialize the database connection using the asynchronous Database module.
        """
        await Database.initialize()
        logging.info("Database connection initialized in RoadDataProcessor.")

    @staticmethod
    async def query_collection(
            collection: str,
            query: Optional[Dict[str, Any]] = None,
            method: str = "find",
            pipeline: Optional[List[Dict[str, Any]]] = None
    ) -> List[dict]:
        """
        Execute a query on a given collection using a specified method.

        :param collection: Name of the MongoDB collection.
        :param query: Query dictionary; defaults to empty dict if None.
        :param method: Query method to use ("find", "find_one", or "aggregate").
        :param pipeline: Aggregation pipeline if method is "aggregate".
        :return: List of documents returned from the query.
        """
        if query is None:
            query = {}

        if method == "find":
            documents = await Database.find(collection, query)
        elif method == "find_one":
            document = await Database.find_one(collection, query)
            documents = [document] if document else []
        elif method == "aggregate":
            if pipeline is None:
                pipeline = []
            documents = await Database.aggregate(collection, pipeline)
        else:
            raise ValueError(f"Unsupported query method: {method}")

        logging.info(f"Retrieved {len(documents)} documents from collection '{collection}' using method '{method}'.")
        return documents

    async def load_all_data(self, start_time=None, end_time=None) -> None:
        """
        Load data asynchronously from the 'road', 'weather', and 'traffic' collections.
        """
        logging.info("Starting to load all data (road, traffic, weather).")
        self.road_data = await self._query_road_data()
        logging.info(f"Loaded {len(self.road_data)} road documents.")
        self.weather_data = self.process_weather_data(start_time, end_time)
        logging.info(f"Loaded weather data.")
        if self.weather_data['rain'] == 0:
            self.traffic_data = await self._query_traffic_data(start_time, end_time)
            logging.info(f"Loaded {len(self.traffic_data)} traffic documents.")
        else:  # todo: query to other 5t traffic data
            self.traffic_data = await self._query_traffic_data(start_time, end_time)
            logging.info(f"Loaded {len(self.traffic_data)} traffic documents.")

    async def _query_road_data(self):
        """
        Query road data from the designated ROAD_COLLECTION.
        """
        logging.info("Querying road data...")
        documents = await self.query_collection(self.ROAD_COLLECTION, {})
        logging.info(f"Queried road data: {len(documents)} documents found.")
        return documents

    async def _query_traffic_data(self, start_time=None, end_time=None):
        """
        Query traffic data using an aggregation pipeline based on a specific hour.
        The hour is determined from start_time, end_time, or the current time.
        """
        logging.info("Querying traffic data...")
        if start_time is not None:
            hour = get_hour(start_time)
            logging.info(f"Using start_time for traffic query: computed hour = {hour}")
        elif end_time is not None:
            hour = get_hour(end_time)
            logging.info(f"Using end_time for traffic query: computed hour = {hour}")
        else:
            hour = get_hour()
            logging.info(f"No specific time provided; using current hour for traffic query: computed hour = {hour}")

        pipeline = [
            {"$match": {"hour": hour}},
            {"$group": {
                "_id": "$road_id",
                "avgSpeed": {"$avg": "$avg_speed"}
            }}
        ]
        documents = await self.query_collection(collection=self.TRAFFIC_COLLECTION, method='aggregate',
                                                pipeline=pipeline)
        logging.info(f"Queried traffic data: {len(documents)} documents found.")
        return documents

    @staticmethod
    def process_weather_data(start_time, end_time) -> pd.DataFrame:
        weather_file_path = os.path.join(os.getenv('DATA_PATH'), "weather.csv")
        df = pd.read_csv(weather_file_path)
        df.set_index('datetime', inplace=True)
        df.index = pd.to_datetime(df.index)

        if start_time is not None:
            time = start_time
        elif end_time is not None:
            time = end_time
        elif start_time is None and end_time is None:
            time = datetime.now()

        pos = df.index.get_indexer([time], method='nearest')[0]
        closest_row = df.iloc[pos]

        return closest_row

    @staticmethod
    def process_road_data(documents: List[dict]) -> gpd.GeoDataFrame:
        """
        Process road collection documents into a GeoDataFrame.
        Converts GeoJSON geometry to shapely objects.
        """
        if not documents:
            logging.warning("No road documents to process.")
            return gpd.GeoDataFrame()
        df = pd.DataFrame(documents)
        logging.info("Converting road geometry from GeoJSON to shapely objects.")
        df['geometry'] = df['geometry'].apply(lambda geom: shape(geom))
        geo_df = gpd.GeoDataFrame(df, geometry='geometry')
        logging.info("Processed road data into a GeoDataFrame.")
        return geo_df

    @staticmethod
    def process_traffic_data(documents: List[dict]) -> gpd.GeoDataFrame:
        """
        Process traffic collection documents into a GeoDataFrame.
        Renames the '_id' column to 'road_id' for merging purposes.
        """
        if not documents:
            logging.warning("No traffic documents to process.")
            return gpd.GeoDataFrame()
        traffic_df = pd.DataFrame(documents)
        logging.info("Renaming '_id' column to 'road_id' in traffic data.")
        traffic_df.rename(columns={"_id": "road_id"}, inplace=True)
        return traffic_df

    def build_network_geodataframe(self) -> Optional[gpd.GeoDataFrame]:
        """
        Process and merge road, traffic, and weather data into a single GeoDataFrame for the network.
        """
        logging.info("Building network GeoDataFrame from road and. traffic data, and weather data.")
        road_gdf = self.process_road_data(self.road_data)
        traffic_df = self.process_traffic_data(self.traffic_data)
        weather_df = self.weather_data

        if 'road_id' in road_gdf.columns:
            if not traffic_df.empty and 'road_id' in traffic_df.columns:
                road_gdf = road_gdf.merge(traffic_df, on="road_id", how="left", suffixes=("", "_traffic"))
                logging.info("Merged traffic data with road GeoDataFrame.")
            else:
                logging.info("Traffic data is empty or missing 'road_id'; skipping merge.")
        else:
            logging.warning("Common key 'road_id' not found in road data; merge skipped.")

        if not weather_df.empty:
            road_gdf['weather_condition'] = weather_df['weather_condition']
        else:
            logging.warning("Empy weather data; skipping merge.")

        return road_gdf
