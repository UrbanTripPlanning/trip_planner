import logging
import pandas as pd
import geopandas as gpd
from shapely.geometry import shape
from typing import Optional, List, Dict, Any

from modules.interact_with_db import Database


class ComputeWeights:
    """
    Class to create a GeoDataFrame by merging information from multiple collections:
    e.g., road, weather, and traffic.

    Provides modular methods for:
      - Establishing an asynchronous database connection.
      - Querying collections using different query methods.
      - Processing query results into DataFrames/GeoDataFrames.
      - Merging data based on common keys or via spatial joins.

    For now, only road data is available; the algorithm will behave as the shortest path.
    """

    def __init__(self):
        self.road_data: Optional[List[dict]] = None
        self.weather_data: Optional[List[dict]] = None
        self.traffic_data: Optional[List[dict]] = None
        self.geo_df: Optional[gpd.GeoDataFrame] = None

    @staticmethod
    async def initialize_db() -> None:
        """
        Initialize the database connection using the asynchronous Database module.
        """
        await Database.initialize()
        logging.info("Database connection initialized in ComputeWeights.")

    @staticmethod
    async def query_collection(
            collection: str,
            query: Optional[Dict[str, Any]] = None,
            method: str = "find",
            pipeline: Optional[List[Dict[str, Any]]] = None
    ) -> List[dict]:
        """
        Execute a query on a given collection using a specified method.

        :param collection: Name of the collection.
        :param query: MongoDB query as a dictionary (default is {}).
        :param method: Query method to use ("find", "find_one", or "aggregate").
        :param pipeline: List of aggregation pipeline stages if method is "aggregate".
        :return: List of documents matching the query.
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

        logging.info(
            f"Retrieved {len(documents)} documents from collection '{collection}' using method '{method}'."
        )
        return documents

    async def load_all_data(self) -> None:
        """
        Load data asynchronously from the 'road', 'weather', and 'traffic' collections.
        """
        self.road_data = await self.query_collection("road", {})
        # Weather and traffic data are not yet available.
        self.weather_data = []  # Alternatively, await self.query_collection("weather", {}) if needed
        self.traffic_data = []  # Alternatively, await self.query_collection("traffic", {}) if needed

    @staticmethod
    def process_road_data(documents: List[dict]) -> gpd.GeoDataFrame:
        """
        Process road collection documents into a GeoDataFrame.

        :param documents: List of road documents.
        :return: GeoDataFrame containing road data.
        """
        if not documents:
            logging.warning("No road documents to process.")
            return gpd.GeoDataFrame()
        df = pd.DataFrame(documents)
        # Convert GeoJSON geometry to shapely objects
        df['geometry'] = df['geometry'].apply(lambda geom: shape(geom))
        geo_df = gpd.GeoDataFrame(df, geometry='geometry')
        logging.info("Processed road data into a GeoDataFrame.")
        return geo_df

    def merge_data(self) -> None:
        """
        Merge the processed road, weather, and traffic data into a single GeoDataFrame.
        Since weather and traffic data are not available, this method returns only the road data.
        """
        # Process the road data into a GeoDataFrame
        road_gdf = self.process_road_data(self.road_data) if self.road_data else gpd.GeoDataFrame()

        # TODO 1: merge with weather and traffic data when they will be available.
        """
        # Example merge: if a common key 'road_id' exists, merge the additional data.
        if 'road_id' in road_gdf.columns:
            if not weather_df.empty and 'road_id' in weather_df.columns:
                road_gdf = road_gdf.merge(weather_df, on="road_id", how="left", suffixes=("", "_weather"))
                logging.info("Merged weather data with road GeoDataFrame.")
            if not traffic_df.empty and 'road_id' in traffic_df.columns:
                road_gdf = road_gdf.merge(traffic_df, on="road_id", how="left", suffixes=("", "_traffic"))
                logging.info("Merged traffic data with road GeoDataFrame.")
        else:
            logging.warning("Common key 'road_id' not found in road data; merge skipped.")
        """

        # TODO 2: compute overall cost (try both simple and complex methods, i.e. PCA or NN).
        """
        # --- Compute dynamic cost based on average speed ---
        # For example, if the average speed is low, the dynamic cost is higher.
        def compute_dynamic_cost(avg_speed):
            # Avoid division by zero; adjust the formula as needed.
            return 50 / avg_speed if avg_speed > 0 else 1000


        if not df_traffic.empty:
            df_traffic["dynamic_cost"] = df_traffic["avg_speed"].apply(compute_dynamic_cost)
        else:
            df_traffic["dynamic_cost"] = 0

        # --- Merge static and dynamic data to compute combined weights ---
        # Here, we assume the static cost is provided by the 'leng' attribute.
        df_merged = pd.merge(df_roads, df_traffic[["road_id", "dynamic_cost"]], on="road_id", how="left")
        df_merged["dynamic_cost"] = df_merged["dynamic_cost"].fillna(0)
        df_merged["combined_weight"] = df_merged["leng"] + df_merged["dynamic_cost"]

        print("Computed combined weights for each road segment.")
        print(df_merged[["road_id", "leng", "dynamic_cost", "combined_weight"]].head())
        """

        self.geo_df = road_gdf
        logging.info("Merged data: using only road data.")

    def get_geo_dataframe(self) -> Optional[gpd.GeoDataFrame]:
        """
        Return the merged GeoDataFrame.
        """
        return self.geo_df
