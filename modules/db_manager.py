import logging
import os
from typing import Optional, Dict, List, Any

from motor.motor_asyncio import AsyncIOMotorClient


class Database:
    """
    Asynchronous MongoDB interface using Motor.
    Provides methods for CRUD operations and aggregation pipelines.
    """

    URI: str = os.getenv("DB_URI", "")
    NAME: str = os.getenv("DB_NAME", "")
    DATABASE = None

    @staticmethod
    async def initialize() -> None:
        """
        Establish a connection to MongoDB and store the database object.
        """
        if not Database.URI or not Database.NAME:
            raise EnvironmentError("DB_URI and/or DB_NAME environment variables are not set.")

        client = AsyncIOMotorClient(Database.URI)
        Database.DATABASE = client[Database.NAME]
        logging.info(f"Connected to MongoDB database: {Database.NAME}")

    @staticmethod
    async def insert(collection: str, data: Dict[str, Any]) -> Any:
        """
        Insert a single document into the specified collection.

        :param collection: MongoDB's collection name.
        :param data: Document to be inserted.
        :return: The result of the insert operation.
        """
        result = await Database.DATABASE[collection].insert_one(data)
        logging.info(f"Inserted document into '{collection}' with _id: {result.inserted_id}")
        return result

    @staticmethod
    async def find(collection: str, query: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Find all documents matching the query.

        :param collection: MongoDB's collection name.
        :param query: Filter query.
        :return: List of matching documents.
        """
        cursor = Database.DATABASE[collection].find(query)
        return await cursor.to_list(length=None)

    @staticmethod
    async def find_one(collection: str, query: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Find a single document matching the query.

        :param collection: MongoDB's collection name.
        :param query: Filter query.
        :return: The matching document or None.
        """
        return await Database.DATABASE[collection].find_one(query)

    @staticmethod
    async def aggregate(collection: str, pipeline: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Execute an aggregation pipeline on the specified collection.

        :param collection: MongoDB's collection name.
        :param pipeline: Aggregation pipeline stages.
        :return: Aggregated documents as a list.
        """
        cursor = Database.DATABASE[collection].aggregate(pipeline)
        return await cursor.to_list(length=None)

    @staticmethod
    async def query(
        collection: str,
        query: Optional[Dict[str, Any]] = None,
        method: str = "find",
        pipeline: Optional[List[Dict[str, Any]]] = None
    ) -> List[Dict[str, Any]]:
        """
        Unified query interface supporting 'find', 'find_one', and 'aggregate'.

        :param collection: MongoDB's collection name.
        :param query: Filter query for find/find_one.
        :param method: One of 'find', 'find_one', or 'aggregate'.
        :param pipeline: Aggregation pipeline (used for 'aggregate').
        :return: A list of documents matching the operation.
        """
        if method == "find":
            query = query or {}
            docs = await Database.find(collection, query)
        elif method == "find_one":
            query = query or {}
            doc = await Database.find_one(collection, query)
            docs = [doc] if doc else []
        elif method == "aggregate":
            docs = await Database.aggregate(collection, pipeline or [])
        else:
            raise ValueError(f"Unsupported query method: '{method}'")

        logging.info(f"{method.upper()} query returned {len(docs)} documents from '{collection}'")
        return docs
