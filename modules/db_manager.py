from motor.motor_asyncio import AsyncIOMotorClient
import logging
import os


class Database:
    URI = os.getenv('DB_URI')
    NAME = os.getenv('DB_NAME')
    DATABASE = None

    @staticmethod
    async def initialize():
        """
        Connect to MongoDB asynchronously and initialize the DATABASE attribute.
        """
        client = AsyncIOMotorClient(Database.URI)
        Database.DATABASE = client[Database.NAME]
        logging.info(f"Connected to MongoDB database: {Database.NAME}")

    @staticmethod
    async def insert(collection: str, data: dict):
        """
        Insert a single document into the specified collection asynchronously.

        :param collection: Name of the collection.
        :param data: Dictionary representing the document to insert.
        :return: The result of the insert_one operation.
        """
        result = await Database.DATABASE[collection].insert_one(data)
        logging.info(f"Inserted document with _id: {result.inserted_id}")
        return result

    @staticmethod
    async def find(collection: str, query: dict):
        """
        Find documents in the specified collection that match the query asynchronously.

        :param collection: Name of the collection.
        :param query: Dictionary representing the query.
        :return: A list of documents that match the query.
        """
        cursor = Database.DATABASE[collection].find(query)
        documents = await cursor.to_list(length=None)
        return documents

    @staticmethod
    async def find_one(collection: str, query: dict):
        """
        Find a single document in the specified collection that matches the query asynchronously.

        :param collection: Name of the collection.
        :param query: Dictionary representing the query.
        :return: A single document or None if no match is found.
        """
        document = await Database.DATABASE[collection].find_one(query)
        return document

    @staticmethod
    async def aggregate(collection: str, pipeline: list):
        """
        Run an aggregation pipeline on the specified collection asynchronously.

        :param collection: Name of the collection.
        :param pipeline: List of aggregation pipeline stages.
        :return: A list of documents resulting from the aggregation.
        """
        cursor = Database.DATABASE[collection].aggregate(pipeline)
        documents = await cursor.to_list(length=None)
        return documents
