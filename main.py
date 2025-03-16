from utils import read_env, setup_logger_and_check_folders
read_env()
setup_logger_and_check_folders()

import asyncio
import logging

from modules.routing import RoadNetwork


async def main():

    COLLECTION_NAME = "road"
    network = RoadNetwork(COLLECTION_NAME)
    await network.async_init()

    # --- Define source and target points ---
    source_point = (7.705189, 45.068828)
    target_point = (7.657668, 45.065126)

    logging.info(f"Source point: {source_point}")
    logging.info(f"Target point: {target_point}")

    # --- Compute shortest paths ---
    dijkstra_path, dijkstra_time = network.compute_dijkstra(source_point, target_point)
    astar_path, astar_time = network.compute_astar(source_point, target_point)

    # --- Plot and save the results ---
    network.plot_path(dijkstra_path, "Shortest Path with Dijkstra", "graph_with_dijkstra.png", "red")
    network.plot_path(astar_path, "Shortest Path with A*", "graph_with_astar.png", "blue")


if __name__ == "__main__":
    asyncio.run(main())
