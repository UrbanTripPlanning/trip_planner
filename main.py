"""
This module serves as the entry point for the route planning application.

Workflow:
1. Read environment variables and set up logging and output directories.
2. Initialize the RoadNetwork instance asynchronously, which loads road and traffic data from the database, processes it, and builds a NetworkX graph.
3. Instantiate a RoutePlanner for different transport modes (e.g., walking and driving).
   - For Foot mode, the cost is based on route length.
   - For Car mode, the cost is based on travel time (using the 'car_travel_time' field).
4. Compute routes between specified source and target points.
5. Display route statistics (including computed start and end times if time constraints are provided) and plot the routes.

The optional time parameters allow planning for a future departure (start_time), a deadline arrival (end_time), or immediate departure when both are None.
"""

import asyncio
from utils import read_env, setup_logger_and_check_folders
read_env()
setup_logger_and_check_folders()
from datetime import datetime
from modules.road_network import RoadNetwork
from modules.routing import RoutePlanner, TransportMode


async def main():

    # Settings
    algorithm = 'A*'  # 'A*' or 'Dijkstra'
    GCN = False  # Compute weights with GCN
    start_time = datetime(2025, 5, 13, 13, 0, 0)  # Planned departure datetime; None means immediate departure.
    end_time = None  # Desired arrival datetime; None means no deadline.
    source_point = (7.705189, 45.068828)  # Departure point (longitude, latitude)
    target_point = (7.657668, 45.065126)  # Arrival point (longitude, latitude)

    # Initialize the road network (loads data and builds the graph).
    network = RoadNetwork(GCN)
    await network.async_init(start_time, end_time)

    # Compute and plot the walking route.
    walking_planner = RoutePlanner(network, transport_mode=TransportMode.FOOT, algorithm=algorithm)
    walking_path, _ = walking_planner.compute(source_point, target_point)
    walking_planner.display_statistics(walking_path, start_time, end_time)
    walking_planner.plot_path(walking_path, path_color="blue")

    # Compute and plot the driving route.
    driving_planner = RoutePlanner(network, transport_mode=TransportMode.CAR, algorithm=algorithm, GCN=GCN)
    driving_path, _ = driving_planner.compute(source_point, target_point)
    driving_planner.display_statistics(driving_path, start_time, end_time)
    driving_planner.plot_path(driving_path, path_color="red")


if __name__ == "__main__":
    asyncio.run(main())
