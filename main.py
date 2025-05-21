"""
Entry point for the route planning application.

Workflow:
1. Load environment and check output folders.
2. Initialize RoadNetwork with or without GNN.
3. Instantiate RoutePlanner for a selected TransportMode.
4. Compute and plot optimal route.
5. Display route statistics (optional start/end time).
"""

import asyncio
from datetime import datetime
from modules.utils import read_env, setup_logger_and_check_folders
read_env()
setup_logger_and_check_folders()
from modules.road_network import RoadNetwork, Model
from modules.routing import RoutePlanner, TransportMode, Algorithm


async def main():
    start = datetime.now()

    # ========== SETTINGS ==========
    algorithm = Algorithm.A.name()       # or 'Dijkstra'
    transport_mode = TransportMode.CAR  # FOOT, BIKE, or CAR
    gnn_model = Model.GCN if transport_mode == TransportMode.CAR else Model.SIMPLE  # or 'LSTM' OR 'SIMPLE'
    use_gnn = gnn_model.use_gnn  # Enable weights computation for CAR only
    start_time = None    # datetime(2025, 9, 1, 00, 30)
    end_time = None      # datetime(2025, 5, 1, 9, 0)
    source_point = (7.705189, 45.068828)   # Departure (lon, lat)
    target_point = (7.657668, 45.065126)   # Arrival (lon, lat)
    # ==============================

    # Step 1: Load road network and build graph
    network = RoadNetwork(gnn_model)
    await network.async_init(start_time, end_time)

    # Step 2: Plan route
    planner = RoutePlanner(
        network=network,
        transport_mode=transport_mode,
        algorithm_name=algorithm,
        use_gnn=use_gnn if transport_mode == TransportMode.CAR else False
    )

    path, stats = planner.compute(
        source_point=source_point,
        target_point=target_point,
        start_time=start_time,
        end_time=end_time
    )

    if path:
        planner.plot_path(path, path_color="blue" if transport_mode != TransportMode.CAR else "red")
        planner.display_statistics()
    else:
        print("No route found.")

    print(f"Finished in {datetime.now() - start}")


if __name__ == "__main__":
    asyncio.run(main())
