import os
import asyncio
import pandas as pd
from utils import read_env, setup_logger_and_check_folders

read_env()
setup_logger_and_check_folders()
from datetime import datetime
from modules.road_network import RoadNetwork
from GCN.utils import save_graph_snapshot

# Percorso del tuo file CSV
CSV_PATH = "./data/weather.csv"

# Dove salvare gli snapshot
OUT_DIR = "./data/snapshots"
os.makedirs(OUT_DIR, exist_ok=True)


async def save_for_datetime(weather: str, dt: datetime):
    """
    Build network at a given datetime and save a full snapshot.
    """
    net = RoadNetwork()
    await net.async_init(start_time=dt, end_time=None)
    net.build_graph()
    tag = f"{weather}_{dt:%Y%m%dT%H}"  # esempio: 'clear_20250429T14'
    save_graph_snapshot(net.graph, OUT_DIR, tag=tag)


async def main():
    # Carica il CSV
    df = pd.read_csv(CSV_PATH)
    df['datetime'] = pd.to_datetime(df['datetime'])

    for idx, row in df.iterrows():
        dt = row['datetime']
        weather = row['weather_condition']
        print(f"Saving snapshot for {weather} at {dt}")
        await save_for_datetime(weather, dt)


if __name__ == "__main__":
    asyncio.run(main())
