import os
import math
import logging
from datetime import datetime
from dotenv import load_dotenv
from functools import lru_cache
from typing import Tuple


def setup_logger_and_check_folders():
    # Create output directory if it doesn't exist and configure logging
    os.makedirs(os.getenv("OUTPUT_PATH"), exist_ok=True)
    os.environ[
        "CURRENT_OUT_PATH"] = f"{os.getenv('OUTPUT_PATH')}/{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"
    os.makedirs(os.getenv("CURRENT_OUT_PATH"))  # Create the output path for the current run
    logging.basicConfig(filename=os.path.join(os.getenv("CURRENT_OUT_PATH"), "logs.log"),
                        level=logging.INFO,
                        format="%(asctime)s - %(levelname)s: %(message)s")


def read_env():
    try:
        load_dotenv()
    except FileNotFoundError as e:
        raise FileNotFoundError(
            f"Error with reading the '.env' file in the root folder: {e}")  # Handle file not found specifically
    except Exception as e:
        print(f"An unexpected error occurred: {e}")  # Catch other exceptions


def get_time_info(time: datetime = None):
    if time is None:
        time = datetime.now()
    current_hour = (time.hour + (1 if time.minute >= 50 else 0)) % 24
    weekday = (time.isoweekday())  # 1 = Monday, 7 = Sunday
    month = time.month  # 1 = January, 12 = December
    return current_hour, weekday, month


@lru_cache(maxsize=None)
def euclidean_distance(p1: Tuple[float, float], p2: Tuple[float, float]) -> float:
    """
    Calculate the Euclidean distance between two points.

    :param p1: Tuple (x, y) representing the first point.
    :param p2: Tuple (x, y) representing the second point.
    :return: Euclidean distance.
    """
    return math.hypot(p1[0] - p2[0], p1[1] - p2[1])


def format_dt(dt: datetime) -> str:
    """
    Format datetime to string for statistics.
    """
    return dt.strftime('%Y-%m-%d %H:%M:%S')
