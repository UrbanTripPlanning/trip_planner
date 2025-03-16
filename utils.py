import os
import logging
from datetime import datetime
from dotenv import load_dotenv


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
