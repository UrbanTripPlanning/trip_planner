# Trip Planner

A Python-based route planning application that lets you compute and visualize optimal routes for cars, bikes, or pedestrians. It supports classic shortest-path algorithms (A\* and Dijkstra) and can optionally leverage machine learning models (GCN or LSTM) to learn travel time weights for car routing.

---

## Features

- **Graph Algorithms**: A*, Dijkstra
- **Transport Modes**: Car (with optional GCN/LSTM weighting), Bike, Foot
- **Machine Learning Models**:  
  - **GCN**: Graph Convolutional Network  
  - **LSTM**: Long Short-Term Memory autoencoder  
- **Visualization**: Plot routes on a map
- **Statistics**: Compute and display travel time, distance, and other metrics

---

## Prerequisites

- Python 3.12
- MongoDB instance (for road network storage)
- (Optional) GPU for faster model inference/training

---

## Installation

1. **Clone or extract the repository**  
   ```bash
   git clone https://your.repo.url/trip_planner.git
   cd trip_planner

---

2. **Create & activate your virtual environment**  
   ```bash
   python3 -m venv venv
   source venv/bin/activate    # Linux/macOS
   venv\Scripts\activate       # Windows
   
---

3. **Install required packages**  
   ```bash
   pip install --upgrade pip
   pip install -r requirements.txt

---

## Configuration

1. **Copy the example environment file and edit it**  
   ```bash
   cp .env.example .env

---

2. **Set your MongoDB connection & collections in .env**  
   ```ini
   DB_URI="mongodb://localhost:27017"
   DB_NAME="your_database"
   GRAPH_COLLECTION="your_graph_collection"
   WEATHER_COLLECTION="your_weather_collection"
   OUTPUT_PATH="./output"

---

## Usage

1. **Prepare your road network**  
- Store your traffic and weather data into MongoDB.
- The modules/road_data_processor.py will ingest and process them.

---

2. **Configure the run parameters**  
Open *main.py* and adjust:
   ```python
    algorithm      = Algorithm.DIJKTRA.name()   # Algorithm.DIJKTRA, or A
    gnn_model      = Model.GCN                  # Model.GCN, LSTM, or SIMPLE
    transport_mode = TransportMode.CAR          # TransportMode.FOOT, BIKE, or CAR
    use_gnn        = gnn_model.use_gnn          # bool value if chosen model use neural network
    start_time     = None                       # e.g. datetime(2025, 9, 1,  8, 30)
    end_time       = None                       # e.g. datetime(2025, 9, 1, 10, 30)
    source_point   = (lon, lat)                 # e.g. (7.705189, 45.068828)
    target_point   = (lon, lat)                 # e.g. (7.657668, 45.065126)

---

3. **Run the application**  
   ```bash
   python main.py

---

The script will:

1. Load environment variables and initialize logging/output folders

2. Build or load the road network graph

3. Compute the optimal path

4. Plot the route and print statistics

5. Output (maps, statistics) will be saved under the ./output folder.

## Project Structure

.  
├── .env.example  
├── main.py              # Entry point: customize parameters & run  
├── requirements.txt     # Python dependencies  
├── modules/  
│   ├── db_manager.py    # MongoDB interactions  
│   ├── road_data_processor.py  # raw data → processed data  
│   ├── road_network.py  # processed data → graph  
│   └── routing.py       # solution provider  
├── GCN/                  # Graph Convolutional Network implementation  
│   ├── autoencoder.py  
│   ├── dataset.py  
│   ├── inference.py  
│   └── train.py  
├── LSTM/                 # LSTM autoencoder implementation  
│   ├── lstm_autoencoder.py  
│   ├── dataset.py  
│   ├── inference.py  
│   └── train.py  
└── utils.py             # Helpers (env loader, logger, etc.)  
