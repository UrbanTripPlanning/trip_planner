# Trip Planner

Core algorithm for the TurinGO urban route planning project.
A Python-based route planning application that lets you compute and visualize optimal routes for cars, bikes, or pedestrians. It supports classic shortest-path algorithms (A\* and Dijkstra) and can optionally leverage machine learning models (GCN or LSTM) to learn travel time weights for car routing.

---

> **Repository scope**
>
> This repository contains the **route-planning core** of a larger academic project.  
> It focuses on graph construction, route computation, and experimental ML-based edge weighting.  
> It does **not** include the full product stack. You can find the frontend in [here](https://github.com/UrbanTripPlanning/turin_go_frontend) and backend [here](https://github.com/UrbanTripPlanning/turin_go_backend).

## Overview

This project computes urban routes for **car**, **bike**, and **foot** travel on a preprocessed road network stored in MongoDB.

The codebase supports:

- directed road-network construction from road-segment data
- shortest-path search with **A\*** and **Dijkstra**
- basic weather-aware traffic field selection for car routing
- experimental learned edge weighting through:
  - a **Graph Convolutional Network (GCN)**
  - an **LSTM autoencoder**
- static route visualization and summary statistics export

The repository is best understood as an **algorithmic/research backend prototype**, not a complete deployable navigation platform.

## Implemented capabilities

- Build a directed NetworkX road graph from MongoDB records
- Select traffic fields based on time and weather context
- Compute routes between two coordinates
- Support three travel modes:
  - **Car**
  - **Bike**
  - **Foot**
- Use either:
  - raw travel time
  - experimental ML-derived edge weights
- Plot the resulting path and save it to disk
- Print route statistics including:
  - total length
  - total duration
  - inferred start/end timestamps


## Method summary

### Graph representation

Road segments are loaded from MongoDB and converted into a directed NetworkX graph.  
Each edge stores route-relevant attributes such as:

- `length`
- `speed`
- `time`
- `geometry`

Each node stores its spatial position for point snapping and path search.

### Routing

- **Car** mode uses directed routing and minimizes either:
  - raw edge `time`, or
  - experimental learned `weight`
- **Bike** and **Foot** use an undirected view of the graph and minimize physical `length`
- A fixed intersection-delay model is added to car-mode duration estimates

### Experimental ML weighting

Two model families are included:

- **GCN**: graph-based travel-time prediction on a line-graph representation
- **LSTM**: sequence-based embedding of edge-level temporal features


## Input data expectations

### MongoDB traffic / graph collection

The runtime path expects a graph collection with fields consistent with the code path in `modules/road_data_processor.py`, including:

- `road_id`
- `tail`
- `head`
- `length`
- `geometry`
- `hour`
- `week`
- `month`
- `speed_clear`
- `speed_rain`
- `time_clear`
- `time_rain`

`geometry` is expected to be GeoJSON-like and convertible with `shapely.geometry.shape`.

### MongoDB weather collection

The weather collection is expected to expose at least:

- `date`
- `hour`
- `condition`

The current implementation uses the weather record matching the chosen reference date/hour and derives a simple rain flag from `condition`.



## Prerequisites

- Python 3.12
- MongoDB instance (for road network storage)
- pip / virtualenv
- (Optional) GPU for faster model inference/training

---

## Installation

### Setup

1. **Clone or extract the repository**  
   ```bash
   git clone git@github.com:UrbanTripPlanning/trip_planner.git
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

```text
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
```
